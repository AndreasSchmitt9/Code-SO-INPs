"""
One figure with four subplots:
  (a)- (c) are comparisons between Measured INP concentration against predicted INP concentrations from parameterizations by 
  McCluskey2018, Moore 2024 and Vignon 2021 for Sea Spray Aerosol/Southern Ocean aerosol. 
  (d) shows INAS density (INP concentration/surface area concentration) of CAPE-k INSEKT, PINE and CSU-IS data against parameterizations,
  this figure includes the new fit, N12 dust parameterization and W25 antarctic parameterization
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import glob
import xarray as xr

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

sns.set_palette("tab10")

DEFAULT_WIND_SPEED = 7.0          # m/s for M24 parameterization
MAX_TEMPERATURE   = 251.15        # K (applied only to PINE)
DEFAULT_SA        = 3.30e7        # Default surface area (µm²/cm³) 
SA_MATCH_HOURS    = 1             # ±1 hour window for PINE
FILTER_SA_HOURS   = 24            # ±12 hours averaging window for INSEKT & CSU


def import_data(file_path):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')
    df = df.dropna(subset=['datetime'])
    return df


def import_INSEKT_data(data_folder, metadata_file):
    metadata_df = pd.read_csv(metadata_file, header=None)
    metadata_df = metadata_df.rename(columns={
        1: "insekt_id", 8: "exp_name",
        metadata_df.shape[1] - 2: "sampling_start",
        metadata_df.shape[1] - 1: "sampling_end"
    })
    metadata_df["sampling_start"] = pd.to_datetime(metadata_df["sampling_start"], format='mixed', errors='coerce')

    heat_ids = set(metadata_df.loc[metadata_df["exp_name"].str.contains("_heat", na=False), "insekt_id"].astype(str))
    blank_ids = set(metadata_df.loc[metadata_df["exp_name"].str.contains("Blank", na=False), "insekt_id"].astype(str))
    actris_ids = set(metadata_df.loc[metadata_df["exp_name"].str.contains("ACTRIS", na=False), "insekt_id"].astype(str))
    all_excluded_ids = heat_ids.union(blank_ids).union(actris_ids)

    cutoff_date = datetime(2025, 2, 15)
    valid_meta = metadata_df[(metadata_df["sampling_start"] >= cutoff_date) &
                             (~metadata_df["insekt_id"].astype(str).isin(all_excluded_ids))]
    valid_ids = set(valid_meta["insekt_id"].astype(str))

    corrected_files = glob.glob(os.path.join(data_folder, "*_corrected.csv"))
    INSEKT_data = []
    for file in corrected_files:
        fname = os.path.basename(file)
        parts = fname.split('_')
        if len(parts) < 2:
            continue
        insekt_id = parts[1]
        if insekt_id not in valid_ids:
            continue
        sampling_time = valid_meta.loc[valid_meta["insekt_id"].astype(str) == insekt_id, "sampling_start"].iloc[0]
        exp_name = valid_meta.loc[valid_meta["insekt_id"].astype(str) == insekt_id, "exp_name"].iloc[0]
        df_temp = pd.read_csv(file)
        df_temp = df_temp.drop(df_temp.columns[:2], axis=1)
        df_temp["T_min"] = df_temp["T_mean / K"]
        df_temp["INP_cn_0"] = df_temp["c_INP / 1/l"]
        df_temp["experiment_id"] = insekt_id
        df_temp["exp_name"] = exp_name
        df_temp["sampling_time"] = sampling_time
        df_temp["data_source"] = "INSEKT"
        INSEKT_data.append(df_temp)

    return pd.concat(INSEKT_data, ignore_index=True)

def import_CSU_data(file_path):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['start_utc'], unit='s', errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['T_min'] = df['temperature'] + 273.15
    df['INP_cn_0'] = df['n_inp_stp']
    df['data_source'] = 'CSU-IS'
    return df[['datetime', 'T_min', 'INP_cn_0', 'data_source']].copy()


def process_SA_concentration_data(file_path):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    for col in ['SMPS_SA', 'APS_SA', 'total_SA_conc']:
        if col in df.columns:
            df[col] *= 1e6       # m²/m³ → µm²/cm³
    dfc = df[['datetime', 'total_SA_conc', 'SMPS_SA', 'APS_SA']].dropna(subset=['datetime', 'total_SA_conc'])
    dfc = dfc[dfc['total_SA_conc'] > 0].sort_values('datetime').reset_index(drop=True)
    sa_hourly = dfc
    sa_daily = dfc.set_index('datetime').resample('D').mean().reset_index()
    sa_daily['date'] = sa_daily['datetime'].dt.date
    return {'10min': sa_hourly[['datetime', 'total_SA_conc']], 'daily': sa_daily, 'raw': dfc}


def process_MET_data(met_folder):
    cdf_files = glob.glob(os.path.join(met_folder, "*.cdf"))
    all_met = []
    for file_path in cdf_files:
        ds = xr.open_dataset(file_path)
        df_met = ds[['time', 'wspd_vec_mean']].to_dataframe().reset_index()
        df_met = df_met.rename(columns={'time': 'datetime', 'wspd_vec_mean': 'wind_speed'})
        df_met = df_met[df_met['wind_speed'].notna() & (df_met['wind_speed'] > 0)]
        all_met.append(df_met)
        ds.close()

    met_combined = pd.concat(all_met, ignore_index=True).sort_values('datetime')
    met_combined['datetime'] = pd.to_datetime(met_combined['datetime'])

    met_10min = met_combined.set_index('datetime').resample('10min')['wind_speed'].mean().reset_index()
    met_10min = met_10min[met_10min['wind_speed'].notna()]

    met_10min['date'] = met_10min['datetime'].dt.date
    daily_met = met_10min.groupby('date')['wind_speed'].mean().reset_index()
    daily_met['datetime'] = pd.to_datetime(daily_met['date'].astype(str) + ' 12:00:00')

    return {'daily': daily_met, 'raw': met_10min}


def hourly_average_pine_data(df_pine):
    pine_df = df_pine.copy()
    hourly_stats = pine_df.resample('1h', on='datetime').agg({
        'T_min': 'mean',
        'INP_cn_0': ['mean', 'std', 'count', 'min', 'max']
    }).reset_index()
    hourly_stats.columns = ['datetime', 'T_min', 'INP_mean', 'INP_std', 'INP_count', 'INP_min', 'INP_max']
    hourly_stats = hourly_stats[hourly_stats['T_min'] < 251.15]
    hourly_stats['INP_cn_0'] = hourly_stats['INP_mean']
    hourly_stats['data_source'] = 'PINE_HOURLY'
    return hourly_stats[['datetime', 'T_min', 'INP_cn_0', 'INP_std', 'INP_count', 'data_source']]


def match_pine_sa(pine_df, sa_raw):
    df = pine_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce').dropna()
    sa = sa_raw.set_index('datetime').sort_index()
    matched = df.copy()
    matched['total_SA_conc'] = np.nan
    for idx, row in df.iterrows():
        t_inp = row['datetime']
        t_min = t_inp - timedelta(hours=SA_MATCH_HOURS)
        t_max = t_inp + timedelta(hours=SA_MATCH_HOURS)
        mask = (sa.index >= t_min) & (sa.index <= t_max)
        if mask.any():
            candidates = sa.loc[mask].copy()
            candidates['time_diff'] = abs(candidates.index - t_inp)
            best = candidates.loc[candidates['time_diff'].idxmin()]
            matched.loc[idx, 'total_SA_conc'] = best['total_SA_conc']
    matched['total_SA_conc'] = matched['total_SA_conc'].fillna(DEFAULT_SA)
    high_sa = matched['total_SA_conc'] > 1e9
    if high_sa.any():
        matched.loc[high_sa, 'total_SA_conc'] = DEFAULT_SA
    return matched


def match_filter_sa_24h(filter_df, sa_raw):
    df = filter_df.copy()
    if 'sampling_time' in df.columns and df['sampling_time'].notna().any():
        df['datetime'] = pd.to_datetime(df['sampling_time'], errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    sa = sa_raw.set_index('datetime').sort_index()
    matched = df.copy()
    matched['total_SA_conc'] = np.nan
    for idx, row in df.iterrows():
        t_center = row['datetime']
        t_start = t_center - timedelta(hours=FILTER_SA_HOURS/2)
        t_end   = t_center + timedelta(hours=FILTER_SA_HOURS/2)
        mask = (sa.index >= t_start) & (sa.index <= t_end)
        if mask.any():
            matched.loc[idx, 'total_SA_conc'] = sa.loc[mask, 'total_SA_conc'].mean()
    matched['total_SA_conc'] = matched['total_SA_conc'].fillna(DEFAULT_SA)

    return matched


def match_all_with_SA(df_inp, df_sa, df_met):
    pine = df_inp[df_inp['data_source'] == 'PINE_HOURLY'][['datetime','T_min','INP_cn_0','INP_std','INP_count','data_source']]
    insekt = df_inp[df_inp['data_source'] == 'INSEKT'][['T_min','INP_cn_0','sampling_time','data_source']]
    csu = df_inp[df_inp['data_source'] == 'CSU-IS'][['datetime','T_min','INP_cn_0','data_source']]

    sa_raw = df_sa['raw']

    pine_m   = match_pine_sa(pine, sa_raw)
    insekt_m = match_filter_sa_24h(insekt, sa_raw)
    csu_m    = match_filter_sa_24h(csu, sa_raw)

    df_m = pd.concat([pine_m, insekt_m, csu_m], ignore_index=True)

    df_m['date'] = df_m['datetime'].dt.date
    df_m = pd.merge(df_m, df_met['daily'][['date','wind_speed']], on='date', how='left')
    df_m['wind_speed'] = df_m['wind_speed'].fillna(DEFAULT_WIND_SPEED)
    df_m = df_m.drop(columns=['date'])
    if 'sampling_time' in df_m.columns:
        df_m = df_m.drop(columns=['sampling_time'])
    return df_m


def filter_by_temperature(df, max_temp=251.15):
    df_pine = df[df['data_source'] == 'PINE_HOURLY'].copy()
    df_other = df[~df['data_source'].isin(['PINE_HOURLY'])].copy()
    if len(df_pine) > 0:
        df_pine = df_pine[df_pine['T_min'] <= max_temp]
        df_pine = df_pine[df_pine['INP_cn_0'].notna()]
    return pd.concat([df_pine, df_other], ignore_index=True)


def calculate_parametrizations_with_APS(df):
    def parametrization_1(T_min, SSA): # New fit function
        a, b = -0.545, 1.0125
        SSA_used = SSA * 1e-12
        ΔT = T_min - 273.16
        n_inp = np.exp(a * ΔT + b) * SSA_used
        return n_inp * 1e-3

    def moore_curve_with_wind(T_min, SSA, wind_speed_series=None): # Moore 2024 
        MO_A, MO_B, MO_C, MO_D = -0.66, -3.11, 0.51, 6.75
        SSA_used = SSA * 1e-12
        T_C = T_min - 273.15
        windSpeed = np.where(pd.notna(wind_speed_series), wind_speed_series, DEFAULT_WIND_SPEED)
        n_inp = (np.exp(MO_A * T_C + MO_B) + np.exp(MO_C * windSpeed + MO_D)) * SSA_used
        return n_inp * 1e-3

    def vignon_2021_parametrization(T_min): # Vignon 2021 
        T_C = T_min - 273.15
        t1, t2 = -21.06, -30.35
        return np.where(T_C > t1,
                        10**(-0.14 * (T_C - t1) - 2.88),
                        np.where(T_C >= t2,
                                 10**(-0.31 * (T_C - t1) - 2.88),
                                 1.0))

    df['predicted_INP_1'] = parametrization_1(df['T_min'], df['total_SA_conc'])
    df['predicted_INP_2'] = moore_curve_with_wind(df['T_min'], df['total_SA_conc'], df.get('wind_speed', None))
    df['predicted_INP_3'] = vignon_2021_parametrization(df['T_min'])
    return df


def create_comparison_plots(df, max_temp=255.0):
    df_clean = df.dropna(subset=['INP_cn_0'] + [f'predicted_INP_{i}' for i in range(1, 4)])

    fig = plt.figure(figsize=(7, 6))
    gs = plt.GridSpec(2, 2, figure=fig, wspace=0.4, hspace=-0.2,
                      top=1, bottom=0, left=0.1, right=0.77,
                      width_ratios=[1, 1], height_ratios=[1, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
    ax_custom = fig.add_subplot(gs[1, 1])
    ax_custom.set_box_aspect(1)
    panel_labels = ['a)', 'b)', 'c)', 'd)']

    for ax, label in zip(axes + [ax_custom], panel_labels):
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va='top', ha='left')

    param_names = ['Mc Cluskey 2018', 'Moore 2024', 'Vignon 2021']
    temp_ranges = {
        'M18': (-27, -10),
        'M24': (-31, -10),
        'V21': (-34, -10),
        'new': (-34, -10),
        'N12': (-36, -12),
        'W25': (-36, -10)
    }
    df_clean['T_C'] = df_clean['T_min'] - 273.15
    df_m18 = df_clean[(df_clean['T_C'] >= temp_ranges['M18'][0]) & (df_clean['T_C'] <= temp_ranges['M18'][1])]
    df_m24 = df_clean[(df_clean['T_C'] >= temp_ranges['M24'][0]) & (df_clean['T_C'] <= temp_ranges['M24'][1])]
    df_v21 = df_clean[(df_clean['T_C'] >= temp_ranges['V21'][0]) & (df_clean['T_C'] <= temp_ranges['V21'][1])]
    df_new = df_clean[(df_clean['T_C'] >= temp_ranges['new'][0]) & (df_clean['T_C'] <= temp_ranges['new'][1])]

    temp_min, temp_max = df_clean['T_C'].min(), df_clean['T_C'].max()
    norm = mcolors.Normalize(vmin=temp_min, vmax=temp_max)
    cmap = plt.cm.viridis
    markers = {'PINE_HOURLY': ('o', 2), 'INSEKT': ('s', 3), 'CSU-IS': ('^', 1)}

    for i, (ax, param_name, filtered_df) in enumerate(zip(axes, param_names, [df_m18, df_m24, df_v21]), 1):
        predicted_col = f'predicted_INP_{i}'
        axis_limits = [1e-5, 1e2]
        for source, (m, z) in markers.items():
            mask = filtered_df['data_source'] == source
            if mask.any():
                ax.scatter(filtered_df.loc[mask, 'INP_cn_0'], filtered_df.loc[mask, predicted_col],
                           c=filtered_df.loc[mask, 'T_C'], cmap=cmap, norm=norm,
                           alpha=1, s=30, marker=m, edgecolors='k', linewidth=0.5, zorder=z+100)
        ax.plot(axis_limits, axis_limits, 'k--', lw=1.5)
        x_range = np.array(axis_limits)
        ax.fill_between(x_range, 0.1*x_range, 10*x_range, color='grey', alpha=0.15)
        ax.plot(x_range, 10*x_range, 'grey', ls=':', lw=0.4, alpha=0.7)
        ax.plot(x_range, 0.1*x_range, 'grey', ls=':', lw=0.4, alpha=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Measured INP (L⁻¹)')
        ax.set_ylabel('Predicted INP (L⁻¹)')
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(param_name)
        ax.tick_params(axis='both', which='minor', width=0, length=0)
        ticks = [1e-4, 1e-2, 1, 1e2]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    df_new['ns'] = (df_new['INP_cn_0'] / df_new['total_SA_conc']) * 1e15
    scatter_for_cbar = None
    for source, (m, z) in markers.items():
        mask = df_new['data_source'] == source
        if mask.any():
            sc = ax_custom.scatter(df_new.loc[mask, 'T_C'], df_new.loc[mask, 'ns'],
                                   c=df_new.loc[mask, 'T_C'], cmap=cmap, norm=norm,
                                   alpha=1, s=30, marker=m, edgecolors='k', linewidth=0.5,
                                   zorder=z)

    T_new = np.linspace(temp_ranges['new'][0], temp_ranges['new'][1], 100)
    ax_custom.plot(T_new, np.exp(-0.617 * T_new + 1.291),
                   color='darkred', lw=2, label='fit function', zorder=10)

    T_m18 = np.linspace(temp_ranges['M18'][0], temp_ranges['M18'][1], 100)
    ax_custom.plot(T_m18, np.exp(-0.545 * T_m18 + 1.0125),
                   color='darkred', lw=2, ls='--', label='M18', zorder=10)

    T_v21 = np.linspace(temp_ranges['V21'][0], temp_ranges['V21'][1], 100)
    t1, t2 = -21.06, -30.35
    vignon = np.where(T_v21 > t1, 10**(-0.14*(T_v21 - t1) - 2.88),
                      np.where(T_v21 >= t2, 10**(-0.31*(T_v21 - t1) - 2.88), 1.0))
    ax_custom.plot(T_v21, vignon / DEFAULT_SA * 1e15,
                   color='darkred', lw=2, ls=':', label='V21', zorder=10)

    T_m24 = np.linspace(temp_ranges['M24'][0], temp_ranges['M24'][1], 100)
    m24_ns = np.exp(-0.66*T_m24 - 3.11) + np.exp(0.51*7.0 + 6.75)
    ax_custom.plot(T_m24, m24_ns, color='darkred', lw=2, ls=(0, (10, 1, 1, 1)), label='M24 (7 m/s)', zorder=10)

    T_N12 = np.linspace(temp_ranges['N12'][0], temp_ranges['N12'][1], 100)
    ax_custom.plot(T_N12, np.exp(-0.517 * T_N12 + 8.934),
                   color='darkorange', lw=2, linestyle='--', label='N12', zorder=10)

    T_W25 = np.linspace(temp_ranges['W25'][0], temp_ranges['W25'][1], 100)
    w25_conc = np.exp(-0.636 * T_W25 - 15.744) * 1e-3
    ax_custom.plot(T_W25, w25_conc / DEFAULT_SA * 1e15,
                   color='darkblue', lw=2, linestyle='--', label='W25', zorder=10)

    ax_custom.set_yscale('log')
    ax_custom.set_xlabel('Temperature (°C)')
    ax_custom.set_ylabel('$n_s$ (m⁻²)')
    ax_custom.set_xlim(-35, -4)
    ax_custom.set_ylim(1e3, 1e11)
    ax_custom.legend(
        loc='upper center',
        bbox_to_anchor=(0.88, 0.47),
        bbox_transform=fig.transFigure,
        handlelength=3
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6,
               label='PINE', markeredgecolor='k', markeredgewidth=1),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=6,
               label='INSEKT', markeredgecolor='k', markeredgewidth=1),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=6,
               label='CSU-IS', markeredgecolor='k', markeredgewidth=1),
        Line2D([0], [0], color='k', linestyle='--', lw=1, label='1:1 line'),
        Patch(facecolor='grey', alpha=0.15, label='10x/0.1x area')
    ]
    if scatter_for_cbar is not None:
        cbar_ax = fig.add_axes([0.80, 0.55, 0.015, 0.3])
        cbar = fig.colorbar(scatter_for_cbar, cax=cbar_ax)
        cbar.ax.invert_yaxis()
        cbar.set_label('Temperature (°C)')
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.88, 0.15),
               bbox_transform=fig.transFigure, ncol=1, frameon=True)
    plt.subplots_adjust(left=0.15, right=0.77, top=1, bottom=0, wspace=0.3, hspace=0.1)
    return fig


def main():
    pine_file    = '/mnt/d/Cape24-data/Cape24/processedPINE/RadonBaselineINP/pine_baseline_by_radon.csv'
    insekt_folder = '/mnt/c/Users/ro7338/Documents/Masterarbeit/py_raw_insekt_original/Examples/DATA_Cape24'
    insekt_meta  = '/mnt/c/Users/ro7338/Documents/Masterarbeit/py_raw_insekt_original/Examples/DATA_Cape24/exp_lst_Cape24.csv'
    sa_file      = '/mnt/d/Cape24-data/Model/SizeDistributions/surface_area_concentrations.csv'
    met_folder   = '/mnt/d/Cape24-data/Met'
    csu_file     = '/mnt/d/Cape24-data/CSU-CAPEk/baselineCSU.csv'

    df_sa = process_SA_concentration_data(sa_file)
    df_met = process_MET_data(met_folder)
    df_pine_raw = import_data(pine_file)
    df_insekt_raw = import_INSEKT_data(insekt_folder, insekt_meta)
    df_csu_raw = import_CSU_data(csu_file)

    pine_hourly = hourly_average_pine_data(df_pine_raw)
    df_final = pd.concat([pine_hourly, df_insekt_raw, df_csu_raw], ignore_index=True)

    df_matched = match_all_with_SA(df_final, df_sa, df_met)
    df_filtered = filter_by_temperature(df_matched, MAX_TEMPERATURE)
    df_filtered = calculate_parametrizations_with_APS(df_filtered)

    fig = create_comparison_plots(df_filtered, MAX_TEMPERATURE)
    fig.savefig('Figure6.png')
    fig.savefig('Figure6.pdf')


if __name__ == "__main__":
    main()