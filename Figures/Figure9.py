
"""
Figure 7b : Median and IQR of historic campaigns in the Southern Ocean binned by 1K
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import re
from datetime import datetime
from matplotlib.patches import Rectangle

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

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
MIN_POINTS_FOR_COLOR = 3      # bins with fewer points get special treatment
MAX_TEMP_BIN = -4             # discard data above this temperature
HIDE_LOW_DATA = True         # True  -> remove low‑data bins completely (lines break)
                              # False -> draw them as white‑filled markers

# =============================================================================
# COLOUR PALETTES
# =============================================================================
CAPEK_PINE_COLOR   = '#377eb8'
CAPEK_INSEKT_COLOR = '#e41a1c'
CAPEK_CSU_COLOR    = '#4daf4a'

HISTORIC_PALETTE = [
    '#1b9e77', '#d95f02', '#7570b3', '#e7298a',
    '#66a61e', '#e6ab02', '#a6761d', '#666666'
]
LINESTYLES = ['-', '--', '-.', ':',
              (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)),
              (0, (3, 1, 1, 1, 1, 1))]
DATASET_MARKERS = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'D', 'X', 'o']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def kelvin_to_celsius(k):
    return k - 273.15

def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
        if f"{name}[]" in df.columns:
            return f"{name}[]"
    return None

def process_bigg_data():
    bigg_raw = [
        {'temp': -10, 'inp': 0.00804489}, {'temp': -10, 'inp': 0.001508771},
        {'temp': -15, 'inp': 0.054962642}, {'temp': -15, 'inp': 0.01570258},
        {'temp': -20, 'inp': 0.40643097}, {'temp': -20, 'inp': 0.12970896}
    ]
    df = pd.DataFrame(bigg_raw)
    grouped = df.groupby('temp')['inp'].agg(
        Median=('mean'), Q1=('min'), Q3=('max'), Count=('count')
    ).reset_index().rename(columns={'temp': 'Temp_bin'})
    grouped['IQR'] = grouped['Q3'] - grouped['Q1']
    grouped = grouped[grouped['Temp_bin'] <= MAX_TEMP_BIN]
    return {
        'data': grouped, 'label': 'Bigg 1973', 'color': 'black',
        'linestyle': ':', 'marker': 'o',
    }

def process_pine_hourly_average(file_config):
    try:
        df = pd.read_csv(file_config['path'], delimiter=',')
        temp_col = file_config.get('temp_col')
        inp_col = file_config.get('inp_col')
        if temp_col is None or temp_col not in df.columns:
            temp_col = find_column(df, ['Temp', 'Temperature', 'T', 'CFDC_Temp', 'IS_Temp_C', 'T_min'])
        if inp_col is None or inp_col not in df.columns:
            inp_col = find_column(df, ['N_INP', 'INP', 'CFDC_N_INP', 'IS_N_INP', 'INP_cn_0'])

        df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
        df[inp_col] = pd.to_numeric(df[inp_col], errors='coerce')
        if file_config.get('max_temp_kelvin') is not None:
            df = df[df[temp_col] < file_config['max_temp_kelvin']]
        if file_config.get('convert_kelvin', False):
            df[temp_col] = kelvin_to_celsius(df[temp_col])
        datetime_col = None
        for col in ['datetime', 'DateTime', 'Date', 'Time', 'Timestamp']:
            if col in df.columns:
                datetime_col = col; break
        if datetime_col:
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
            df = df.dropna(subset=[datetime_col])
            df['hourly_bin'] = df[datetime_col].dt.floor('1H')
            hourly = df.groupby('hourly_bin')[[temp_col, inp_col]].mean().reset_index()
            hourly.columns = ['datetime', 'Temp', 'INP']
            working = hourly[['Temp', 'INP']].copy()
        else:
            working = df[[temp_col, inp_col]].copy()
            working.columns = ['Temp', 'INP']
        working = working.dropna()
        if file_config.get('exclude_zeros', False):
            working = working[working['INP'] > 0]
        working['Temp_bin'] = np.floor(working['Temp'])
        grouped = working.groupby('Temp_bin')['INP'].agg(
            Median='median', Q1=lambda x: np.percentile(x, 25),
            Q3=lambda x: np.percentile(x, 75),
            IQR=lambda x: np.percentile(x, 75) - np.percentile(x, 25),
            Count='count'
        ).reset_index()
        grouped = grouped[grouped['Temp_bin'] <= MAX_TEMP_BIN]
        if grouped.empty: return None
        grouped = grouped.sort_values('Temp_bin', ascending=False)
        return {
            'data': grouped, 'label': file_config['label'],
            'color': file_config['color'], 'marker': file_config.get('marker', 'o'),
            'linestyle': file_config.get('linestyle', '-'),
        }
    except Exception:
        return None

def process_insekt_binned(file_config):
    folder = file_config['path']
    exp_list_path = file_config.get('exp_list_path')
    if not exp_list_path or not os.path.exists(exp_list_path):
        return None
    metadata = pd.read_csv(exp_list_path, header=None)
    metadata = metadata.rename(columns={
        1: "insekt_id", 8: "exp_name",
        metadata.shape[1] - 2: "sampling_start",
        metadata.shape[1] - 1: "sampling_end"
    })
    metadata["sampling_start"] = pd.to_datetime(metadata["sampling_start"], errors='coerce')
    metadata["insekt_id"] = pd.to_numeric(metadata["insekt_id"], errors='coerce')
    metadata = metadata.dropna(subset=["insekt_id"])
    metadata["insekt_id"] = metadata["insekt_id"].astype(int)
    heat_ids = set(metadata.loc[metadata["exp_name"].str.contains("_heat", na=False), "insekt_id"])
    blank_ids = set(metadata.loc[metadata["exp_name"].str.contains("Blank", na=False), "insekt_id"])
    actris_ids = set(metadata.loc[metadata["exp_name"].str.contains("ACTRIS", na=False), "insekt_id"])
    excluded = heat_ids.union(blank_ids).union(actris_ids)
    cutoff = datetime(2025, 2, 15)
    valid_meta = metadata[(metadata["sampling_start"] >= cutoff) &
                          (~metadata["insekt_id"].isin(excluded))]
    valid_ids = set(valid_meta["insekt_id"].astype(str))
    all_files = os.listdir(folder)
    corrected = [f for f in all_files if f.endswith('.csv') and 'INSEKT' in f and '_corrected' in f]
    filtered_files = [f for f in corrected
                      if re.search(r'INSEKT_(\d+)_', f) and re.search(r'INSEKT_(\d+)_', f).group(1) in valid_ids]
    if not filtered_files: return None
    all_points = []
    for fname in filtered_files:
        df = pd.read_csv(os.path.join(folder, fname))
        df = df.drop(df.columns[:2], axis=1)
        temp_col = inp_col = None
        for col in ['T_mean / K', 'T_mean', 'Temperature', 'Temp']:
            if col in df.columns: temp_col = col; break
        for col in ['c_INP / 1/l', 'c_INP', 'INP', 'INP_conc']:
            if col in df.columns: inp_col = col; break
        if temp_col is None or inp_col is None: continue
        temp_c = kelvin_to_celsius(pd.to_numeric(df[temp_col], errors='coerce'))
        inp = pd.to_numeric(df[inp_col], errors='coerce')
        valid = ~(temp_c.isna() | inp.isna())
        temp_c = temp_c[valid]
        inp = np.maximum(inp[valid], 1e-6)
        all_points.append(pd.DataFrame({'Temp': temp_c, 'INP': inp}))
    if not all_points: return None
    combined = pd.concat(all_points, ignore_index=True)
    combined['Temp_bin'] = np.floor(combined['Temp'])
    grouped = combined.groupby('Temp_bin')['INP'].agg(
        Median='median', Q1=lambda x: np.percentile(x, 25),
        Q3=lambda x: np.percentile(x, 75),
        IQR=lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        Count='count'
    ).reset_index()
    grouped = grouped[grouped['Temp_bin'] <= MAX_TEMP_BIN]
    if grouped.empty: return None
    grouped = grouped.sort_values('Temp_bin', ascending=False)
    return {
        'data': grouped, 'label': file_config['label'],
        'color': file_config['color'], 'marker': file_config.get('marker', 'X'),
        'linestyle': file_config.get('linestyle', '-'),
    }
# ACE data is processed seperately, as the ship was close to the harbor between the legs, which is not accounted for in the dataset
def process_acespace(file_config):
    try:
        df = pd.read_csv(file_config['path'])
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col; break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            exclude_ranges = [
                ('2016-12-20', '2016-12-24'),
                ('2017-01-20', '2017-01-24'),
                ('2017-02-20', '2017-02-28')
            ]
            mask = pd.Series(True, index=df.index)
            for start, end in exclude_ranges:
                mask = mask & ~((df[date_col] >= start) & (df[date_col] <= end))
            df = df[mask]
            if df.empty: return None
        pairs = []
        for col in df.columns:
            if 'concentration_of_ice_nucleating_particles_at_' in col:
                try:
                    temp = float(col.split('_')[-2].replace('degreeC', ''))
                    for val in df[col]:
                        if pd.notna(val) and val > 0:
                            pairs.append((temp, val * file_config.get('convert_units', 1)))
                except ValueError: continue
        if not pairs: return None
        df_proc = pd.DataFrame(pairs, columns=['Temp', 'INP'])
        df_proc['Temp_bin'] = np.floor(df_proc['Temp'])
        grouped = df_proc.groupby('Temp_bin')['INP'].agg(
            Median='median', Q1=lambda x: np.percentile(x, 25),
            Q3=lambda x: np.percentile(x, 75),
            IQR=lambda x: np.percentile(x, 75) - np.percentile(x, 25),
            Count='count'
        ).reset_index()
        grouped = grouped[grouped['Temp_bin'] <= MAX_TEMP_BIN]
        grouped = grouped.sort_values('Temp_bin', ascending=False)
        return {
            'data': grouped, 'label': file_config['label'],
            'color': file_config['color'], 'marker': file_config.get('marker', 'o'),
            'linestyle': file_config.get('linestyle', '-'),
        }
    except Exception:
        return None

def process_file_historic(file_config):
    if file_config.get('insekt_data', False):
        return process_insekt_binned(file_config)
    if 'PINE' in file_config.get('label', '') and not file_config.get('plot_raw', False):
        return process_pine_hourly_average(file_config)
    if file_config.get('temp_specific_cols', False):
        return process_acespace(file_config)
    try:
        df = pd.read_csv(file_config['path'],
                         delimiter=file_config.get('delimiter', ','),
                         encoding=file_config.get('encoding', 'utf-8'),
                         skipinitialspace=file_config.get('skipinitialspace', False))
        df.columns = df.columns.str.strip()
        temp_col = file_config.get('temp_col')
        inp_col = file_config.get('inp_col')
        if temp_col is None or temp_col not in df.columns:
            temp_col = find_column(df, ['Temp', 'Temperature', 'T', 'CFDC_Temp', 'IS_Temp_C', 'T_min'])
        if inp_col is None or inp_col not in df.columns:
            inp_col = find_column(df, ['N_INP', 'INP', 'CFDC_N_INP', 'IS_N_INP', 'INP_cn_0'])
 
        df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
        df[inp_col] = pd.to_numeric(df[inp_col], errors='coerce')
        if file_config.get('convert_kelvin', False):
            df[temp_col] = kelvin_to_celsius(df[temp_col])
        if file_config.get('exclude_zeros', False):
            df = df[df[inp_col] > 0]
        if 'max_temp' in file_config:
            df = df[df[temp_col] <= file_config['max_temp']]
        df = df.dropna(subset=[temp_col, inp_col])
        df['Temp_bin'] = np.floor(df[temp_col])
        grouped = df.groupby('Temp_bin')[inp_col].agg(
            Median='median', Q1=lambda x: np.percentile(x, 25),
            Q3=lambda x: np.percentile(x, 75),
            IQR=lambda x: np.percentile(x, 75) - np.percentile(x, 25),
            Count='count'
        ).reset_index()
        grouped = grouped[grouped['Temp_bin'] <= MAX_TEMP_BIN]
        grouped = grouped.sort_values('Temp_bin', ascending=False)
        return {
            'data': grouped, 'label': file_config['label'],
            'color': file_config['color'], 'marker': file_config.get('marker', 'o'),
            'linestyle': file_config.get('linestyle', '-'),
        }
    except Exception:
        return None

# =============================================================================
# FILE CONFIGURATIONS
# =============================================================================
file_configs_historic = [
    {
        'path': '/mnt/d/Cape24-data/Historic/CAP1/IS/IS_aerosol_untreated_CAPRICORN.csv',
        'temp_col': 'Temp', 'inp_col': 'nINP',
        'label': 'CAPRICORN 1 Filters 2016',
        'color': HISTORIC_PALETTE[0], 'linestyle': LINESTYLES[0],
        'marker': DATASET_MARKERS[0], 'exclude_zeros': True,
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/CAP1/CFDC/CFDC_CAPRICORN.csv',
        'temp_col': 'CFDC Temp', 'inp_col': 'nINP',
        'label': 'CAPRICORN 1 CFDC 2016',
        'color': HISTORIC_PALETTE[1], 'linestyle': LINESTYLES[1],
        'marker': DATASET_MARKERS[1], 'exclude_zeros': True,
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/ACESPACE/ACESPACE.csv',
        'label': 'ACE Filters 2017',
        'color': HISTORIC_PALETTE[2], 'linestyle': LINESTYLES[2],
        'marker': DATASET_MARKERS[2],
        'temp_specific_cols': True, 'convert_units': 1e-3,
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/MARCUS/CSU_all_data.csv',
        'temp_col': 'CFDC_Temp', 'inp_col': 'CFDC_N_INP',
        'label': 'MARCUS Filters 2017',
        'color': HISTORIC_PALETTE[3], 'linestyle': LINESTYLES[3],
        'marker': DATASET_MARKERS[3],
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/CAP2/CSU_all_data.csv',
        'temp_col': 'IS_Temp_C[]', 'inp_col': 'IS_N_INP[]',
        'label': 'CAPRICORN 2 Filters 2018',
        'color': HISTORIC_PALETTE[4], 'linestyle': LINESTYLES[4],
        'marker': DATASET_MARKERS[4],
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/CAP2/CFDC_all_data.csv',
        'temp_col': 'CFDC_Temp', 'inp_col': 'CFDC_N_INP',
        'label': 'CAPRICORN 2 CFDC 2018',
        'color': HISTORIC_PALETTE[5], 'linestyle': LINESTYLES[5],
        'marker': DATASET_MARKERS[5],
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/SOCRATES/clean_time_temp_inp.csv',
        'temp_col': 'temp_c', 'inp_col': 'n_inp',
        'label': 'SOCRATES Filters 2018',
        'color': HISTORIC_PALETTE[6], 'linestyle': LINESTYLES[6],
        'marker': DATASET_MARKERS[6], 'exclude_zeros': True, 'max_temp': -3,
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/MICRE/CSU_all.csv',
        'temp_col': 'Temp_C[]', 'inp_col': 'N_INP[]',
        'label': 'MICRE Filters 2016 - 2018',
        'color': HISTORIC_PALETTE[7], 'linestyle': LINESTYLES[7],
        'marker': DATASET_MARKERS[7], 'exclude_zeros': True,
    },
    {
        'path': "/mnt/d/Cape24-data/Cape24/processedPINE/RadonBaselineINP/pine_baseline_by_radon.csv",
        'temp_col': 'T_min', 'inp_col': 'INP_cn_0',
        'label': 'CAPE-k PINE 2025',
        'color': CAPEK_PINE_COLOR, 'marker': DATASET_MARKERS[8],
        'linestyle': '-', 'convert_kelvin': True, 'max_temp_kelvin': 251.15,
    },
    {
        'path': "/mnt/c/Users/ro7338/Documents/Masterarbeit/py_raw_insekt_original/Examples/DATA_Cape24",
        'label': 'CAPE-k INSEKT 2025',
        'color': CAPEK_INSEKT_COLOR, 'marker': DATASET_MARKERS[9],
        'linestyle': '-', 'insekt_data': True,
        'exp_list_path': "/mnt/c/Users/ro7338/Documents/Masterarbeit/py_raw_insekt_original/Examples/DATA_Cape24/exp_lst_Cape24.csv",
    },
    {
        'path': "/mnt/d/Cape24-data/CSU-CAPEk/baselineCSU.csv",
        'temp_col': 'temperature', 'inp_col': 'n_inp_stp',
        'label': 'CAPE-k CSU-IS 2025',
        'color': CAPEK_CSU_COLOR, 'marker': DATASET_MARKERS[10],
        'linestyle': '-',
    },
]

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    bigg_dataset = process_bigg_data()
    right_datasets = [bigg_dataset]
    for cfg in file_configs_historic:
        res = process_file_historic(cfg)
        if res is not None:
            right_datasets.append(res)

    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    fig_b.subplots_adjust(top=0.82)

    for ds in right_datasets:
        if ds['data'] is None or ds['data'].empty:
            continue
        grouped = ds['data'].sort_values('Temp_bin', ascending=True)
        color = ds['color']
        marker = ds.get('marker', 'o')
        linestyle = ds.get('linestyle', '-')
        box_width = 0.8

        # IQR rectangles for all bins
        for _, row in grouped.iterrows():
            rect = Rectangle((row['Temp_bin'] - box_width/2, row['Q1']),
                             box_width, row['Q3'] - row['Q1'],
                             facecolor=color, alpha=0.3, edgecolor='none', zorder=3)
            ax_b.add_patch(rect)

        # Determine low-count bins
        mask_ok = grouped['Count'] >= MIN_POINTS_FOR_COLOR
        mask_low = grouped['Count'] < MIN_POINTS_FOR_COLOR
        if ds['label'] == 'Bigg 1973':
            mask_ok = pd.Series(True, index=grouped.index)
            mask_low = pd.Series(False, index=grouped.index)

        # Prepare median values for line (break line at hidden bins)
        median_for_line = grouped['Median'].copy()
        if HIDE_LOW_DATA:
            median_for_line[mask_low] = np.nan

        ax_b.plot(grouped['Temp_bin'], median_for_line,
                  color=color, linestyle=linestyle, linewidth=3.5, zorder=5)

        # Valid markers
        if mask_ok.any():
            ax_b.scatter(grouped.loc[mask_ok, 'Temp_bin'],
                         grouped.loc[mask_ok, 'Median'],
                         color=color, marker=marker,
                         edgecolor='black', linewidth=1.0,
                         s=plt.rcParams['lines.markersize']**2, zorder=6)

        # Low‑count markers only if not hiding them
        if not HIDE_LOW_DATA and mask_low.any():
            ax_b.scatter(grouped.loc[mask_low, 'Temp_bin'],
                         grouped.loc[mask_low, 'Median'],
                         facecolor='white', edgecolor='black',
                         marker=marker, linewidth=1.0,
                         s=plt.rcParams['lines.markersize']**2, zorder=6)

    ax_b.set_xlabel('Temperature (°C)')
    ax_b.set_ylabel('INP concentration (L⁻¹)')
    ax_b.set_xlim(-35, -4)
    ax_b.set_ylim(0.00006, 200)
    ax_b.set_yscale('log')
    ax_b.tick_params(which='minor', length=0)

    # Legend construction
    bright_labels = {'CAPE-k PINE 2025', 'CAPE-k INSEKT 2025', 'CAPE-k CSU-IS 2025'}
    iqr_handle = mlines.Line2D([], [], color='gray', marker='s',
                               linestyle='None', markerfacecolor='gray', alpha=0.3,
                               markeredgecolor='none')
    handles_right, labels_right = [], []

    # Bigg 1973 first
    for ds in right_datasets:
        if ds['label'] == 'Bigg 1973':
            handles_right.append(mlines.Line2D([], [], color=ds['color'],
                                               marker=ds.get('marker','o'),
                                               linestyle=ds.get('linestyle','-'),
                                               markerfacecolor=ds['color'],
                                               markeredgecolor='black'))
            labels_right.append('Bigg 1973')
            break

    # Historic non‑bright
    for cfg in file_configs_historic:
        lbl = cfg['label']
        if lbl in bright_labels or lbl == 'Bigg 1973':
            continue
        ds = next((d for d in right_datasets if d['label'] == lbl), None)
        if ds:
            handles_right.append(mlines.Line2D([], [], color=ds['color'],
                                               marker=ds.get('marker','o'),
                                               linestyle=ds.get('linestyle','-'),
                                               markerfacecolor=ds['color'],
                                               markeredgecolor='black'))
            labels_right.append(lbl)

    # Bright Cape‑K
    for lbl in ['CAPE-k PINE 2025', 'CAPE-k INSEKT 2025', 'CAPE-k CSU-IS 2025']:
        ds = next((d for d in right_datasets if d['label'] == lbl), None)
        if ds:
            handles_right.append(mlines.Line2D([], [], color=ds['color'],
                                               marker=ds.get('marker','o'),
                                               linestyle=ds.get('linestyle','-'),
                                               markerfacecolor=ds['color'],
                                               markeredgecolor='black'))
            labels_right.append(lbl)

    handles_right.append(iqr_handle)
    labels_right.append('IQR (25th-75th percentile)')

    if not HIDE_LOW_DATA:
        low_count_handle = mlines.Line2D([], [], color='black', marker='o',
                                         markerfacecolor='white', markeredgecolor='black',
                                         linestyle='None', markersize=7, markeredgewidth=1.0)
        handles_right.append(low_count_handle)
        labels_right.append('Not enough data')

    ax_b.legend(handles_right, labels_right,
                loc='lower center', bbox_to_anchor=(0.5, 1.02),
                ncol=3, fontsize=9, framealpha=1,
                frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout(rect=[0, 0, 1, 1])
    fig_b.savefig('Figure9.png')
    fig_b.savefig('Figure9.pdf')
    plt.close(fig_b)