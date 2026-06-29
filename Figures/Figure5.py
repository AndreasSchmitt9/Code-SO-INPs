#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One figure with two subplots:
  (a) Cape‑K 2025 – radon‑based classification (baseline points only)
  (b) 18‑19 Feb 2025 – PINE (10‑10 UTC, binned 1°C, x‑error),
      INSEKT (18 Feb, binned 0.5°C), CSU‑IS (18 Feb, binned 0.5°C, CI error bars)
      Double‑layer markers, thick error bars.

PINE timestamps are already in UTC – no local timezone conversion needed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import re
import glob
from datetime import datetime, timezone
import pytz
import xarray as xr

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1,
    "lines.markersize": 6,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})


UTC = pytz.utc  # for defining time windows

# Colours (left panel)
BRIGHT_BLUE   = '#4477AA'
BRIGHT_RED    = '#EE6677'
BRIGHT_GREEN  = '#228833'
GREY          = 'grey'

# Right panel colours
PINE_COLOR   = BRIGHT_BLUE
INSEKT_COLOR = BRIGHT_RED
CSU_COLOR    = BRIGHT_GREEN

# Styling constants
OUTER_MARKER_SIZE = 50
MARKER_EDGEWIDTH  = 1
ERRORBAR_CAPSIZE  = 3
ERRORBAR_CAPTHICK = 1.5
ERRORBAR_ELINEWIDTH = 1.5
LEGEND_MARKER_SIZE = 6 
BIN_SIZE_PINE   = 1.0     # 1°C bins for PINE
BIN_SIZE_OTHER  = 0.5     # 0.5°C bins for INSEKT and CSU

LEFT_EDGEWIDTH = {
    'PINE baseline':     1.0,
    'INSEKT baseline':   1.0,
    'CSU-IS baseline':   1.0,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def kelvin_to_celsius(k):
    return k - 273.15

def safe_load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

# ---------- INSEKT binning (weighted mean) ----------
def bin_insekt_data(temp_c, inp, inp_unc_minus, inp_unc_plus, temp_unc, bin_size=0.5):
    if len(temp_c) == 0:
        return pd.DataFrame()
    min_temp = np.floor(temp_c.min())
    max_temp = np.ceil(temp_c.max())
    bins = np.arange(min_temp, max_temp + bin_size, bin_size)
    bin_indices = pd.cut(temp_c, bins=bins, include_lowest=True, labels=False)

    binned_data = []
    for bin_idx in range(len(bins) - 1):
        mask = bin_indices == bin_idx
        if np.sum(mask) > 0:
            bin_temp = temp_c[mask]
            bin_inp = inp[mask]
            bin_inp_unc_minus = inp_unc_minus[mask]
            bin_inp_plus = inp_unc_plus[mask]
            bin_temp_unc = temp_unc[mask]

            weights = 1 / (bin_inp_plus + bin_inp_unc_minus)
            temp_mean = np.average(bin_temp, weights=weights)
            temp_unc_mean =np.sqrt(np.average(bin_temp_unc**2, weights=weights)) #np.average(bin_temp_unc)#
            inp_mean = np.average(bin_inp)#, weights=weights)
            inp_unc_minus_mean = np.average(bin_inp_unc_minus)#np.sqrt(np.average(bin_inp_unc_minus**2, weights=weights))
            inp_unc_plus_mean = np.average(bin_inp_plus)#np.sqrt(np.average(bin_inp_plus**2, weights=weights))
            count = np.sum(mask)

            binned_data.append({
                'temp_mean': temp_mean,
                'temp_unc': temp_unc_mean,
                'inp_mean': inp_mean,
                'inp_unc_minus': inp_unc_minus_mean,
                'inp_unc_plus': inp_unc_plus_mean,
                'count': count
            })
    return pd.DataFrame(binned_data)

# ---------- Left panel helpers ----------
def get_sample_date_from_name(samplename):
    try:
        m = re.search(r'(\d{8})$', samplename)
        if m:
            return datetime.strptime(m.group(1), '%d%m%Y').date()
    except Exception:
        pass
    return None

def process_insekt_raw(folder_path, exp_list_df):
    insekt_files = [f for f in glob.glob(os.path.join(folder_path, '*_corrected.csv'))
                    if os.path.basename(f).startswith('INSEKT')]
    if not insekt_files:
        return {}

    processed_data = {}
    for file_path in insekt_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        number = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None
        samplename = "Not found"

        if number is not None and not exp_list_df.empty:
            matched = exp_list_df[exp_list_df['exp_id'] == number]
            if not matched.empty:
                samplename = matched['samplename'].iloc[0]

        if number == 13868 or samplename == "Cape24_Water_ACTRIS":
            continue

        lower_name = samplename.lower()
        if any(kw in lower_name for kw in ['_heat', 'blank', 'blk', 'none', 'na', 'nan', 'null']):
            continue

        sample_date = get_sample_date_from_name(samplename)
        if sample_date and sample_date == datetime(2025, 2, 14).date():
            continue

        try:
            df = pd.read_csv(file_path, skiprows=[1])
            req = ['T_mean / K', 'c_INP / 1/l',
                   'c_INP_unc_minus / 1/l', 'c_INP_unc_plus / 1/l']
            if all(c in df.columns for c in req) and not df.empty:
                df['T_mean_C'] = kelvin_to_celsius(df['T_mean / K'])
                df['c_INP_unc_minus'] = df['c_INP_unc_minus / 1/l']
                df['c_INP_unc_plus']  = df['c_INP_unc_plus / 1/l']
                processed_data[filename] = {'df': df, 'sample_date': sample_date}
        except Exception:
            continue
    return processed_data

def calculate_hourly_averages_baseline(df, freq='1h'):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour_bin'] = df['datetime'].dt.floor(freq)
    hourly = df.groupby('hour_bin').agg(
        T_min_C_mean=('T_min_C', 'mean'),
        INP_mean=('INP_cn_0', 'mean'),
        INP_count=('INP_cn_0', 'count'),
        INP_iqr=('INP_cn_0', lambda x: x.quantile(0.75) - x.quantile(0.25))
    ).reset_index()
    hourly = hourly[hourly['INP_mean'] >= 0.01]
    return hourly

def load_radon_daily_average(radon_csv_path):
    try:
        df = pd.read_csv(radon_csv_path)
        datetime_col = None
        for col in df.columns:
            if any(kw in col.lower() for kw in ['datetime', 'time', 'date']):
                datetime_col = col; break
        if datetime_col is None:
            datetime_col = df.columns[0]
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df = df.dropna(subset=[datetime_col])
        radon_col = None
        for col in df.columns:
            if 'radon' in col.lower():
                radon_col = col; break
        if radon_col is None:
            radon_col = df.columns[1] if len(df.columns) > 1 else None
        if radon_col is None:
            raise ValueError("Could not identify radon concentration column.")
        df[radon_col] = pd.to_numeric(df[radon_col], errors='coerce')
        radon_series = df.set_index(datetime_col)[radon_col].sort_index()
        daily_avg = radon_series.resample('D').mean().dropna()
        return daily_avg
    except Exception as e:
        print(f"Error loading radon data: {e}")
        return pd.Series(dtype=float)

def get_daily_radon(daily_radon, date):
    if daily_radon.empty:
        return np.nan
    target = pd.Timestamp(date).date()
    try:
        val = daily_radon.get(pd.Timestamp(target))
        return val if not pd.isna(val) else np.nan
    except Exception:
        return np.nan

def load_csu_capek_data(nc_directory, daily_radon):
    """Returns csu_red (baseline) and csu_grey (others).
    Keys: temperature, concentration, time, ci_lower, ci_upper."""
    csu_red, csu_grey = [], []
    nc_files = sorted(glob.glob(os.path.join(nc_directory, "*.nc")))
    if not nc_files:
        return csu_red, csu_grey
    radon_threshold = 0.1
    for file_path in nc_files:
        try:
            with xr.open_dataset(file_path) as ds:
                temp = ds['temperature'].values
                n_inp = ds['n_inp_stp'].values
                flag = ds['treatment_flag'].values
                time_coord = ds['time'].values

                ci_lower = ds['lower_ci'].values if 'lower_ci' in ds else None
                ci_upper = ds['upper_ci'].values if 'upper_ci' in ds else None

                if n_inp.ndim == 1:
                    n_inp = n_inp[np.newaxis, :]
                    flag = flag[np.newaxis, :]
                    time_coord = time_coord[np.newaxis]
                    if ci_lower is not None:
                        ci_lower = ci_lower[np.newaxis, :]
                    if ci_upper is not None:
                        ci_upper = ci_upper[np.newaxis, :]

                for t_idx in range(n_inp.shape[0]):
                    conc = n_inp[t_idx, :]
                    flag_row = flag[t_idx, :]
                    mask_untreated = (flag_row == 0) & ~np.isnan(conc)
                    if np.any(mask_untreated):
                        temp_untreated = temp[mask_untreated]
                        conc_untreated = conc[mask_untreated]
                        sort_idx = np.argsort(temp_untreated)
                        sample_time = pd.to_datetime(time_coord[t_idx])

                        ci_l = ci_lower[t_idx, mask_untreated][sort_idx] if ci_lower is not None else None
                        ci_u = ci_upper[t_idx, mask_untreated][sort_idx] if ci_upper is not None else None

                        radon_val = get_daily_radon(daily_radon, sample_time.date())
                        data_dict = {
                            'temperature': temp_untreated[sort_idx],
                            'concentration': conc_untreated[sort_idx],
                            'filename': os.path.basename(file_path),
                            'time': sample_time,
                            'daily_radon': radon_val,
                            'ci_lower': ci_l,
                            'ci_upper': ci_u
                        }
                        if (not np.isnan(radon_val)) and radon_val < radon_threshold:
                            csu_red.append(data_dict)
                        else:
                            csu_grey.append(data_dict)
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
    return csu_red, csu_grey

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # ------------------ data paths ------------------
    data_dir_cape = '/mnt/c/Users/ro7338/Documents/Masterarbeit/py_raw_insekt_original/Examples/DATA_Cape24'
    baseline_file_path = '/mnt/d/Cape24-data/Cape24/processedPINE/RadonBaselineINP/pine_baseline_by_radon.csv'
    exp_list_path = '/mnt/c/Users/ro7338/Documents/Masterarbeit/py_raw_insekt_original/Examples/DATA_Cape24/exp_lst_Cape24.csv'
    csu_nc_dir = "/mnt/d/Cape24-data/CSU-CAPEk/kcginpS3.a1/original_format"
    radon_csv_path = "/mnt/d/Cape24-data/Model/Radon/release-v2/combined_radon_data.csv"

    daily_radon = load_radon_daily_average(radon_csv_path)
    exp_list_cape24_df = safe_load_csv(exp_list_path)

    # ------------------ PINE data (all) – already UTC ------------------
    capek_df = safe_load_csv('/mnt/d/Cape24-data/Data/processedPINE/Capek.csv')
    PINE_COLS = ['datetime', 'T_min', 'INP_cn_0']
    empty_pine = pd.DataFrame(columns=PINE_COLS + ['T_min_C', 'datetime_utc'])
    if not capek_df.empty and all(c in capek_df.columns for c in PINE_COLS):
        capek_df = capek_df[PINE_COLS].copy()
        # PINE timestamps are already UTC – parse directly
        capek_df['datetime_utc'] = pd.to_datetime(capek_df['datetime'], utc=True)
        capek_df['T_min'] = pd.to_numeric(capek_df['T_min'], errors='coerce')
        capek_df['T_min_C'] = kelvin_to_celsius(capek_df['T_min'])
        capek_df = capek_df.dropna(subset=['T_min'])
    else:
        capek_df = empty_pine

    cutoff_temp_k = 251.15
    capek_filtered = capek_df[capek_df['T_min'] <= cutoff_temp_k] if not capek_df.empty else empty_pine

    # ------------------ PINE baseline (hourly) for left panel ------------------
    baseline_raw = safe_load_csv(baseline_file_path)
    if not baseline_raw.empty:
        baseline_raw['T_min'] = pd.to_numeric(baseline_raw['T_min'], errors='coerce')
        baseline_raw['INP_cn_0'] = pd.to_numeric(baseline_raw['INP_cn_0'], errors='coerce')
        baseline_raw['datetime'] = pd.to_datetime(baseline_raw['datetime'], errors='coerce')
        baseline_clean = baseline_raw.dropna(subset=['T_min', 'INP_cn_0', 'datetime']).copy()
        baseline_clean['T_min_C'] = kelvin_to_celsius(baseline_clean['T_min'])
        baseline_filt = baseline_clean[baseline_clean['T_min'] <= cutoff_temp_k]
        baseline_hourly = calculate_hourly_averages_baseline(baseline_filt, freq='2h')
    else:
        baseline_hourly = pd.DataFrame()

    # ------------------ INSEKT data (for left panel) ------------------
    capek_insekt_processed = process_insekt_raw(data_dir_cape, exp_list_cape24_df)
    CAPE_INSEKT_HIGHLIGHT_DATE = datetime(2025, 2, 14).date()
    red_insekt_raw, grey_insekt_raw = [], []
    for fname, info in capek_insekt_processed.items():
        is_red = info['sample_date'] and info['sample_date'] > CAPE_INSEKT_HIGHLIGHT_DATE
        if is_red:
            red_insekt_raw.append(info)
        else:
            grey_insekt_raw.append(info)

    # ------------------ CSU‑IS data ------------------
    csu_red, csu_grey = load_csu_capek_data(csu_nc_dir, daily_radon)

    # ================== RIGHT PANEL DATA ==================
    target_date = datetime(2025, 2, 18).date()

    # ---- Collect raw INSEKT points for 18 Feb ----
    insekt_raw_all = []
    for fname, info in capek_insekt_processed.items():
        if info['sample_date'] == target_date:
            df = info['df']
            if not df.empty:
                cols = ['T_mean_C', 'c_INP / 1/l',
                        'c_INP_unc_minus', 'c_INP_unc_plus']
                if 'T_std / K' in df.columns:
                    temp_unc = df['T_std / K']
                else:
                    temp_unc = pd.Series(np.ones(len(df)) * 0.5)
                available = [c for c in cols if c in df.columns]
                tmp = df[available].copy()
                tmp['temp_unc'] = temp_unc
                tmp = tmp.dropna()
                insekt_raw_all.append(tmp)
    insekt_raw_df = pd.concat(insekt_raw_all, ignore_index=True) if insekt_raw_all else pd.DataFrame()

    # ---- Bin INSEKT (0.5°C) ----
    if not insekt_raw_df.empty:
        insekt_binned = bin_insekt_data(
            insekt_raw_df['T_mean_C'].values,
            insekt_raw_df['c_INP / 1/l'].values,
            insekt_raw_df['c_INP_unc_minus'].values,
            insekt_raw_df['c_INP_unc_plus'].values,
            insekt_raw_df['temp_unc'].values,
            bin_size=BIN_SIZE_OTHER
        )
        insekt_binned = insekt_binned[insekt_binned['inp_mean'] > 0]
    else:
        insekt_binned = pd.DataFrame()

    # ---- Collect raw CSU‑IS points for 18 Feb ----
    csu_raw_all = []
    for d in csu_red + csu_grey:
        if d['time'].date() == target_date:
            temp = d['temperature']
            conc = d['concentration']
            ci_l = d['ci_lower'] if d['ci_lower'] is not None else np.full_like(temp, np.nan)
            ci_u = d['ci_upper'] if d['ci_upper'] is not None else np.full_like(temp, np.nan)
            mask = (temp >= -35) & (temp <= 0)
            t_arr = temp[mask]
            c_arr = conc[mask]
            l_arr = ci_l[mask]
            u_arr = ci_u[mask]
            csu_raw_all.append(pd.DataFrame({
                'temperature': t_arr,
                'concentration': c_arr,
                'ci_lower': l_arr,
                'ci_upper': u_arr
            }))
    csu_raw_df = pd.concat(csu_raw_all, ignore_index=True) if csu_raw_all else pd.DataFrame()

    # ---- Bin CSU‑IS (0.5°C) ----
    if not csu_raw_df.empty:
        csu_raw_df['T_bin'] = (np.floor(csu_raw_df['temperature'] / BIN_SIZE_OTHER) * BIN_SIZE_OTHER).round(1)
        csu_binned = csu_raw_df.groupby('T_bin').agg(
            mean_conc=('concentration', 'mean'),
            mean_lower=('ci_lower', 'mean'),
            mean_upper=('ci_upper', 'mean'),
            count=('concentration', 'count')
        ).reset_index()
        csu_binned['yerr_lower'] =  np.maximum(csu_binned['mean_conc'] - csu_binned['mean_lower'], 1e-6)
        csu_binned['yerr_upper'] =  np.maximum(csu_binned['mean_upper'] - csu_binned['mean_conc'], 1e-6)
        csu_binned = csu_binned[csu_binned['mean_conc'] > 0]
    else:
        csu_binned = pd.DataFrame()

    # ---- PINE 10-10 UTC (already UTC) ----
    pine_start = datetime(2025, 2, 18, 10, 0, tzinfo=timezone.utc)
    pine_end   = datetime(2025, 2, 19, 10, 0, tzinfo=timezone.utc)
    pine_24h = capek_filtered[
        (capek_filtered['datetime_utc'] >= pine_start) &
        (capek_filtered['datetime_utc'] <= pine_end)
    ].copy()

    if not pine_24h.empty:
        pine_24h['T_bin'] = (np.floor(pine_24h['T_min_C'] / BIN_SIZE_PINE) * BIN_SIZE_PINE).round(1)
        pine_binned = pine_24h.groupby('T_bin').agg(
            mean_INP=('INP_cn_0', 'mean'),
            std_INP=('INP_cn_0', 'std'),
            count=('INP_cn_0', 'count')
        ).reset_index()
        pine_binned['error_lower'] = np.maximum(
            pine_binned['mean_INP'] - np.maximum(pine_binned['mean_INP'] - pine_binned['std_INP'], 0.1),
            0.001
        )
        pine_binned['error_upper'] = np.maximum(pine_binned['std_INP'], 0.001)
        pine_binned['xerr'] = 1
        pine_binned = pine_binned[pine_binned['mean_INP'] > 0]
    else:
        pine_binned = pd.DataFrame()

    # ================== FIGURE ==================
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, 4))
    fig.subplots_adjust(top=0.8, wspace=0.06, right=0.99, left=0.1)

    # ---------- LEFT SUBPLOT (a) ----------
    # Only plot baseline data (red/green/blue), no grey "all" data.
    labeled_left = {
        'BASELINE_HOURLY': False,
        'INSEKT_BASELINE': False,
        'CSU_BASELINE': False
    }

    # CSU-IS baseline (green)
    for d in csu_red:
        ax_a.scatter(d['temperature'], d['concentration'],
                     color=BRIGHT_GREEN, marker='o', s=OUTER_MARKER_SIZE,
                     edgecolor='black', linewidth=1.0, alpha=1, zorder=1.5,
                     label='CSU-IS baseline' if not labeled_left['CSU_BASELINE'] else "")
        labeled_left['CSU_BASELINE'] = True

    # INSEKT baseline (red)
    for info in red_insekt_raw:
        df = info['df']
        if not df.empty:
            ax_a.scatter(df['T_mean_C'], df['c_INP / 1/l'],
                         color=BRIGHT_RED, marker='X', s=OUTER_MARKER_SIZE,
                         label='INSEKT baseline' if not labeled_left['INSEKT_BASELINE'] else "",
                         linewidths=1, zorder=5, alpha=1.0, edgecolor='black')
            labeled_left['INSEKT_BASELINE'] = True

    # PINE baseline hourly (blue)
    if not baseline_hourly.empty:
        ax_a.scatter(baseline_hourly['T_min_C_mean'], baseline_hourly['INP_mean'],
                     color=BRIGHT_BLUE, marker='D', s=OUTER_MARKER_SIZE,
                     edgecolors='black', linewidths=1,
                     label='PINE baseline' if not labeled_left['BASELINE_HOURLY'] else "",
                     zorder=2, alpha=1)
        labeled_left['BASELINE_HOURLY'] = True

    ax_a.set_xlabel('Temperature (°C)')
    ax_a.set_ylabel('INP concentration (L⁻¹)')
    ax_a.set_xlim(-35, -4)
    ax_a.set_ylim(0.00006, 600)
    ax_b.set_ylabel('')  # remove y-axis label

    ax_b.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_a.set_yscale('log')
    #ax_a.tick_params(axis='both', which='major', labelsize=18, length=6)
    ax_a.tick_params(which='minor', length=0)

    # Only baseline items in legend
    desired_order_left = ['PINE baseline', 'INSEKT baseline', 'CSU-IS baseline']
    left_styles = {
        'PINE baseline': (BRIGHT_BLUE, 'D'),
        'INSEKT baseline': (BRIGHT_RED, 'X'),
        'CSU-IS baseline': (BRIGHT_GREEN, 'o')
    }
    handles_left = []
    labels_left = []
    for label in desired_order_left:
        if label in left_styles:
            color, marker = left_styles[label]
            edge_w = LEFT_EDGEWIDTH.get(label, 1.0)
            handle = mlines.Line2D([], [], color=color, marker=marker,
                                markersize=LEGEND_MARKER_SIZE, linestyle='None',
                                markeredgecolor='black', markeredgewidth=edge_w)
            handles_left.append(handle)
            labels_left.append(label)
    ax_a.legend(handles_left, labels_left,
                loc='lower center', bbox_to_anchor=(0.5, 1.02),
                ncol=1, framealpha=1,
                frameon=True, fancybox=False, edgecolor='black')

    ax_a.text(0.01, 0.98, 'a)', transform=ax_a.transAxes,
               va='top')

    # ---------- RIGHT SUBPLOT (b) – binned data ----------
    # PINE
    if not pine_binned.empty:
        ax_b.scatter(pine_binned['T_bin'], pine_binned['mean_INP'],
                     color='black', marker='D', s=OUTER_MARKER_SIZE,
                     zorder=4, alpha=1.0, linewidth=1)
        ax_b.scatter(pine_binned['T_bin'], pine_binned['mean_INP'],
                     color=PINE_COLOR, marker='D', s=OUTER_MARKER_SIZE,
                     label='PINE 10‑10 UTC (1°C bins)',
                     zorder=5, alpha=1.0,
                     edgecolor='black', linewidth=MARKER_EDGEWIDTH)
        ax_b.errorbar(pine_binned['T_bin'], pine_binned['mean_INP'],
                      xerr=pine_binned['xerr'],
                      yerr=[pine_binned['error_lower'], pine_binned['error_upper']],
                      fmt='none', color='black',
                      capsize=ERRORBAR_CAPSIZE,
                      capthick=ERRORBAR_CAPTHICK,
                      elinewidth=ERRORBAR_ELINEWIDTH,
                      zorder=3, alpha=1)

    # INSEKT
    if not insekt_binned.empty:
        ax_b.scatter(insekt_binned['temp_mean'], insekt_binned['inp_mean'],
                     color='black', marker='X', s=OUTER_MARKER_SIZE,
                     zorder=4, alpha=1.0, linewidth=1)
        ax_b.scatter(insekt_binned['temp_mean'], insekt_binned['inp_mean'],
                     color=INSEKT_COLOR, marker='X', s=OUTER_MARKER_SIZE,
                     label='INSEKT 18 Feb (0.5°C bins)',
                     zorder=5, alpha=1.0,
                     edgecolor='black', linewidth=MARKER_EDGEWIDTH)
        ax_b.errorbar(insekt_binned['temp_mean'], insekt_binned['inp_mean'],
                      xerr=insekt_binned['temp_unc'],
                      yerr=[insekt_binned['inp_unc_minus'], insekt_binned['inp_unc_plus']],
                      fmt='none', color='black',
                      capsize=ERRORBAR_CAPSIZE,
                      capthick=ERRORBAR_CAPTHICK,
                      elinewidth=ERRORBAR_ELINEWIDTH,
                      zorder=3, alpha=1)

    # CSU‑IS
    if not csu_binned.empty:
        ax_b.scatter(csu_binned['T_bin'], csu_binned['mean_conc'],
                     color='black', marker='o', s=OUTER_MARKER_SIZE,
                     zorder=4, alpha=1.0, linewidth=1)
        ax_b.scatter(csu_binned['T_bin'], csu_binned['mean_conc'],
                     color=CSU_COLOR, marker='o', s=OUTER_MARKER_SIZE,
                     label='CSU-IS 18 Feb',
                     zorder=5, alpha=1.0,
                     edgecolor='black', linewidth=MARKER_EDGEWIDTH)
        ax_b.errorbar(csu_binned['T_bin'], csu_binned['mean_conc'],
                      yerr=[csu_binned['yerr_lower'], csu_binned['yerr_upper']],
                      fmt='none', color='black',
                      capsize=ERRORBAR_CAPSIZE,
                      capthick=ERRORBAR_CAPTHICK,
                      elinewidth=ERRORBAR_ELINEWIDTH,
                      zorder=3, alpha=1)

    ax_b.set_xlabel('Temperature (°C)')
    ax_b.set_ylabel('')
    ax_b.set_xlim(-35, -4)
    ax_b.set_ylim(0.00006, 600)
    ax_b.set_yscale('log')
    ax_b.tick_params(which='minor', length=0)

    handles_b = [
        mlines.Line2D([], [], color=PINE_COLOR, marker='D', markersize=LEGEND_MARKER_SIZE,
                    linestyle='None', markeredgecolor='black', markeredgewidth=MARKER_EDGEWIDTH),
        mlines.Line2D([], [], color=INSEKT_COLOR, marker='X', markersize=LEGEND_MARKER_SIZE,
                    linestyle='None', markeredgecolor='black', markeredgewidth=MARKER_EDGEWIDTH),
        mlines.Line2D([], [], color=CSU_COLOR, marker='o', markersize=LEGEND_MARKER_SIZE,
                    linestyle='None', markeredgecolor='black', markeredgewidth=MARKER_EDGEWIDTH)
    ]
    
    labels_b = ['PINE 24h avg. 18 Feb', 'INSEKT 18 Feb', 'CSU-IS 18 Feb']
    ax_b.legend(handles_b, labels_b,
                loc='lower center', bbox_to_anchor=(0.5, 1.02),
                frameon=True, fancybox=False, edgecolor='black')

    ax_b.text(0.01, 0.98, 'b)', transform=ax_b.transAxes,
            va='top')
    fig.savefig('Figure5.png')
    fig.savefig('Figure5.pdf')
    print("Saved Figure5.png")
    plt.close(fig)