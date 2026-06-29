#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two separate figures:
  Figure 7a : Cape‑K 2025 – radon‑based classification (raw points)
  Figure 7b : Historic campaigns + Cape‑K 2025 – binned median with IQR

Both figures share the same x‑ and y‑limits (y up to 1000).
Legends are placed above the axes in 3 columns,
with marker sizes matching those in the plots.
Panel markers (a, b) are removed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import re
import glob
from datetime import datetime
import xarray as xr
from pathlib import Path
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

# Minimum number of points per temperature bin required for a coloured marker.
# Bins with fewer points will be shown as white-filled markers with a black edge.
MIN_POINTS_FOR_COLOR = 3

# Upper temperature limit for plotting (data above this is discarded)
MAX_TEMP_BIN = -4

# =============================================================================
# SCIENTIFIC COLOUR PALETTES
# =============================================================================
# Cape‑K 2025 datasets (colourblind‑friendly Set1)
CAPEK_PINE_COLOR  = '#377eb8'   # blue
CAPEK_INSEKT_COLOR = '#e41a1c'  # red
CAPEK_CSU_COLOR   = '#4daf4a'   # green

# Historic datasets (Dark2 – 8 distinguishable colours)
HISTORIC_PALETTE = [
    '#1b9e77',  # teal
    '#d95f02',  # orange
    '#7570b3',  # purple
    '#e7298a',  # magenta
    '#66a61e',  # lime green
    '#e6ab02',  # golden yellow
    '#a6761d',  # brown
    '#666666'   # grey
]

LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)),
              (0, (5, 2)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]

DATASET_MARKERS = [
    'o', 's', '^', 'v', '<', '>', 'p', '*', 'D', 'X', 'o'
]

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

def find_column(df, possible_names):
    """Return the first column name that exists in df."""
    for name in possible_names:
        if name in df.columns:
            return name
        if f"{name}[]" in df.columns:
            return f"{name}[]"
    return None

# ---------- Left panel helper functions ----------
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
    used_samples = []

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
            req = ['T_mean / K', 'c_INP / 1/l', 'c_INP_unc_minus / 1/l', 'c_INP_unc_plus / 1/l']
            if all(c in df.columns for c in req) and not df.empty:
                df['T_mean_C'] = kelvin_to_celsius(df['T_mean / K'])
                processed_data[filename] = {'df': df, 'sample_date': sample_date}
                used_samples.append(f"{filename} (exp_id: {number}, sample: {samplename}, date: {sample_date})")
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
        T_min_C_std=('T_min_C', 'std'),
        T_min_C_count=('T_min_C', 'count'),
        INP_mean=('INP_cn_0', 'mean'),
        INP_count=('INP_cn_0', 'count'),
        INP_iqr=('INP_cn_0', lambda x: x.quantile(0.75) - x.quantile(0.25))
    ).reset_index()
    hourly['INP_yerr'] = hourly['INP_iqr'].fillna(0)
    hourly['T_min_C_yerr'] = np.maximum(
        hourly['T_min_C_std'].fillna(0) / np.sqrt(hourly['T_min_C_count']), 0.1)
    hourly = hourly[hourly['INP_mean'] >= 0.01]
    return hourly

def load_radon_daily_average(radon_csv_path):
    try:
        df = pd.read_csv(radon_csv_path)
        datetime_col = None
        for col in df.columns:
            if any(kw in col.lower() for kw in ['datetime', 'time', 'date']):
                datetime_col = col
                break
        if datetime_col is None:
            datetime_col = df.columns[0]
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df = df.dropna(subset=[datetime_col])
        radon_col = None
        for col in df.columns:
            if 'radon' in col.lower():
                radon_col = col
                break
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
                if n_inp.ndim == 1:
                    n_inp = n_inp[np.newaxis, :]
                    flag = flag[np.newaxis, :]
                    time_coord = time_coord[np.newaxis]
                for t_idx in range(n_inp.shape[0]):
                    conc = n_inp[t_idx, :]
                    flag_row = flag[t_idx, :]
                    mask_untreated = (flag_row == 0) & ~np.isnan(conc)
                    if np.any(mask_untreated):
                        temp_untreated = temp[mask_untreated]
                        conc_untreated = conc[mask_untreated]
                        sort_idx = np.argsort(temp_untreated)
                        sample_time = pd.to_datetime(time_coord[t_idx])
                        radon_val = get_daily_radon(daily_radon, sample_time.date())
                        data_dict = {
                            'temperature': temp_untreated[sort_idx],
                            'concentration': conc_untreated[sort_idx],
                            'filename': os.path.basename(file_path),
                            'time': sample_time,
                            'daily_radon': radon_val
                        }
                        if (not np.isnan(radon_val)) and radon_val < radon_threshold:
                            csu_red.append(data_dict)
                        else:
                            csu_grey.append(data_dict)
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
    return csu_red, csu_grey

# ---------- Right panel helper functions ----------
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
    # Filter: keep only bins <= MAX_TEMP_BIN
    grouped = grouped[grouped['Temp_bin'] <= MAX_TEMP_BIN]
    return {
        'data': grouped, 'label': 'Bigg 1973', 'color': 'black',
        'linestyle': ':', 'marker': 'o', 'scale_size': False,
        'raw_data': None, 'plot_raw': False,
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
        if temp_col is None or inp_col is None:
            return None
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
        # Filter bins above MAX_TEMP_BIN
        grouped = grouped[grouped['Temp_bin'] <= MAX_TEMP_BIN]
        if grouped.empty:
            return None
        grouped = grouped.sort_values('Temp_bin', ascending=False)
        return {
            'data': grouped, 'label': file_config['label'],
            'color': file_config['color'], 'marker': file_config.get('marker', 'o'),
            'linestyle': file_config.get('linestyle', '-'),
            'scale_size': file_config.get('scale_size', True),
            'raw_data': None, 'plot_raw': False,
        }
    except Exception as e:
        print(f"Error processing PINE {file_config['path']}: {e}")
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
    if not filtered_files:
        return None
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
    # Filter bins above MAX_TEMP_BIN
    grouped = grouped[grouped['Temp_bin'] <= MAX_TEMP_BIN]
    if grouped.empty: return None
    grouped = grouped.sort_values('Temp_bin', ascending=False)
    return {
        'data': grouped, 'label': file_config['label'],
        'color': file_config['color'], 'marker': file_config.get('marker', 'X'),
        'linestyle': file_config.get('linestyle', '-'),
        'scale_size': file_config.get('scale_size', True),
        'raw_data': None, 'plot_raw': False,
    }

def process_acespace(file_config):
    try:
        df = pd.read_csv(file_config['path'])
        # === Date filtering for ACE data ===
        # Look for a column that contains 'date' in its name (case insensitive)
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            # Define exclusion date ranges (inclusive)
            exclude_ranges = [
                ('2016-12-20', '2016-12-24'),
                ('2017-01-20', '2017-01-24'),
                ('2017-02-20', '2017-02-28')
            ]
            mask = pd.Series(True, index=df.index)
            for start, end in exclude_ranges:
                mask = mask & ~((df[date_col] >= start) & (df[date_col] <= end))
            df = df[mask]
            if df.empty:
                print("WARNING: All ACE data removed after date filtering.")
                return None
        # ==================================
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
        # Filter bins above MAX_TEMP_BIN
        grouped = grouped[grouped['Temp_bin'] <= MAX_TEMP_BIN]
        grouped = grouped.sort_values('Temp_bin', ascending=False)
        return {
            'data': grouped, 'label': file_config['label'],
            'color': file_config['color'], 'marker': file_config.get('marker', 'o'),
            'linestyle': file_config.get('linestyle', '-'),
            'scale_size': file_config.get('scale_size', True),
            'raw_data': None, 'plot_raw': False,
        }
    except Exception as e:
        print(f"Error processing ACE SPACE: {e}")
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
        if temp_col is None or inp_col is None:
            return None
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
        # Filter bins above MAX_TEMP_BIN
        grouped = grouped[grouped['Temp_bin'] <= MAX_TEMP_BIN]
        grouped = grouped.sort_values('Temp_bin', ascending=False)
        return {
            'data': grouped, 'label': file_config['label'],
            'color': file_config['color'], 'marker': file_config.get('marker', 'o'),
            'linestyle': file_config.get('linestyle', '-'),
            'scale_size': file_config.get('scale_size', True),
            'raw_data': None, 'plot_raw': False,
        }
    except Exception as e:
        print(f"Error processing {file_config['path']}: {e}")
        return None

# =============================================================================
# FILE CONFIGURATIONS FOR RIGHT SUBPLOT
# =============================================================================
file_configs_historic = [
    {
        'path': '/mnt/d/Cape24-data/Historic/CAP1/IS/IS_aerosol_untreated_CAPRICORN.csv',
        'temp_col': 'Temp', 'inp_col': 'nINP',
        'label': 'CAPRICORN 1 Filters 2016',
        'color': HISTORIC_PALETTE[0], 'linestyle': LINESTYLES[0],
        'marker': DATASET_MARKERS[0], 'convert_kelvin': False,
        'exclude_zeros': True, 'scale_size': False,
        'plot_raw': False, 'delimiter': ',',
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/CAP1/CFDC/CFDC_CAPRICORN.csv',
        'temp_col': 'CFDC Temp', 'inp_col': 'nINP',
        'label': 'CAPRICORN 1 CFDC 2016',
        'color': HISTORIC_PALETTE[1], 'linestyle': LINESTYLES[1],
        'marker': DATASET_MARKERS[1], 'convert_kelvin': False,
        'exclude_zeros': True, 'scale_size': False,
        'plot_raw': False, 'delimiter': ',',
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/ACESPACE/ACESPACE.csv',
        'label': 'ACE Filters 2017',
        'color': HISTORIC_PALETTE[2], 'linestyle': LINESTYLES[2],
        'marker': DATASET_MARKERS[2], 'scale_size': False,
        'temp_specific_cols': True, 'convert_units': 1e-3,
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/MARCUS/CSU_all_data.csv',
        'temp_col': 'CFDC_Temp', 'inp_col': 'CFDC_N_INP',
        'label': 'MARCUS Filters 2017',
        'color': HISTORIC_PALETTE[3], 'linestyle': LINESTYLES[3],
        'marker': DATASET_MARKERS[3], 'scale_size': False,
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/CAP2/CSU_all_data.csv',
        'temp_col': 'IS_Temp_C[]', 'inp_col': 'IS_N_INP[]',
        'label': 'CAPRICORN 2 Filters 2018',
        'color': HISTORIC_PALETTE[4], 'linestyle': LINESTYLES[4],
        'marker': DATASET_MARKERS[4], 'scale_size': False,
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/CAP2/CFDC_all_data.csv',
        'temp_col': 'CFDC_Temp', 'inp_col': 'CFDC_N_INP',
        'label': 'CAPRICORN 2 CFDC 2018',
        'color': HISTORIC_PALETTE[5], 'linestyle': LINESTYLES[5],
        'marker': DATASET_MARKERS[5], 'scale_size': False,
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/SOCRATES/clean_time_temp_inp.csv',
        'temp_col': 'temp_c', 'inp_col': 'n_inp',
        'label': 'SOCRATES Filters 2018',
        'color': HISTORIC_PALETTE[6], 'linestyle': LINESTYLES[6],
        'marker': DATASET_MARKERS[6], 'convert_kelvin': False,
        'exclude_zeros': True, 'scale_size': False,
        'plot_raw': False, 'delimiter': ',',
        'max_temp': -3,
    },
    {
        'path': '/mnt/d/Cape24-data/Historic/MICRE/CSU_all.csv',
        'temp_col': 'Temp_C[]', 'inp_col': 'N_INP[]',
        'label': 'MICRE Filters 2016 - 2018',
        'color': HISTORIC_PALETTE[7], 'linestyle': LINESTYLES[7],
        'marker': DATASET_MARKERS[7], 'convert_kelvin': False,
        'exclude_zeros': True, 'scale_size': False,
        'plot_raw': False, 'delimiter': ',',
    },
    {
        'path': "/mnt/d/Cape24-data/Cape24/processedPINE/RadonBaselineINP/pine_baseline_by_radon.csv",
        'temp_col': 'T_min', 'inp_col': 'INP_cn_0',
        'label': 'CAPE-k PINE 2025',
        'color': CAPEK_PINE_COLOR, 'marker': DATASET_MARKERS[8],
        'linestyle': '-',
        'convert_kelvin': True, 'exclude_zeros': False,
        'scale_size': False, 'plot_raw': False,
        'delimiter': ',', 'max_temp_kelvin': 251.15,
    },
    {
        'path': "/mnt/c/Users/ro7338/Documents/Masterarbeit/py_raw_insekt_original/Examples/DATA_Cape24",
        'label': 'CAPE-k INSEKT 2025',
        'color': CAPEK_INSEKT_COLOR, 'marker': DATASET_MARKERS[9],
        'linestyle': '-', 'scale_size': False,
        'plot_raw': False, 'insekt_data': True,
        'exp_list_path': "/mnt/c/Users/ro7338/Documents/Masterarbeit/py_raw_insekt_original/Examples/DATA_Cape24/exp_lst_Cape24.csv",
    },
    {
        'path': "/mnt/d/Cape24-data/CSU-CAPEk/baselineCSU.csv",
        'temp_col': 'temperature', 'inp_col': 'n_inp_stp',
        'label': 'CAPE-k CSU-IS 2025',
        'color': CAPEK_CSU_COLOR, 'marker': DATASET_MARKERS[10],
        'linestyle': '-', 'convert_kelvin': False,
        'exclude_zeros': False, 'scale_size': False,
        'plot_raw': False, 'delimiter': ',',
    },
]

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # ------------------ left panel data ------------------
    data_dir_cape = '/mnt/c/Users/ro7338/Documents/Masterarbeit/py_raw_insekt_original/Examples/DATA_Cape24'
    baseline_file_path = '/mnt/d/Cape24-data/Cape24/processedPINE/RadonBaselineINP/pine_baseline_by_radon.csv'
    exp_list_path = '/mnt/c/Users/ro7338/Documents/Masterarbeit/py_raw_insekt_original/Examples/DATA_Cape24/exp_lst_Cape24.csv'
    csu_nc_dir = "/mnt/d/Cape24-data/CSU-CAPEk/kcginpS3.a1/original_format"
    radon_csv_path = "/mnt/d/Cape24-data/Model/Radon/release-v2/combined_radon_data.csv"

    daily_radon = load_radon_daily_average(radon_csv_path)
    exp_list_cape24_df = safe_load_csv(exp_list_path)

    capek_df = safe_load_csv('/mnt/d/Cape24-data/Data/processedPINE/Capek.csv')
    PINE_COLS = ['datetime', 'T_min', 'INP_cn_0']
    empty_pine = pd.DataFrame(columns=PINE_COLS + ['T_min_C'])
    if not capek_df.empty and all(c in capek_df.columns for c in PINE_COLS):
        capek_df = capek_df[PINE_COLS].copy()
        capek_df['datetime'] = pd.to_datetime(capek_df['datetime'])
        capek_df['T_min'] = pd.to_numeric(capek_df['T_min'], errors='coerce')
        capek_df = capek_df.dropna(subset=['T_min'])
        capek_df['T_min_C'] = kelvin_to_celsius(capek_df['T_min'])
    else:
        capek_df = empty_pine

    cutoff_temp_k = 251.15
    capek_filtered = capek_df[capek_df['T_min'] <= cutoff_temp_k] if not capek_df.empty else empty_pine

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

    capek_insekt_processed = process_insekt_raw(data_dir_cape, exp_list_cape24_df)
    CAPE_INSEKT_HIGHLIGHT_DATE = datetime(2025, 2, 14).date()
    red_insekt_raw, grey_insekt_raw = [], []
    for fname, info in capek_insekt_processed.items():
        is_red = info['sample_date'] and info['sample_date'] > CAPE_INSEKT_HIGHLIGHT_DATE
        if is_red:
            red_insekt_raw.append(info)
        else:
            grey_insekt_raw.append(info)

    csu_red, csu_grey = load_csu_capek_data(csu_nc_dir, daily_radon)

    # ------------------ right panel data ------------------
    bigg_dataset = process_bigg_data()
    right_datasets = [bigg_dataset]
    for cfg in file_configs_historic:
        res = process_file_historic(cfg)
        if res is not None:
            right_datasets.append(res)

    # ================== FIGURE 7b ==================
    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    fig_b.subplots_adjust(top=0.82)

    added_labels_right = set()
    for ds in right_datasets:
        if ds['data'] is None or ds['data'].empty:
            continue
        grouped = ds['data'].sort_values('Temp_bin', ascending=True)
        color = ds['color']
        marker = ds.get('marker', 'o')
        linestyle = ds.get('linestyle', '-')
        box_width = 0.8

        # 1. Draw IQR rectangles for every bin
        for _, row in grouped.iterrows():
            rect = Rectangle((row['Temp_bin'] - box_width/2, row['Q1']),
                             box_width, row['Q3'] - row['Q1'],
                             facecolor=color, alpha=0.3, edgecolor='none', zorder=3)
            ax_b.add_patch(rect)

        # 2. Draw the median line (no markers)
        ax_b.plot(grouped['Temp_bin'], grouped['Median'],
                  color=color, linestyle=linestyle, linewidth=3.5,
                  zorder=5)

        # 3. Overlay markers with conditional face color.
        mask_ok = grouped['Count'] >= MIN_POINTS_FOR_COLOR
        mask_low = grouped['Count'] < MIN_POINTS_FOR_COLOR

        if ds['label'] == 'Bigg 1973':
            mask_ok = pd.Series(True, index=grouped.index)
            mask_low = pd.Series(False, index=grouped.index)

        if mask_ok.any():
            ax_b.scatter(grouped.loc[mask_ok, 'Temp_bin'],
                         grouped.loc[mask_ok, 'Median'],
                         color=color, marker=marker,
                         edgecolor='black', linewidth=1.0,
                         s=plt.rcParams['lines.markersize']**2,
                         zorder=6)
        if mask_low.any():
            ax_b.scatter(grouped.loc[mask_low, 'Temp_bin'],
                         grouped.loc[mask_low, 'Median'],
                         facecolor='white', edgecolor='black',
                         marker=marker, linewidth=1.0,
                         s=plt.rcParams['lines.markersize']**2,
                         zorder=6)

        if ds['label'] not in added_labels_right:
            ax_b.plot([], [], color=color, linestyle=linestyle,
                      marker=marker, markerfacecolor=color,
                      markeredgecolor='black', linewidth=1.8, markersize=7,
                      label=ds['label'])
            added_labels_right.add(ds['label'])

    ax_b.set_xlabel('Temperature (°C)')
    ax_b.set_ylabel('INP concentration (L⁻¹)')
    ax_b.set_xlim(-35, -4)
    ax_b.set_ylim(0.00006, 200)
    ax_b.set_yscale('log')
    ax_b.tick_params(which='minor', length=0)

    bright_labels = {'CAPE-k PINE 2025', 'CAPE-k INSEKT 2025', 'CAPE-k CSU-IS 2025'}
    iqr_handle = mlines.Line2D([], [], color='gray', marker='s',
                               linestyle='None',
                               markerfacecolor='gray', alpha=0.3,
                               markeredgecolor='none')
    handles_right = []
    labels_right = []
    for ds in right_datasets:
        if ds['label'] == 'Bigg 1973':
            handles_right.append(mlines.Line2D([], [], color=ds['color'],
                                               marker=ds.get('marker','o'),
                                               linestyle=ds.get('linestyle','-'),
                                               markerfacecolor=ds['color'],
                                               markeredgecolor='black', markeredgewidth=1.0))
            labels_right.append('Bigg 1973')
            break
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
                                               markeredgecolor='black', markeredgewidth=1.0))
            labels_right.append(lbl)
    for lbl in ['CAPE-k PINE 2025', 'CAPE-k INSEKT 2025', 'CAPE-k CSU-IS 2025']:
        ds = next((d for d in right_datasets if d['label'] == lbl), None)
        if ds:
            handles_right.append(mlines.Line2D([], [], color=ds['color'],
                                               marker=ds.get('marker','o'),
                                               linestyle=ds.get('linestyle','-'),
                                               markerfacecolor=ds['color'],
                                               markeredgecolor='black', markeredgewidth=1.0))
            labels_right.append(lbl)
    handles_right.append(iqr_handle)
    labels_right.append('IQR (25th-75th percentile)')

    low_count_handle = mlines.Line2D([], [], color='black', marker='o',
                                     markerfacecolor='white', markeredgecolor='black',
                                     linestyle='None', markersize=7,
                                     markeredgewidth=1.0)
    handles_right.append(low_count_handle)
    labels_right.append('Not enough data')

    ax_b.legend(handles_right, labels_right,
                loc='lower center', bbox_to_anchor=(0.5, 1.02),
                ncol=3, fontsize=9, framealpha=1,
                frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout(rect=[0, 0, 1, 1])
    fig_b.savefig('Figure7_corrected.png')
    fig_b.savefig('Figure7_corrected.pdf')
    print("Saved Figure7_corrected.png")
    plt.close(fig_b)