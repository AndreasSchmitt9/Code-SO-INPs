import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from datetime import datetime

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

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
RADON_CSV_PATH = "/mnt/d/Cape24-data/Model/Radon/release-v2/combined_radon_data.csv"
RADON_THRESHOLD = 0.1   # Bq/m³ (≤ threshold = Baseline, > threshold = Continental)

TEMP_LOW  = -26.0
TEMP_HIGH = -24.0

MIN_INTERVALS_PER_DAY = 10   # minimum 30‑min intervals inside T window

# ----------------------------------------------------------------------
# 1. LOAD RADON DATA – daily averages
# ----------------------------------------------------------------------
def load_radon_daily_average(csv_path):
    """Return a Series of daily mean radon concentrations, indexed by date."""
    try:
        df = pd.read_csv(csv_path)
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

daily_radon = load_radon_daily_average(RADON_CSV_PATH)
if daily_radon.empty:
    print("WARNING: No radon data available. All days will be treated as Continental.")
    daily_radon = pd.Series(dtype=float)

# ----------------------------------------------------------------------
# 2. LOAD PINE RAW DATA, QUALITY FILTERS, THEN 30‑MIN AVERAGING
# ----------------------------------------------------------------------
csv_file = '/mnt/d/Cape24-data/Data/processedPINE/Capek.csv'
df = pd.read_csv(csv_file)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT-10').dt.tz_localize(None)
df = df.dropna(subset=['datetime', 'T_min', 'INP_cn_0'])

# Quality filters
df = df[
    (df['T_min'] < 257) & 
    (df['operation_number'] >= 11) & 
    (~df['operation_number'].between(24, 27, inclusive='both')) &
    (~df['operation_number'].isin([242, 244, 43, 74])) 
]

df['T_min_C'] = df['T_min'] if df['T_min'].max() < 100 else df['T_min'] - 273.15

# 30‑minute averaging
df['time_bin'] = df['datetime'].dt.floor('30min')
avg_30min = df.groupby('time_bin').agg({
    'INP_cn_0': 'mean',
    'T_min_C': 'mean',
    'datetime': 'count'
}).reset_index()
avg_30min.columns = ['datetime', 'INP_mean', 'T_mean', 'count']
avg_30min = avg_30min[avg_30min['count'] >= 1]   # bins with at least one measurement
avg_30min['date'] = avg_30min['datetime'].dt.date

# ----------------------------------------------------------------------
# 3. ADD DAILY RADON AND CLASSIFY
# ----------------------------------------------------------------------
avg_30min['radon_daily'] = avg_30min['date'].apply(
    lambda d: daily_radon.get(pd.Timestamp(d), np.nan)
)
avg_30min = avg_30min.dropna(subset=['radon_daily'])
if avg_30min.empty:
    print("No data after merging radon – check paths/dates.")
    exit()

avg_30min['airmass'] = np.where(
    avg_30min['radon_daily'] <= RADON_THRESHOLD, 'Baseline', 'Continental'
)

# ----------------------------------------------------------------------
# 4. RESTRICT TO TEMPERATURE WINDOW
# ----------------------------------------------------------------------
temp_mask = (avg_30min['T_mean'] >= TEMP_LOW) & (avg_30min['T_mean'] <= TEMP_HIGH)
temp_data = avg_30min[temp_mask].copy()
if temp_data.empty:
    print(f"No data in temperature range {TEMP_LOW} to {TEMP_HIGH} °C.")
    exit()

# ----------------------------------------------------------------------
# 5. EXCLUDE DAYS WITH FEWER THAN MIN_INTERVALS_PER_DAY 30‑MIN INTERVALS
# ----------------------------------------------------------------------
day_counts = temp_data.groupby('date')['datetime'].count()
valid_dates = day_counts[day_counts >= MIN_INTERVALS_PER_DAY].index
temp_data = temp_data[temp_data['date'].isin(valid_dates)]

if temp_data.empty:
    print(f"No days with at least {MIN_INTERVALS_PER_DAY} 30‑min intervals in the temperature window.")
    exit()

# ----------------------------------------------------------------------
# 6. EXTRACT SINGLE MAXIMUM AND SINGLE MINIMUM PER DAY (30‑min averages)
# ----------------------------------------------------------------------
def day_extreme_single(group):
    idx_max = group['INP_mean'].idxmax()
    idx_min = group['INP_mean'].idxmin()
    
    max_hour = group.loc[idx_max, 'datetime'].hour
    min_hour = group.loc[idx_min, 'datetime'].hour
    
    max_inp = group.loc[idx_max, 'INP_mean']
    min_inp = group.loc[idx_min, 'INP_mean']
    
    radon_val = group['radon_daily'].iloc[0]
    airmass_label = group['airmass'].iloc[0]
    
    return pd.Series({
        'max_hour': max_hour,
        'min_hour': min_hour,
        'max_inp': max_inp,
        'min_inp': min_inp,
        'radon_daily': radon_val,
        'airmass': airmass_label
    })

daily = temp_data.groupby('date').apply(day_extreme_single).reset_index()

# Split by airmass
baseline_max_hours    = daily.loc[daily['airmass'] == 'Baseline', 'max_hour']
continental_max_hours = daily.loc[daily['airmass'] == 'Continental', 'max_hour']

baseline_min_hours    = daily.loc[daily['airmass'] == 'Baseline', 'min_hour']
continental_min_hours = daily.loc[daily['airmass'] == 'Continental', 'min_hour']

# Daily INP range (absolute)
daily['diff_abs'] = daily['max_inp'] - daily['min_inp']
baseline_diff_abs     = daily.loc[daily['airmass'] == 'Baseline', 'diff_abs']
continental_diff_abs  = daily.loc[daily['airmass'] == 'Continental', 'diff_abs']

# --- Compute total days per category for percentage normalisation ---
total_baseline = (daily['airmass'] == 'Baseline').sum()
total_continental = (daily['airmass'] == 'Continental').sum()

# ----------------------------------------------------------------------
# 7. PLOT – Custom GridSpec with spacer rows for tailored vertical gaps
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(7, 4.5))

# 5 rows: [top plots, tiny gap, middle plots, bigger gap, bottom plots]
gs = gridspec.GridSpec(5, 2, figure=fig,
                       height_ratios=[1, 0.25, 1, 0.5, 1],
                       hspace=0.0,
                       wspace=0.15,
                       left=0.08, right=0.98, top=0.99, bottom=0.1)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax4 = fig.add_subplot(gs[2, 1], sharex=ax2, sharey=ax3)

ax5 = fig.add_subplot(gs[4, 0])
ax6 = fig.add_subplot(gs[4, 1], sharey=ax5)

# 4‑hour bins for max/min histograms
time_bins_4h = np.arange(0, 25, 3)

# ---------- Top row: Maximum time (percentage of days) ----------
if total_baseline > 0 and not baseline_max_hours.empty:
    weights_max_bl = np.ones_like(baseline_max_hours, dtype=float) * 100.0 / total_baseline
    ax1.hist(baseline_max_hours, bins=time_bins_4h, density=False,
             weights=weights_max_bl, alpha=0.7, color='steelblue', edgecolor='black')
else:
    ax1.text(0.5, 0.5, 'No data', transform=ax1.transAxes, ha='center', va='center')

ax1.set_title('Baseline')
ax1.set_ylabel('% of days')
ax1.set_xlim(0, 24)
ax1.set_xticks(np.arange(0, 24, 3))
ax1.set_xlabel('Local time of Maximum INP conc')
ax1.tick_params(bottom=False, labelbottom=False)

if total_continental > 0 and not continental_max_hours.empty:
    weights_max_co = np.ones_like(continental_max_hours, dtype=float) * 100.0 / total_continental
    ax2.hist(continental_max_hours, bins=time_bins_4h, density=False,
             weights=weights_max_co, alpha=0.7, color='darkorange', edgecolor='black')
else:
    ax2.text(0.5, 0.5, 'No data', transform=ax2.transAxes, ha='center', va='center')

ax2.set_title('Continental')
ax2.set_xlim(0, 24)
ax2.set_xticks(np.arange(0, 24, 3))
ax2.set_xlabel('Local time of Maximum INP conc')
ax2.tick_params(bottom=False, labelbottom=False)

# ---------- Middle row: Minimum time (percentage of days) ----------
if total_baseline > 0 and not baseline_min_hours.empty:
    weights_min_bl = np.ones_like(baseline_min_hours, dtype=float) * 100.0 / total_baseline
    ax3.hist(baseline_min_hours, bins=time_bins_4h, density=False,
             weights=weights_min_bl, alpha=0.7, color='steelblue', edgecolor='black')
else:
    ax3.text(0.5, 0.5, 'No data', transform=ax3.transAxes, ha='center', va='center')

ax3.set_ylabel('% of days')
ax3.set_xlabel('Local time of Minimum INP conc')
ax3.set_xlim(0, 24)
ax3.set_xticks(np.arange(0, 24, 3))

if total_continental > 0 and not continental_min_hours.empty:
    weights_min_co = np.ones_like(continental_min_hours, dtype=float) * 100.0 / total_continental
    ax4.hist(continental_min_hours, bins=time_bins_4h, density=False,
             weights=weights_min_co, alpha=0.7, color='darkorange', edgecolor='black')
else:
    ax4.text(0.5, 0.5, 'No data', transform=ax4.transAxes, ha='center', va='center')

ax4.set_xlabel('Local time of Minimum INP conc')
ax4.set_xlim(0, 24)
ax4.set_xticks(np.arange(0, 24, 3))

# ---------- Bottom row: Absolute difference (probability per bin, log x) ----------
baseline_abs = baseline_diff_abs[baseline_diff_abs > 0]
continental_abs = continental_diff_abs[continental_diff_abs > 0]

if not baseline_abs.empty or not continental_abs.empty:
    all_pos_abs = np.concatenate([baseline_abs.values if not baseline_abs.empty else [],
                                  continental_abs.values if not continental_abs.empty else []])
    min_abs = all_pos_abs.min()
    max_abs = all_pos_abs.max()
    if min_abs == max_abs:
        min_abs /= 2
        max_abs *= 2
    abs_bins = np.logspace(np.log10(min_abs), np.log10(max_abs), 15)
else:
    abs_bins = np.logspace(-3, 0, 10)

if not baseline_abs.empty:
    ax5.hist(baseline_abs, bins=abs_bins, density=False,
             weights=np.ones_like(baseline_abs) / len(baseline_abs),
             alpha=0.7, color='steelblue', edgecolor='black')
else:
    ax5.text(0.5, 0.5, 'No data', transform=ax5.transAxes, ha='center', va='center')
ax5.set_xlabel('max − min (L⁻¹)')
ax5.set_ylabel('% of days')
ax5.set_xscale('log')

if not continental_abs.empty:
    ax6.hist(continental_abs, bins=abs_bins, density=False,
             weights=np.ones_like(continental_abs) / len(continental_abs),
             alpha=0.7, color='darkorange', edgecolor='black')
else:
    ax6.text(0.5, 0.5, 'No data', transform=ax6.transAxes, ha='center', va='center')
ax6.set_xlabel('max − min (L⁻¹)')
ax6.set_xscale('log')

# ----------------------------------------------------------------------
# Final adjustments
# ----------------------------------------------------------------------
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.grid(False)
ax1.legend(handles=[plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.7)],
           labels=['Baseline'], loc='upper right')
ax2.legend(handles=[plt.Rectangle((0,0),1,1, color='darkorange', alpha=0.7)],
           labels=['Continental'], loc='upper right')

# --- Middle row: Minimum time ---
ax3.legend(handles=[plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.7)],
           labels=['Baseline'], loc='upper right')
ax4.legend(handles=[plt.Rectangle((0,0),1,1, color='darkorange', alpha=0.7)],
           labels=['Continental'], loc='upper right')

# --- Bottom row: Absolute difference ---
ax5.legend(handles=[plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.7)],
           labels=['Baseline'], loc='upper right')  # different loc to avoid clash
ax6.legend(handles=[plt.Rectangle((0,0),1,1, color='darkorange', alpha=0.7)],
           labels=['Continental'], loc='upper right')
fig.savefig('DiurnalPDF.png')
fig.savefig('DiurnalPDF.pdf')
print("Saved: 30minAvg_SingleExtremes_1hBars_min5intervals.png/pdf")
print(f"Baseline days: {total_baseline}, Continental days: {total_continental}")
print(f"Baseline abs. diff points: {len(baseline_abs)}, Continental abs. diff points: {len(continental_abs)}")