"""
One figure with six subplots:
  (a) and (b) show the diurnal percentage of maximum INP concentration laying at a specific time of day. 
  (c) and (d) show the percentage of days in which minimum INP conentration lay at a specific time of day. 
  If there was a real diurnal change, the maximum would always be in the middle of the day, and the minimum would be at night.
  (e) and (f) show the absolute change in INP concentrations during one day. 
  The plots are split up to baseline and continental based on radon concentration. 
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

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

RADON_CSV_PATH = "/mnt/d/Cape24-data/Model/Radon/release-v2/combined_radon_data.csv"
RADON_THRESHOLD = 0.1
TEMP_LOW  = -26.0
TEMP_HIGH = -24.0
MIN_INTERVALS_PER_DAY = 10 # All days with less than 10 valid 30min averages at the same temperature are discarded. 

def load_radon_daily_average(csv_path):
    df = pd.read_csv(csv_path)
    datetime_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in ['datetime', 'time', 'date']):
            datetime_col = col
            break
        
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df = df.dropna(subset=[datetime_col])
    radon_col = None
    for col in df.columns:
        if 'radon' in col.lower():
            radon_col = col
            break
    
    df[radon_col] = pd.to_numeric(df[radon_col], errors='coerce')
    radon_series = df.set_index(datetime_col)[radon_col].sort_index()
    daily_avg = radon_series.resample('D').mean().dropna() # Daily average for Baseline or Continental days. 
    return daily_avg

daily_radon = load_radon_daily_average(RADON_CSV_PATH)

csv_file = '/mnt/d/Cape24-data/Data/processedPINE/Capek.csv' # PINE data
df = pd.read_csv(csv_file)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT-10').dt.tz_localize(None) #Convert to local Time
df = df.dropna(subset=['datetime', 'T_min', 'INP_cn_0'])

#PINE quality control. Manually selected based on PIA data.
df = df[
    (df['T_min'] < 257) &
    (df['operation_number'] >= 11) &
    (~df['operation_number'].between(24, 27, inclusive='both')) &
    (~df['operation_number'].isin([242, 244, 43, 74]))
]

df['T_min_C'] = df['T_min'] - 273.15

df['time_bin'] = df['datetime'].dt.floor('30min')
avg_30min = df.groupby('time_bin').agg({
    'INP_cn_0': 'mean',
    'T_min_C': 'mean',
    'datetime': 'count'
}).reset_index()
avg_30min.columns = ['datetime', 'INP_mean', 'T_mean', 'count']
avg_30min = avg_30min[avg_30min['count'] >= 1]
avg_30min['date'] = avg_30min['datetime'].dt.date

avg_30min['radon_daily'] = avg_30min['date'].apply(
    lambda d: daily_radon.get(pd.Timestamp(d), np.nan)
)
avg_30min = avg_30min.dropna(subset=['radon_daily'])

avg_30min['airmass'] = np.where(
    avg_30min['radon_daily'] <= RADON_THRESHOLD, 'Baseline', 'Continental'
)

temp_mask = (avg_30min['T_mean'] >= TEMP_LOW) & (avg_30min['T_mean'] <= TEMP_HIGH)
temp_data = avg_30min[temp_mask].copy()

day_counts = temp_data.groupby('date')['datetime'].count()
valid_dates = day_counts[day_counts >= MIN_INTERVALS_PER_DAY].index
temp_data = temp_data[temp_data['date'].isin(valid_dates)]

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

baseline_max_hours    = daily.loc[daily['airmass'] == 'Baseline', 'max_hour']
continental_max_hours = daily.loc[daily['airmass'] == 'Continental', 'max_hour']
baseline_min_hours    = daily.loc[daily['airmass'] == 'Baseline', 'min_hour']
continental_min_hours = daily.loc[daily['airmass'] == 'Continental', 'min_hour']

daily['diff_abs'] = daily['max_inp'] - daily['min_inp']
baseline_diff_abs    = daily.loc[daily['airmass'] == 'Baseline', 'diff_abs']
continental_diff_abs = daily.loc[daily['airmass'] == 'Continental', 'diff_abs']

total_baseline = (daily['airmass'] == 'Baseline').sum()
total_continental = (daily['airmass'] == 'Continental').sum()

fig = plt.figure(figsize=(7, 4.5))
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

time_bins_4h = np.arange(0, 25, 3)

# Top row: Maximum time in percentage of days
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


weights_max_co = np.ones_like(continental_max_hours, dtype=float) * 100.0 / total_continental
ax2.hist(continental_max_hours, bins=time_bins_4h, density=False,
             weights=weights_max_co, alpha=0.7, color='darkorange', edgecolor='black')


ax2.set_title('Continental')
ax2.set_xlim(0, 24)
ax2.set_xticks(np.arange(0, 24, 3))
ax2.set_xlabel('Local time of Maximum INP conc')
ax2.tick_params(bottom=False, labelbottom=False)

# Middle row: Minimum time
weights_min_bl = np.ones_like(baseline_min_hours, dtype=float) * 100.0 / total_baseline
ax3.hist(baseline_min_hours, bins=time_bins_4h, density=False,
             weights=weights_min_bl, alpha=0.7, color='steelblue', edgecolor='black')


ax3.set_ylabel('% of days')
ax3.set_xlabel('Local time of Minimum INP conc')
ax3.set_xlim(0, 24)
ax3.set_xticks(np.arange(0, 24, 3))

weights_min_co = np.ones_like(continental_min_hours, dtype=float) * 100.0 / total_continental
ax4.hist(continental_min_hours, bins=time_bins_4h, density=False,
             weights=weights_min_co, alpha=0.7, color='darkorange', edgecolor='black')


ax4.set_xlabel('Local time of Minimum INP conc')
ax4.set_xlim(0, 24)
ax4.set_xticks(np.arange(0, 24, 3))

# Bottom panel: Absolute difference
baseline_abs = baseline_diff_abs[baseline_diff_abs > 0]
continental_abs = continental_diff_abs[continental_diff_abs > 0]


all_pos_abs = np.concatenate([baseline_abs.values,continental_abs.values])
min_abs = all_pos_abs.min()
max_abs = all_pos_abs.max()
abs_bins = np.logspace(np.log10(min_abs), np.log10(max_abs), 15)


ax5.hist(baseline_abs, bins=abs_bins, density=False,
             weights=np.ones_like(baseline_abs) / len(baseline_abs),
             alpha=0.7, color='steelblue', edgecolor='black')

ax5.set_xlabel('max − min (L⁻¹)')
ax5.set_ylabel('% of days')
ax5.set_xscale('log')


ax6.hist(continental_abs, bins=abs_bins, density=False,
             weights=np.ones_like(continental_abs) / len(continental_abs),
             alpha=0.7, color='darkorange', edgecolor='black')

ax6.set_xlabel('max − min (L⁻¹)')
ax6.set_xscale('log')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.grid(False)

ax1.legend(handles=[plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.7)],
           labels=['Baseline'], loc='upper right')
ax2.legend(handles=[plt.Rectangle((0,0),1,1, color='darkorange', alpha=0.7)],
           labels=['Continental'], loc='upper right')
ax3.legend(handles=[plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.7)],
           labels=['Baseline'], loc='upper right')
ax4.legend(handles=[plt.Rectangle((0,0),1,1, color='darkorange', alpha=0.7)],
           labels=['Continental'], loc='upper right')
ax5.legend(handles=[plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.7)],
           labels=['Baseline'], loc='upper right')
ax6.legend(handles=[plt.Rectangle((0,0),1,1, color='darkorange', alpha=0.7)],
           labels=['Continental'], loc='upper right')

fig.savefig('Figure3.png')
fig.savefig('Figure3.pdf')