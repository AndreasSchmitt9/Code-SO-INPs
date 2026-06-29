"""
One figure with two subplots showing the relative change in INP concentrations during one day
((max − min) / min), split by baseline and continental air masses based on radon concentration.
For each day:
  - max = maximum 30‑min INP concentration
  - min = minimum **non‑zero** 30‑min INP concentration (days with all zeros are excluded)
Histogram uses a logarithmic x‑axis and shows percentage of days.
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
MIN_INTERVALS_PER_DAY = 10

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
    daily_avg = radon_series.resample('D').mean().dropna()
    return daily_avg

daily_radon = load_radon_daily_average(RADON_CSV_PATH)

csv_file = '/mnt/d/Cape24-data/Data/processedPINE/Capek.csv'
df = pd.read_csv(csv_file)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT-10').dt.tz_localize(None)
df = df.dropna(subset=['datetime', 'T_min', 'INP_cn_0'])

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
    max_inp = group['INP_mean'].max()
    pos_vals = group['INP_mean'][group['INP_mean'] > 0]
    if len(pos_vals) == 0:
        return pd.Series({
            'max_inp': max_inp,
            'min_inp': np.nan,
            'radon_daily': group['radon_daily'].iloc[0],
            'airmass': group['airmass'].iloc[0]
        })
    min_inp = pos_vals.min()
    radon_val = group['radon_daily'].iloc[0]
    airmass_label = group['airmass'].iloc[0]
    return pd.Series({
        'max_inp': max_inp,
        'min_inp': min_inp,
        'radon_daily': radon_val,
        'airmass': airmass_label
    })

daily = temp_data.groupby('date').apply(day_extreme_single, include_groups=False).reset_index()
daily = daily.dropna(subset=['min_inp'])

daily['diff_rel'] = (daily['max_inp'] - daily['min_inp']) / daily['min_inp']

baseline_diff_rel    = daily.loc[daily['airmass'] == 'Baseline', 'diff_rel']
continental_diff_rel = daily.loc[daily['airmass'] == 'Continental', 'diff_rel']

total_baseline = (daily['airmass'] == 'Baseline').sum()
total_continental = (daily['airmass'] == 'Continental').sum()

fig = plt.figure(figsize=(7, 2))
gs = gridspec.GridSpec(1, 2, figure=fig,
                       wspace=0.15,
                       left=0.08, right=0.98, top=0.95, bottom=0.25)

ax5 = fig.add_subplot(gs[0, 0])
ax6 = fig.add_subplot(gs[0, 1], sharey=ax5)

all_rel = pd.concat([baseline_diff_rel, continental_diff_rel])
min_rel = all_rel.min()
max_rel = all_rel.max()
if min_rel <= 0:
    min_rel = 1e-6
rel_bins = np.logspace(np.log10(min_rel), np.log10(max_rel), 12)

weights_baseline = np.ones_like(baseline_diff_rel) * 100.0 / total_baseline
weights_continental = np.ones_like(continental_diff_rel) * 100.0 / total_continental

ax5.hist(baseline_diff_rel, bins=rel_bins, weights=weights_baseline,
         alpha=0.7, color='steelblue', edgecolor='black')
ax6.hist(continental_diff_rel, bins=rel_bins, weights=weights_continental,
         alpha=0.7, color='darkorange', edgecolor='black')

ax5.set_xlabel('(max INP − min INP) / min INP')
ax5.set_ylabel('% of days')
ax5.set_xscale('log')

ax6.set_xlabel('(max INP − min INP) / min INP')
ax6.set_xscale('log')

ax5.grid(False)
ax6.grid(False)

ax5.legend(handles=[plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.7)],
           labels=['Baseline'], loc='upper right')
ax6.legend(handles=[plt.Rectangle((0,0),1,1, color='darkorange', alpha=0.7)],
           labels=['Continental'], loc='upper right')

fig.savefig('Figure3.png')
fig.savefig('Figure3.pdf')