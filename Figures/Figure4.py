"""
One figure with two subplots:
  (a) CAPE-k PINE INP concentration for a selected week. Same as Figure3 timeline but 2K, 30min average to show daily changes.
  (b) radon concentration during the selected week
  (c) Bivariate Polar plot showing INP concentration between -24 and -26 over wind direction and wind speed."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  
from matplotlib.colors import BoundaryNorm, LogNorm
import numpy as np
import matplotlib
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from pathlib import Path
import xarray as xr
import glob

matplotlib.use('Agg')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# =============================================================================
# Load PINE data
# =============================================================================
csv_file = '/mnt/d/Cape24-data/Data/processedPINE/Capek.csv'
df = pd.read_csv(csv_file)

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime', 'T_min', 'INP_cn_0'])

#Manual Quality control based on PIA data
df = df[
    (df['T_min'] < 253.15) & 
    (df['operation_number'] >= 11) & 
    (~df['operation_number'].between(24, 27, inclusive='both')) &
    (~df['operation_number'].isin([242, 244])) & 
    (~df['operation_number'].isin([43])) & 
    (~df['operation_number'].isin([74])) 
]


df['T_min_K'] = df['T_min']
df['T_min_C'] = df['T_min'] - 273.15
temperature_col_for_filtering = 'T_min_K'
temperature_col_for_display = 'T_min_C'

df = df[df[temperature_col_for_display] < -22]
#df = df[~df['datetime'].dt.month.isin([9, 11])] 

# =============================================================================
# Left panel: time series PINE
# =============================================================================
start_date = pd.Timestamp('2025-01-28')
end_date = pd.Timestamp('2025-02-05')
df_period = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].copy()

bin_edges_2K = np.arange(-34, -22 + 2, 2)
bin_labels_2K = [f"{int(bin_edges_2K[i])} to {int(bin_edges_2K[i+1])}°C" for i in range(len(bin_edges_2K)-1)]

df_period['temp_bin_2K'] = pd.cut(df_period[temperature_col_for_display],
                                  bins=bin_edges_2K,
                                  labels=bin_labels_2K)

df_period['30min_group'] = df_period['datetime'].dt.floor('30min')
fourhour_avg = df_period.groupby(['30min_group', 'temp_bin_2K'], observed=False).agg({
    'INP_cn_0': ['mean', 'count'],
    temperature_col_for_display: 'mean',
}).reset_index()
fourhour_avg.columns = ['datetime', 'temp_bin_2K', 'INP_cn_0_mean', 'INP_cn_0_count', 'T_C_mean']
fourhour_avg = fourhour_avg.dropna(subset=['temp_bin_2K'])

fourhour_nonzero_by_bin = {}
for bin_name in fourhour_avg['temp_bin_2K'].unique():
    bin_data = fourhour_avg[fourhour_avg['temp_bin_2K'] == bin_name].copy()
    fourhour_nonzero_by_bin[bin_name] = bin_data[bin_data['INP_cn_0_mean'] > 0] #Same as Figure2.py, just this time no zeroes are plotted.

# =============================================================================
# Radon data
# =============================================================================
RADON_FOLDER = "/mnt/d/Cape24-data/Model/Radon/release-v2"
BASELINE_RADON_THRESHOLD = 100

def load_radon_data():
    radon_files = ["2024.xlsx", "2025.xlsx"]
    all_data = []
    for filename in radon_files:
        file_path = Path(RADON_FOLDER) / filename
        df_r = pd.read_excel(file_path)
        datetime_col = None
        for col in df_r.columns:
            if str(col).lower() == 'datetime':
                datetime_col = col
                break
        radon_col = None
        for col in df_r.columns:
            if str(col).lower() == 'radon_stp':
                radon_col = col
                break
        if radon_col is None:
            for col in df_r.columns:
                if str(col).lower() == 'radon':
                    radon_col = col
                    break
        tmp = df_r[[datetime_col, radon_col]].copy()
        tmp.columns = ['Datetime', 'Radon_Concentration']
        tmp['Datetime'] = pd.to_datetime(tmp['Datetime'], errors='coerce')
        tmp = tmp.dropna()
        tmp = tmp[tmp['Radon_Concentration'] > 0] 
        tmp = tmp.sort_values('Datetime')
        all_data.append(tmp)
    return pd.concat(all_data, ignore_index=True).sort_values('Datetime')

radon_df = load_radon_data()
radon_period = radon_df[(radon_df['Datetime'] >= start_date) & (radon_df['Datetime'] <= end_date)].copy()

# =============================================================================
# meteorological data for polar plot
# =============================================================================
met_folder = '/mnt/d/Cape24-data/Met'
met_files = sorted(glob.glob(f'{met_folder}/*.cdf'))

def preprocess_met(ds):
    needed = ['wspd_vec_mean', 'wdir_vec_mean']
    return ds[needed]

met_ds = xr.open_mfdataset(met_files, combine='by_coords', preprocess=preprocess_met)
met_df = met_ds.to_dataframe().reset_index()
met_df['time'] = pd.to_datetime(met_df['time'])
met_df = met_df.set_index('time').sort_index()

wind_speed_col = 'wspd_vec_mean'
wind_dir_col   = 'wdir_vec_mean'

df['datetime_30min'] = df['datetime'].dt.round('30min')
met_df['time_30min'] = met_df.index.round('30min')
met_unique = met_df.drop_duplicates(subset='time_30min')
met_unique = met_unique.reset_index()[['time_30min', wind_speed_col, wind_dir_col]]

merged = pd.merge(df, met_unique, how='inner', left_on='datetime_30min', right_on='time_30min')

polar_temp_min = -26
polar_temp_max = -24
polar_data = merged[(merged[temperature_col_for_display] >= polar_temp_min) & 
                    (merged[temperature_col_for_display] <= polar_temp_max)]

# =============================================================================
# Create combined figure
# =============================================================================
fig = plt.figure(figsize=(7, 2.5))

gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1],
                       hspace=0.18, wspace=0.28,
                       left=0.165, right=0.84, top=0.93, bottom=0.09)

# ----- Left top: INP 4‑h averages -----
ax_inp = fig.add_subplot(gs[0, 0])

bin_edges = np.arange(-34, -20, 2)
cmap_temp = plt.cm.viridis
norm_temp = BoundaryNorm(bin_edges, cmap_temp.N)

for bin_name, bin_data in fourhour_nonzero_by_bin.items():
    
    temp_range = bin_name.replace('°C', '').split(' to ')
    temp_mid = (float(temp_range[0]) + float(temp_range[1])) / 2
    color = cmap_temp(norm_temp(temp_mid))
    ax_inp.scatter(bin_data['datetime'], bin_data['INP_cn_0_mean'], 
                       c=[color] * len(bin_data), alpha=0.9, s=30,
                       edgecolors='black', linewidths=0.3)

ax_inp.set_yscale('log')
ax_inp.set_ylabel('   INP conc. (stdL⁻¹)')
ax_inp.tick_params(axis='both', which='minor', width=0, length=0)
ax_inp.set_xlim(start_date, end_date)
ax_inp.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
ax_inp.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax_inp.tick_params(axis='x', labelbottom=False, bottom=False)
ax_inp.set_ylim(5e-2, 100)

ax_inp.text(0.02, 0.95, 'a)', transform=ax_inp.transAxes,
            weight='normal', va='top', ha='left')

# ----- Left bottom: Radon concentration -----
ax_radon = fig.add_subplot(gs[1, 0], sharex=ax_inp)
ax_radon.plot(radon_period['Datetime'], radon_period['Radon_Concentration'] * 1000,
                  color='darkred', linewidth=1.2, label='Radon')
ax_radon.axhline(BASELINE_RADON_THRESHOLD, color='grey', linestyle='--',
                     linewidth=1.0, alpha=0.8, label=f'{BASELINE_RADON_THRESHOLD} mBq m⁻³')

ax_radon.set_yscale('log')
ax_radon.set_ylim(0.2,10000)
ax_radon.set_ylabel('Radon (mBq m⁻³)')
ax_radon.legend(loc='lower right', framealpha=0.8)
ax_radon.tick_params(axis='both', which='minor', width=0, length=0)

ax_radon.text(0.02, 0.95, 'b)', transform=ax_radon.transAxes,
              weight='normal', va='top', ha='left')

fig.align_ylabels([ax_inp, ax_radon])

# ----- Right: polar wind plot -----
#Bivariate Polar Plot: This creates a polar plot of INP concentrations, Wind direction (angle) and wind speed (radius). It smoothes the datapoints by fitting a function over it.

ax_polar = fig.add_subplot(gs[:, 1], projection='polar')
ax_polar.set_aspect('equal')

pos_polar = ax_polar.get_position()
new_pos = [pos_polar.x0 - 0.03, pos_polar.y0, pos_polar.width, pos_polar.height]
ax_polar.set_position(new_pos)

ws = polar_data[wind_speed_col].values
wd = polar_data[wind_dir_col].values
inp = polar_data['INP_cn_0'].values


ws_spread, wd_spread, min_bin = 3.5, 5, 1 # Parameters for bivariate polar plot. Smoothes the Polar plot. I played around with it until it looked ok

wd_grid = np.arange(0, 360, 2)
ws_grid = np.arange(0, np.ceil(ws.max()) + 2, 0.5)
smoothed = np.full((len(wd_grid), len(ws_grid)), np.nan, dtype=float)

for i, wd0 in enumerate(wd_grid):
    for j, ws0 in enumerate(ws_grid):
        diff_ang = np.abs(wd - wd0)
        diff_ang = np.minimum(diff_ang, 360 - diff_ang)
        dist_wd = diff_ang / wd_spread
        dist_ws = (ws - ws0) / ws_spread
        weights = np.exp(-0.5 * (dist_wd**2 + dist_ws**2))
        total_weight = np.sum(weights)
        if total_weight >= min_bin:
            smoothed[i, j] = np.average(inp, weights=weights)

Theta, R = np.meshgrid(np.deg2rad(wd_grid), ws_grid)
vmin = np.nanmin(smoothed[smoothed > 0]) if np.any(smoothed > 0) else 1e-3
vmax = np.nanmax(smoothed)
pcm = ax_polar.pcolormesh(Theta, R, smoothed.T,
                              cmap='coolwarm', norm=LogNorm(vmin=vmin, vmax=vmax),
                              shading='auto')

#Calculates the polar plot angle for Melbourne from Cape Grim to add a line in the Plot for reference
cape_grim_lat = -40.6831
cape_grim_lon = 144.6897
melbourne_lat = -37.8136
melbourne_lon = 144.9631

lat1, lon1 = np.radians(cape_grim_lat), np.radians(cape_grim_lon)
lat2, lon2 = np.radians(melbourne_lat), np.radians(melbourne_lon)
dlon = lon2 - lon1

x = np.sin(dlon) * np.cos(lat2)
y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
bearing_rad = np.arctan2(x, y)
bearing_deg = (np.degrees(bearing_rad) + 360) % 360

ax_polar.plot([np.deg2rad(bearing_deg), np.deg2rad(bearing_deg)], [28, 30],
            color='black', linewidth=2.5, solid_capstyle='butt')
ax_polar.annotate('Melbourne', xy=(np.deg2rad(bearing_deg), 30),
                      xytext=(0, 2), textcoords='offset points',
                      ha='center', va='bottom', color='black', weight='normal')

ax_polar.set_theta_zero_location('N')
ax_polar.set_theta_direction(-1)
ax_polar.set_thetagrids(np.arange(0, 360, 45), labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
radial_ticks = np.arange(0, 31, 5)
labels = [f"{r:.0f}" for r in radial_ticks]
labels = ["  ws. (m s$^{-1}$)" if r == 30 else f"{r:.0f}" for r in radial_ticks]
ax_polar.set_rgrids(radial_ticks, labels=labels, angle=135)
ax_polar.set_ylim(0, 30)
ax_polar.text(-0.15, 0.97, 'c)', transform=ax_polar.transAxes,
                  weight='normal', va='top', ha='left')

pos_polar_new = ax_polar.get_position()
cax_wind = fig.add_axes([pos_polar_new.x1 + 0.08, pos_polar_new.y0, 0.015, pos_polar_new.height])
cbar_wind = fig.colorbar(pcm, cax=cax_wind, label='INP conc. for T in [-24°C, -26 °C] \n  (stdL⁻¹) ')
ticks = [0.1, 0.5, 1, 5, 10]
ticks = [t for t in ticks if vmin <= t <= vmax]
cbar_wind.set_ticks(ticks)
cbar_wind.set_ticklabels([f"{t:g}" for t in ticks])

#Temperature colorbar for left subplot
pos_inp = ax_inp.get_position()
pos_radon = ax_radon.get_position()
cbar_ax = fig.add_axes([pos_inp.x0 - 0.092, pos_radon.y0, 0.015, pos_inp.y1 - pos_radon.y0])
sm = plt.cm.ScalarMappable(norm=norm_temp, cmap=cmap_temp)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cbar_ax, boundaries=bin_edges, ticks=bin_edges)
cbar.set_ticklabels([f"{int(edge)}" for edge in bin_edges])
cbar.set_label('Temperature (°C)')
cbar.ax.invert_yaxis()
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')


plt.savefig('Figure4.png', dpi=300)
plt.savefig('Figure4.pdf', dpi=300)
plt.close()