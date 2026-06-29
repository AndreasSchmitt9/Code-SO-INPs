import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
from matplotlib.colors import BoundaryNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# =============================================================================
# 1. Read and filter PINE INP data
# =============================================================================
csv_pine = '/mnt/d/Cape24-data/Data/processedPINE/Capek.csv'
df_pine = pd.read_csv(csv_pine)

df_pine['datetime'] = pd.to_datetime(df_pine['datetime'], errors='coerce')
df_pine = df_pine.dropna(subset=['datetime', 'T_min', 'INP_cn_0'])

df_pine = df_pine[
    (df_pine['T_min'] < 253.15) &
    (df_pine['operation_number'] >= 11) &
    (~df_pine['operation_number'].between(24, 27, inclusive='both')) &
    (~df_pine['operation_number'].isin([242, 244])) &
    (~df_pine['operation_number'].isin([43])) &
    (~df_pine['operation_number'].isin([74]))
]

if df_pine['T_min'].max() < 100:
    df_pine['T_min_C'] = df_pine['T_min']
else:
    df_pine['T_min_C'] = df_pine['T_min'] - 273.15

df_pine = df_pine[df_pine['T_min_C'] < -22]
df_pine = df_pine[~df_pine['datetime'].dt.month.isin([9])]

print(f"PINE data filtered: {len(df_pine)} measurements remaining")

# =============================================================================
# 2. PINE temperature binning & 4‑hour averages
# =============================================================================
pine_bin_edges = np.arange(-34, -22 + 2, 2)
pine_bin_labels = [f"{int(pine_bin_edges[i])} to {int(pine_bin_edges[i+1])}°C"
                   for i in range(len(pine_bin_edges) - 1)]
pine_bin_centres = (pine_bin_edges[:-1] + pine_bin_edges[1:]) / 2.0

df_pine['temp_bin_2K'] = pd.cut(df_pine['T_min_C'], bins=pine_bin_edges,
                                labels=pine_bin_labels, include_lowest=True)
df_pine['4h_group'] = df_pine['datetime'].dt.floor('4h')

fourhour_avg = df_pine.groupby(['4h_group', 'temp_bin_2K'], observed=False).agg(
    INP_cn_0_mean=('INP_cn_0', 'mean'),
    INP_cn_0_count=('INP_cn_0', 'count'),
    INP_cn_0_frac_zero=('INP_cn_0', lambda x: np.mean(x == 0)),
    T_C_mean=('T_min_C', 'mean'),
).reset_index()
fourhour_avg = fourhour_avg.dropna(subset=['temp_bin_2K'])

fourhour_nonzero = {}
fourhour_zeros = {}
for bin_name in fourhour_avg['temp_bin_2K'].unique():
    bin_data = fourhour_avg[fourhour_avg['temp_bin_2K'] == bin_name]
    fourhour_nonzero[bin_name] = bin_data[bin_data['INP_cn_0_mean'] > 0]
    fourhour_zeros[bin_name] = bin_data[bin_data['INP_cn_0_mean'] == 0]

# ---------------------------------------------------------------------------
# PINE monthly median and IQR (computed from all data)
# ---------------------------------------------------------------------------
def pine_monthly_stats(bin_name):
    bin_data = fourhour_avg[fourhour_avg['temp_bin_2K'] == bin_name].copy()
    if bin_data.empty:
        return None
    bin_data['month_start'] = bin_data['4h_group'].dt.to_period('M').apply(lambda r: r.start_time)
    monthly = bin_data.groupby('month_start')['INP_cn_0_mean'].quantile(
        [0.25, 0.5, 0.75]).unstack()
    monthly.columns = ['q25', 'median', 'q75']
    monthly = monthly.reset_index()
    monthly['plot_date'] = monthly['month_start'] + pd.Timedelta(days=14)
    monthly['temp_bin_2K'] = bin_name
    temps = bin_name.replace('°C', '').split(' to ')
    monthly['temp_min_C'] = float(temps[0])
    monthly['temp_max_C'] = float(temps[1])
    monthly['temp_mid'] = (monthly['temp_min_C'] + monthly['temp_max_C']) / 2
    return monthly

pine_monthly = pd.concat([pine_monthly_stats(b) for b in pine_bin_labels])
pine_monthly = pine_monthly.dropna(subset=['median'])

# =============================================================================
# 3. Read and filter ARM-INS (CSU-IS) data
# =============================================================================
csv_csu = "/mnt/d/Cape24-data/CSU-CAPEk/combined_CSU_all.csv"
df_csu = pd.read_csv(csv_csu)
df_csu['datetime'] = pd.to_datetime(df_csu['base_time'], unit='s', errors='coerce')
df_csu = df_csu.dropna(subset=['datetime', 'temperature', 'n_inp_stp'])

df_csu = df_csu[(df_csu['temperature'] >= -28.0) & (df_csu['temperature'] <= -10.0)].copy()
print(f"ARM-INS data filtered: {len(df_csu)} measurements")

csu_bin_edges = np.arange(-28.0, -10.0 + 2.0, 2.0)
csu_bin_centres = (csu_bin_edges[:-1] + csu_bin_edges[1:]) / 2.0

df_csu['temp_bin_centre'] = pd.cut(df_csu['temperature'], bins=csu_bin_edges,
                                   labels=csu_bin_centres, include_lowest=True).astype(float)

selected_csu_centres = [-25.0, -23.0]
csu_monthly_stats_dict = {}
for centre in selected_csu_centres:
    csu_sel = df_csu[df_csu['temp_bin_centre'] == centre].copy()
    if not csu_sel.empty:
        csu_sel['month'] = csu_sel['datetime'].dt.to_period('M')
        monthly = csu_sel.groupby('month')['n_inp_stp'].quantile(
            [0.25, 0.5, 0.75]).unstack()
        monthly.columns = ['q25', 'median', 'q75']
        monthly = monthly.reset_index()
        monthly['month_start'] = monthly['month'].dt.to_timestamp()
        monthly['plot_date'] = monthly['month_start'] + pd.Timedelta(days=14)
        csu_monthly_stats_dict[centre] = monthly

# =============================================================================
# 4. Set up figure with 3 vertical panels
# =============================================================================
fig = plt.figure(figsize=(7, 4.5))
gs = gridspec.GridSpec(3, 1, height_ratios=[1.8, 0.2, 1.8],
                       hspace=0.3,
                       left=0.09, right=0.85, top=0.98, bottom=0.12)

ax_pine_4h = fig.add_subplot(gs[0])
ax_zeros = fig.add_subplot(gs[1])
ax_combined = fig.add_subplot(gs[2])

# =============================================================================
# 5. PINE colour map
# =============================================================================
pine_cmap = plt.cm.viridis
pine_norm = BoundaryNorm(pine_bin_edges, pine_cmap.N)

# ---------------------------------------------------------------------------
# a) 4‑hour averages (non-zero only) – no x‑axis ticks
# ---------------------------------------------------------------------------
for bin_name in pine_bin_labels:
    temp_range = bin_name.replace('°C', '').split(' to ')
    temp_mid = (float(temp_range[0]) + float(temp_range[1])) / 2
    color = pine_cmap(pine_norm(temp_mid))

    nonzero = fourhour_nonzero.get(bin_name)
    if nonzero is not None and not nonzero.empty:
        ax_pine_4h.scatter(nonzero['4h_group'], nonzero['INP_cn_0_mean'],
                           color=color, edgecolors='black', linewidths=0.4,
                           s=50, alpha=0.9, marker='o')

ax_pine_4h.set_yscale('log')
ax_pine_4h.set_ylabel('4‑h Avg. INP conc. (stdL⁻¹)')
ax_pine_4h.set_ylim(0.02, 800)
ax_pine_4h.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
ax_pine_4h.yaxis.set_major_formatter(FixedFormatter(['10⁻²', '10⁻¹', '10⁰', '10¹', '10²']))
ax_pine_4h.tick_params(axis='y', which='minor', width=0, length=0)

ax_pine_4h.set_xlim(pd.Timestamp('2024-11-01'), pd.Timestamp('2025-09-30'))
ax_pine_4h.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# ---------------------------------------------------------------------------
# Zero values scatter – remove November 2024 tick
# ---------------------------------------------------------------------------
has_zeros = False
for bin_name in pine_bin_labels:
    temp_range = bin_name.replace('°C', '').split(' to ')
    temp_mid = (float(temp_range[0]) + float(temp_range[1])) / 2
    color = pine_cmap(pine_norm(temp_mid))
    zero_data = fourhour_zeros.get(bin_name)
    if zero_data is not None and not zero_data.empty:
        has_zeros = True
        ax_zeros.scatter(zero_data['4h_group'], [0] * len(zero_data),
                         color=color, edgecolors='black', linewidths=0.3,
                         s=50, alpha=0.9, marker='o')

if has_zeros:
    ax_zeros.set_ylim(-0.5, 0.5)
    ax_zeros.set_yticks([0])
    ax_zeros.set_yticklabels(['0'])
else:
    ax_zeros.text(0.5, 0.5, 'No zero values', ha='center', va='center',
                  transform=ax_zeros.transAxes)

ax_zeros.tick_params(axis='both', which='minor', width=0, length=0)
ax_zeros.set_xlim(pd.Timestamp('2024-11-01'), pd.Timestamp('2025-09-30'))

# Build tick list, then remove November 2024
tick_dates_restricted = []
for year in [2024, 2025]:
    for month in range(1, 13):
        if (year == 2024 and month >= 11) or (year == 2025 and month <= 9):
            tick_dates_restricted.append(pd.Timestamp(year=year, month=month, day=1))

tick_dates_restricted = [d for d in tick_dates_restricted if d != pd.Timestamp('2024-11-01')]

ax_zeros.set_xticks(tick_dates_restricted)
ax_zeros.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax_zeros.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax_zeros.tick_params(axis='x', which='major', labelsize=9)

# ---------------------------------------------------------------------------
# Vertical lines
# ---------------------------------------------------------------------------
black_dates = [
    pd.Timestamp('2024-11-27'),
    pd.Timestamp('2025-05-26'),
    pd.Timestamp('2025-06-06'),
    pd.Timestamp('2025-06-23'),
    pd.Timestamp('2025-07-07'),
    pd.Timestamp('2025-08-26'),
    pd.Timestamp('2025-02-13')
    #pd.Timestamp('2025-03-06'),
    #pd.Timestamp('2025-03-28')
]

for ax in [ax_pine_4h, ax_zeros]:
    for d in black_dates:
        ax.axvline(d, color='black', linestyle='--', linewidth=1, alpha=0.7)
    #ax.axvline(pd.Timestamp('2025-02-13'), color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(pd.Timestamp('2025-03-06'), color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(pd.Timestamp('2025-03-28'), color='red', linestyle='--', linewidth=1, alpha=0.7)

handles_vlines = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='A. Mainland'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Tasmania')
]
ax_pine_4h.legend(handles=handles_vlines, loc='upper right', frameon=True,
                  facecolor='white', edgecolor='none', fontsize=8)
# ---------------------------------------------------------------------------
# b) Combined plot
# ---------------------------------------------------------------------------
highlight_bin = "-26 to -24°C"

for tmin in sorted(pine_monthly['temp_min_C'].unique()):
    bin_data = pine_monthly[pine_monthly['temp_min_C'] == tmin].sort_values('plot_date')
    if bin_data.empty:
        continue
    mask = ~((bin_data['month_start'].dt.month == 11) & (bin_data['month_start'].dt.year == 2024) |
             (bin_data['month_start'].dt.month == 9) & (bin_data['month_start'].dt.year == 2025))
    bin_data_plot = bin_data[mask]
    if bin_data_plot.empty:
        continue

    temp_mid = tmin + 1
    color = pine_cmap(pine_norm(temp_mid))

    ax_combined.plot(bin_data_plot['plot_date'], bin_data_plot['median'],
                     color=color, linewidth=3.0, marker='o', markersize=6,
                     markeredgecolor='black', markeredgewidth=0.4,
                     zorder=10, alpha=0.9, label='')

    if bin_data_plot['temp_bin_2K'].iloc[0] == highlight_bin:
        ax_combined.fill_between(bin_data_plot['plot_date'],
                                 bin_data_plot['q25'], bin_data_plot['q75'],
                                 color=color, alpha=0.25, edgecolor='none', zorder=5, label='')

for centre in selected_csu_centres:
    if centre not in csu_monthly_stats_dict:
        continue
    df_csu_stats = csu_monthly_stats_dict[centre]
    csu_color = pine_cmap(pine_norm(centre))
    if centre == -25.0:
        ax_combined.fill_between(df_csu_stats['plot_date'],
                                 df_csu_stats['q25'], df_csu_stats['q75'],
                                 color=csu_color, alpha=0.2, edgecolor='none', zorder=5, label='')
    ax_combined.plot(df_csu_stats['plot_date'], df_csu_stats['median'],
                     color=csu_color, linestyle='--', linewidth=2.5, marker='s', markersize=5,
                     markeredgecolor='black', markeredgewidth=0.3, label='')

ax_combined.set_yscale('log')
ax_combined.set_ylabel('Monthly Median INP conc. (stdL⁻¹)')
ax_combined.set_ylim(0.04, 40)
ax_combined.tick_params(axis='both', which='minor', width=0, length=0)
ax_combined.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax_combined.xaxis.get_majorticklabels(), rotation=30, ha='right')

# Remove the old inside-the-axes legend — we will create it outside later
green_color = pine_cmap(pine_norm(-25.0))
handles_green = [
    Line2D([0], [0], color=green_color, marker='o', linestyle='-',
           linewidth=2.5, markersize=6,
           markeredgecolor='black', markeredgewidth=0.4,   # ← added black edge
           label='PINE'),
    Line2D([0], [0], color=green_color, marker='s', linestyle='--',
           linewidth=2.5, markersize=5,
           markeredgecolor='black', markeredgewidth=0.3,   # ← added black edge
           label='ARM-INS'),
    Patch(facecolor=green_color, alpha=0.3, edgecolor='none', label='IQR')
]
x_start_full = pd.Timestamp('2024-04-01')
x_end_full   = pd.Timestamp('2025-10-31')
ax_combined.set_xlim(x_start_full, x_end_full)

# Bi‑monthly ticks, but remove April 2024
tick_dates_2month = []
current = pd.Timestamp('2024-04-01')
while current <= x_end_full:
    if current != pd.Timestamp('2024-04-01'):
        tick_dates_2month.append(current)
    year = current.year
    month = current.month + 2
    if month > 12:
        year += 1
        month -= 12
    current = pd.Timestamp(year=year, month=month, day=1)
ax_combined.set_xticks(tick_dates_2month)

# =============================================================================
# 6. Color bar – shortened at bottom to leave room for legend
# =============================================================================
pos_top = ax_pine_4h.get_position()
pos_bottom = ax_combined.get_position()

# Reserve space at the bottom for the legend below the colorbar
legend_height = 0.12
gap = 0.02
cbar_bottom = pos_bottom.y0 + legend_height + gap   # colorbar starts above legend area
cbar_top = pos_top.y1

cbar_pine_ax = fig.add_axes([0.87, cbar_bottom, 0.015, cbar_top - cbar_bottom])
sm_pine = plt.cm.ScalarMappable(cmap=pine_cmap, norm=pine_norm)
sm_pine.set_array([])
cbar_pine = fig.colorbar(sm_pine, cax=cbar_pine_ax, boundaries=pine_bin_edges,
                         ticks=pine_bin_edges)
cbar_pine.set_ticklabels([f"{int(e)}" for e in pine_bin_edges])
cbar_pine.set_label('Temperature (°C)', fontsize=8)
cbar_pine.ax.invert_yaxis()

# -------------------------------------------------------------------------
# Legend for the combined panel – placed below the colorbar, right of subplot
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Legend for the combined panel – placed below the colorbar, right of subplot
# -------------------------------------------------------------------------
legend_ax = fig.add_axes([0.89, pos_bottom.y0, 0.07, legend_height])
legend_ax.axis('off')   # hide the axes frame
legend_ax.legend(handles=handles_green, loc='center', frameon=True,
                 facecolor='white', edgecolor='none', fontsize=8,
                 ncol=1)

# =============================================================================
# 7. Panel labels
# =============================================================================
ax_pine_4h.text(0.01, 0.96, 'a)', transform=ax_pine_4h.transAxes,
                va='top', ha='left')
ax_combined.text(0.01, 0.96, 'b)', transform=ax_combined.transAxes,
                 va='top', ha='left')

# -------------------------------------------------------------------------
# Move the zero‑values subplot further up
# -------------------------------------------------------------------------
pos_pine = ax_pine_4h.get_position()
pos_zeros = ax_zeros.get_position()
desired_gap = 0.005
new_y0_zeros = pos_pine.y0 - desired_gap - pos_zeros.height
ax_zeros.set_position([pos_zeros.x0, new_y0_zeros, pos_zeros.width, pos_zeros.height])

# Align y‑axis labels of the two main subplots
fig.align_ylabels([ax_pine_4h, ax_combined])

# =============================================================================
# 8. Save
# =============================================================================
plt.savefig('ResultsPINETimelineCSU_final.png', dpi=300)
plt.savefig('ResultsPINETimelineCSU_final.pdf', dpi=300)
print("Figure saved as 'ResultsPINETimelineCSU_final.png' and .pdf")
plt.close()

# Summary output
print("\nPINE temperature bins (2K):")
for bin_name in fourhour_avg['temp_bin_2K'].unique():
    count = len(fourhour_avg[fourhour_avg['temp_bin_2K'] == bin_name])
    nonzero = len(fourhour_nonzero.get(bin_name, []))
    zeros = len(fourhour_zeros.get(bin_name, []))
    print(f"  {bin_name}: {count} 4‑h periods ({nonzero} nonzero, {zeros} zero)")

print(f"\nPINE monthly medians and IQR for highlighted bin '{highlight_bin}':")
print(pine_monthly[pine_monthly['temp_bin_2K'] == highlight_bin][
          ['month_start', 'median', 'q25', 'q75']].to_string(index=False))

print("\nARM-INS monthly medians and IQR for the three cold bins (IQR only for -26 to -24):")
for centre in selected_csu_centres:
    if centre in csu_monthly_stats_dict:
        df_csu_stats = csu_monthly_stats_dict[centre]
        print(f"\nBin centre {centre}°C:")
        print(df_csu_stats[['month_start', 'median', 'q25', 'q75']].to_string(index=False))
    else:
        print(f"\nBin centre {centre}°C: No data.")