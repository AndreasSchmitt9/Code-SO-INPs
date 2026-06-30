'''
This script bins the PINE data in 2K bins, calculates the 4h average and prints the minimum (non-zero), maximum, mean, median and IQR for each bin
'''
import pandas as pd
import numpy as np


csv_file = '/mnt/d/Cape24-data/Data/processedPINE/Capek.csv'
df = pd.read_csv(csv_file)


df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime', 'T_min', 'INP_cn_0'])

# Quality control
df = df[
    (df['T_min'] < 253.15) &                     # keep only T_min < -20 °C (in Kelvin)
    (df['operation_number'] >= 11) & 
    (~df['operation_number'].between(24, 27, inclusive='both')) &
    (~df['operation_number'].isin([242, 244])) & 
    (~df['operation_number'].isin([43])) & 
    (~df['operation_number'].isin([74]))
]

df['T_min_K'] = df['T_min']
df['T_min_C'] = df['T_min'] - 273.15

df = df[df['T_min_C'] < -22]

bin_edges = np.arange(-34, -22 + 2, 2)   # [-34, -32, -30, -28, -26, -24, -22]
bin_labels = [f"{int(bin_edges[i])} to {int(bin_edges[i+1])}°C" for i in range(len(bin_edges)-1)]

df['temp_bin'] = pd.cut(df['T_min_C'], bins=bin_edges, labels=bin_labels, right=True)

# 4‑hour averages 
df['4h_group'] = df['datetime'].dt.floor('4h')


fourhour_avg = df.groupby(['4h_group', 'temp_bin'], observed=False)['INP_cn_0'].mean().reset_index()
fourhour_avg.rename(columns={'INP_cn_0': 'INP_4h_mean'}, inplace=True)

# nonzero
fourhour_nonzero = fourhour_avg[fourhour_avg['INP_4h_mean'] > 0]

# Statistics
grouped = fourhour_nonzero.groupby('temp_bin', observed=False)['INP_4h_mean'].agg([
    'min', 'max', 'mean', 'std', 'median',
    ('IQR', lambda x: x.quantile(0.75) - x.quantile(0.25))
])

# Print the results
print("Statistics of NON‑ZERO 4‑hour average INP values per 2°C bin:")
print("(Bins: -34 to -22°C in 2° steps, only T < -22°C)\n")

for bin_name, row in grouped.iterrows():
    if pd.notna(bin_name):
        print(f"{bin_name}:  min = {row['min']:.3f},  max = {row['max']:.3f},  "
              f"mean = {row['mean']:.3f},  std = {row['std']:.3f},  "
              f"median = {row['median']:.3f},  IQR = {row['IQR']:.3f}")