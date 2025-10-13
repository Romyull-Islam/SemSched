import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
df = pd.read_csv("complete_performance_table 7B model.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace('"', '')

# Extract DRAM sizes
def parse_dram(config):
    parts = config.split(',')
    host = int(parts[0].split('GB')[0].strip())
    cxl = int(parts[1].split('GB')[0].strip())
    return host, cxl

df[['Host_DRAM_GB', 'CXL_DRAM_GB']] = df['Configuration'].apply(parse_dram).apply(pd.Series)
df['Total_DRAM_GB'] = df['Host_DRAM_GB'] + df['CXL_DRAM_GB']

# Map data types to numeric
data_type_map = {'FP32': 0, 'FP16': 1, 'INT8': 2, 'INT4': 3}
df['Data_Type_Num'] = df['Data_Type'].map(data_type_map)

# Define simulators
simulators = [
    'Baseline',
    'CXL_Sequential',
    'CXL_Adaptive',
    'CXL_Parallel_IO'
]

# Extract time values (convert "X.XX s" → float)
for sim in simulators:
    time_col = f"{sim}_Time"
    if time_col in df.columns:
        df[f'{sim}_Time_Num'] = df[time_col].str.extract(r'(\d+\.\d+)').astype(float)

colors = ['red', 'blue', 'green', 'orange']

# --- PLOT 1 ---
fig1 = make_subplots(specs=[[{"type": "scene"}]])

for i, sim in enumerate(simulators):
    time_col = f'{sim}_Time_Num'
    if time_col in df.columns:
        pivot_df = df.pivot_table(
            index='Data_Type_Num',
            columns='Host_DRAM_GB',
            values=time_col,
            aggfunc='mean'
        )
        fig1.add_trace(
            go.Surface(
                z=pivot_df.values,
                x=pivot_df.index,
                y=pivot_df.columns,
                colorscale=[[0, colors[i]], [1, colors[i]]],
                name=sim,
                showscale=False,
                opacity=0.7,
                showlegend=True
            )
        )

fig1.update_scenes(
    xaxis_title='Data Type (0=FP32, 1=FP16, 2=INT8, 3=INT4)',
    yaxis_title='Host DRAM (GB)',
    zaxis_title='Total Time (s)'
)
fig1.update_layout(
    title="3D Surface: Data Type vs Host DRAM vs Total Time (All Simulators)",
    width=1000,
    height=900,
    margin=dict(l=60, r=60, t=100, b=130),  # ← KEY FIX
    showlegend=True,
    legend=dict(
        x=0.85,
        y=0.9,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    ),
    scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1)))
)

fig1.write_image("plot1_data_type_vs_dram_vs_time.png", scale=2)
fig1.write_image("plot1_data_type_vs_dram_vs_time.svg")
print("✅ Plot 1 saved with visible legend")

# --- PLOT 2 ---
fig2 = make_subplots(specs=[[{"type": "scene"}]])

fp32_df = df[df['Data_Type'] == 'FP32'].copy()

for i, sim in enumerate(simulators):
    time_col = f'{sim}_Time_Num'
    if time_col in fp32_df.columns:
        pivot_df = fp32_df.pivot_table(
            index='Host_DRAM_GB',
            columns='CXL_DRAM_GB',
            values=time_col,
            aggfunc='mean'
        )
        fig2.add_trace(
            go.Surface(
                z=pivot_df.values,
                x=pivot_df.index,
                y=pivot_df.columns,
                colorscale=[[0, colors[i]], [1, colors[i]]],
                name=sim,
                showscale=False,
                opacity=0.7,
                showlegend=True
            )
        )

fig2.update_scenes(
    xaxis_title='Host DRAM (GB)',
    yaxis_title='CXL DRAM (GB)',
    zaxis_title='Total Time (s)'
)
fig2.update_layout(
    title="3D Surface: Host DRAM vs CXL DRAM vs FP32 Time (All Simulators)",
    width=1000,
    height=900,
    margin=dict(l=60, r=60, t=100, b=0),  # ← KEY FIX
    showlegend=True,
    legend=dict(
        x=0.85,
        y=0.9,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    ),
    scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1)))
)

fig2.write_image("plot2_host_cxl_vs_fp32_time.png", scale=2)
fig2.write_image("plot2_host_cxl_vs_fp32_time.svg")
print("✅ Plot 2 saved with visible legend")