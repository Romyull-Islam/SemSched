import plotly.graph_objects as go
import pandas as pd

# Data for FP32 quantization (shows most variation)
memory_sizes = [4, 8, 16]
baseline = [174.7, 160.7, 138.0]
cxl_sequential = [161.1, 150.24, 124.78]
cxl_adaptive = [139.4, 130.4, 109.3]
cxl_parallel = [57.42, 52.85, 49.32]
#scheduling_async = [13.18, 13.51, 13.46]

# Create figure
fig = go.Figure()

# Add traces for each inference method with larger markers
fig.add_trace(go.Scatter(
    x=memory_sizes,
    y=baseline,
    mode='lines+markers',
    name='Baseline',
    line=dict(width=3),
    marker=dict(size=10)
))

fig.add_trace(go.Scatter(
    x=memory_sizes,
    y=cxl_sequential,
    mode='lines+markers',
    name='CXL Sequential',
    line=dict(width=3),
    marker=dict(size=10)
))

fig.add_trace(go.Scatter(
    x=memory_sizes,
    y=cxl_adaptive,
    mode='lines+markers',
    name='CXL Adaptive',
    line=dict(width=3),
    marker=dict(size=10)
))

fig.add_trace(go.Scatter(
    x=memory_sizes,
    y=cxl_parallel,
    mode='lines+markers',
    name='CXL Parallel',
    line=dict(width=3),
    marker=dict(size=10)
))

"""fig.add_trace(go.Scatter(
    x=memory_sizes,
    y=scheduling_async,
    mode='lines+markers',
    name='Sched Async',
    line=dict(width=3),
    marker=dict(size=10)
))"""

# Update layout with linear scale and better formatting
fig.update_layout(
    title='Inference Time vs Memory Size (FP32)',
    xaxis_title='Memory (GB)',
    yaxis_title='Time (sec)',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update axes with better formatting and grid
fig.update_xaxes(
    tickvals=[4, 8, 16], 
    ticktext=['4', '8', '16'],
    tickfont=dict(size=14, color='black'),
    showgrid=True,
    gridcolor='lightgray',
    gridwidth=1
)

fig.update_yaxes(
    tickfont=dict(size=14, color='black'),
    showgrid=True,
    gridcolor='lightgray',
    gridwidth=1,
    range=[0, 180]
)

fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("inference_chart.png")
fig.write_image("inference_chart.svg", format="svg")