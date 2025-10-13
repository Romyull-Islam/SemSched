import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Parse the data
data = {
    "Memory_Size_GB": ["4GB", "8GB", "16GB"],
    "Quantizations": ["FP32", "FP16", "INT8", "INT4"],
    "Baseline": [[174.7, 80.2, 36.4, 16.7], [160.7, 69.25, 25.73, 13.17], [137.98, 43.82, 14.59, 13.17]],
    "CXL_Sequential": [[161.1, 70.14, 25.86, 13.17], [150.24, 57.57, 16.1, 13.17], [124.78, 34.22, 14.59, 13.17]],
    "CXL_Adaptive_Prefetch": [[139.4, 61.03, 23.34, 13.17], [130.4, 50.64, 15.77, 13.17], [109.3, 31.32, 14.59, 13.17]],
    "CXL_Parallel_IO_Prefetch": [[57.42, 26.02, 14.59, 13.17], [52.85, 23.51, 14.59, 13.17], [49.32, 19.47, 14.59, 13.17]],
    #"Scheduling_Async_Simulation": [[13.18, 9.7, 14.45, 13.1], [13.51, 14.22, 14.45, 13.1], [13.46, 17.16, 14.45, 13.1]]
}

# Create a structured dataset
methods = ["Baseline", "CXL_Sequential", "CXL_Adaptive_Prefetch", "CXL_Parallel_IO_Prefetch"] #, "Scheduling_Async_Simulation"]
method_labels = ["Baseline", "CXL Sequential", "CXL Adaptive", "CXL Parallel"] #, "Scheduling"]
memory_sizes = data["Memory_Size_GB"]
quantizations = data["Quantizations"]

# Colors for the 5 methods
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F'] #, '#D2BA4C']

# Create the grouped bar chart
fig = go.Figure()

# Create x-axis positions with clear grouping
x_positions = []
x_labels = []
group_centers = []
group_labels = []

position = 0
group_width = 8  # 4 quantizations per memory size
group_spacing = 2  # spacing between memory groups

for mem_idx, memory in enumerate(memory_sizes):
    if mem_idx > 0:
        position += group_spacing
    
    group_start = position
    
    for quant_idx, quant in enumerate(quantizations):
        x_positions.append(position)
        x_labels.append(quant)
        position += 1
    
    # Calculate group center for memory size label
    group_center = group_start + (group_width - 1) / 2
    group_centers.append(group_center)
    group_labels.append(memory)

# Add bars for each method
for method_idx, method in enumerate(method_labels):
    values = []
    x_pos = []
    text_labels = []
    
    pos_idx = 0
    for mem_idx, memory in enumerate(memory_sizes):
        for quant_idx, quant in enumerate(quantizations):
            value = data[methods[method_idx]][mem_idx][quant_idx]
            values.append(value)
            x_pos.append(x_positions[pos_idx])
            
            # Format text labels
            if value >= 100:
                text_labels.append(f"{value:.0f}")
            elif value >= 10:
                text_labels.append(f"{value:.1f}")
            else:
                text_labels.append(f"{value:.2f}")
            
            pos_idx += 1
    
    fig.add_trace(go.Bar(
        name=method,
        x=x_pos,
        y=values,
        marker_color=colors[method_idx],
        text=text_labels,
        textposition='outside',
        textfont_size=15,
        width=0.25
    ))

# Update layout
fig.update_layout(
    title='Inference Time by Quantization & Memory',
    xaxis_title='',
    yaxis_title='Inference Time (s)',
    barmode='group',
    bargap=0.2,
    bargroupgap=0.1,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    ),
    xaxis=dict(
        tickmode='array',
        tickvals=x_positions,
        ticktext=x_labels,
        tickangle=0,
        showgrid=False
    ),
    yaxis=dict(
        range=[0, max([max(data[method][i]) for method in methods for i in range(len(memory_sizes))]) + 15],
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1
    )
)

# Add memory size group labels below x-axis
for center, label in zip(group_centers, group_labels):
    fig.add_annotation(
        x=center,
        y=-25,
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(size=16, color="black"),
        xanchor="center",
        yanchor="top"
    )

# Add vertical separators between memory groups
separator_positions = []
for i in range(len(group_centers) - 1):
    separator_x = (group_centers[i] + group_centers[i+1]) / 2
    separator_positions.append(separator_x)
    fig.add_vline(
        x=separator_x, 
        line_dash="solid", 
        line_color="gray", 
        line_width=0, 
        opacity=0.6
    )

# Add subtle background rectangles to highlight memory groups
for i, center in enumerate(group_centers):
    rect_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
    fig.add_shape(
        type="rect",
        x0=center - 2.3,
        x1=center + 2.3,
        y0=0,
        y1=max([max(data[method][i]) for method in methods for i in range(len(memory_sizes))]) + 15,
        fillcolor=rect_color,
        opacity=0.3,
        layer="below",
        line_width=0
    )

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("inference_time_chart.png")
fig.write_image("inference_time_chart.svg", format="svg")

print("Final improved chart saved successfully!")