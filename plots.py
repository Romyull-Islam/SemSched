import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pandas.api.types import CategoricalDtype # Import this

def load_and_prep_data(filepath, model_name):
    """Loads CSV, cleans data, and adds model/config labels."""
    
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Add a column for the model
    df['Model'] = model_name
    
    # Filter out any [Repeat] rows
    df = df[~df['Simulation Name'].str.contains("Repeat", na=False)]
    
    # Rename columns for easier access
    df = df.rename(columns={
        'Simulation Name': 'Simulator',
        'Throughput (tok/s)': 'Throughput',
        'Host Cap (GiB)': 'Host',
        'CXL Pool Cap (GiB)': 'CXL'
    })
    
    # Create the categorical X-axis label
    def get_mem_config(row):
        # Handle the Baseline (N/A) case
        if pd.isna(row['CXL']):
            # Create a Baseline entry for each Host Cap
            return f"{int(row['Host'])}GiB Host (Baseline)"
        # Handle CXL cases
        return f"{int(row['Host'])}GiB Host + {int(row['CXL'])}GiB CXL"
        
    df['Memory Config'] = df.apply(get_mem_config, axis=1)
    
    # Select only the columns we need
    return df[['Memory Config', 'Simulator', 'Model', 'Throughput']]

def create_plot(df):
    """Creates and saves the line plot."""
    
    # --- 1. Define Plot Aesthetics ---
    
    # Define the exact order for the x-axis
    x_axis_order = [
        '16GiB Host (Baseline)', '16GiB Host + 32GiB CXL', '16GiB Host + 64GiB CXL',
        '32GiB Host (Baseline)', '32GiB Host + 32GiB CXL', '32GiB Host + 64GiB CXL'
    ]
    
    # --- FIX: Convert 'Memory Config' to a Categorical type with the correct order ---
    mem_config_dtype = CategoricalDtype(categories=x_axis_order, ordered=True)
    df['Memory Config'] = df['Memory Config'].astype(mem_config_dtype)
    # --- END FIX ---

    # Define colors for each simulator (as requested)
    simulator_colors = {
        "Baseline (No CXL)": "#ef4444",                 # red-500
        "ADAPTIVE Prefetching": "#3b82f6",              # blue-500
        "Async Prefetch Sequential": "#22c55e",         # green-500
        "Semantic-Aware Duplex Scheduler": "#a855f7"    # purple-500
    }
    
    # --- FIX: Define linestyles for each model in a format seaborn/matplotlib understands ---
    model_linestyles = {
        "20B Model": (5, 5), # Dashed line
        "72B Model": ""      # Solid line
    }
    # --- END FIX ---

    # --- 2. Create Plot ---
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Use lineplot with hue for simulator (color) and style for model (linestyle)
    ax = sns.lineplot(
        data=df,
        x='Memory Config',
        y='Throughput',
        hue='Simulator',      # Simulator determines color
        style='Model',        # Model determines linestyle
        palette=simulator_colors,
        dashes=model_linestyles, # Use the corrected dictionary
        markers=True,         # Add markers to data points
        markersize=8,
        linewidth=2.5
    )
    
    # --- 3. Customize Legend and Labels (NEW: Two separate legends) ---
    
    # Get handles and labels from the full legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Create two separate legends
    
    # Legend 1: Simulators (Colors)
    # Find the handles/labels that correspond to Simulators
    sim_handles = [handles[i] for i, label in enumerate(labels) if label in simulator_colors]
    sim_labels = [label for label in labels if label in simulator_colors]
    # Create the first legend (for simulators)
    leg1 = ax.legend(handles=sim_handles, labels=sim_labels, title="Simulator Type", loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Legend 2: Model (Linestyles)
    # Manually create handles for the line styles
    model_handles = [
        mlines.Line2D([], [], color='black', linestyle=model_linestyles['72B Model'], label='72B Model'),
        mlines.Line2D([], [], color='black', linestyle=model_linestyles['20B Model'], label='20B Model')
    ]
    # Add the second legend (for models)
    ax.add_artist(leg1) # Add the first legend back
    ax.legend(handles=model_handles, title="Model Size", loc='upper left', bbox_to_anchor=(1.02, 0.75))

    
    # --- 4. Final Touches and Save ---
    ax.set_title("Simulator Throughput vs. Memory Configuration (20B vs. 72B)", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Memory Configuration", fontsize=12, fontweight='medium')
    ax.set_ylabel("Throughput (tok/s) - Higher is Better", fontsize=12, fontweight='medium')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0) 
    
    plt.xticks(rotation=15, ha='right')
    # Use tight_layout to adjust plot, but save with bbox_inches to include legend
    plt.tight_layout() 
    
    # Save the figure
    output_filename = "throughput_vs_memory.png"
    # Use bbox_inches='tight' to ensure the external legend is saved
    plt.savefig(output_filename, dpi=300, bbox_inches='tight') 
    
    print(f"Graph successfully saved as '{output_filename}'")
    plt.show()

def main():
    # --- 1. Define File Paths ---
    file_20b = '/home/mislam22/Desktop/CXL/outputs/cxl_final_results_20B.csv'
    file_72b = '/home/mislam22/Desktop/CXL/outputs/cxl_simulation_results_72B.csv'
    
    # --- 2. Load and Combine Data ---
    try:
        df_20b = load_and_prep_data(file_20b, '20B Model')
        df_72b = load_and_prep_data(file_72b, '72B Model')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'cxl_final_results_20B.csv' and 'cxl_simulation_results_72B.csv' are in the same directory.")
        return
        
    df_all = pd.concat([df_20b, df_72b]).reset_index(drop=True)
    
    # --- 3. Create Plot ---
    create_plot(df_all)

if __name__ == "__main__":
    main()