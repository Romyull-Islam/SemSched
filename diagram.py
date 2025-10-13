from diagrams import Diagram, Cluster, Edge
# UPDATED: Changed all imports to use the stable 'onprem' module
from diagrams.onprem.compute import Server as CPU
from diagrams.onprem.database import PostgreSQL as SSD
from diagrams.onprem.inmemory import Redis
from diagrams.programming.language import Python
# CORRECTED: Changed the import path for the Python icon
# The Python icon is used for both the scheduler and I/O threads

# Use a custom node for the scheduler logic for clarity
class SchedulerNode(Python):
    _type = "scheduler"
    _icon_dir = "."  # Use current directory
    fontcolor = "#FFFFFF"

# Diagram generation starts here
with Diagram("Asynchronous Scheduler Workflow", show=False, filename="scheduler_workflow", direction="TB"):
    
    # Define a logical group for the asynchronous I/O workers
    with Cluster("Async I/O Pool (Background Workers)"):
        io_threads = [
            Python(f"I/O Thread {i+1}") for i in range(3)
        ]

    # Define the main host system components
    with Cluster("Host System"):
        with Cluster("CPU"):
            scheduler = SchedulerNode("Scheduler Logic")
            ready_queue = Redis("Ready Queue")
            cpu_cores = [
                CPU("Core 1"),
                CPU("Core 2")
            ]
        
        host_dram = Redis("Host DRAM")
        
    # Define the CXL memory device components
    with Cluster("CXL Memory Device"):
        cxl_dram_cache = Redis("CXL DRAM Cache")
        cxl_nand = SSD("CXL NAND Storage")

    # --- Define the data and control flows with numbered labels ---
    
    # 1. The scheduler identifies a future layer needed from CXL NAND
    scheduler >> Edge(label="1. Identifies future layer needed", style="dashed", color="grey") >> cxl_nand

    # 2. It dispatches a prefetch request to the I/O pool
    scheduler >> Edge(label="2. Dispatch Prefetch Request", style="dashed", color="darkgreen") >> io_threads[0]

    # 3. An I/O thread fetches the data from NAND and places it in the CXL DRAM cache
    io_threads[0] >> Edge(label="3. Fetch Data (Slow Read)", color="red") >> cxl_nand
    cxl_nand >> Edge(label="3. Stage in Cache (Write)", color="blue") >> cxl_dram_cache

    # 4. Once data is in the cache, the layer is marked as 'ready'
    cxl_dram_cache >> Edge(label="4. Layer is Ready", style="dashed", color="darkgreen") >> ready_queue
    
    # 5. The scheduler assigns the ready layer to a free CPU core for computation
    ready_queue >> Edge(label="5. Assign to Core", style="dashed", color="black") >> cpu_cores[0]

    # 6a. SUCCESS PATH (Cache Hit): The CPU core fetches the data quickly from the cache
    cpu_cores[0] >> Edge(label="6a. SUCCESS: Cache Hit (Fast Read)", color="blue") >> cxl_dram_cache

    # 6b. FAILURE PATH (Cache Miss): If prefetch failed, the CPU stalls and reads directly from slow NAND
    cpu_cores[1] >> Edge(label="6b. FAILURE: Cache Miss / Stall (Slow Read)", style="dotted", color="red") >> cxl_nand
    
    # Also show data access from Host DRAM for layers placed there
    cpu_cores[1] >> Edge(label="Fast Read", color="blue") >> host_dram

