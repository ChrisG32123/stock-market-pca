import pandas as pd
import matplotlib.pyplot as plt

# Load the results from the CSV file
df = pd.read_csv('simulation_results.csv')

# Plotting execution time vs. number of MPI tasks
plt.figure(figsize=(10, 6))
for key, grp in df.groupby(['portfolio_size']):
    plt.plot(grp['mpi_tasks'], grp['execution_time'], marker='o', label=f'Portfolio Size {key}')

plt.xlabel('Number of MPI Tasks')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. Number of MPI Tasks')
plt.legend()
plt.grid(True)
plt.show()

# Plotting memory usage vs. number of MPI tasks
plt.figure(figsize=(10, 6))
for key, grp in df.groupby(['portfolio_size']):
    plt.plot(grp['mpi_tasks'], grp['memory_usage'], marker='o', label=f'Portfolio Size {key}')

plt.xlabel('Number of MPI Tasks')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage vs. Number of MPI Tasks')
plt.legend()
plt.grid(True)
plt.show()

# Additional plots can be added here based on specific requirements and data availability
# Calculate thread-to-thread speedup
plt.figure(figsize=(10, 6))
base_times = df[df['omp_threads'] == 1].set_index('mpi_tasks')['execution_time']
for key, grp in df.groupby(['mpi_tasks']):
    speedup = base_times[key] / grp.set_index('omp_threads')['execution_time']
    plt.plot(speedup.index, speedup, marker='o', label=f'MPI Tasks {key}')

plt.xlabel('Number of OpenMP Threads')
plt.ylabel('Speedup')
plt.title('Thread-to-Thread Speedup')
plt.legend()
plt.grid(True)
plt.show()

# Strong and Weak Scaling Analysis
# For strong scaling
plt.figure(figsize=(10, 6))
constant_size = 100  # Example fixed portfolio size
strong_scaling_df = df[df['portfolio_size'] == constant_size]
plt.plot(strong_scaling_df['mpi_tasks'], strong_scaling_df['execution_time'], marker='o', label='Strong Scaling')

# For weak scaling
weak_scaling_df = df.groupby('mpi_tasks').mean()  # Averaging over different portfolio sizes
plt.plot(weak_scaling_df.index, weak_scaling_df['execution_time'], marker='o', linestyle='--', label='Weak Scaling')

plt.xlabel('Number of MPI Tasks')
plt.ylabel('Execution Time (seconds)')
plt.title('Scaling Studies')
plt.legend()
plt.grid(True)
plt.show()
