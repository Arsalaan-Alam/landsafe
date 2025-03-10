import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "glif.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Identify multi-numeric columns (precip, temp, air, humidity, wind)
metric_types = ["precip", "temp", "air", "humidity", "wind"]
average_metrics = {}

for metric in metric_types:
    metric_columns = [col for col in df.columns if col.startswith(metric)]
    df[f"avg_{metric}"] = df[metric_columns].mean(axis=1)
    average_metrics[f"avg_{metric}"] = df[f"avg_{metric}"].mean()

# Save the updated dataframe with averages
df.to_csv("updated_dataset_with_averages.csv", index=False)

# Print calculated averages
for key, value in average_metrics.items():
    print(f"{key}: {value}")

# Plot the average values
plt.figure(figsize=(10, 6))
plt.bar(average_metrics.keys(), average_metrics.values(), color=['blue', 'red', 'green', 'purple', 'orange'])
plt.xlabel("Metrics")
plt.ylabel("Average Value")
plt.title("Average Meteorological Metrics")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Compute correlation matrix for averaged metrics
corr_matrix = df[[f"avg_{metric}" for metric in metric_types]].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Average Meteorological Metrics")
plt.show()
