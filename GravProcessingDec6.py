import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Function to load XYZ file from GitHub repository
def load_xyz_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        xyz_data = response.text
        cleaned_data = []

        # Track malformed lines
        malformed_lines = []

        # Process each line
        for i, line in enumerate(xyz_data.splitlines(), start=1):
            columns = line.split()
            if len(columns) == 3:  # Keep valid lines
                cleaned_data.append(" ".join(columns))
            else:
                malformed_lines.append((i, line))

        # Log malformed lines
        if malformed_lines:
            print(f"Found {len(malformed_lines)} malformed lines. Skipping these:")
            for idx, malformed in malformed_lines[:10]:  # Display first 10 errors
                print(f"Line {idx}: {malformed}")
            print("... (skipping remaining malformed lines)")

        # Join cleaned data
        cleaned_xyz_data = "\n".join(cleaned_data)

        # Load into pandas
        try:
            df = pd.read_csv(StringIO(cleaned_xyz_data), sep='\s+', header=None, names=['x', 'y', 'value'])
            return df
        except Exception as e:
            print(f"Error reading the cleaned data: {e}")
            return None
    else:
        print(f"Failed to load data, status code: {response.status_code}")
        return None

# Function to get user input for the range
def get_range_input(prompt):
    while True:
        try:
            user_input = input(prompt)
            range_values = user_input.split(',')
            min_value = float(range_values[0].strip())
            max_value = float(range_values[1].strip())
            if min_value > max_value:
                print("Minimum value cannot be greater than maximum value. Please try again.")
                continue
            return min_value, max_value
        except ValueError:
            print("Invalid input. Please enter two comma-separated numbers (e.g., -117, -110).")

# Function to plot data
def plot_data(ax, filtered_data, title, show_ylabel=False):
    if filtered_data.empty:
        print(f"No data to display for {title}, check your filtering conditions.")
    else:
        # Normalize the value column for circle size (adjust size range)
        min_value = filtered_data['value'].min()
        max_value = filtered_data['value'].max()
        size_scaled = 100 * (filtered_data['value'] - min_value) / (max_value - min_value)  # Scale values for circle size

        # Scatter plot where each point is a circle
        scatter = ax.scatter(
            filtered_data['x'], filtered_data['y'], 
            s=size_scaled, c=filtered_data['value'], cmap='viridis', alpha=0.6, 
            edgecolors='w', linewidth=0.5
        )

        # Set titles and labels
        ax.set_title(title, fontsize=12)
        if show_ylabel:
            ax.set_ylabel('Latitude', fontsize=12)

        return scatter

# GitHub raw URLs
url1 = "https://raw.githubusercontent.com/maxfollett/AirBorneInsight2/main/USbougerGravData.xyz"
url2 = "https://raw.githubusercontent.com/maxfollett/AirBorneInsight2/main/isograv.xyz"

# Get user-defined latitude and longitude ranges
latitude_min, latitude_max = get_range_input("Enter the latitude range (min,max) (e.g., 40,42) for both plots: ")
longitude_min, longitude_max = get_range_input("Enter the longitude range (min,max) (e.g., -114,-112) for both plots: ")

# Get survey name
survey_name1 = input("Enter the name of the survey: ")

# Load datasets
data1 = load_xyz_from_github(url1)
data2 = load_xyz_from_github(url2)

# Filter datasets
filtered_data1 = data1[(data1['x'] >= longitude_min) & (data1['x'] <= longitude_max) & 
                       (data1['y'] >= latitude_min) & (data1['y'] <= latitude_max)] if data1 is not None else pd.DataFrame()
filtered_data2 = data2[(data2['x'] >= longitude_min) & (data2['x'] <= longitude_max) & 
                       (data2['y'] >= latitude_min) & (data2['y'] <= latitude_max)] if data2 is not None else pd.DataFrame()

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
fig.suptitle(f"Gravity Anomaly Map of {survey_name1}", fontsize=16, y=0.95)

# Plot Bouguer Correction
scatter1 = plot_data(axes[0], filtered_data1, "Bouguer Correction", show_ylabel=True)

# Plot Isostatic Correction
scatter2 = plot_data(axes[1], filtered_data2, "Isostatic Correction")

# Add colorbar to the right of the second plot
cbar = fig.colorbar(scatter2, ax=axes, location='right', label='Gravity Value (milligal)')
cbar.ax.tick_params(labelsize=10)

# Unified X-axis label
fig.supxlabel('Longitude', fontsize=12, y=0.02)

plt.show()
