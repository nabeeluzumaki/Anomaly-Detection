import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
from datetime import datetime, timedelta

# This class detects anomalies in a data stream using a moving average approach
class AnomalyDetector:
    def __init__(self, window_size=50, threshold=3.0):
        # window_size: How many recent data points to consider when calculating average and std deviation
        # threshold: How many standard deviations away a point needs to be from the mean to be flagged as an anomaly
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)  # Stores the most recent data points, up to 'window_size' limit

    def detect(self, new_value):
        # Add new value to the window
        if len(self.data_window) < self.window_size:
            # If we haven't collected enough data yet, just add the new value and skip anomaly check
            self.data_window.append(new_value)
            return False  # Not enough data yet to check for anomalies

        # Calculate mean and standard deviation of current window
        mean = np.mean(self.data_window)
        std_dev = np.std(self.data_window)

        # Calculate Z-score to see if new value deviates significantly from mean
        # Z-score = (new_value - mean) / standard deviation
        if std_dev > 0:
            z_score = abs(new_value - mean) / std_dev
            # If the Z-score is higher than the threshold, we flag it as an anomaly
            if z_score > self.threshold:
                return True  # Anomaly detected!

        # Add the new value to the window (keeps the window sliding forward)
        self.data_window.append(new_value)
        return False  # No anomaly detected

# Simulates a data stream with values around a stable mean (100), plus some noise
def simulate_data_stream(num_points=300):
    data_stream = []
    base_time = datetime.now()  # Starting point in time for the stream
    
    for i in range(num_points):
        # Generate a baseline value of 100 with random noise added
        value = 100 + random.normalvariate(0, 2)  # Normal distribution with mean=100, std deviation=2
        if i % 50 == 0:  # Every 50th value, add a large spike to simulate an anomaly
            value += 15
        timestamp = base_time + timedelta(seconds=i)  # Increment time for each data point
        data_stream.append((timestamp, value))  # Append timestamp and value as a tuple
    
    return pd.DataFrame(data_stream, columns=["timestamp", "value"])

# Plots the data stream in real-time, highlighting anomalies in red
def plot_data_stream(data_stream, detector):
    plt.ion()  # Turn on interactive mode to update the plot dynamically
    fig, ax = plt.subplots()
    timestamps = []  # List to hold timestamps for the x-axis
    values = []      # List to hold values for the y-axis
    anomalies = []   # List to hold detected anomalies for separate plotting

    for i, row in data_stream.iterrows():
        timestamp, value = row['timestamp'], row['value']
        timestamps.append(timestamp)
        values.append(value)

        # Check if the current value is an anomaly
        if detector.detect(value):
            anomalies.append((timestamp, value))  # Record anomaly points

        # Clear and redraw the plot for each new data point to give a real-time effect
        ax.clear()
        ax.plot(timestamps, values, label="Data Stream", color='blue')
        
        # Plot any detected anomalies in red
        if anomalies:
            anomaly_times, anomaly_values = zip(*anomalies)  # Separate timestamps and values for plotting
            ax.scatter(anomaly_times, anomaly_values, color='red', label="Anomalies")
        
        ax.legend(loc="upper left")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        plt.pause(0.01)  # Pause to allow plot to update

    plt.ioff()  # Turn off interactive mode once done
    plt.show()

if __name__ == "__main__":
    # Create a simulated data stream
    data_stream = simulate_data_stream(num_points=300)

    # Initialize the anomaly detector
    detector = AnomalyDetector(window_size=50, threshold=3.0)

    # Run the real-time anomaly detection and visualization
    plot_data_stream(data_stream, detector)
