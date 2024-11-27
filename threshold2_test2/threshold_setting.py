import numpy as np
import pandas as pd

def calculate_theta_power(data, sampling_rate=256):
    theta_band = (4, 8)  # Theta 대역: 4-8 Hz
    power = []
    for i in range(data.shape[1]):
        fft_vals = np.fft.fft(data[:, i])
        freqs = np.fft.fftfreq(len(fft_vals), 1 / sampling_rate)
        theta_indices = np.where((freqs >= theta_band[0]) & (freqs < theta_band[1]))
        power.append(np.mean(np.abs(fft_vals[theta_indices])))
    return np.mean(power)

def set_threshold(drowsy_data, awake_data):
    theta_power_drowsy = calculate_theta_power(drowsy_data)
    theta_power_awake = calculate_theta_power(awake_data)
    return (theta_power_drowsy + theta_power_awake) / 2

def main():
    print("Loading preprocessed EEG data...")
    drowsy_data = pd.read_csv('drowsy_preprocessed.csv').values
    awake_data = pd.read_csv('awake_preprocessed.csv').values

    print("Calculating drowsiness threshold...")
    threshold = set_threshold(drowsy_data, awake_data)
    print(f"Calculated Drowsiness Threshold: {threshold}")

if __name__ == "__main__":
    main()