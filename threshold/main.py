import threading
import time
import sys
from collect_awake_data import collect_awake_data
from collect_drowsy_data import collect_drowsy_data
from calculate_threshold import (
    calculate_threshold, visualize_threshold, visualize_data,
    plot_frequency_bands_comparison_in_one_plot, plot_stft
)
from muselsl import stream, list_muses
import asyncio

def start_muse_stream(muse_id):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream(muse_id))

def main():
    muses = list_muses()
    if not muses:
        print("No Muse devices found. Please connect a Muse device.")
        return

    muse_id = muses[0]['address']
    print(f"Connecting to Muse: {muse_id}")

    stream_thread = threading.Thread(target=start_muse_stream, args=(muse_id,))
    stream_thread.start()
    time.sleep(2)

    awake_data = collect_awake_data(10)
    if awake_data is None:
        print("Failed to collect Awake EEG data. Exiting program.")
        sys.exit(1)

    drowsy_data = collect_drowsy_data(10)
    if drowsy_data is None:
        print("Failed to collect Drowsy EEG data. Exiting program.")
        sys.exit(1)

    print("Visualizing Awake and Drowsy EEG data...")
    visualize_data(awake_data, drowsy_data, title="Awake vs Drowsy EEG Data")
    plot_frequency_bands_comparison_in_one_plot(awake_data, drowsy_data)
    plot_stft(awake_data, title="STFT of Awake EEG Data")
    plot_stft(drowsy_data, title="STFT of Drowsy EEG Data")

    threshold, avg_theta_energy_awake, avg_theta_energy_drowsy = calculate_threshold(awake_data, drowsy_data)
    print(f"Dynamic threshold set to: {threshold}")

    visualize_threshold(threshold, avg_theta_energy_awake, avg_theta_energy_drowsy)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0)