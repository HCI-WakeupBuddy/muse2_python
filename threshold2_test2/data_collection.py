import time
import numpy as np
from pylsl import StreamInlet, resolve_stream
import pandas as pd
from muselsl import stream, list_muses
import asyncio
import threading

def start_muse_stream(muse_id):
    """Start streaming from Muse2 device with proper asyncio loop management."""
    try:
        print(f"Attempting to start Muse stream for device: {muse_id}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(stream(muse_id))
    except Exception as e:
        print(f"Error in start_muse_stream: {e}")
    finally:
        asyncio.set_event_loop(None)

def check_stream():
    """Check if EEG stream is available."""
    print("Searching for EEG stream...")
    streams = resolve_stream('type', 'EEG')
    if not streams:
        print("No EEG stream found. Check your Muse device connection.")
        return None
    else:
        print("EEG stream found!")
        return streams

def collect_eeg_data(duration, sampling_rate=256):
    """Collect EEG data for the specified duration."""
    streams = check_stream()
    if not streams:
        return None  # 종료

    inlet = StreamInlet(streams[0])

    data_buffer = []
    start_time = time.time()
    print(f"Collecting EEG data for {duration} seconds...")
    while time.time() - start_time < duration:
        eeg_data, _ = inlet.pull_sample()
        data_buffer.append(eeg_data[:4])  # TP9, AF7, AF8, TP10만 가져옴
        time.sleep(1 / sampling_rate)
    
    print(f"Collected {len(data_buffer)} samples.")
    return np.array(data_buffer)

def save_data(data, filename):
    """Save the collected EEG data to a CSV file."""
    if data is None or len(data) == 0:
        print("No data to save. Exiting.")
        return
    df = pd.DataFrame(data, columns=['TP9', 'AF7', 'AF8', 'TP10'])
    df.to_csv(filename, index=False)
    print(f"Data saved successfully to {filename}")

def main():
    # Check for connected Muse devices
    muses = list_muses()
    if not muses:
        print("No Muse devices found. Please connect a Muse device.")
        return

    muse_id = muses[0]['address']
    print(f"Connecting to Muse: {muse_id}")

    # Start Muse stream in a separate thread
    stream_thread = threading.Thread(target=start_muse_stream, args=(muse_id,))
    stream_thread.start()
    time.sleep(2)  # Allow stream to initialize

    # Collect EEG data
    duration = 10  # EEG 데이터를 10초 동안 수집
    filename = "eeg_data.csv"

    print("Starting EEG data collection process...")
    try:
        data = collect_eeg_data(duration=duration)
        if data is not None:
            print("Saving collected data...")
            save_data(data, filename)
        else:
            print("No data collected. Please check your device.")
    except Exception as e:
        print(f"Error during EEG data collection: {e}")
    finally:
        stream_thread.join()  # Ensure thread completes

if __name__ == "__main__":
    main()