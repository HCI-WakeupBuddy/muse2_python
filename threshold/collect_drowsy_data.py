import numpy as np
from pylsl import StreamInlet, resolve_stream
import time
from calculate_threshold import bandpass_filter, preprocess_eeg_with_mne

def collect_eeg_data(duration, sampling_rate=256, timeout=5):
    streams = resolve_stream('type', 'EEG')
    if not streams:
        print("No EEG stream found.")
        return None
    inlet = StreamInlet(streams[0])

    data_buffer = []
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            eeg_data, timestamp = inlet.pull_sample(timeout=timeout)
            if eeg_data is None:
                print("Timeout: No data received.")
                return None
            data_buffer.append(eeg_data[:4])
            time.sleep(1 / sampling_rate)
        except KeyboardInterrupt:
            print("Data collection interrupted by user.")
            break

    if not data_buffer:
        print("No data collected.")
        return None
    
    return np.array(data_buffer)

def collect_drowsy_data(duration):
    print("Collecting EEG data for Drowsy state...")
    raw_data = collect_eeg_data(duration)
    print("Preprocessing Drowsy EEG data...")
    filtered_data = bandpass_filter(raw_data)
    preprocessed_data = preprocess_eeg_with_mne(filtered_data)
    return preprocessed_data