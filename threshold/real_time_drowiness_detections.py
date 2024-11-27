import numpy as np
from muselsl import stream, list_muses
from pylsl import StreamInlet, resolve_stream
from scipy.signal import butter, filtfilt, detrend
from mne.preprocessing import ICA
from mne import create_info, EpochsArray
from mne.time_frequency import psd_array_multitaper
import asyncio
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt

# 이벤트 객체 생성 (스레드 종료 신호)
stop_event = threading.Event()

# Muse2 스트리밍 시작 함수
def start_muse_stream(muse_id):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        print("Starting Muse stream...")
        loop.run_until_complete(stream(muse_id))
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()
        print("Muse stream stopped.")

# 밴드패스 필터
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=4.0, highcut=40.0, fs=256, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

# 전처리 함수
def preprocess_eeg(data, sampling_rate=256):
    filtered_data = bandpass_filter(data, lowcut=4.0, highcut=40.0, fs=sampling_rate)
    detrended_data = detrend(filtered_data, axis=0)

    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
    ch_types = ['eeg'] * len(ch_names)
    info = create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
    data = detrended_data.T[np.newaxis, :, :]
    raw = EpochsArray(data, info)
    ica = ICA(n_components=len(ch_names), random_state=0, max_iter="auto")
    ica.fit(raw)
    ica.exclude = [0]
    processed_data = ica.apply(raw).get_data()[0].T

    return processed_data

# 특징 추출 함수
def extract_features(data, sampling_rate=256):
    freq_bands = {"Theta": (4, 8), "Alpha": (8, 12), "Beta": (13, 30)}
    psds, freqs = psd_array_multitaper(data.T, sfreq=sampling_rate, fmin=0.5, fmax=40, verbose=False)
    band_powers = {
        band: np.sum(psds[(freqs >= low) & (freqs < high)], axis=0)
        for band, (low, high) in freq_bands.items()
    }
    theta_alpha_ratio = band_powers["Theta"] / band_powers["Alpha"]
    theta_beta_ratio = band_powers["Theta"] / band_powers["Beta"]

    return band_powers, theta_alpha_ratio, theta_beta_ratio

# EEG 데이터 수집 함수
def collect_eeg_data(duration, sampling_rate=256):
    try:
        streams = resolve_stream('type', 'EEG')
        if not streams:
            raise RuntimeError("No EEG streams found.")
        inlet = StreamInlet(streams[0])

        data_buffer = []
        start_time = time.time()
        while time.time() - start_time < duration:
            eeg_data, _ = inlet.pull_sample()
            data_buffer.append(eeg_data[:4])  # TP9, AF7, AF8, TP10만 사용
            time.sleep(1 / sampling_rate)

        return np.array(data_buffer)
    except Exception as e:
        print(f"Error during data collection: {e}")
        return None

# 실시간 졸음 탐지
def real_time_drowsiness_detection(thresholds, duration_minutes=60, sampling_rate=256):
    theta_alpha_threshold, theta_beta_threshold = thresholds
    drowsy_events = []
    awake_events = []

    start_time = time.time()
    end_time = start_time + duration_minutes * 60
    print(f"Starting real-time drowsiness detection for {duration_minutes} minutes...")

    while time.time() < end_time:
        print("Collecting EEG data for 5 seconds...")
        eeg_data = collect_eeg_data(duration=5)
        if eeg_data is None or eeg_data.shape[0] == 0:
            print("Failed to collect EEG data. Skipping this cycle.")
            continue

        preprocessed_data = preprocess_eeg(eeg_data)

        # 특징 추출
        _, theta_alpha, theta_beta = extract_features(preprocessed_data)

        # 졸음 판별
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if theta_alpha > theta_alpha_threshold or theta_beta > theta_beta_threshold:
            print(f"Drowsy detected at {timestamp}! Theta/Alpha: {theta_alpha:.2f}, Theta/Beta: {theta_beta:.2f}")
            drowsy_events.append({"Timestamp": timestamp, "Theta/Alpha": theta_alpha, "Theta/Beta": theta_beta})
        else:
            print(f"Awake at {timestamp}. Theta/Alpha: {theta_alpha:.2f}, Theta/Beta: {theta_beta:.2f}")
            awake_events.append({"Timestamp": timestamp, "Theta/Alpha": theta_alpha, "Theta/Beta": theta_beta})

    # 결과 저장 및 시각화
    save_and_visualize(drowsy_events, awake_events)

# 결과 저장 및 시각화 함수
def save_and_visualize(drowsy_events, awake_events):
    if drowsy_events:
        drowsy_df = pd.DataFrame(drowsy_events)
        drowsy_df.to_csv("drowsy_log.csv", index=False)
        print("Drowsy log saved to 'drowsy_log.csv'.")

    if awake_events:
        awake_df = pd.DataFrame(awake_events)
        awake_df.to_csv("awake_log.csv", index=False)
        print("Awake log saved to 'awake_log.csv'.")

    visualize_results(drowsy_events, awake_events)

# 시각화 함수
def visualize_results(drowsy_events, awake_events):
    plt.figure(figsize=(14, 7))

    if drowsy_events:
        drowsy_timestamps = [event["Timestamp"] for event in drowsy_events]
        drowsy_theta_alpha = [event["Theta/Alpha"] for event in drowsy_events]
        drowsy_theta_beta = [event["Theta/Beta"] for event in drowsy_events]
        plt.scatter(drowsy_timestamps, drowsy_theta_alpha, color='red', label='Drowsy Theta/Alpha', alpha=0.7)
        plt.scatter(drowsy_timestamps, drowsy_theta_beta, color='darkred', label='Drowsy Theta/Beta', alpha=0.7)

    if awake_events:
        awake_timestamps = [event["Timestamp"] for event in awake_events]
        awake_theta_alpha = [event["Theta/Alpha"] for event in awake_events]
        awake_theta_beta = [event["Theta/Beta"] for event in awake_events]
        plt.scatter(awake_timestamps, awake_theta_alpha, color='blue', label='Awake Theta/Alpha', alpha=0.7)
        plt.scatter(awake_timestamps, awake_theta_beta, color='darkblue', label='Awake Theta/Beta', alpha=0.7)

    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("Ratio")
    plt.title("Real-Time Drowsiness Detection")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    thresholds = (3.50, 3.08)  # 설정된 임곗값
    muses = list_muses()
    if not muses:
        print("No Muse devices found. Please connect a Muse device and try again.")
    else:
        muse_id = muses[0]['address']
        stream_thread = threading.Thread(target=start_muse_stream, args=(muse_id,), daemon=True)
        stream_thread.start()

        try:
            duration_minutes = int(input("Enter detection duration in minutes (max 60): "))
            if 1 <= duration_minutes <= 60:
                real_time_drowsiness_detection(thresholds, duration_minutes=duration_minutes)
            else:
                print("Please enter a valid duration between 1 and 60.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        finally:
            stop_event.set()
            stream_thread.join()