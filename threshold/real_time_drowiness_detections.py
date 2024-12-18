import numpy as np
from muselsl import list_muses
from pylsl import StreamInlet, resolve_stream, resolve_byprop
from scipy.signal import butter, filtfilt, detrend
from mne.preprocessing import ICA
from mne import create_info, EpochsArray
from mne.time_frequency import psd_array_multitaper
import subprocess
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
import logging

def resolve_stream_with_timeout(prop, value, timeout=10):
    """Timeout을 적용한 resolve_stream"""
    start_time = time.time()
    while True:
        streams = resolve_byprop(prop, value)
        if streams:
            return streams
        if time.time() - start_time > timeout:
            raise RuntimeError(f"No streams of type {value} found within {timeout} seconds.")
        time.sleep(0.5)  # 스트림을 주기적으로 확인

# 이벤트 객체 생성 (스레드 종료 신호)
stop_event = threading.Event()

# Muse 스트림 실행 함수
def start_muse_stream():
    print("Starting Muse stream as a subprocess...")
    process = subprocess.Popen(['muselsl', 'stream'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

# Muse 스트림 종료 함수
def stop_muse_stream(process):
    print("Stopping Muse stream...")
    process.terminate()
    process.wait()

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

# 특징 추출 함수 (Theta/Alpha, Theta/Beta 비율 계산)
def extract_features(data):
    # 데이터는 각 채널(TP9, AF7, AF8, TP10)의 신호가 열로 구성된 데이터프레임
    freq_bands = {"Theta": (4, 8), "Alpha": (8, 12), "Beta": (13, 30)}
    sampling_rate = 256

    # FFT를 통해 주파수 스펙트럼 계산
    fft_data = np.fft.rfft(data, axis=0)
    freqs = np.fft.rfftfreq(len(data), d=1 / sampling_rate)

    # 각 주파수 대역의 파워 계산
    band_powers = {
        band: np.sum(np.abs(fft_data[(freqs >= low) & (freqs < high)])**2, axis=0)
        for band, (low, high) in freq_bands.items()
    }

    # Theta/Alpha 및 Theta/Beta 비율 계산
    theta_alpha_ratio = band_powers["Theta"] / band_powers["Alpha"]
    theta_beta_ratio = band_powers["Theta"] / band_powers["Beta"]

    return theta_alpha_ratio.mean(), theta_beta_ratio.mean()

# EEG 데이터 수집 함수
# def collect_eeg_data(duration, sampling_rate=256):
#     try:
#         print("Resolving EEG stream...")
#         streams = resolve_stream('type', 'EEG')
#         if not streams:
#             raise RuntimeError("No EEG streams found. Please ensure Muse is streaming.")
#         inlet = StreamInlet(streams[0])

#         print("EEG stream resolved. Collecting data...")
#         data_buffer = []
#         start_time = time.time()
#         while time.time() - start_time < duration:
#             eeg_data, _ = inlet.pull_sample(timeout=1.0)
#             if eeg_data:
#                 data_buffer.append(eeg_data[:4])
#             else:
#                 print("No data received during this cycle.")
#             time.sleep(1 / sampling_rate)

#         return np.array(data_buffer)
#     except Exception as e:
#         print(f"Error during data collection: {e}")
#         return None

def collect_eeg_data(duration, sampling_rate=256):
    print("Resolving EEG stream...")
    try:
        # resolve_stream 대신 resolve_stream_with_timeout 사용
        streams = resolve_stream_with_timeout('type', 'EEG', timeout=10)
        if not streams:
            raise RuntimeError("No EEG streams found. Please ensure Muse is streaming.")
    except RuntimeError as e:
        print(f"Error: {e}")
        return None

    inlet = StreamInlet(streams[0])

    print("EEG stream resolved. Collecting data...")
    data_buffer = []
    start_time = time.time()
    while time.time() - start_time < duration:
        eeg_data, _ = inlet.pull_sample(timeout=1.0)
        if eeg_data:
            data_buffer.append(eeg_data[:4])  # AUX 채널 제외
        else:
            print("No data received during this cycle.")
        time.sleep(1 / sampling_rate)

    return np.array(data_buffer)

# 실시간 졸음 탐지
def real_time_drowsiness_detection(thresholds, duration_minutes, sampling_rate=256):
    theta_alpha_threshold, theta_beta_threshold = thresholds
    drowsy_events = []
    awake_events = []

    start_time = time.time()
    end_time = start_time + duration_minutes * 60

    while time.time() < end_time:
        print("Collecting EEG data for 5 seconds...")
        eeg_data = collect_eeg_data(duration=5)
        if eeg_data is None or eeg_data.shape[0] == 0:
            print("Failed to collect EEG data. Skipping this cycle.")
            continue

        preprocessed_data = preprocess_eeg(eeg_data)

        # 특징 추출
        theta_alpha, theta_beta = extract_features(preprocessed_data)

        # 졸음 탐지
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if theta_alpha > theta_alpha_threshold or theta_beta > theta_beta_threshold:
            print(f"졸음이 감지되었습니다! 시간: {timestamp}")
            drowsy_events.append({"Timestamp": timestamp, "Theta/Alpha": theta_alpha, "Theta/Beta": theta_beta})
        else:
            print(f"Awake at {timestamp}. Theta/Alpha: {theta_alpha:.2f}, Theta/Beta: {theta_beta:.2f}")
            awake_events.append({"Timestamp": timestamp, "Theta/Alpha": theta_alpha, "Theta/Beta": theta_beta})

    save_and_visualize(drowsy_events, awake_events)


# 결과 저장 및 시각화
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

# 시각화
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

def main():

    thresholds = (3.50, 3.08)  # 사용자 맞춤 임곗값  (Theta/Alpha, Theta/Beta)

    muses = list_muses()
    if not muses:
        print("No Muse devices found. Please connect a Muse device.")
        return

    muse_id = muses[0]['address']
    print(f"Connecting to Muse: {muse_id}")
    
    stream_thread = threading.Thread(target=start_muse_stream)
    stream_thread.start()
    time.sleep(2)

    try:
        duration_minutes = int(input("Enter detection duration in minutes (max 60): "))
        if 1 <= duration_minutes <= 60:
            real_time_drowsiness_detection(thresholds, duration_minutes=duration_minutes)
        else:
            print("Please enter a valid duration between 1 and 60.")
    except ValueError:
        print("Invalid input. Please enter a number.")
    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        # 스레드 종료 신호 설정
        stop_event.set()
        stream_thread.join()
        print("Stopping Muse stream...")

if __name__ == "__main__":
    # Pylsl 네트워크 로그 억제
    logging.getLogger('pylsl').setLevel(logging.ERROR)
    main()