import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from muselsl import stream, list_muses
from pylsl import StreamInlet, resolve_stream
from scipy.signal import butter, filtfilt, stft
from mne.preprocessing import ICA
from mne import create_info, EpochsArray
import threading
import asyncio
import time

def start_muse_stream(muse_id):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream(muse_id))

# 고주파/저주파 잡음 제거
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# 밴드패스 필터
def bandpass_filter(data, lowcut=4.0, highcut=40.0, fs=256, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

# 눈 깜빡임 제거
def preprocess_eeg_with_mne(data, sampling_rate=256):
    # MNE를 사용한 데이터 전처리
    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']  # AUX 채널 제외
    ch_types = ['eeg'] * len(ch_names)
    info = create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
    
    # 데이터 형식 변환 (MNE EpochsArray 객체 생성)
    data = data.T[np.newaxis, :, :]  # MNE 형식에 맞게 변환
    raw = EpochsArray(data, info)
    
    # ICA 적용
    ica = ICA(n_components=len(ch_names), random_state=0, max_iter="auto")
    ica.fit(raw)  # 데이터에 ICA를 적용하여 아티팩트를 식별
    
    # 눈 깜빡임 아티팩트로 간주되는 성분 자동 제거
    ica.exclude = [0]  
    raw_corrected = ica.apply(raw)  # 아티팩트를 제거한 데이터를 생성
    
    return raw_corrected.get_data()[0].T  # 원래 형태로 변환하여 반환

def collect_eeg_data(duration, sampling_rate=256):
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    data_buffer = []
    start_time = time.time()
    while time.time() - start_time < duration:
        eeg_data, _ = inlet.pull_sample()
        data_buffer.append(eeg_data[:4])  # AUX 채널을 제외하고 TP9, AF7, AF8, TP10만 가져옴
        time.sleep(1 / sampling_rate)
    
    return np.array(data_buffer)

def save_data(data, filename):
    df = pd.DataFrame(data, columns=['TP9', 'AF7', 'AF8', 'TP10'])
    df.to_csv(filename, index=False)

def visualize_data(before_data, after_data, channel_names=['TP9', 'AF7', 'AF8', 'TP10']):
    plt.figure(figsize=(14, 6))

    # 전처리 전 데이터 시각화
    plt.subplot(1, 2, 1)
    for i, channel in enumerate(channel_names):
        plt.plot(before_data[:, i], label=channel)
    plt.title("Raw EEG Data")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")

    # 전처리 후 데이터 시각화
    plt.subplot(1, 2, 2)
    for i, channel in enumerate(channel_names):
        plt.plot(after_data[:, i], label=channel)
    plt.title("Preprocessed EEG Data (MNE ICA Applied)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")

    plt.show()

def plot_frequency_bands_comparison_in_one_plot(before_data, after_data, sampling_rate=256, channel_names=['TP9', 'AF7', 'AF8', 'TP10']):
    # 주파수 대역 정의
    freq_bands = {
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (13, 30)
    }

    # 각 채널의 주파수 스펙트럼 계산
    plt.figure(figsize=(14, 12))
    for i, channel in enumerate(channel_names):
        # 전처리 전 FFT 수행
        fft_vals_before = np.fft.fft(before_data[:, i])
        freqs = np.fft.fftfreq(len(fft_vals_before), 1 / sampling_rate)
        positive_freqs = freqs[freqs >= 0]
        fft_vals_before = np.abs(fft_vals_before[freqs >= 0])

        # 전처리 후 FFT 수행
        fft_vals_after = np.fft.fft(after_data[:, i])
        fft_vals_after = np.abs(fft_vals_after[freqs >= 0])

        # 전처리 전후를 그래프로 표시
        plt.subplot(len(channel_names), 2, i * 2 + 1)
        for band, (low, high) in freq_bands.items():
            band_indices = np.where((positive_freqs >= low) & (positive_freqs < high))
            plt.plot(positive_freqs[band_indices], fft_vals_before[band_indices], label=f"{band} ({low}-{high} Hz)")
        
        plt.title(f"Channel {channel} Frequency Bands (Before Preprocessing)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()

        plt.subplot(len(channel_names), 2, i * 2 + 2)
        for band, (low, high) in freq_bands.items():
            band_indices = np.where((positive_freqs >= low) & (positive_freqs < high))
            plt.plot(positive_freqs[band_indices], fft_vals_after[band_indices], label=f"{band} ({low}-{high} Hz)")

        plt.title(f"Channel {channel} Frequency Bands (After Preprocessing)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_stft(data, sampling_rate=256, channel_names=['TP9', 'AF7', 'AF8', 'TP10']):
    plt.figure(figsize=(14, 10))
    
    for i, channel in enumerate(channel_names):
        # STFT 수행
        f, t, Zxx = stft(data[:, i], fs=sampling_rate, nperseg=64)  # nperseg를 작게 설정
        
        # 0-30 Hz 범위로 제한
        f_indices = np.where(f <= 30)  # 0-30 Hz 범위 선택
        f = f[f_indices]
        Zxx = Zxx[f_indices]
        
        # STFT 결과를 시각화 (컬러맵 범위를 수동으로 조정)
        plt.subplot(len(channel_names), 1, i + 1)
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', vmin=0, vmax=100)  # vmax 조정
        plt.colorbar(label='Magnitude')
        plt.title(f"STFT Magnitude of {channel} (0-30 Hz)")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


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

    print("Collecting EEG data for 10 seconds...")
    raw_data = collect_eeg_data(10)

    print("Saving raw EEG data...")
    save_data(raw_data, 'raw_eeg_data.csv')

    print("Preprocessing EEG data with MNE ICA...")
    preprocessed_data = preprocess_eeg_with_mne(raw_data)

    print("Saving preprocessed EEG data...")
    save_data(preprocessed_data, 'preprocessed_eeg_data.csv')

    print("Visualizing EEG data...")
    visualize_data(raw_data, preprocessed_data)

    print("Plotting Frequency Bands Comparison (Before and After Preprocessing)...")
    plot_frequency_bands_comparison_in_one_plot(raw_data, preprocessed_data)

    print("Performing and Plotting STFT...")
    plot_stft(preprocessed_data)

if __name__ == "__main__":
    main()
