import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, stft
from mne.preprocessing import ICA
from mne import create_info, EpochsArray

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=4.0, highcut=40.0, fs=256, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

def preprocess_eeg_with_mne(data, sampling_rate=256):
    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
    ch_types = ['eeg'] * len(ch_names)
    info = create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
    
    data = data.T[np.newaxis, :, :]
    raw = EpochsArray(data, info)
    
    ica = ICA(n_components=len(ch_names), random_state=0, max_iter="auto")
    ica.fit(raw)
    
    ica.exclude = [0]  
    raw_corrected = ica.apply(raw)
    
    return raw_corrected.get_data()[0].T

# 임계값 계산 함수
def calculate_threshold(data_awake, data_drowsy, sampling_rate=256):
    theta_band = (4, 8)         # 세타 주파수 대역 정의 (4-8Hz)
    theta_energy_awake = []     # 비졸음 상태의 세타 대역 에너지를 저장할 리스트
    theta_energy_drowsy = []    # 졸음 상태의 세타 대역 에너지를 저장할 리스트
    
    # 비졸음 데이터와 졸음 데이터를 각각 처리
    for data in [data_awake, data_drowsy]:
        fft_vals = np.fft.fft(data)                                 # FFT(고속 푸리에 변환)을 통해 주파수 성분 추출
        freqs = np.fft.fftfreq(len(fft_vals), 1 / sampling_rate)    # 주파수 계산
        positive_freqs = freqs[freqs >= 0]                          # 양수 주파수만 사용
        fft_vals = np.abs(fft_vals[freqs >= 0])
        
        # 세타 대역에 해당하는 인덱스 찾기
        theta_indices = np.where((positive_freqs >= theta_band[0]) & (positive_freqs < theta_band[1]))
        
        # 세타 대역 에너지 계산 (해당 주파수 성분의 합)
        theta_energy = np.sum(fft_vals[theta_indices])
        
         # 비졸음 데이터와 졸음 데이터를 구분하여 각각의 에너지를 저장
        if np.array_equal(data, data_awake):
            theta_energy_awake.append(theta_energy)
        else:
            theta_energy_drowsy.append(theta_energy)

    # 비졸음 상태와 졸음 상태에서의 평균 세타 대역 에너지 계산
    avg_theta_energy_awake = np.mean(theta_energy_awake)
    avg_theta_energy_drowsy = np.mean(theta_energy_drowsy)

     # 동적 임계값 설정: 비졸음 상태의 평균 에너지 + 졸음 상태와의 차이의 절반
    threshold = avg_theta_energy_awake + 0.5 * (avg_theta_energy_drowsy - avg_theta_energy_awake)
    return threshold, avg_theta_energy_awake, avg_theta_energy_drowsy

# 임계값 및 에너지 시각화 함수
def visualize_threshold(threshold, avg_theta_energy_awake, avg_theta_energy_drowsy):
    plt.figure(figsize=(10, 6))     # 그래프 크기 설정

    # 비졸음과 졸음 상태의 세타 대역 에너지를 막대 그래프로 시각화
    plt.bar(['Awake', 'Drowsy'], [avg_theta_energy_awake, avg_theta_energy_drowsy], color=['blue', 'red'])
    plt.axhline(y=threshold, color='g', linestyle='--', label='Threshold')      # 임계값을 표시하는 선 추가 (녹색 점선)
    plt.ylabel('Theta Band Energy') 
    plt.title('Theta Band Energy Comparison and Threshold')
    plt.legend()
    plt.show()

def visualize_data(data1, data2, title="EEG Data Comparison", channel_names=['TP9', 'AF7', 'AF8', 'TP10']):
    plt.figure(figsize=(14, 10))

    for i, channel in enumerate(channel_names):
        plt.subplot(2, 2, i+1)
        plt.plot(data1[:, i], label='Awake')
        plt.plot(data2[:, i], label='Drowsy')
        plt.title(f"{channel} - {title}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_frequency_bands_comparison_in_one_plot(before_data, after_data, sampling_rate=256, channel_names=['TP9', 'AF7', 'AF8', 'TP10']):
    freq_bands = {
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (13, 30)
    }

    plt.figure(figsize=(14, 12))
    for i, channel in enumerate(channel_names):
        fft_vals_before = np.fft.fft(before_data[:, i])
        freqs = np.fft.fftfreq(len(fft_vals_before), 1 / sampling_rate)
        positive_freqs = freqs[freqs >= 0]
        fft_vals_before = np.abs(fft_vals_before[freqs >= 0])

        fft_vals_after = np.fft.fft(after_data[:, i])
        fft_vals_after = np.abs(fft_vals_after[freqs >= 0])

        plt.subplot(len(channel_names), 2, i * 2 + 1)
        for band, (low, high) in freq_bands.items():
            band_indices = np.where((positive_freqs >= low) & (positive_freqs < high))
            plt.plot(positive_freqs[band_indices], fft_vals_before[band_indices], label=f"{band} ({low}-{high} Hz)")
        
        plt.title(f"Channel {channel} Frequency Bands (Awake)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()

        plt.subplot(len(channel_names), 2, i * 2 + 2)
        for band, (low, high) in freq_bands.items():
            band_indices = np.where((positive_freqs >= low) & (positive_freqs < high))
            plt.plot(positive_freqs[band_indices], fft_vals_after[band_indices], label=f"{band} ({low}-{high} Hz)")

        plt.title(f"Channel {channel} Frequency Bands (Drowsy)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_stft(data, sampling_rate=256, channel_names=['TP9', 'AF7', 'AF8', 'TP10']):
    plt.figure(figsize=(14, 10))
    
    for i, channel in enumerate(channel_names):
        f, t, Zxx = stft(data[:, i], fs=sampling_rate, nperseg=64)
        
        f_indices = np.where(f <= 30)
        f = f[f_indices]
        Zxx = Zxx[f_indices]
        
        plt.subplot(len(channel_names), 1, i + 1)
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', vmin=0, vmax=100)
        plt.colorbar(label='Magnitude')
        plt.title(f"STFT Magnitude of {channel} (0-30 Hz)")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.show()