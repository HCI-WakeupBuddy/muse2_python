import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from muselsl import stream, list_muses
from pylsl import StreamInlet, resolve_stream
import threading
import asyncio
import time

def start_muse_stream(muse_id):
    # Muse 스트리밍을 비동기적으로 시작
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream(muse_id))

def collect_eeg_data(duration, sampling_rate=256):
    # 스트림을 찾기 위해 대기
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    data_buffer = []
    start_time = time.time()
    while time.time() - start_time < duration:
        eeg_data, _ = inlet.pull_sample()
        data_buffer.append(eeg_data)
        time.sleep(1 / sampling_rate)
    
    return np.array(data_buffer)

def analyze_eeg_psd(data, sampling_rate):
    # Welch's method를 사용하여 파워 스펙트럼 밀도(PSD) 계산
    freqs, psd = welch(data[:, 0], fs=sampling_rate, nperseg=256)  # 첫 번째 채널 예시

    theta_band = np.where((freqs >= 3.5) & (freqs <= 7))[0]
    alpha_band = np.where((freqs >= 8) & (freqs <= 12))[0]
    beta_band = np.where((freqs >= 13) & (freqs <= 30))[0]

    return psd, freqs, theta_band, alpha_band, beta_band

def save_data(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def visualize_psd(psd, freqs, bands, labels):
    plt.figure(figsize=(12, 6))
    for band, label in zip(bands, labels):
        plt.plot(freqs[band], psd[band], label=label)
    plt.title("EEG Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V^2/Hz)")
    plt.legend()
    plt.show()

def main():
    muses = list_muses()
    if not muses:
        print("No Muse devices found. Please connect a Muse device.")
        return

    muse_id = muses[0]['address']
    print(f"Connecting to Muse: {muse_id}")

    # Muse 스트리밍을 백그라운드에서 시작
    stream_thread = threading.Thread(target=start_muse_stream, args=(muse_id,))
    stream_thread.start()
    time.sleep(2)

    print("Collecting EEG data for 10 seconds...")
    data = collect_eeg_data(10)

    print("Saving EEG data...")
    save_data(data, 'eeg_data.csv')

    sampling_rate = 256
    psd, freqs, theta_band, alpha_band, beta_band = analyze_eeg_psd(data, sampling_rate)

    print("Visualizing EEG data as Power Spectral Density...")
    visualize_psd(psd, freqs, [theta_band, alpha_band, beta_band], 
                  ["Theta", "Alpha", "Beta"])

if __name__ == "__main__":
    main()
