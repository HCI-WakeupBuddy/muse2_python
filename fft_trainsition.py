import numpy as np
from time import sleep
import asyncio
from muselsl import stream, list_muses
from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import threading
import time  # time 모듈 추가

def analyze_eeg(data, sampling_rate):
    # FFT 변환
    fft_values = np.fft.fft(data, axis=0)
    fft_magnitudes = np.abs(fft_values)[:data.shape[0] // 2]  # 절반만 사용

    # 주파수 구하기
    freqs = np.fft.fftfreq(data.shape[0], d=1/sampling_rate)[:data.shape[0] // 2]  # 절반만 사용

    # 관심 주파수 대역 설정 (예시: 알파, 베타, 감마, 세타)
    delta_band = np.where((freqs >= 0) & (freqs <= 3.5))[0]
    theta_band = np.where((freqs >= 3.5) & (freqs <= 7))[0]
    alpha_band = np.where((freqs >= 8) & (freqs <= 12))[0]
    beta_band = np.where((freqs >= 13) & (freqs <= 30))[0]
    gamma_band = np.where((freqs >= 30))[0]

    return fft_magnitudes, freqs, delta_band, theta_band, alpha_band, beta_band, gamma_band

def start_stream(muse_id):
    # Muse 스트리밍을 백그라운드에서 시작
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream(muse_id))


def main():
    # Muse 장치 목록 가져오기
    muses = list_muses()
    if not muses:
        print("No Muse devices found. Please connect a Muse device.")
        return

    # 첫 번째 Muse 장치와 연결
    muse_id = muses[0]['address']
    print(f"Connecting to Muse: {muse_id}")
    
    # Muse 스트리밍 시작 (백그라운드 스레드)
    stream_thread = threading.Thread(target=start_stream, args=(muse_id,))
    stream_thread.start()
    
    # 스트림을 찾기 위해 대기 시간 추가
    sleep(2)

    streams = resolve_stream('type', 'EEG')
    if not streams:
        print("No EEG stream found.")
        return

    inlet = StreamInlet(streams[0])

    # 데이터 수집
    sampling_rate = 256
    data_buffer = []
    start_time = time.time()

    print("Collecting EEG data for 10 seconds...")
    for _ in range(10 * sampling_rate):  # 10초 동안 수집
        # EEG 데이터를 실시간으로 수집
        eeg_data, _ = inlet.pull_sample()
        
        # 데이터 버퍼에 추가
        data_buffer.append(eeg_data)

        sleep(1/sampling_rate)  # 샘플링 주파수에 맞게 대기

    # Muse 스트리밍 중지 (중지하는 방법을 정의해야 함)
    print("Stopped streaming.")

    # 수집한 데이터를 numpy 배열로 변환
    data_array = np.array(data_buffer)

    # FFT 분석 수행
    fft_magnitudes, freqs, delta_band, theta_band, alpha_band, beta_band, gamma_band = analyze_eeg(data_array, sampling_rate)

    # 결과 출력 
    print("FFT Magnitudes:", fft_magnitudes)
    print("Frequencies:", freqs)
    print("Delta Band Indices:", delta_band)
    print("Theta Band Indices:", theta_band)
    print("Alpha Band Indices:", alpha_band)
    print("Beta Band Indices:", beta_band)
    print("Gamma Band Indices:", gamma_band)

    # FFT 결과 시각화 (옵션)
    plt.figure(figsize=(12, 6))
    for channel in range(data_array.shape[1]):
        plt.plot(freqs, fft_magnitudes[:, channel], label=f"Channel {channel+1}")
    plt.title("FFT of EEG Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 50)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()