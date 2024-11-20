import numpy as np
import threading
import os
import asyncio
from muselsl import stream, list_muses, view

# 환경 변수 설정 (필요시 추가 설정)
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib'

def analyze_eeg(data):
    # 채널별 평균 계산
    mean_values = np.mean(data, axis=1)
    print(f"Mean for each channel: {mean_values}")

def start_muse_stream(address):
    # 명시적으로 이벤트 루프 생성
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print("Starting Muse stream...")
    stream(address)  # 스트림 시작

def stream_eeg():
    # Muse 장치 검색
    muses = list_muses()
    if not muses:
        print("No Muse devices found")
        return

    address = muses[0]['address']
    print(f"Connecting to Muse at address {address}")

    # Muse 스트리밍을 별도 스레드에서 실행
    stream_thread = threading.Thread(target=start_muse_stream, args=(address,))
    stream_thread.start()
    
    # 데이터 뷰어를 통한 EEG 데이터 실시간 시각화
    view()  # EEG 스트림을 실시간으로 관찰 가능

# 메인 실행
if __name__ == "__main__":
    stream_eeg()