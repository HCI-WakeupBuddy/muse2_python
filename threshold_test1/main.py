import asyncio
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


async def start_muse_stream(muse_id):
    """Muse 스트리밍 시작."""
    await stream(muse_id)


def run_async_stream(muse_id):
    """비동기 스트리밍 실행."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # 이미 실행 중인 루프에 태스크 추가
        loop.create_task(start_muse_stream(muse_id))
    else:
        # 새로운 루프 실행
        asyncio.run(start_muse_stream(muse_id))


def main():
    try:
        # Muse 기기 검색
        muses = list_muses()
        if not muses:
            print("No Muse devices found. Please connect a Muse device.")
            sys.exit(1)

        muse_id = muses[0]['address']
        print(f"Connecting to Muse: {muse_id}")

        # 스트림 시작
        stream_thread = threading.Thread(target=run_async_stream, args=(muse_id,))
        stream_thread.daemon = True
        stream_thread.start()
        time.sleep(2)  # 연결 대기

        # 데이터 수집 시도
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 비졸음 데이터 수집
                print("Collecting EEG data for Awake state...")
                awake_data = collect_awake_data(10)
                if not awake_data or len(awake_data) == 0:
                    raise ValueError("No data collected for Awake state")

                # 졸음 데이터 수집
                print("Collecting EEG data for Drowsy state...")
                drowsy_data = collect_drowsy_data(10)
                if not drowsy_data or len(drowsy_data) == 0:
                    raise ValueError("No data collected for Drowsy state")

                break  # 데이터 수집 성공 시 반복 종료
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print("Retrying...")
                    time.sleep(2)
                else:
                    print("Failed to collect EEG data after multiple attempts")
                    sys.exit(1)

        # 데이터 분석 및 시각화
        print("\nAnalyzing collected EEG data...")
        thresholds = calculate_threshold(awake_data, drowsy_data)
        theta_threshold, alpha_threshold, avg_theta_awake, avg_theta_drowsy, avg_alpha_awake, avg_alpha_drowsy = thresholds

        averages = (avg_theta_awake, avg_theta_drowsy, avg_alpha_awake, avg_alpha_drowsy)
        thresholds = (theta_threshold, alpha_threshold)
        visualize_threshold(thresholds, averages)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
