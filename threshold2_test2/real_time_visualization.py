import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from threshold_setting import calculate_theta_power
from muselsl import stream, list_muses
import asyncio
import threading

def visualize_real_time(data_stream, threshold, sampling_rate=256):
    plt.ion()
    fig, ax = plt.subplots()

    theta_powers = []
    timestamps = []

    for data in data_stream:
        theta_power = calculate_theta_power(data)
        theta_powers.append(theta_power)
        timestamps.append(len(theta_powers))

        ax.clear()
        ax.plot(timestamps, theta_powers, label="Theta Power")
        ax.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
        ax.set_title("Real-Time Theta Power with Threshold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Theta Power")
        ax.legend()
        plt.pause(0.1)
    plt.ioff()
    plt.show()

def start_muse_stream(muse_id):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream(muse_id))

def main():
    muses = list_muses()
    if not muses:
        print("No Muse devices found. Please connect a Muse device.")
        return

    muse_id = muses[0]['address']
    print(f"Connecting to Muse: {muse_id}")

    stream_thread = threading.Thread(target=start_muse_stream, args=(muse_id,))
    stream_thread.start()

    print("Loading preprocessed data for simulation...")
    preprocessed_data = pd.read_csv('drowsy_preprocessed.csv').values
    data_stream = (preprocessed_data for _ in range(100))  # 반복 스트림 생성

    threshold = 10.0  # 이전 단계에서 계산한 임곗값 (여기서는 예제)
    print("Starting real-time visualization...")
    visualize_real_time(data_stream, threshold)

if __name__ == "__main__":
    main()