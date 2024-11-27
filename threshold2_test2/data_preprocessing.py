import numpy as np
import pandas as pd
from mne import create_info, EpochsArray
from mne.preprocessing import ICA

def preprocess_eeg_with_mne(data, sampling_rate=256):
    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']  # AUX 채널 제외
    ch_types = ['eeg'] * len(ch_names)
    info = create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
    
    data = data.T[np.newaxis, :, :]  # MNE 형식에 맞게 변환
    raw = EpochsArray(data, info)
    
    ica = ICA(n_components=len(ch_names), random_state=0, max_iter="auto")
    ica.fit(raw)
    ica.exclude = [0]  
    raw_corrected = ica.apply(raw)
    
    return raw_corrected.get_data()[0].T  # 원래 형태로 반환

def main():
    print("Loading raw EEG data...")
    drowsy_data = pd.read_csv('drowsy_data.csv').values
    awake_data = pd.read_csv('awake_data.csv').values

    print("Preprocessing EEG data...")
    preprocessed_drowsy = preprocess_eeg_with_mne(drowsy_data)
    preprocessed_awake = preprocess_eeg_with_mne(awake_data)

    print("Saving preprocessed EEG data...")
    pd.DataFrame(preprocessed_drowsy, columns=['TP9', 'AF7', 'AF8', 'TP10']).to_csv('drowsy_preprocessed.csv', index=False)
    pd.DataFrame(preprocessed_awake, columns=['TP9', 'AF7', 'AF8', 'TP10']).to_csv('awake_preprocessed.csv', index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()