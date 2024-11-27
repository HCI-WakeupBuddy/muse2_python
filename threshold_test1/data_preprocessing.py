import numpy as np
import mne
from scipy.signal import stft

def preprocess_eeg_with_mne(raw_data, sfreq=256):
    """
    EEG 데이터를 전처리합니다.
    
    Parameters:
        raw_data: numpy array 형태의 EEG 데이터 (채널 x 샘플 수)
        sfreq: 샘플링 주파수 (기본값: 256Hz)
    
    Returns:
        preprocessed_data: 전처리된 EEG 데이터 (numpy array)
        stft_data: 전처리된 데이터의 STFT 결과 (frequency x time x channels)
    """
    # 1. Muse2 채널 정보 (TP9, AF7, AF8, TP10만 사용)
    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
    ch_types = ['eeg'] * len(ch_names)

    # 2. MNE 객체 생성
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(raw_data.T, info)

    # 3. 델타(저주파) 및 감마(고주파) 제거 (4-30Hz 필터링)
    raw.filter(l_freq=4, h_freq=30, fir_design='firwin', skip_by_annotation='edge')

    # 4. 독립 성분 분석(ICA)을 통해 눈 깜빡임 아티팩트 제거
    ica = mne.preprocessing.ICA(n_components=len(ch_names), random_state=42, max_iter=800)
    ica.fit(raw)
    raw_ica = ica.apply(raw)

    # 5. 전처리된 데이터 가져오기
    preprocessed_data = raw_ica.get_data().T  # (샘플 x 채널)

    # 6. STFT 변환
    f, t, Zxx = stft(preprocessed_data, fs=sfreq, nperseg=256)
    stft_data = np.abs(Zxx)  # STFT 결과 (frequency x time x channels)

    return preprocessed_data, stft_data
