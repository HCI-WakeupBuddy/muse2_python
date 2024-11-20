import pandas as pd

# CSV 파일 불러오기
data = pd.read_csv('eeg_data.csv')

# 데이터 확인
print(data.head())  # 처음 5줄 확인
print(data.describe())  # 기본 통계 정보 확인
print(data.info())  # 데이터 구조 및 컬럼 정보 확인