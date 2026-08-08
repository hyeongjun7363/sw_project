import numpy as np
from scipy.signal import butter, filtfilt
import time

class RPPGAnalyzer:
    def __init__(self, buffer_size=150, min_hz=0.8, max_hz=3.0):
        # 30fps 기준 150 프레임이면 약 5초 분량의 데이터
        # min_hz=0.8 (48 BPM), max_hz=3.0 (180 BPM)
        self.buffer_size = buffer_size
        self.min_hz = min_hz
        self.max_hz = max_hz
        
        self.times = []
        self.signal_buffer = []
        self.current_bpm = 0.0

    def process_frame(self, roi_left, roi_right, current_time=None):
        """
        양쪽 뺨 ROI 이미지를 받아 실시간으로 BPM을 연산합니다.
        roi_left, roi_right: OpenCV BGR 포맷의 Numpy Array
        """
        if current_time is None:
            current_time = time.time()
            
        # 1. 공간 평균 (Spatial Averaging)
        # OpenCV는 BGR 순서이므로 녹색 채널은 인덱스 1입니다.
        mean_g_left = np.mean(roi_left[:, :, 1]) if roi_left.size > 0 else 0
        mean_g_right = np.mean(roi_right[:, :, 1]) if roi_right.size > 0 else 0
        
        # 양쪽 뺨의 Green 채널 평균의 평균값을 사용
        mean_g = (mean_g_left + mean_g_right) / 2.0
        
        self.times.append(current_time)
        self.signal_buffer.append(mean_g)
        
        # 버퍼 사이즈 유지 (오래된 데이터 삭제)
        if len(self.signal_buffer) > self.buffer_size:
            self.times.pop(0)
            self.signal_buffer.pop(0)
            
        # 버퍼가 어느 정도 찼을 때만 BPM 연산 수행 (예: 100프레임 이상)
        if len(self.signal_buffer) >= 100:
            self.current_bpm = self._calculate_bpm()
            
        return self.current_bpm

    def _calculate_bpm(self):
        # 시간 간격 계산을 통해 실제 FPS 추정
        time_diffs = np.diff(self.times)
        avg_time_diff = np.mean(time_diffs)
        if avg_time_diff == 0:
            return self.current_bpm
            
        fps = 1.0 / avg_time_diff
        
        # 1. 신호 전처리: Z-score Normalization (평균 0, 분산 1로 정규화)
        # 근거: 조명 변화로 인한 진폭의 흔들림을 보정하기 위함
        signal = np.array(self.signal_buffer)
        mean_sig = np.mean(signal)
        std_sig = np.std(signal)
        if std_sig == 0:
            return self.current_bpm
        signal = (signal - mean_sig) / std_sig
        
        # 2. 나이퀴스트 주파수 기반 Bandpass Filter (0.8 ~ 3.0 Hz)
        nyq = 0.5 * fps
        low = self.min_hz / nyq
        high = self.max_hz / nyq
        
        if low > 0 and high < 1.0:
            # 3차 버터워스 필터 적용 (생체 신호 처리의 표준)
            b, a = butter(3, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, signal)
            
            # 3. 해밍 창(Hamming Window) 적용
            # 근거: 유한한 버퍼 데이터(Truncated Signal)를 FFT할 때 발생하는 스펙트럼 누수(Spectral Leakage) 최소화
            windowed_signal = filtered_signal * np.hamming(len(filtered_signal))
            
            # 4. FFT (고속 푸리에 변환) 적용
            n = len(windowed_signal)
            fft_result = np.fft.rfft(windowed_signal)
            fft_freqs = np.fft.rfftfreq(n, d=1.0/fps)
            
            # 0.8 Hz ~ 3.0 Hz (48 ~ 180 BPM) 범위 내의 주파수만 탐색
            valid_idx = np.where((fft_freqs >= self.min_hz) & (fft_freqs <= self.max_hz))[0]
            if len(valid_idx) > 0:
                valid_fft = np.abs(fft_result[valid_idx])
                
                # 5. Peak Detection
                peak_idx = valid_idx[np.argmax(valid_fft)]
                peak_freq = fft_freqs[peak_idx]
                
                # Hz를 BPM으로 변환
                estimated_bpm = peak_freq * 60.0
                
                # 급격한 변화 방지를 위한 이동 평균(Moving Average) 성격의 스무딩
                if self.current_bpm == 0:
                    return estimated_bpm
                else:
                    # 이전 BPM 70%, 새로운 BPM 30% 반영 (자연스러운 UI 전환을 위함)
                    return (self.current_bpm * 0.7) + (estimated_bpm * 0.3)
                    
        return self.current_bpm
