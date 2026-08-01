import cv2
import numpy as np
import mediapipe as mp
import collections

class VitalAnalyzer:
    """rPPG 및 피부 질감(홍조, 발한) 분석 모듈 (근거리 모드)"""
    
    def __init__(self, fps=30):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
        self.fps = fps
        self.g_channel_history = collections.deque(maxlen=fps * 10) # 10초 데이터
        self.baseline_redness = None
        self.baseline_sweatness = None
        
    def analyze(self, frame):
        """
        프레임을 받아 생체 정보(심박, 홍조, 발한)를 분석합니다.
        Returns:
            frame: ROI가 표시된 프레임
            vitals: 측정 결과 딕셔너리
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        vitals = {"hr": "Calculating...", "redness": "Normal", "sweat": "Normal"}
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            
            # 간이 ROI 설정 (양 볼 부위 랜드마크 대략적 위치: 234(좌측 끝), 454(우측 끝))
            # 단순화를 위해 전체 얼굴 바운딩 박스의 중앙 50% 영역을 ROI로 사용
            x_coords = [int(lm.x * w) for lm in landmarks]
            y_coords = [int(lm.y * h) for lm in landmarks]
            
            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
            
            roi_x1 = int(x_min + (x_max - x_min) * 0.3)
            roi_x2 = int(x_min + (x_max - x_min) * 0.7)
            roi_y1 = int(y_min + (y_max - y_min) * 0.4)
            roi_y2 = int(y_min + (y_max - y_min) * 0.6)
            
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            if roi.size > 0:
                # --- 1. 심박수 (rPPG) ---
                # G 채널 평균값 추출 (가장 혈류 변화에 민감)
                g_mean = np.mean(roi[:, :, 1])
                self.g_channel_history.append(g_mean)
                
                if len(self.g_channel_history) > self.fps * 3: # 3초 이상 데이터가 모이면
                    # 단순화된 피크 검출 또는 변화율 기반 임의 심박수 계산 로직
                    # (실제 환경에서는 FFT나 Bandpass Filter 적용 필요)
                    signal = np.array(self.g_channel_history)
                    signal_detrend = signal - np.mean(signal)
                    zero_crossings = np.where(np.diff(np.sign(signal_detrend)))[0]
                    peaks_count = len(zero_crossings) / 2
                    duration = len(self.g_channel_history) / self.fps
                    hr = int((peaks_count / duration) * 60)
                    
                    # 지나치게 튀는 값 보정 (60~100 사이 맵핑으로 시뮬레이션 안정성 확보)
                    if hr < 40 or hr > 150:
                        vitals["hr"] = "Measuring..."
                    else:
                        vitals["hr"] = f"{hr} bpm"

                # --- 2. 홍조 및 발한 분석 (HSV 색공간) ---
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # 홍조: S채널(채도) 평균값 증가로 간이 판단
                current_redness = np.mean(roi_hsv[:, :, 1])
                if self.baseline_redness is None:
                    self.baseline_redness = current_redness
                
                if current_redness > self.baseline_redness * 1.2:
                    vitals["redness"] = "High (Flushed)"
                else:
                    vitals["redness"] = "Normal"
                    
                # 발한: V채널(명도) 중 임계값 이상의 밝은 픽셀(난반사) 비율로 판단
                v_channel = roi_hsv[:, :, 2]
                specular_pixels = np.sum(v_channel > 200)
                total_pixels = v_channel.size
                current_sweatness = specular_pixels / total_pixels
                
                if self.baseline_sweatness is None:
                    self.baseline_sweatness = current_sweatness
                
                if current_sweatness > self.baseline_sweatness + 0.05: # 5% 이상 증가
                    vitals["sweat"] = "Sweating"
                else:
                    vitals["sweat"] = "Normal"
            
            # ROI 렌더링
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            
        return frame, vitals

if __name__ == "__main__":
    analyzer = VitalAnalyzer()
    print("VitalAnalyzer 모듈 초기화 완료.")
