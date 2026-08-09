import cv2
import numpy as np

class LabHeatStressAnalyzer:
    def __init__(self):
        # 기저 상태(정상) 저장용 변수
        self.baseline_L = None
        self.baseline_a = None
        
        # UI 표출을 위한 스무딩(이동 평균) 변수
        self.current_L = 0.0
        self.current_a = 0.0

    def set_baseline(self, roi_left, roi_right):
        """외부에서 주어진 정상 상태 ROI 이미지로 기준점을 설정합니다."""
        if roi_left is None or roi_right is None or roi_left.size == 0 or roi_right.size == 0:
            return False
            
        L_left, a_left = self._extract_lab_means(roi_left)
        L_right, a_right = self._extract_lab_means(roi_right)
        
        self.baseline_L = (L_left + L_right) / 2.0
        self.baseline_a = (a_left + a_right) / 2.0
        
        # 스무딩 변수도 함께 초기화
        self.current_L = self.baseline_L
        self.current_a = self.baseline_a
        return True

    def _extract_lab_means(self, roi_bgr):
        """
        BGR 이미지를 LAB로 변환하여 L*(명도)와 a*(적색도)의 평균값을 추출합니다.
        """
        if roi_bgr.size == 0:
            return 0.0, 0.0
            
        # 1. BGR -> LAB 색 공간 변환
        lab_img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        
        # OpenCV LAB 채널 인덱스: 0 -> L* (명도), 1 -> a* (적녹), 2 -> b* (황청)
        L_channel = lab_img[:, :, 0]
        a_channel = lab_img[:, :, 1]
        
        return np.mean(L_channel), np.mean(a_channel)

    def process_frame(self, roi_left, roi_right):
        """
        양볼 ROI를 받아 델타(Delta) 값을 계산하고, 의심되는 질환 상태를 반환합니다.
        """
        # 기저 상태가 없으면 분석을 수행하지 않음
        if self.baseline_L is None or self.baseline_a is None:
            return {"status": "Baseline needed (Press 'b')", "delta_L": 0.0, "delta_a": 0.0}
            
        L_left, a_left = self._extract_lab_means(roi_left)
        L_right, a_right = self._extract_lab_means(roi_right)
        
        # 양쪽 볼의 평균값
        mean_L = (L_left + L_right) / 2.0
        mean_a = (a_left + a_right) / 2.0
        
        # 웹캠 노이즈 방지를 위한 스무딩 (이전 값 80%, 새 값 20%)
        self.current_L = (self.current_L * 0.8) + (mean_L * 0.2) if self.current_L != 0 else mean_L
        self.current_a = (self.current_a * 0.8) + (mean_a * 0.2) if self.current_a != 0 else mean_a

        # 2. 변화량(Delta) 계산
        delta_L = self.current_L - self.baseline_L  # 명도 변화
        delta_a = self.current_a - self.baseline_a  # 적색도 변화

        # 3. 생리적 기전에 따른 상태 분류 로직
        # 임계값(Threshold)은 웹캠 화질에 따라 조정 필요
        threshold = 2.0 
        
        state = "정상"
        
        # 일사병(열탈진): 적색도 감소(음수) & 명도 증가(땀으로 인한 빛 반사, 양수)
        if delta_a < -threshold and delta_L > threshold:
            state = "일사병(열탈진) 패턴"
            
        # 열사병: 적색도 급증(양수) & 명도 하락(짙어짐, 음수)
        elif delta_a > threshold and delta_L < -threshold:
            state = "열사병 패턴"
            
        # 단순히 붉어지기만 하는 경우 (초기 열 스트레스)
        elif delta_a > threshold and delta_L >= -threshold:
            state = "열 스트레스(체온상승)"

        return {
            "status": state,
            "delta_L": round(delta_L, 2),
            "delta_a": round(delta_a, 2)
        }
