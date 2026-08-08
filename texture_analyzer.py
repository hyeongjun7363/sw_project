import cv2
import numpy as np

class TextureAnalyzer:
    def __init__(self):
        # 기저 상태
        self.baseline_lap_var = None
        self.baseline_specular = None
        
        # 스무딩된 현재 상태
        self.current_lap_var = 0.0
        self.current_specular = 0.0

    def set_baseline(self, roi_left, roi_right):
        """외부에서 주어진 정상 상태 ROI 이미지로 기준점을 설정합니다."""
        if roi_left is None or roi_right is None or roi_left.size == 0 or roi_right.size == 0:
            return False
            
        lap_l, spec_l = self._extract_texture_features(roi_left)
        lap_r, spec_r = self._extract_texture_features(roi_right)
        
        self.baseline_lap_var = (lap_l + lap_r) / 2.0
        self.baseline_specular = (spec_l + spec_r) / 2.0
        
        self.current_lap_var = self.baseline_lap_var
        self.current_specular = self.baseline_specular
        return True

    def _extract_texture_features(self, roi_bgr):
        """
        피부의 질감(거칠기)과 정반사(땀으로 인한 번들거림)를 추출합니다.
        """
        if roi_bgr.size == 0:
            return 0.0, 0.0
            
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. 텍스처 복잡도 (Laplacian Variance)
        # 피부가 건조할수록 미세 주름/거칠기가 부각되어 값이 커지고, 땀이 나면 매끄러워져 값이 작아짐
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. 정반사(Specular Reflection) 비율 (%)
        # 땀이 맺히면 조명을 반사하는 아주 밝은 픽셀이 생김 (임계값 220 이상)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        specular_ratio = (cv2.countNonZero(thresh) / gray.size) * 100.0
        
        return lap_var, specular_ratio

    def process_frame(self, roi_left, roi_right):
        # 기저 상태가 없으면 분석을 수행하지 않음
        if self.baseline_lap_var is None or self.baseline_specular is None:
            return {
                "status": "Baseline needed (Press 'b')",
                "lap_var": 0.0,
                "delta_lap": 0.0,
                "spec_ratio": 0.0,
                "delta_spec": 0.0
            }
            
        lap_l, spec_l = self._extract_texture_features(roi_left)
        lap_r, spec_r = self._extract_texture_features(roi_right)
        
        mean_lap = (lap_l + lap_r) / 2.0
        mean_spec = (spec_l + spec_r) / 2.0
        
        # 노이즈 방지를 위한 EMA 스무딩 (이전 80%, 신규 20%)
        self.current_lap_var = (self.current_lap_var * 0.8) + (mean_lap * 0.2) if self.current_lap_var != 0 else mean_lap
        self.current_specular = (self.current_specular * 0.8) + (mean_spec * 0.2) if self.current_specular != 0 else mean_spec
        
        # 기저 상태와의 변화량(Delta) 계산
        delta_lap = self.current_lap_var - self.baseline_lap_var
        delta_spec = self.current_specular - self.baseline_specular
        
        status = "정상"
        # 땀(열탈진 증상): 번들거림(Specular) 증가 & 거칠기(Lap_var) 감소
        if delta_spec > 1.0 and delta_lap < -30.0:
            status = "땀(번들거림) 감지"
        # 피부 건조(열사병 증상): 번들거림 감소 & 거칠기 증가
        elif delta_lap > 30.0:
            status = "피부 건조(거칠어짐)"
            
        return {
            "status": status,
            "lap_var": round(self.current_lap_var, 1),
            "delta_lap": round(delta_lap, 1),
            "spec_ratio": round(self.current_specular, 2),
            "delta_spec": round(delta_spec, 2)
        }
