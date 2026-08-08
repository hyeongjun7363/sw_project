import cv2
import numpy as np

from get_face_roi import FaceROIExtractor
from rppg_analyzer import RPPGAnalyzer
from redness_analyzer import LabHeatStressAnalyzer
from texture_analyzer import TextureAnalyzer

class VitalAnalyzer:
    """고도화된 rPPG 및 생체 신호(홍조, 발한) 분석 통합 모듈"""
    
    def __init__(self, fps=30):
        self.fps = fps
        
        # 전문 분석기 및 ROI 추출기 인스턴스화
        self.roi_extractor = FaceROIExtractor()
        self.rppg = RPPGAnalyzer(buffer_size=fps*5) # 5초 분량 데이터 버퍼링
        self.redness = LabHeatStressAnalyzer()
        self.texture = TextureAnalyzer()
        
    def trigger_baseline_capture(self):
        """'b' 키를 눌렀을 때 기준점 저장을 트리거합니다."""
        success = self.roi_extractor.save_baseline_now()
        if success:
            # ROI 추출기에서 성공적으로 기준점을 캡처했으면, 각 분석기에 세팅
            self.redness.set_baseline(self.roi_extractor.baseline_roi_left, self.roi_extractor.baseline_roi_right)
            self.texture.set_baseline(self.roi_extractor.baseline_roi_left, self.roi_extractor.baseline_roi_right)
        return success
        
    def analyze(self, frame, landmarks):
        """
        프레임과 FaceMesh 랜드마크를 받아 생체 정보(심박, 홍조, 발한)를 분석합니다.
        Returns:
            frame: (필요시 렌더링된) 프레임
            vitals: 측정 결과 딕셔너리
        """
        # 초기 상태 반환값 설정
        vitals = {
            "hr": "Calculating...", 
            "redness": "Baseline needed", 
            "sweat": "Baseline needed"
        }
        
        if landmarks:
            # 1. 뺨 영역(ROI) 정밀 다각형 추출
            roi_left, roi_right = self.roi_extractor.extract_roi(frame, landmarks)
            
            if roi_left.size > 0 and roi_right.size > 0:
                # 2. rPPG 심박수 분석 (항상 실행)
                bpm = self.rppg.process_frame(roi_left, roi_right)
                if bpm > 0:
                    vitals["hr"] = f"{int(bpm)} bpm"
                    
                # 3. 홍조 및 땀 분석 (내부적으로 Baseline이 없으면 "Baseline needed" 문자열 리턴)
                redness_result = self.redness.process_frame(roi_left, roi_right)
                texture_result = self.texture.process_frame(roi_left, roi_right)
                
                vitals["redness"] = redness_result["status"]
                vitals["sweat"] = texture_result["status"]
                
                # 얼굴 박스 대신 양뺨의 대략적인 중심 위치에 점 표시 (디버깅/안내용)
                h, w, _ = frame.shape
                left_cheek_pt = (int(landmarks[118].x * w), int(landmarks[118].y * h))
                right_cheek_pt = (int(landmarks[347].x * w), int(landmarks[347].y * h))
                cv2.circle(frame, left_cheek_pt, 3, (0, 255, 0), -1)
                cv2.circle(frame, right_cheek_pt, 3, (0, 255, 0), -1)

        return frame, vitals

if __name__ == "__main__":
    analyzer = VitalAnalyzer()
    print("고도화된 VitalAnalyzer 모듈 초기화 완료.")
