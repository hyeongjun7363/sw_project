import cv2
import numpy as np

# MediaPipe Face Mesh 기준 양쪽 뺨을 둘러싸는 핵심 랜드마크 인덱스
LEFT_CHEEK_INDICES = [116, 117, 118, 119, 100, 126, 209, 49, 129, 104, 116]
RIGHT_CHEEK_INDICES = [345, 346, 347, 348, 329, 355, 429, 279, 358, 333, 345]

class FaceROIExtractor:
    def __init__(self):
        self.current_roi_left = None
        self.current_roi_right = None
        
        self.baseline_roi_left = None
        self.baseline_roi_right = None

    def _extract_polygon_roi(self, frame, landmarks, indices):
        """특정 랜드마크 인덱스들로 이루어진 다각형 영역(ROI)을 추출합니다."""
        h, w, _ = frame.shape
        points = []
        for idx in indices:
            lm = landmarks[idx]
            points.append([int(lm.x * w), int(lm.y * h)])
        points = np.array(points, dtype=np.int32)
        
        # 뺨 이외의 영역을 날리기 위한 마스크 생성
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        
        # 원본 이미지에 마스크 씌우기
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 바운딩 박스로 타이트하게 자르기
        x, y, bw, bh = cv2.boundingRect(points)
        if bw == 0 or bh == 0:
            return np.array([])
            
        roi = masked_frame[y:y+bh, x:x+bw]
        return roi

    def extract_roi(self, frame, landmarks):
        """왼쪽, 오른쪽 뺨 ROI를 추출하여 현재 상태에 저장하고 반환합니다."""
        self.current_roi_left = self._extract_polygon_roi(frame, landmarks, LEFT_CHEEK_INDICES)
        self.current_roi_right = self._extract_polygon_roi(frame, landmarks, RIGHT_CHEEK_INDICES)
        
        return self.current_roi_left, self.current_roi_right

    def save_baseline_now(self):
        """현재 ROI 이미지를 기준으로 Baseline을 저장합니다."""
        if self.current_roi_left is not None and self.current_roi_right is not None:
            if self.current_roi_left.size > 0 and self.current_roi_right.size > 0:
                self.baseline_roi_left = self.current_roi_left.copy()
                self.baseline_roi_right = self.current_roi_right.copy()
                print("기준점(Baseline) 픽셀 데이터가 성공적으로 저장되었습니다!")
                return True
        print("경고: 얼굴을 찾을 수 없어 기준점을 저장하지 못했습니다.")
        return False

    def has_baseline(self):
        """기준점이 설정되어 있는지 여부를 반환합니다."""
        return self.baseline_roi_left is not None and self.baseline_roi_right is not None
