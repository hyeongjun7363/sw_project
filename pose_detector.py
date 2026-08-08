import cv2
import mediapipe as mp

class PoseDetector:
    """자세 추정 및 낙상 감지 모듈 (원거리 모드)"""

    def __init__(self, min_detection_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 연산량 최소화 (Lite 모델 사용)
            min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_fall(self, frame):
        """
        프레임을 받아 낙상 여부를 판별합니다.
        Returns:
            processed_frame: 랜드마크가 그려진 이미지
            status: 'Standing', 'Sitting', 'Lying down', 또는 'Unknown'
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        status = "Unknown"
        face_area_ratio = 0.0
        
        if results.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = results.pose_landmarks.landmark
            
            # 얼굴(머리) 크기 비율 계산 (Pose의 0~10번 랜드마크 활용)
            try:
                face_landmarks = [landmarks[i] for i in range(11)]
                face_x = [lm.x for lm in face_landmarks]
                face_y = [lm.y for lm in face_landmarks]
                face_area_ratio = (max(face_x) - min(face_x)) * (max(face_y) - min(face_y))
            except Exception:
                pass
                
            # 좌우 평균값을 사용하여 기준점 설정 (정확도를 위해)
            try:
                y_shoulder = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + 
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2 * h
                y_pelvis = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y + 
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2 * h
                y_knee = (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y + 
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2 * h
                
                y_coords = [y_shoulder, y_pelvis, y_knee]
                y_max, y_min = max(y_coords), min(y_coords)
                
                # (c) 누워있는 자세 (낙상) - y값 차이가 60픽셀 이내 (H=640 기준 10% 정도)
                if (y_max - y_min) <= 60:
                    status = "Lying down (Fall Detected!)"
                # (a) 서 있는 자세 - 어깨 < 골반 < 무릎 (y좌표는 아래로 갈수록 큼)
                elif y_shoulder < y_pelvis and y_pelvis < y_knee:
                    # 무릎과 골반이 충분히 떨어져 있는지 확인 (예: 40픽셀 이상)
                    if (y_knee - y_pelvis) > 40:
                        status = "Standing"
                    else:
                        status = "Sitting"
                # (b) 앉은 자세 - 어깨가 가장 위, 골반과 무릎이 비슷한 선상
                elif y_shoulder < y_pelvis:
                    status = "Sitting"
                
            except IndexError:
                pass # 일부 랜드마크가 화면 밖인 경우

            # 랜드마크 렌더링
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
        return frame, status, face_area_ratio

if __name__ == "__main__":
    detector = PoseDetector()
    print("PoseDetector 모듈 초기화 완료.")
