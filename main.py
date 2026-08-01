import cv2
import mediapipe as mp
from camera_manager import CameraManager
from vital_analyzer import VitalAnalyzer
from pose_detector import PoseDetector

def main():
    # 초기화
    cam = CameraManager()
    vital_analyzer = VitalAnalyzer()
    pose_detector = PoseDetector()
    
    # 거리 판별용 Face Detection 초기화 (MediaPipe)
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    print("시스템이 시작되었습니다. 종료하려면 'q'를 누르세요.")
    
    while True:
        frame = cam.read_frame()
        if frame is None:
            print("웹캠 프레임을 읽을 수 없습니다.")
            break
            
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. 거리 판별 루프 (Main Loop)
        results = face_detection.process(image_rgb)
        
        mode = "Searching..."
        is_near = False
        
        if results.detections:
            # 가장 신뢰도 높은 첫 번째 얼굴 사용
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            
            # Bounding Box 면적 비율로 근거리/원거리 판별
            area_ratio = bboxC.width * bboxC.height
            
            # 임계값: 전체 화면의 10% 이상이면 근거리 (필요에 따라 조정)
            DISTANCE_THRESHOLD = 0.10
            
            if area_ratio >= DISTANCE_THRESHOLD:
                is_near = True
                mode = "Near Mode (Vital Signs)"
            else:
                is_near = False
                mode = "Far Mode (Fall Detection)"
                
            # 얼굴 바운딩 박스 렌더링 (거리 참고용)
            bbox_x = int(bboxC.xmin * w)
            bbox_y = int(bboxC.ymin * h)
            bbox_w = int(bboxC.width * w)
            bbox_h = int(bboxC.height * h)
            cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (255, 0, 0), 2)
            cv2.putText(frame, f"Area: {area_ratio:.2f}", (bbox_x, bbox_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            mode = "Far Mode (Fall Detection)"
            is_near = False # 얼굴이 안보이면 기본적으로 원거리(낙상) 모드로 동작
            
        # 2. 모드 스위칭 병렬 처리
        if is_near:
            # 근거리 -> 기능 1 (생체 정보 분석)
            frame, vitals = vital_analyzer.analyze(frame)
            
            # 결과 표시
            cv2.putText(frame, f"HR: {vitals['hr']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Redness: {vitals['redness']}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Sweat: {vitals['sweat']}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 원거리 -> 기능 2 (낙상 감지)
            frame, pose_status = pose_detector.detect_fall(frame)
            
            # 결과 표시
            color = (0, 0, 255) if "Fall" in pose_status else (255, 255, 0)
            cv2.putText(frame, f"Pose: {pose_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        # 현재 모드 표시
        cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        # 화면 렌더링
        cv2.imshow("Hybrid Vision System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
