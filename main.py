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
    
    # 통합 Face Mesh 초기화 (거리 판별 + 생체 분석 공용)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    print("시스템이 시작되었습니다. 종료하려면 'q'를 누르세요.")
    
    while True:
        frame = cam.read_frame()
        if frame is None:
            print("웹캠 프레임을 읽을 수 없습니다.")
            break
            
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. 통합 Face Mesh 연산 (프레임당 단 1회)
        results = face_mesh.process(image_rgb)
        
        mode = "Searching..."
        is_near = False
        landmarks = None
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 랜드마크들로부터 가상의 Bounding Box 계산
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            
            # 최소/최대 좌표로 박스의 폭과 높이 비율(0~1) 계산
            bbox_w = max(x_coords) - min(x_coords)
            bbox_h = max(y_coords) - min(y_coords)
            
            # Bounding Box 면적 비율로 근거리/원거리 판별
            area_ratio = bbox_w * bbox_h
            
            # 임계값: 전체 화면의 10% 이상이면 근거리 (필요에 따라 조정)
            DISTANCE_THRESHOLD = 0.10
            
            if area_ratio >= DISTANCE_THRESHOLD:
                is_near = True
                mode = "Near Mode (Vital Signs)"
            else:
                is_near = False
                mode = "Far Mode (Fall Detection)"
                
            # 얼굴 바운딩 박스 렌더링 (거리 참고용)
            bbox_x_px = int(min(x_coords) * w)
            bbox_y_px = int(min(y_coords) * h)
            bbox_w_px = int(bbox_w * w)
            bbox_h_px = int(bbox_h * h)
            
            cv2.rectangle(frame, (bbox_x_px, bbox_y_px), (bbox_x_px + bbox_w_px, bbox_y_px + bbox_h_px), (255, 0, 0), 2)
            cv2.putText(frame, f"Area: {area_ratio:.2f}", (bbox_x_px, bbox_y_px - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            mode = "Far Mode (Fall Detection)"
            is_near = False
            
        # 2. 모드 스위칭 병렬 처리
        if is_near and landmarks:
            # 근거리 -> 기능 1 (생체 정보 분석)
            # 랜드마크를 넘겨주어 vital_analyzer 내부에서 다시 AI 연산하는 것을 방지
            frame, vitals = vital_analyzer.analyze(frame, landmarks)
            
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
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            # 'b' 키를 누르면 기준점(Baseline) 저장 트리거
            if is_near:
                vital_analyzer.trigger_baseline_capture()
            else:
                print("얼굴이 가까이 있어야 기준점을 저장할 수 있습니다.")
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
