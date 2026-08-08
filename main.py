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
    
    # 현재 시스템 모드 상태 변수 (초기값은 FAR 모드로 시작)
    current_mode = "FAR"
    
    print("시스템이 시작되었습니다. 종료하려면 'q'를 누르세요.")
    
    while True:
        frame = cam.read_frame()
        if frame is None:
            print("웹캠 프레임을 읽을 수 없습니다.")
            break
            
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        display_mode = ""
        
        # ----------------------------------------------------
        # 1. FAR 모드 (전신 자세 및 낙상 감지)
        # ----------------------------------------------------
        if current_mode == "FAR":
            display_mode = "Far Mode (Fall Detection)"
            
            # 오직 Pose AI만 실행!
            frame, pose_status, face_area_ratio = pose_detector.detect_fall(frame)
            
            # 결과 표시
            color = (0, 0, 255) if "Fall" in pose_status else (255, 255, 0)
            cv2.putText(frame, f"Pose: {pose_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Face Area: {face_area_ratio:.3f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 사람이 가까이 오면 상태 변경 (임계값 0.05 이상)
            if face_area_ratio >= 0.05:
                current_mode = "NEAR"
                print("사람이 접근했습니다 -> NEAR 모드로 전환")
                
        # ----------------------------------------------------
        # 2. NEAR 모드 (얼굴 생체 신호 분석)
        # ----------------------------------------------------
        else:
            display_mode = "Near Mode (Vital Signs)"
            
            # 오직 FaceMesh AI만 실행!
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # 얼굴 넓이 계산
                x_coords = [lm.x for lm in landmarks]
                y_coords = [lm.y for lm in landmarks]
                area_ratio = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                
                # 얼굴이 멀어지면 상태 변경 (깜빡임 방지를 위해 0.04 이하로 떨어질 때 전환)
                if area_ratio < 0.04:
                    current_mode = "FAR"
                    print("사람이 멀어졌습니다 -> FAR 모드로 전환")
                else:
                    # 근거리 -> 생체 정보 분석
                    frame, vitals = vital_analyzer.analyze(frame, landmarks)
                    
                    # 결과 표시
                    cv2.putText(frame, f"HR: {vitals['hr']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Redness: {vitals['redness']}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Sweat: {vitals['sweat']}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Face Area: {area_ratio:.3f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 얼굴 바운딩 박스 표시
                    bbox_x = int(min(x_coords) * w)
                    bbox_y = int(min(y_coords) * h)
                    cv2.rectangle(frame, (bbox_x, bbox_y), (int(max(x_coords)*w), int(max(y_coords)*h)), (255, 0, 0), 2)
            else:
                # 얼굴이 아예 안 보이면 즉시 FAR 모드로 전환
                current_mode = "FAR"
                print("얼굴 감지 안됨 -> FAR 모드로 전환")
            
        # ----------------------------------------------------
        
        # 현재 모드 텍스트 렌더링
        cv2.putText(frame, f"Mode: {display_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        # 화면 렌더링
        cv2.imshow("Hybrid Vision System", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            if current_mode == "NEAR":
                vital_analyzer.trigger_baseline_capture()
            else:
                print("얼굴이 가까이 있어야 기준점을 저장할 수 있습니다.")
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
