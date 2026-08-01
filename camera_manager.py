import cv2

class CameraManager:
    """웹캠 프레임 I/O 처리를 위한 클래스"""
    
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with ID {camera_id}")

    def read_frame(self):
        """프레임을 읽어옵니다. 실패 시 None 반환"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """카메라 자원을 해제합니다."""
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    # 간단한 동작 테스트
    cam = CameraManager()
    print("카메라 실행. 종료하려면 'q'를 누르세요.")
    while True:
        frame = cam.read_frame()
        if frame is None:
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
