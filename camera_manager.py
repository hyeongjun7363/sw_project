import cv2


def gstreamer_pipeline_usb(device="/dev/video0", width=1280, height=720, framerate=30):
    """Jetson Nano 하드웨어 가속 전용 USB 웹캠 GStreamer 파이프라인"""
    return (
        f"v4l2src device={device} ! "
        f"image/jpeg, width={width}, height={height}, framerate={framerate}/1 ! "
        # 1. CPU가 하던 jpegdec 대신 하드웨어 엔진인 nvjpegdec 사용
        "nvjpegdec ! "
        # 2. 크기 조절 등을 하드웨어 가속기인 nvvidconv에게 맡김 (결과물은 BGRx 포맷)
        "video/x-raw(memory:NVMM) ! nvvidconv ! "
        # 3. OpenCV가 읽을 수 있는 BGR 포맷으로 최종 변환
        "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    )
class CameraManager:
    """웹캠 프레임 I/O 처리를 위한 클래스"""

    def __init__(self):
        # 1. GStreamer 파이프라인으로 USB 웹캠 열기 시도
        pipeline = gstreamer_pipeline_usb()
        print("GStreamer 파이프라인으로 웹캠 연결을 시도합니다...")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        # 2. GStreamer로 열리지 않으면 기본 VideoCapture(0)으로 폴백(Fallback)
        if not self.cap.isOpened():
            print("GStreamer로 카메라를 여는 데 실패했습니다. 기본 VideoCapture(0)을 시도합니다.")
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                raise RuntimeError("웹캠을 찾을 수 없습니다. (GStreamer 및 기본 0번 모두 실패)")

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
        frame = cam.read_frame() # 반환된 frame
        if frame is None:
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
