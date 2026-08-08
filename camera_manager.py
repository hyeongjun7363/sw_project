import cv2


def gstreamer_pipeline_usb_jpeg(device="/dev/video0", width=1280, height=720, framerate=30):
    """Jetson Nano 하드웨어 가속 전용 USB 웹캠 (JPEG 포맷 지원용)"""
    return (
        f"v4l2src device={device} ! "
        f"image/jpeg, width={width}, height={height}, framerate={framerate}/1 ! "
        "nvjpegdec ! "
        "video/x-raw(memory:NVMM) ! nvvidconv ! "
        "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    )

def gstreamer_pipeline_usb_yuyv(device="/dev/video0", width=640, height=480, framerate=30):
    """로지텍 웹캠 등 YUYV 기본 포맷용 GStreamer 파이프라인 (안전 모드)"""
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw, width={width}, height={height}, framerate={framerate}/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    )

class CameraManager:
    """웹캠 프레임 I/O 처리를 위한 클래스"""

    def __init__(self):
        print("1. 고화질 GStreamer 파이프라인(JPEG)으로 웹캠 연결을 시도합니다...")
        pipeline = gstreamer_pipeline_usb_jpeg()
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("2. JPEG 포맷 연결 실패. 로지텍 YUYV 호환 파이프라인으로 시도합니다...")
            pipeline_yuyv = gstreamer_pipeline_usb_yuyv()
            self.cap = cv2.VideoCapture(pipeline_yuyv, cv2.CAP_GSTREAMER)
            
        # GStreamer로 끝내 열리지 않으면 기본 VideoCapture(0)으로 폴백(V4L2 강제 지정)
        if not self.cap.isOpened():
            print("3. GStreamer 파이프라인 모두 실패. V4L2 드라이버로 직접 연결을 시도합니다...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                raise RuntimeError("웹캠을 전혀 찾을 수 없습니다. (크롬 웹캠 테스트 등 다른 프로그램이 카메라를 켜놓고 있는지 확인하세요!)")

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
