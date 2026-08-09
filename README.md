# 실시간 웹캠 기반 생체 정보 분석 및 낙상 감지 시스템

본 프로젝트는 실시간 웹캠 피드를 활용하여 사용자의 거리에 따라 두 가지 모드(생체 정보 분석 / 낙상 감지)를 자동으로 전환하며 동작하는 하이브리드 비전 시스템입니다. **NVIDIA Jetson Nano** 환경에 최적화되어 있습니다.

## 🚀 설치 및 시작하기 (Jetson Nano 환경)

젯슨 나노의 하드웨어 가속(GPU)을 사용하기 위해 일반적인 `pip install` 대신 아래 절차를 반드시 따라야 합니다.

### 1. 가상환경 세팅 (필수)
젯슨 나노에 기본 내장된 최적화 라이브러리(OpenCV 등)를 사용하기 위해 시스템 패키지를 상속받는 가상환경을 생성합니다.
```bash
python3 -m venv myenv --system-site-packages
source myenv/bin/activate
```

### 2. 필수 시스템 라이브러리 설치
일반 pip 대신 반드시 `apt-get`을 통해 엔비디아 최적화 버전을 설치해야 합니다. (잘못 설치된 pip 버전이 있다면 미리 지워야 `core dumped` 에러가 발생하지 않습니다.)
```bash
sudo apt-get update
sudo apt-get install -y python3-numpy python3-scipy python3-opencv
```

### 3. MediaPipe 전용 파일(Wheel) 설치
젯슨 나노(ARM64)용으로 빌드된 전용 `.whl` 파일을 다운로드하여 설치해야 합니다. (파이썬 3.8 기준)
```bash
pip3 install dataclasses
wget https://github.com/KumaTea/Mediapipe-AARCH64/releases/download/v0.8.4/mediapipe-0.8.4-cp38-cp38-linux_aarch64.whl
pip3 install mediapipe-0.8.4-cp38-cp38-linux_aarch64.whl
```

### 4. 프로그램 실행
가상환경이 활성화된 상태에서 메인 스크립트를 실행합니다.
```bash
python3 main.py
```
- 생체 신호(홍조/발한) 측정을 위한 기준점(Baseline)을 저장하려면 얼굴이 가까이 있을 때 영문 **`b`** 키를 누르세요.
- 프로그램을 종료하려면 영문 **`q`** 키를 누르세요.

---

## ⚙️ 시스템 구동 방식 (State Machine 최적화)
단 한 프레임에서도 2개의 AI가 동시에 돌아가는 것을 막아 프레임 드랍을 완벽하게 방어하는 **상태 머신(State Machine)** 구조로 작동합니다.

1. **원거리 (FAR 모드): 낙상 감지**
   - **`Pose` AI만 단독 실행** (연산량 최소화).
   - 사용자의 어깨, 골반, 무릎 좌표를 기반으로 서 있는지, 앉아 있는지, 낙상(누워 있는지)을 감지합니다.
   - `Pose`가 추적하는 얼굴(귀, 코) 면적을 계산해 사용자가 다가오면 즉시 NEAR 모드로 전환합니다.

2. **근거리 (NEAR 모드): 생체 정보 분석**
   - **`FaceMesh` AI만 단독 실행**.
   - 얼굴 뺨 영역(ROI) 다각형을 정밀하게 추출하여 색상 변화로 심박수(rPPG)를 유추합니다.
   - 사전에 'b' 키로 저장된 정상 상태(Baseline)와 비교하여 붉은기(홍조)와 난반사 픽셀(발한)의 변화량을 실시간으로 추적합니다.
   - 얼굴이 너무 작아지면 즉시 FAR 모드로 전환합니다.

---

## 📁 파일 구조

코드는 크게 웹캠 제어부와 AI 분석부로 역할에 맞게 분리되어 있습니다.

- `main.py`: 프로그램 진입점. 상태 머신 기반 모드 스위칭 및 화면 렌더링을 총괄합니다.

### 📸 `camera/` (웹캠 담당)
- `camera_manager.py`: 로지텍 웹캠(YUYV) 및 하드웨어 디코딩(JPEG)을 위한 GStreamer 파이프라인과 V4L2 폴백(Fallback)을 관리합니다.

### 🧠 `analyzer/` (AI 분석 담당)
- `vital_analyzer.py`: 생체 신호(심박, 홍조, 땀) 분석을 총괄하는 컨트롤 타워입니다.
- `pose_detector.py`: MediaPipe Pose 모델을 사용하여 사용자 관절 좌표 기반 낙상을 감지합니다.
- `get_face_roi.py`: FaceMesh 랜드마크를 기반으로 정확한 양뺨의 영역(ROI)을 추출합니다.
- `rppg_analyzer.py`: 녹색 채널 변화량과 Bandpass Filter를 통해 심박수(BPM)를 계산합니다.
- `redness_analyzer.py`: Lab 색공간을 통해 실시간 붉은기(스트레스/홍조)를 분석합니다.
- `texture_analyzer.py`: HSV 색공간을 통해 난반사(발한/땀) 정도를 분석합니다.
