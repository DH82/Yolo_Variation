# NVIDIA CUDA 12.1 + cuDNN 런타임 베이스 이미지
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git \
    libgl1 libglib2.0-0 \
    && apt-get clean

RUN apt-get install -y nano

# python3를 기본 파이썬으로 연결
RUN ln -s /usr/bin/python3 /usr/bin/python

# pip 최신화
RUN pip install --upgrade pip

# PyTorch (CUDA 12.1 버전) 설치
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ultralytics (YOLOv8 포함) 설치
RUN pip install ultralytics

RUN git clone https://github.com/DH82/Yolo_Variation.git

RUN pip install timm

# 작업 디렉토리 설정
WORKDIR /app

# 기본 실행을 bash로 설정
CMD ["bash"]
