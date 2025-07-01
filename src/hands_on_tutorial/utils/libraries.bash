#!/bin/bash

pip install opencv-python

pip install ultralytics
pip install mediapipe
pip install pygame

sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev python3-dev python3-pip

pip install dlib
pip install face_recognition

sudo apt install -y tesseract-ocr
pip install pytesseract