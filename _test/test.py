from flask import Flask, render_template
import cv2

app = Flask(__name__)


camera = cv2.VideoCapture(0)
camera.set(3,1280)
camera.set(4,720)
cv2.namedWindow('move_detect', cv2.WINDOW_NORMAL)


# 웹캠 시작
while camera.isOpened():
    ok, frame = camera.read()

    # 0. 웹캠 에러 처리    
    if not ok:
        print('[ERROR] Camera error')
        break

    cv2.imshow('move_detect', frame)

    # 4. 종료 조건
    # 4-1. ESC 키를 누를 경우
    k = cv2.waitKey(1) & 0xFF 
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()

