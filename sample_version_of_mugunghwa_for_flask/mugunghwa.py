import cv2
import numpy as np
import time
from datetime import datetime
import sys
import mediapipe
from flask import Flask, render_template, Response
from util import *

# 변수 설정
app = Flask(__name__)
qts = util()
knn = cv2.createBackgroundSubtractorKNN(history=1, dist2Threshold=10000, detectShadows=False)
num = 3
robot_status = 'Scanning...' 


@app.route('/')
def index():
    now = datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    templateData = {
            'title':'Image Streaming',
            'time': timeString
            }
    return render_template('index.html', **templateData)



@app.route('/move_main')
def move_main():
    return Response(move_main(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/pose_main')
def pose_main():
    return Response(pose_main(), mimetype='multipart/x-mixed-replace; boundary=frame')



###### 기능 구현 함수 ######

def move_main():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:
            mask = knn.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 움직임 탐지 main
            label = qts._move_detect(mask, display=False)

            cv2.putText(mask, robot_status, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            ret, buffer = cv2.imencode('.jpg', mask)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  



def pose_main():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ =  frame.shape
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
            
            # 포즈 탐지 main
            frame, landmarks = qts._pose_detect(frame, display=False)
            if landmarks:   
                label = qts._pose_classify(landmarks, frame, display=False)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  





if __name__ == '__main__':
    app.run(host='localhost', port=5100, threaded=True, use_reloader=False) 