import cv2
import numpy as np
import time
from datetime import datetime
import sys
import mediapipe
from flask import Flask, render_template, Response
from util import *
import simpleaudio as sa
import random
from flask_cors import CORS


# 전역 변수 초기 설정
app = Flask(__name__)
CORS(app)
qts = util()
robot_status = 'Scanning...'
sound_file_name = 'assets_sound.wav'
mission_pose = ['T Pose', 'Tree Pose', 'Warrior Pose']
total_game_time = 20
end_time = 0
now = 0
global result_score
result_score = '0'
timeout_tf = False
result_score_of_move = 0
result_score_of_pose = 0


###### flask 서버 함수 ######
# route : index()
@app.route('/', methods=['GET', 'POST'])
def index():
    # 무궁화 꽃이 피었습니다 음성 출력 (page load delay : 2s)
    time.sleep(2)
    wave_obj = sa.WaveObject.from_wave_file('sound/' + sound_file_name)
    play_obj = wave_obj.play()

    # 전역 변수 초기화
    global timeout_tf
    timeout_tf = False
    global result_score
    result_score = '0'
    qts.move_frame_count = 0
    qts.pose_frame_count = 0
    global end_time
    
    # 게임 종료 시간 초기화
    end_time = time.time() + total_game_time

    # mission pose image 가져오기 (random)
    qts.mission_pose = mission_pose[random.randint(0,0)]

    templateData = {
            'title': 'mission pose',
            'mission_pose': qts.mission_pose   
    }

    return render_template('index.html', **templateData)
    # return str(qts.mission_pose)


# route : pose_main()
@app.route('/pose_main')
def pose_main():
    return Response(pose(), mimetype='multipart/x-mixed-replace; boundary=frame')


# route : move_main()
@app.route('/move_main')
def move_main():
    return Response(move(), mimetype='multipart/x-mixed-replace; boundary=frame')


# route : result_main()
@app.route('/result_main')
def result_main():
    global result_score
    return str(result_score)


###### 기능 구현 함수 ######
# 기능 함수 : move()
def move():
    # knn 필터 설정 (history 초 동안의 움직임을 감지, dist2Threshold 이상의 움직임을 mask로 표현)
    knn = cv2.createBackgroundSubtractorKNN(history=1, dist2Threshold=10000, detectShadows=False)
    # 카메라 설정
    camera = cv2.VideoCapture(0)
    while(True):
        success, frame = camera.read()  
        if not success:
            break
        else:
            # mask에 knn 필터 적용
            mask = knn.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # 이전 frame과 현재 frame을 비교하여 움직임 감지
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 움직임 탐지 main 함수 호출
            label = qts._move_detect(mask, display=False)

            # 결과를 화면에 출력
            cv2.putText(mask, 'robot status : ' + robot_status, (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)

            # 웹캠 화면 출력
            ret, buffer = cv2.imencode('.jpg', mask)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 기능 함수 : pose()
def pose():
    # 카메라 설정
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:
            # 출력 frame 설정
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ =  frame.shape
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

            # 전역 변수 선언
            global end_time
            global timeout_tf
            global result_score
            global result_score_of_move
            global result_score_of_pose

            # 게임 동작 시간 기준 설정
            now = time.time()
            tmp = round(end_time - now,3)

            # 남은 시간이 음수(0)라면, 타이머 종료, 결과 출력
            if tmp < 0:
                if timeout_tf == False:
                    timeout_tf = True
                    result_score_of_move = qts.move_frame_count
                    result_score_of_pose = qts.pose_frame_count
                    result_score_of_total = (result_score_of_move + result_score_of_pose)
                    if result_score_of_total > 600:
                        result_score_of_total = 600
                    # 게임 점수
                    result_score = '%.2f' % (result_score_of_total / 650.0 * 100.0)
                # 결과에 대해 출력                
                cv2.putText(frame, 'score (move) : ' + str(result_score_of_move), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2.3, (255, 0, 0), 2)
                cv2.putText(frame, 'score (pose) : ' + str(result_score_of_pose), (10, 60),cv2.FONT_HERSHEY_PLAIN, 2.3, (255, 0, 0), 2)
                cv2.putText(frame, '!!! Time Out !!!', (10, 90),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            
            else:
                left_time = str(tmp)
                cv2.putText(frame, 'not move frame count : ' + str(qts.move_frame_count), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2.3, (255,0,0), 2)
                cv2.putText(frame, 'correct pose frame count : ' + str(qts.pose_frame_count), (10, 60),cv2.FONT_HERSHEY_PLAIN, 2.3, (255,0,0), 2)
                cv2.putText(frame, left_time + ' seconds left.', (10, 90),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

            
            # 포즈 탐지 main 호출
            frame, landmarks = qts._pose_detect(frame, display=False)
            # 포즈에서 landmark 탐지 시, 포즈 구별 수행
            if landmarks:   
                label = qts._pose_classify(landmarks, frame, display=False)

            # 웹캠 출력
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

        
                

# main()
if __name__ == '__main__':
    app.run(host='localhost', port=5100, threaded=True, use_reloader=False) 