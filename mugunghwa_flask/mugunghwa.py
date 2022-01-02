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


# 변수 설정
app = Flask(__name__)
qts = util()
robot_status = 'Scanning...' 
sound_file_name = 'assets_sound.wav'
result_status = [1,2,3]
mission_pose = ['T', 'Tree', 'Warrior']
knn = cv2.createBackgroundSubtractorKNN(history=1, dist2Threshold=10000, detectShadows=False)

# 게임 시간과 결과 점수에 대한 변수 설정
total_game_time = 10
end_time = 0
now = 0
result_score = 0
timeout_tf = False


@app.route('/', methods=['GET', 'POST'])
def index():
    # 무궁화 꽃이 피었습니다 음성 출력 (page load delay : 2s)
    time.sleep(2)
    wave_obj = sa.WaveObject.from_wave_file('sound/' + sound_file_name)
    play_obj = wave_obj.play()

    global end_time
    end_time = time.time() + total_game_time
    # end_time = time.time() + total_game_time
    templateData = {
            'title': 'mission pose',
            'mission_pose': str(mission_pose[random.randint(0,2)])
    }
    return render_template('index.html', **templateData)




@app.route('/timer_main')
def timer_main():
    return Response(timer(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/pose_main')
def pose_main():
    return Response(pose(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/move_main')
def move_main():
    return Response(move(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/result_main')
def result_main():
    global result_score
    return result_score




###### 기능 구현 함수 ######

def timer():
    camera = cv2.VideoCapture(0)
    while(True):
        success, frame = camera.read()  
        if not success:
            break
        else:
            blank = np.zeros((360, 640, 3), np.uint8)

            # #putText : 남은 시간
            now = time.time()
            global end_time
            tmp = round(end_time - now,3)
            # 남은 시간이 음수(0)라면, 타이머 종료, 결과 보여주기
            if tmp < 0:
                global timeout_tf
                if timeout_tf == False:
                    timeout_tf = True
                    result_score_of_move = qts.move_frame_count
                    result_score_of_pose = qts.pose_frame_count
                    if result_score_of_move < 200 & result_score_of_pose < 100:
                        result_score = result_status[0]
                    elif result_score_of_move < 100 & result_score_of_pose < 50:
                        result_score = result_status[1]
                    else:
                        result_score = result_status[2]
                cv2.putText(blank, 'finish! score : ' + str(result_score), (10, 90),cv2.FONT_HERSHEY_PLAIN, 2.3, (0,0,255), 2)
                    
            else:
                left_time = str(tmp)
                cv2.putText(blank, left_time + ' seconds left.', (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
                cv2.putText(blank, 'not move frame count : ' + str(qts.move_frame_count), (10, 60),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
                cv2.putText(blank, 'correct pose frame count : ' + str(qts.pose_frame_count), (10, 90),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
                
            

            ret, buffer = cv2.imencode('.jpg', blank)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




def move():
    camera = cv2.VideoCapture(0)
    while(True):
        success, frame = camera.read()  
        if not success:
            break
        else:
            mask = knn.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            qts.total_frame_count += 1

            # 움직임 탐지 main
            label = qts._move_detect(mask, display=False)

            cv2.putText(mask, robot_status, (10, 60),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

            ret, buffer = cv2.imencode('.jpg', mask)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def pose():
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