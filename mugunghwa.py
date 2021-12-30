from util import *
import cv2
import math
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import simpleaudio as sa
import random



# 공통 함수(라이브러리) 불러오기
qts = util()


# 변수 및 window frame 설정
robot_status = 'Scanning...' 
total_game_time = 15 # 총 게임 시간
knn = cv2.createBackgroundSubtractorKNN(history=1, dist2Threshold=10000, detectShadows=False)
camera = cv2.VideoCapture(0)
camera.set(3,1280)
camera.set(4,720)
cv2.namedWindow('pose_detect', cv2.WINDOW_NORMAL)
cv2.namedWindow('move_detect', cv2.WINDOW_NORMAL)


# 미션 문제 설정
pose_file_list = ['T Pose', 'Tree Pose', 'Warrior Pose']
pose_file_name = pose_file_list[random.randint(0,len(pose_file_list)-1)]
# pose_file_name = 'T Pose'
qts.mission_pose = pose_file_name
mission_image = cv2.imread('images/' + pose_file_name + '.jpg', cv2.IMREAD_COLOR)
resize_mission_image = cv2.resize(mission_image, dsize=(640, 480))


# 음성 출력 및 미션 문제 출력 (음성이 출력될 동안만 문제 출력)
wave_obj = sa.WaveObject.from_wave_file("sound/assets_sound.wav")
play_obj = wave_obj.play()
while play_obj.is_playing():
    cv2.namedWindow('!!! REMEMBER THIS POSE !!!', cv2.WINDOW_NORMAL)
    cv2.imshow('!!! REMEMBER THIS POSE !!!', resize_mission_image)
    cv2.moveWindow('!!! REMEMBER THIS POSE !!!',400,100)
    k = cv2.waitKey(1) & 0xFF 
    if k == 27:
        break
cv2.destroyAllWindows()


# 카운트 (카운트 후 게임 종료)
endTime = time.time() + total_game_time

# 웹캠 시작
while camera.isOpened():
    ok, frame = camera.read()

    # 웹캠 에러 처리    
    if not ok:
        print('[ERROR] Camera error')
        break

    qts.total_frame_count += 1

    # 움직임 탐지 프레임 설정
    mask = knn.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.putText(mask, robot_status, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)


    # 포즈 탐지 프레임 설정
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ =  frame.shape
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))


    # 움직임 탐지 main 
    label = qts._move_detect(mask, display=False)
    cv2.imshow('move_detect', mask)
    cv2.moveWindow('move_detect',1000,50)
    cv2.resizeWindow('move_detect', 250, 250)


    # 포즈 탐지 main
    frame, landmarks = qts._pose_detect(frame, display=False)
    if landmarks:
        label = qts._pose_classify(landmarks, frame, display=False)
        
    cv2.imshow('pose_detect', frame)
    cv2.moveWindow('pose_detect', 50, 50)
    cv2.resizeWindow('pose_detect', 900, 600)


    # 종료 조건
    # 1. ESC 키를 누를 경우 종료
    k = cv2.waitKey(1) & 0xFF 
    if k == 27:
        break
    
    # 2. endTime이 초과할 경우 종료
    if time.time() > endTime:
        break



camera.release()
cv2.destroyAllWindows()

# moveTime = round(qts.move_total_time,3)
print()
print('====== game result ======')
print()
print('총 게임 시간 (초) : ', total_game_time)
print('총 프레임 수 : ', qts.total_frame_count)
print()
print('움직이지 않은 시간 (초) : ', 15 - qts.move_result_time)
print('움직이지 않은 프레임 수 : ', qts.move_frame_count)
print()
print('포즈를 유지한 시간 (초) : ', qts.pose_result_time)
print('포즈를 유지한 프레임 수 : ', qts.pose_frame_count)
print()
print('=========================')
print()
