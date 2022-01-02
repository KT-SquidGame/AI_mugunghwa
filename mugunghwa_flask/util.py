from datetime import datetime
import threading
import cv2
import math
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import simpleaudio as sa
import random


# 공통 함수(라이브러리) 정의
class util():

    # __init__() : 변수 초기화 함수
    def __init__(self):

        # 게임 시작부터 종료까지의 웹캠 frame 수
        # self.total_frame_count = 0
        
        # mission image에 대한 값
        self.mission_pose = ''

        # not move detect에 대한 시간 측정
        self.move_frame_count = 0
        # self.move_start_time = 0 
        # self.move_result_time = 0 

        # correct pose detect에 대한 시간 측정
        self.pose_frame_count = 0
        # self.pose_start_time = 0 
        # self.pose_result_time = 0 

        # 음악 재생 여부
        # self.play_sound_tf = False

        # 움직임 탐지 임계 값
        self.MOVE_THRESHOLD = 2000 

        # mediapipe 라이브러리 중 pose 클래스의 초기화
        self.mp_pose = mp.solutions.pose

        # pose 클래스를 통한 pose 함수 설정
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

        # mediapipe 라이브러리 중 drawing 클래스의 초기화        
        self.mp_drawing = mp.solutions.drawing_utils 

        # 웹캠 출력을 위한 pose 함수 설정
        self.pose_video = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)


    # _pose_detect() : 영상 속 frame 단위의 image에서 pose를 탐지하여 시각화 결과를 반환하는 함수
    def _pose_detect(self, image, pose='', display=True):
        pose = self.pose_video
        output_image = image.copy()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # pose detection 수행
        results = pose.process(imageRGB)
        
        # input image의 크기 가져옴
        height, width, _ = image.shape
        
        landmarks = []
        # pose_landmarks에 대해 탐지 탐지 성공 시, output image에 draw 함수를 통해 탐지 여부 표시
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=self.mp_pose.POSE_CONNECTIONS)
            
            # 탐지된 landmark에 대해 반복 수행하여 landmarks 변수에 append
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                    (landmark.z * width)))
        
        # 결과 image 표시
        if display:
            plt.figure(figsize=[22,22])
            plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
            plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
            self.mp_drawing.plot_landmarks(results.pose_world_landmarks, self.mp_pose.POSE_CONNECTIONS)
        else:
            return output_image, landmarks




    # _pose_calculate_angle() : 세 개의 landmark 간의 각도(angle)를 계산하는 함수
    def _pose_calculate_angle(self, landmark1, landmark2, landmark3):
        # landmark의 좌표 가져옴
        x1, y1, _ = landmark1
        x2, y2, _ = landmark2
        x3, y3, _ = landmark3

        # landmark 간의 각도 계산
        # 첫 번째 landmark : 첫 번째 라인의 시작점
        # 두 번째 landmark : 첫 번째 라인의 끝점과 두 번째 라인의 시작점
        # 세 번째 landmark : 두 번째 라인의 끝점
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        
        #각도가 음수일 경우 360도를 더함
        if angle < 0:
            angle += 360
        
        return angle




    # _pose_classify() : landmark 간 게산된 각도를 바탕으로 pose를 구별하는 함수
    def _pose_classify(self, landmarks, output_image, display=False):
        label = 'Unknown Pose'
        color = (0, 0, 255)
        
        # 각도 계산 : 왼쪽 어깨, 팔꿈치, 손목
        left_elbow_angle = self._pose_calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value])
        
        # 각도 계산 : 오른쪽 어깨, 팔꿈치, 손목
        right_elbow_angle = self._pose_calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value])   
        
        # 각도 계산 : 왼쪽 팔꿈치, 어깨, 엉덩이
        left_shoulder_angle = self._pose_calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value])

        # 각도 계산 : 오른쪽 팔꿈치, 어깨, 엉덩이
        right_shoulder_angle = self._pose_calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value])

        # 각도 계산 : 왼쪽 엉덩이, 무릎, 발목
        left_knee_angle = self._pose_calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])

        # 각도 계산 : 오른쪽 엉덩이, 무릎, 발목
        right_knee_angle = self._pose_calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        
        
        # [Warrior pose] 체크
        # 특징 : 양팔을 곧게 펴고, 어깨는 일정 각도 유지, 한쪽 다리는 곧게 펴고, 다른쪽 다리는 구부러짐

        # 양팔이 곧게 펴져 있는지 체크
        if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
            # 어깨가 일정 각도를 유지하는지 체크
            if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
                # 한쪽 다리가 곧게 펴져 있는지 체크
                if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                    # 다른쪽 다리가 구부러져 있는지 체크
                    if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                        label = 'Warrior Pose' 



        # [T pose] 체크
        # 특징 : 양팔을 곧게 펴고, 어깨는 일정 각도 유지, 두 다리는 곧게 폄
        
        # 양팔이 곧게 펴져 있는지 체크
        if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
            # 어깨가 일정 각도를 유지하는지 체크
            if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
                # 두 다리가 곧게 펴져 있는지 체크
                if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                    label = 'T Pose'



        # [Tree pose] 체크      
        # 특징 : 한쪽 다리를 곧게 펴고, 다른쪽 다리는 구부러짐
        
        # 한쪽 다리가 곧게 펴져 있는지 체크
        if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
            # 다른쪽 다리가 구부러져 있는지 체크
            if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
                label = 'Tree Pose'

        if label != 'Unknown Pose':
            self.pose_frame_count += 1
            color = (0, 255, 0)  

        # # 시간 계산
        # if label == 'Unknown Pose':
        #     if self.pose_start_time != 0:
        #         end = time.time()
        #         tmp = end - self.pose_start_time
        #         self.pose_result_time += round(tmp, 3)
        #         self.pose_start_time = 0
        # elif label == self.mission_pose:
        #     # 일치하는 pose인 frame일 경우 count
        #     self.pose_frame_count += 1
        #     if self.pose_start_time == 0:
        #         self.pose_start_time = time.time()
            
        # output image에 label 추가
        cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        
        # 결과 image 표시
        if display:
            plt.figure(figsize=[10,10])
            plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        else:
            return label



    # _move_detect() : 움직임을 탐지하는 함수
    def _move_detect(self, mask, display=False):
        label = 'not move'
        
        # 움직임의 정도를 계산하여 diff 변수화
        diff = (mask.astype('float') / 255.).sum()

        # diff 값과 움직임 탐지 임계 값 비교
        if diff > self.MOVE_THRESHOLD:
            label = 'move'
        else:
            # 움직임이 탐지되지 않은 frame인 경우 count
            self.move_frame_count += 1

        # output image에 label 추가
        cv2.putText(mask, 'player status : ' + label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

        # 시간 계산
        # if label == 'not move':
        #     if self.move_start_time != 0:
        #         end = time.time()
        #         tmp = end - self.move_start_time
        #         self.move_result_time += round(tmp, 3)
        #         self.move_start_time = 0
        # elif label == 'move':
        #     if self.move_start_time == 0:
        #         self.move_start_time = time.time()

        return label




    # def _start_game(self, pose_file_list, sound_file_name):
    #     # 미션 문제 설정
    #     pose_file_name = pose_file_list[random.randint(0,len(pose_file_list)-1)]
        
    #     # 함수가 최초 동작할 때만 mission_pose 지정
    #     if self.mission_pose == '':
    #         self.mission_pose = pose_file_name

    #     mission_image = cv2.imread('images/' + pose_file_name + '.jpg', cv2.IMREAD_COLOR)
    #     resize_mission_image = cv2.resize(mission_image, dsize=(640, 480))

    #     # 음성 출력 및 미션 문제 출력
    #     wave_obj = sa.WaveObject.from_wave_file('sound/' + sound_file_name)
    #     play_obj = wave_obj.play()
        
    #     cv2.namedWindow('!!! REMEMBER THIS POSE !!!', cv2.WINDOW_NORMAL)
    #     cv2.imshow('!!! REMEMBER THIS POSE !!!', resize_mission_image)
    #     cv2.moveWindow('!!! REMEMBER THIS POSE !!!',400,100)

    #     # 음성 출력 및 미션 문제 출력 (음성 종료 시 미션 문제 종료)
    #     # while play_obj.is_playing():
    #     #     cv2.namedWindow('!!! REMEMBER THIS POSE !!!', cv2.WINDOW_NORMAL)
    #     #     cv2.imshow('!!! REMEMBER THIS POSE !!!', resize_mission_image)
    #     #     cv2.moveWindow('!!! REMEMBER THIS POSE !!!',400,100)
    #     #     k = cv2.waitKey(1) & 0xFF 
    #     #     if k == 27:
    #     #         break
    #     # cv2.destroyAllWindows()
    
    #     self.play_sound_tf = True
        