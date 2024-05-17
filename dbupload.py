from flask import Flask, render_template, jsonify, Blueprint, Response, request
import torch
import cv2
import cx_Oracle
from numpy import random
import numpy as np
from urllib.request import urlopen
from keras.models import load_model
import torch
import threading
import time
import os
import datetime
from time import sleep

app = Flask(__name__)

abcs = Blueprint(
    "abcs",
    __name__,
    template_folder="templates", 
    static_folder="static"
)

# DB 설정
username = 'abcs'
password = 'abcs'
dsn = '192.168.0.145:1521/xe'
db_connection = cx_Oracle.connect(username, password, dsn)

# 환경 설정
ip = '192.168.137.37'
stream = urlopen('http://' + ip +':81/stream')
buffer = b''

# keras 모델 로드
motor_model = load_model(r"D:\\workspaces\\Arduino\\apps\\abcs\\converted_keras\\keras_model.h5", compile=False)
class_names = open(r"D:\\workspaces\\Arduino\\apps\\abcs\\converted_keras\\labels.txt", "r").readlines()

# YOLOv5 모델 로드
img_model = torch.hub.load('D:\\workspaces\\Arduino\\apps\\abcs\\yolov5', 'custom',
                           path='D:\\workspaces\\Arduino\\apps\\abcs\\best.pt', source='local')

# 차량 상태 및 이미지 플래그 초기화
car_state2 = "go"
thread_frame = None  
image_flag = 0
thread_image_flag = 0
img = None
key = None

#객체 검출
def yolo_thread():
    global image_flag, thread_image_flag, frame, thread_frame, car_state2
    while True:
        if image_flag == 1:
            thread_frame = frame

            # 이미지를 모델에 입력
            results = img_model(thread_frame)
            # 객체 감지 결과 얻기
            detections = results.pandas().xyxy[0]

            if not detections.empty:
                # 결과를 반복하며 객체 표시
                for _, detection in detections.iterrows():
                    x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values
                    label = detection['name']
                    conf = detection['confidence']

                    if "car" in label and conf > 0.8:
                        print("car detected, stopping")
                        car_state2 = "stop"
                        urlopen('http://' + ip + "/action?go=stop")
                    elif "stop" in label and conf > 0.8:
                        print("stop")
                        car_state2 = "stop"

                    # 실시간 정류소 인식 db반영 코드
                    if label in ['one1', 'two2', 'three3', 'four4', 'five5', 'six6']:
                        update_statement = "UPDATE LOCATION SET LOCATION = :label WHERE NO = :no"
                        cursor = db_connection.cursor()
                        cursor.execute(update_statement, {'label': label, 'no': 1})
                        db_connection.commit()
                        cursor.close()
                        print(f"Database updated: {label} detected")
                    
                    #person 감지되면 db에 person 추가               
                    if label in ['person']:
                        update_statement = "UPDATE LOCATION SET DETECT = :label WHERE NO = :no"
                        cursor = db_connection.cursor()
                        cursor.execute(update_statement, {'label': label, 'no': 1})
                        db_connection.commit()
                        cursor.close()
                        print(f"Database updated: {label} detected")
                   
                    #person감지 안되면 no표시
                    else:                  
                        update_statement = "UPDATE LOCATION SET DETECT = 'no' WHERE NO = :no"
                        cursor = db_connection.cursor()
                        cursor.execute(update_statement, {'no': 1})
                        db_connection.commit()
                        cursor.close()
                        print("Database updated: 'no' detected")
                    
                    # 박스와 라벨 표시
                    color = [int(c) for c in random.choice(range(256), size=3)]
                    cv2.rectangle(thread_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(thread_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            thread_image_flag = 1
            image_flag = 0
            

#자율 주행
def image_process_thread():
    global img, ip, image_flag, car_state2, motor_model, class_names
    
    while True:
        if image_flag == 1:
            global img
            img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
            img = (img / 127.5) - 1

            prediction = motor_model.predict(img)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            percent = int(str(np.round(confidence_score * 100))[:-2])
            
            if "go" in class_name[2:] and car_state2=="go" and percent >= 95:
                urlopen('http://' + ip + "/action?go=forward")
            elif "left" in class_name[2:] and car_state2=="go" and percent >= 95:
                urlopen('http://' + ip + "/action?go=left")
            elif "right" in class_name[2:] and car_state2=="go" and percent >= 95:
                urlopen('http://' + ip + "/action?go=right")
            elif car_state2=="stop":
                urlopen('http://' + ip + "/action?go=stop")
                
            image_flag = 0
    

@abcs.route('/video')
def video():
    return render_template('control.html')

@abcs.route('/video_2')
def video_2():
    return render_template('parents.html')

@abcs.route('/control')
def control():
    return render_template('teacher.html')


def send_video():
    global thread_frame
    
    while True:
        ret, buffer11 = cv2.imencode('.jpg', thread_frame)
        frame11 = buffer11.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame11 + b'\r\n')

@abcs.route('/car_video')
def car_video():
    return Response(send_video(),  mimetype='multipart/x-mixed-replace; boundary=frame')


#주행시작 제어
@abcs.route('/start-driving', methods=['POST'])
def start_driving():
    global driving_enabled, car_state2
    driving_enabled = True
    try:
        # urlopen('http://' + ip + '/action?go=forward')  #차량 제어 api
        car_state2="go"
        return jsonify(status="success", message="주행 시작됨", driving_enabled=driving_enabled)
    except Exception as e:
        return jsonify(status="error", message=str(e), driving_enabled=driving_enabled)

#긴급정지 제어
@abcs.route('/stop-driving', methods=['POST'])
def stop_driving():
    global driving_enabled, car_state2
    driving_enabled = False
    try:
        # urlopen('http://' + ip + '/action?go=stop')
        car_state2="stop"
        return jsonify(status="success", message="차량 정지됨", driving_enabled=driving_enabled)
    except Exception as e:
        return jsonify(status="error", message=str(e), driving_enabled=driving_enabled)
    

def video_stream():
    global stream, buffer, thread_image_flag, car_state2, image_flag, frame, img

    # 데몬 스레드 생성
    t1 = threading.Thread(target=yolo_thread)
    t1.daemon = True 
    t1.start()

    t2 = threading.Thread(target=image_process_thread)
    t2.daemon = True 
    t2.start()

    while True:
        buffer += stream.read(4096)
        head = buffer.find(b'\xff\xd8')
        end = buffer.find(b'\xff\xd9')
        
        if head > -1 and end > -1:
            jpg = buffer[head:end+2]
            buffer = buffer[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            # 프레임 크기 조정
            frame = cv2.resize(img, (640, 480))

            # 아래부분의 반만 자르기
            height, width, _ = img.shape
            img = img[height // 2:, :]
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imshow("AI CAR Streaming", img)
            
            image_flag = 1
        
            #쓰레드에서 이미지 처리가 완료되었으면
            if thread_image_flag == 1:
                #cv2.imshow('thread_frame', thread_frame)
                thread_image_flag = 0

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

# 데몬 스레드 생성.
t3 = threading.Thread(target=video_stream)
t3.daemon = True 
t3.start()