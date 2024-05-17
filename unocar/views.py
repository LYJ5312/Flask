from flask import Flask, render_template, jsonify, Blueprint, Response, request
import torch
import cv2
import cx_Oracle
from numpy import random
import numpy as np
from urllib.request import urlopen
from keras.models import load_model
import threading
import time

app = Flask(__name__)

unocar = Blueprint(
    "unocar",
    __name__,
    template_folder="templates",
    static_folder="static"
)

ip = '192.168.137.97'
stream = urlopen('http://' + ip + ':81/stream')
buffer = b''

motor_model = load_model(r"d:\\workspaces\\arduino\\converted_keras\\keras_model.h5", compile=False)
class_names = open(r"d:\\workspaces\\arduino\\converted_keras\\labels.txt", "r").readlines()

img_model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

car_state2 = "stop"
thread_frame = None
image_flag = 0
thread_image_flag = 0
img = None

username = 'uno'
password = 'uno'
dsn = '192.168.0.139:1521/xe'
db_connection = cx_Oracle.connect(username, password, dsn)

def update_database(speed, state):
    select_query = "SELECT MAX(SUNO) FROM AI_TRANS"
    cursor = db_connection.cursor()
    cursor.execute(select_query)
    suno = cursor.fetchone()[0]
    if suno is None:
        suno = 0
    cursor.close()

    update_statement = "UPDATE AI_TRANS SET TP_SPEED = :speed, TP_STATE = :state, TP_TIME = SYSDATE WHERE SUNO = :suno"
    cursor = db_connection.cursor()
    cursor.execute(update_statement, {'speed': speed, 'state': state, 'suno': suno})
    db_connection.commit()
    cursor.close()

def yolo_thread():
    global image_flag, thread_image_flag, frame, thread_frame, car_state2, state, speed
    state = '정지'
    speed = 0

    while True:
        if image_flag == 1:
            thread_frame = frame
            results = img_model(thread_frame)
            detections = results.pandas().xyxy[0]

            if not detections.empty:
                for _, detection in detections.iterrows():
                    x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values
                    label = detection['name']
                    conf = detection['confidence']

                    if "stop" in label and conf > 0.75:
                        print("stop")
                        car_state2 = "stop"
                        time.sleep(1)
                        car_state2 = "go"
                        urlopen('http://' + ip + "/action?go=speed40")

                    elif "speed40" in label and conf > 0.75:
                        state = '운행중'
                        speed = 40
                        update_database(speed, state)
                        urlopen('http://' + ip + "/action?go=speed40")
                        print(f"Database updated: {label} detected")

                    elif "speed60" in label and conf > 0.75:
                        state = '운행중'
                        speed = 60
                        update_database(speed, state)
                        urlopen('http://' + ip + "/action?go=speed60")
                        print(f"Database updated: {label} detected")

                    elif "farm" in label and conf > 0.75:
                        state = '복귀완료'
                        speed = 0
                        update_database(speed, state)
                        urlopen('http://' + ip + "/action?go=stop")
                        print(f"Database updated: {label} detected")

                    elif "storage" in label and conf > 0.69:
                        print("storage")
                        car_state2 = "stop"

                        state = "물품하차중"
                        speed = 0
                        update_database(speed, state)
                        urlopen('http://' + ip + "/action?go=stop")
                        time.sleep(3)

                        state = "하차완료"
                        speed = 0
                        update_database(speed, state)
                        time.sleep(3)

                        car_state2 = "go"
                        state = "복귀중"
                        speed = 40
                        update_database(speed, state)
                        urlopen('http://' + ip + "/action?go=speed40")

                    color = [int(c) for c in random.choice(range(256), size=3)]
                    cv2.rectangle(thread_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(thread_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            thread_image_flag = 1
            image_flag = 0

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

            if "go" in class_name[2:] and car_state2 == "go" and percent >= 95:
                print("직진:", str(np.round(confidence_score * 100))[:-2], "%")
                urlopen('http://' + ip + "/action?go=forward")

            elif "left" in class_name[2:] and car_state2 == "go" and percent >= 95:
                print("왼쪽:", str(np.round(confidence_score * 100))[:-2], "%")
                urlopen('http://' + ip + "/action?go=left")

            elif "right" in class_name[2:] and car_state2 == "go" and percent >= 95:
                print("오른쪽:", str(np.round(confidence_score * 100))[:-2], "%")
                urlopen('http://' + ip + "/action?go=right")

            elif car_state2 == "stop":
                urlopen('http://' + ip + "/action?go=stop")

            image_flag = 0

@unocar.route('/start_delivery', methods=['GET'])
def start_delivery():
    global car_state2, suno
    car_state2 = "go"
    urlopen('http://' + ip + "/action?go=speed40")
    suno += 1
    state = "운행중"
    speed = 40
    update_database(speed, state)
    return jsonify({'수거번호': 'Delivery started','suno':suno})

@unocar.route('/')
def car():
    return render_template('car.html')

def send_video():
    global thread_frame

    while True:
        ret, buffer11 = cv2.imencode('.jpg', thread_frame)
        frame11 = buffer11.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame11 + b'\r\n')

@unocar.route('/video')
def video():
    return Response(send_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def video_stream():
    global stream, buffer, thread_image_flag, car_state2, image_flag, frame, img

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
            jpg = buffer[head:end + 2]
            buffer = buffer[end + 2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            frame = cv2.resize(img, (640, 480))

            height, width, _ = img.shape
            img = img[height // 2:, :]

            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            cv2.imshow("AI CAR Streaming", img)

            image_flag = 1

            if thread_image_flag == 1:
                thread_image_flag = 0

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == 32:
                car_state2 = "go"

t3 = threading.Thread(target=video_stream)
t3.daemon = True
t3.start()