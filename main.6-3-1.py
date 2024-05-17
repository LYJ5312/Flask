import torch
import cv2
from numpy import random
import numpy as np
from urllib.request import urlopen
from keras.models import load_model
import threading
import time
import os
import cx_Oracle

username = 'uno'
password = 'uno'
dsn = '192.168.0.37/xe'

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ip = '192.168.137.169'
stream = urlopen('http://' + ip + ':81/stream')
buffer = b''
urlopen('http://' + ip + "/action?go=speed40")

motor_model = load_model(r"d:\\workspaces\\arduino\\converted_keras\\keras_model.h5", compile=False)
class_names = open(r"d:\\workspaces\\arduino\\converted_keras\\labels.txt", "r").readlines()

img_model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

car_state2 = "go"
car_speed = 40
thread_frame = None  
image_flag = 0
thread_image_flag = 0
img = None

prev_speed = 40
prev_state = "go"
current_suno = None

db_connection = cx_Oracle.connect(username, password, dsn)

def insert_or_update_database(speed, state):
    global current_suno
    try:
        db_cursor = db_connection.cursor()

        select_query = """
            SELECT COUNT(*) FROM AI_TRANS WHERE SUNO = :suno
        """
        db_cursor.execute(select_query, suno=current_suno)
        count = db_cursor.fetchone()[0]

        if count > 0:
            update_query = """
                UPDATE AI_TRANS
                SET TP_SPEED = :speed, TP_STATE = :state, TP_TIME = SYSDATE
                WHERE SUNO = :suno
            """
            db_cursor.execute(update_query, speed=speed, state=state, suno=current_suno)
            print("Data updated in the database successfully.")
        else:
            insert_query = """
                INSERT INTO AI_TRANS (TP_SPEED, TP_STATE, TP_TIME, SUNO)
                VALUES (:speed, :state, SYSDATE, :suno)
            """
            db_cursor.execute(insert_query, speed=speed, state=state, suno=current_suno)
            print("Data inserted into the database successfully.")

        db_connection.commit()
        db_cursor.close()
    except cx_Oracle.Error as error:
        print("Error occurred while inserting or updating data in the database:", error)

label_actions = {
    "speed40": ("go", 40, "운행중", "/action?go=speed40"),
    "speed60": ("go", 60, "운행중", "/action?go=speed60"),
    "storage": ("storage", 0, "물품하차중", "/action?go=speed0"),
    "farm": ("farm", 0, "복귀완료", "/action?go=speed0")
}

def yolo_thread():
    global image_flag, thread_image_flag, frame, thread_frame, car_state2, car_speed, current_suno, prev_speed, prev_state
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

                    if label in label_actions and conf > 0.75:
                        car_state2, car_speed, tp_state, action_url = label_actions[label]
                        print(label)
                        insert_or_update_database(car_speed, tp_state)
                        urlopen('http://' + ip + action_url)
                        
                        if label == "storage":
                            time.sleep(5)
                            tp_state = "하차완료"
                            insert_or_update_database(car_speed, tp_state)
                            time.sleep(3)
                            car_state2 = "복귀중"
                            car_speed = 40
                            insert_or_update_database(car_speed, tp_state)
                            urlopen('http://' + ip + "/action?go=speed40")
                         
                    color = [int(c) for c in random.choice(range(256), size=3)]
                    cv2.rectangle(thread_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(thread_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
            thread_image_flag = 1
            image_flag = 0
            
t1 = threading.Thread(target=yolo_thread)
t1.daemon = True
t1.start()

def image_process_thread():
    global img, image_flag, car_speed, car_state2, prev_speed, prev_state
    while True:
        if image_flag == 1:
            resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            normalized_img = (resized_img / 127.5) - 1
            normalized_img = np.expand_dims(normalized_img, axis=0)

            prediction = motor_model.predict(normalized_img)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            percent = int(str(np.round(confidence_score * 100))[:-2])

            if "go" in class_name[2:] and car_state2 == "go" and percent >= 85:
                urlopen('http://' + ip + "/action?go=forward")
                car_speed = 40
            elif "left" in class_name[2:] and car_state2 == "go" and percent >= 85:
                urlopen('http://' + ip + "/action?go=left")
                car_speed = 40
            elif "right" in class_name[2:] and car_state2 == "go" and percent >= 85:
                urlopen('http://' + ip + "/action?go=right")
                car_speed = 40
            elif car_state2 == "stop":
                urlopen('http://' + ip + "/action?go=stop")
                car_speed = 0

            image_flag = 0

t2 = threading.Thread(target=image_process_thread)
t2.daemon = True 
t2.start()

while True:
    buffer += stream.read(8192)
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')
    
    try: 
        if head > -1 and end > -1:
            jpg = buffer[head:end+2]
            buffer = buffer[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            frame = cv2.resize(img, (640, 480))
            height, width, _ = img.shape
            img = img[height // 2:, :]

            cv2.imshow("AI CAR Streaming", img)
            
            image_flag = 1
           
            if thread_image_flag == 1:
                cv2.imshow('thread_frame', thread_frame)
                thread_image_flag = 0

                prev_speed = car_speed
                prev_state = car_state2

            db_connection = cx_Oracle.connect(username, password, dsn)
            db_cursor = db_connection.cursor()
            select_query = "SELECT MAX(SUNO) FROM TRANS_R"
            db_cursor.execute(select_query)
            suno = db_cursor.fetchone()[0]
            if suno is None:
                suno = 0  # 또는 적절한 초기값 설정
            db_cursor.close()
            db_connection.close()

            current_suno = suno

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == 32:
                car_state2 = "go"

    except Exception as e:
        print("Error:", str(e))

db_connection.close()
urlopen('http://' + ip + "/action?go=stop")
cv2.destroyAllWindows()