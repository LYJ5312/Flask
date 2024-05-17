from flask import Flask, Blueprint, render_template, Response, jsonify, send_file, request
import cv2
import torch
from numpy import random
from flask_login import login_required
import os
from datetime import datetime
import requests
import cx_Oracle
import time



app = Flask(__name__)



unofarm = Blueprint(
    "unofarm", 
    __name__,   
    template_folder="templates", 
    static_folder="static"
)

model = torch.hub.load('apps/unoFarm/yolov5', 'custom', path='apps/unoFarm/best4.pt', source='local', force_reload=True)
if torch.cuda.is_available():
    model = model.cuda()


# Define connection details
username = 'uno'
password = 'uno'
dsn = '192.168.0.139:1521/xe'   #192.168.0.139 #localhost





# Establish a connection
connection = cx_Oracle.connect(username, password, dsn)

log_dir = 'C:/unofarm/logs'
video_dir = 'C:/unofarm/video/file'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Define a flag to indicate whether recording is in progress
recording = False
# Define the start time of the recording
recording_start_time = None
# Define the video writer
video_writer = None
# Define the maximum recording duration (10 seconds)
max_record_duration = 30

# Dictionary to store the last detection time for each label
last_detection_times = {}

def start_recording(frame):
    global recording, recording_start_time, video_writer
    
    # If recording is already in progress, do nothing
    if recording:
        return
    
    # Set recording flag to True
    recording = True
    # Get the current time as the start time of the recording
    recording_start_time = time.time()
    
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the filename with current date and time
    video_filename = f'C:/unofarm/video/file/{current_datetime}.mp4'
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20, (frame.shape[1], frame.shape[0]))

def stop_recording():
    global recording, video_writer
    
    # If recording is not in progress, do nothing
    if not recording:
        return
    
    # Release the video writer
    video_writer.release()
    # Set recording flag to False
    recording = False



def detect_objects():
    cap = cv2.VideoCapture(0)

    while True:

        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]

        if not detections.empty:
            for _, detection in detections.iterrows():
                x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values
                label = detection['name']
                conf = detection['confidence']
                # Replace 'farmer' with the actual ID of the farmer
                id_value = 'nongjang'

# Replace 'danger' with the appropriate status value indicating a dangerous situation
                status_value = 'danger'

                # Check if confidence level is above 80%
                if conf > 0.7 and label in ['man','boar','dog','bird','hen','pig']:
                    color = [int(c) for c in random.choice(range(256), size=3)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                     # Check if this label has been detected before
                    if label in last_detection_times:
                        # Calculate the time elapsed since the last detection of the same label
                        time_elapsed = time.time() - last_detection_times[label]
                        
                        # If less than 10 seconds have passed, skip logging for this detection
                        if time_elapsed < 10:
                            continue

                    # Update the last detection time for this label
                    last_detection_times[label] = time.time()


                    cursor = connection.cursor()
                    vno_sequence_query = "SELECT vno_seq.nextVal FROM dual"
                    cursor.execute(vno_sequence_query)
                    vno = cursor.fetchone()[0]
                    cursor.close()

                    # Retrieve the next value from the sequence for DNO
                    cursor = connection.cursor()
                    dno_sequence_query = "SELECT farm_control_seq.nextVal FROM dual"
                    cursor.execute(dno_sequence_query)
                    dno = cursor.fetchone()[0]
                    cursor.close()
                   
                    # Get current timestamp
                    timestamp = datetime.now()

                    # Insert detection log into the database
                    cursor = connection.cursor()
                    insert_statement = """
                    INSERT INTO FARM_CONTROL(DNO, D_LABEL, D_TIME, ID, D_STATUS)
                    VALUES (:1, :2, :3, :4, :5)
                    """
                    cursor.execute(insert_statement, (dno, label, timestamp, 'nongjang', 'danger'))
                    connection.commit()
                    cursor.close()
                    

                    video_path = "c:/unofarm/video/file"
                    filetype = ".mp4"
                    # 비디오 정보를 데이터베이스에 삽입
                    cursor = connection.cursor()
                    insert_statement = """
                    INSERT INTO video(vno, v_title, pathupload, filetype, dno)
                    VALUES (:1, :2, :3, :4, :5)
                    """
                    cursor.execute(insert_statement, (vno, label, video_path, filetype, dno))
                    connection.commit()
                    cursor.close()




                    # Log detection
                    log_filename = os.path.join(log_dir, f"log_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.txt")
                    with open(log_filename, 'a') as f:
                        f.write(f"객체 감지: {label}, 신뢰도: {conf:.2f}\n")


                    if not recording:
                        start_recording(frame)


             
        if recording:
            video_writer.write(frame)


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        if recording and time.time() - recording_start_time >= max_record_duration:
            stop_recording()


        time.sleep(0.1)


def start_detection():
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    for frame_bytes in detect_objects(cap):
        # Yield the frame bytes for streaming
        yield frame_bytes

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()










@unofarm.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

@unofarm.route('/logs')
def get_logs():
    all_logs = ""
    for log_file in os.listdir(log_dir):
        if log_file.endswith(".txt"):
            with open(os.path.join(log_dir, log_file), 'r') as f:
                all_logs += f.read() + "\n"
    return all_logs

@unofarm.route("/api/logs", methods=['GET'])
def get_logs_api():
    logs = []
    for log_file in os.listdir(log_dir):
        if log_file.endswith(".txt"):
            with open(os.path.join(log_dir, log_file), 'r') as f:
                log_content = f.read()
                logs.append({"file_name": log_file, "content": log_content})
    return jsonify({"logs": logs})

@unofarm.route("/api/object_detection_video", methods=['GET','POST'])
def get_object_detection_video_api():
    video_path = "c:/unofarm/video/file"  # 경로 설정
    return jsonify({"object_detection_video": video_path})

@unofarm.route("/download_video", methods=['GET'])
def download_video_api():
    video_path = "c:/unofarm/video/file"  # 경로 설정
    return send_file(video_path, as_attachment=True)


@app.route('/send_logs', methods=['POST'])
def send_logs():
    # 로그 데이터 수집 및 가공
    log_data = request.json
    
    # 로그 데이터를 Java 애플리케이션으로 전송
    try:
        response = requests.post('http://localhost/control/monitor', json=log_data)
        if response.status_code == 200:
            return jsonify({'message': 'Logs sent successfully'}), 200
        else:
            return jsonify({'message': 'Failed to send logs'}), response.status_code
    except Exception as e:
        return jsonify({'message': 'Error occurred while sending logs', 'error': str(e)}), 500


app.register_blueprint(unofarm, url_prefix="/unofarm")

if __name__ == "__main__":
    app.run(debug=True)
