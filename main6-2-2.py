import torch
import cv2
from numpy import random
import numpy as np
from urllib.request import urlopen

ip = '192.168.137.45'
stream = urlopen('http://' + ip +':81/stream')
buffer = b''



# YOLOv5 모델 정의
model = torch.hub.load('yolov5','custom', path='ssem_best.pt',source='local',force_reload=True)

if torch.cuda.is_available():
   model = model.cuda()

while True:
    buffer += stream.read(4096)
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')
    
    try: 
        if head > -1 and end > -1:
            jpg = buffer[head:end+2]
            buffer = buffer[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            # 프레임 크기 조정
            frame = cv2.resize(img, (640, 480))

            # 이미지를 모델에 입력
            results = model(frame)
           
            # 객체 감지 결과 얻기
            detections = results.pandas().xyxy[0]

            
            if not detections.empty:
                # 결과를 반복하며 객체 표시
                for _, detection in detections.iterrows():
                    x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values
                    label = detection['name']
                    conf = detection['confidence']

                    # 박스와 라벨 표시
                    color = [int(c) for c in random.choice(range(256), size=3)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
      
            # 프레임 표시
            cv2.imshow('frame', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    except:
        print("에러")
        pass

urlopen('http://' + ip + "/action?go=stop")
cv2.destroyAllWindows()

# main6-2-2.py
# 학습학 객체 검출하기