import cv2
import numpy as np
from urllib.request import urlopen
import time

ip = '192.168.137.18'
stream = urlopen('http://' + ip + ':81/stream')
buffer = b''

while True:
    buffer += stream.read(4096)
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')
    
    try:
        if head > -1 and end > -1:
            jpg = buffer[head:end+2]
            buffer = buffer[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            # 아래부분의 반만 자르기
            height, width, _ = img.shape
            img = img[height // 2:, :]

            lower_bound = np.array([0,0,0])
            upper_bound = np.array([255, 255, 80])
            mask = cv2.inRange(img, lower_bound, upper_bound)
            
            cv2.imshow("mask", mask)

            M =cv2.moments(mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX,cY = 0,0
            
            center_offset = width // 2 -cX
            print(center_offset)

            cv2.circle(img,(cX,cY),10(0,255,0),-1)
            cv2.imshow("AI CAR Streaming", img)

            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    except:
        print("에러")
        pass

cv2.destroyAllWindows()

# main5-1-1.py
# 자동차의 영상을 OpenCV를 이용하여 출력하기