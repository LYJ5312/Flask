import cv2
import numpy as np
from urllib.request import urlopen

ip = '192.168.137.123'
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
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                               cv2.IMREAD_UNCHANGED)
            
            height, width, _ = img.shape
            img = img[height // 2:, :]
            
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            cv2.imshow("AI CAR Streaming", img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            input = {
                ord('w'):'전진',
                ord('a'):'왼쪽',
                ord('d'):'오른쪽',
                ord('s'):'후진',
                ord('A'):'왼쪽 회전',
                ord('D'):'오른쪽 회전',
                32:'멈춤',
                ord('1'):'40%속도',
                ord('2'):'50%속도',
                ord('3'):'60%속도',
                ord('4'):'80%속도',
                ord('5'):'100%속도',
            }
            if key in input : print(input[key])
    except:
        print("에러")
        pass

cv2.destroyAllWindows()
