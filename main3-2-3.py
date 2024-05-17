import cv2
import numpy as np
from urllib.request import urlopen

ip = '192.168.137.51'
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
            

            input = {
                ord('q'):'stop',
                ord('w'):'forward',
                ord('a'):'left',
                ord('d'):'right',
                ord('s'):'backward',
                ord('A'):'turn_left',
                ord('D'):'turn_right',
                32:'멈춤',
                ord('1'):'speed40',
                ord('2'):'speed50',
                ord('3'):'speed60',
                ord('4'):'speed80',
                ord('5'):'speed100',
            }

            if key in input:
                urlopen('http://'+ip+"/action?go="+input[key])

            if key == ord('q'):break
            if key in input : print(input[key])

    except:
        print("에러")
        pass

cv2.destroyAllWindows()
