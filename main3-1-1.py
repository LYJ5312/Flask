import cv2
import numpy as np
from urllib.request import urlopen

ip='192.168.137.153'
stream = urlopen('http://'+ip+':81/stream')
buffer = b''

while True:
    buffer += stream.read(4096)
    print(buffer)