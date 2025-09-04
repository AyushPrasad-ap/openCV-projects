import os 
import datetime
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode




with open('./authorized_users.txt', 'r') as f:
    authorized_users = [line.strip() for line in f if len(line.strip()) > 0]
    f.close()

log_path = './log.txt'

most_recent_log = {}
time_between_logs_threshold = 10  #seconds

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    qr_info = decode(frame)

    if len(qr_info) > 0:
        qr = qr_info[0]
        data = qr.data
        rect = qr.rect
        polygon = qr.polygon

        if data.decode() in authorized_users:
            cv2.putText(frame, "ACCESS GRANTED", (rect.left, rect.top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if data.decode() not in most_recent_log.keys() or time.time() - most_recent_log[data.decode()] > time_between_logs_threshold:
                most_recent_log[data.decode()] = time.time()
                with open(log_path, 'a') as f:
                    f.write(f"{data.decode()}, {datetime.datetime.now()}\n")
                    f.close()
        else:
            cv2.putText(frame, "ACCESS DENIED", (rect.left, rect.top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        frame = cv2.rectangle(frame, (rect.left, rect.top), (rect.left + rect.width, rect.top + rect.height), (0,255,0), 3)
        frame = cv2.polylines(frame, [np.array(polygon)], True, (255,0,0), 3)


    cv2.imshow("QR code reader", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()