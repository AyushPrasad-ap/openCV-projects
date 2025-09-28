import cv2
import numpy as np
import hand_tracker as ht
import math

# pycaw imports
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


cap = cv2.VideoCapture(0)
detector = ht.HandDetector(detectionCon = 0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

minVol, maxVol, _ = volume.GetVolumeRange()


while True:
    success, frame = cap.read()

    frame = detector.find_hands(frame)
    lm_list = detector.find_position(frame, draw=False)
    # thumb tip = 4, index tip = 8
    if lm_list:
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2-x1, y2-y1)

        if length < 30:
            cv2.circle(frame, (cx,cy), 10, (0,255,0), cv2.FILLED)
        
        # at approx length=30 set to minVol, at length=200 set to maxVol
        # hand range:    30 to 200
        # volume range: -60 to 0

        vol = np.interp(length, [30, 200], [minVol, maxVol])
        # print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)




    # cv2.imshow("Webcam", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()