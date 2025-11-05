import cv2
import numpy as np
import hand_tracker as ht
import autopy

# 1920x1200
############
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 3
############


pLocX, pLocY = 0,0
cLocX, cLocY = 0,0


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = ht.HandDetector(maxHands=1)
wScr, hScr = autopy.screen.size()


while True:
    # 1. Find hand Landmarks
    success, frame = cap.read()
    if not success:
        break

    frame = detector.find_hands(frame) 
    lmList, bbox = detector.find_position(frame)


    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:] # Index finger tip
        x2, y2 = lmList[12][1:] # Middle finger tip

        # 3. Check which fingers are up
        fingers = detector.fingers_up()

        cv2.rectangle(frame, (frameR, frameR), (wCam-frameR, hCam-frameR), (255,0,255), 2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert coordiates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))


            # 6. Smoothen Values
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening


            # 7. Move Mouse
            autopy.mouse.move(wScr - cLocX, cLocY)
            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            pLocX, pLocY = cLocX, cLocY

        # 8. Both Index and Middle fingers are up : clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, frame, lineInfo = detector.find_distance(8, 12, frame)

            # 10. Click mouse if distance is short
            if length < 35:
                cv2.circle(frame, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()


    # 11. Display
    cv2.imshow("AI Virtual Mouse", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()      
cv2.destroyAllWindows()