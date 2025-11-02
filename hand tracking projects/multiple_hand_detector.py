import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=1, maxHands=2)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    hands, frame = detector.findHands(frame)  # with draw

    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
