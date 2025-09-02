import cv2
from PIL import Image
from utils import getLimits

blue = [255, 0, 0]
lowerLimit, upperLimit = getLimits(color=blue)
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lowerLimit, upperLimit)

    maskPillow = Image.fromarray(mask)

    boundingBox = maskPillow.getbbox()

    if boundingBox:
        x1,y1, x2, y2 = boundingBox
        frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)


    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
