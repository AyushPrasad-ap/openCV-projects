"""
Simple Hand Tracking module using MediaPipe.
Provides a HandDetector class with methods:
 - find_hands(frame, draw=True) -> (frame_with_drawings)
 - find_position(frame, hand_no=0, draw=False) -> lm_list (id, x, y)
 - close() -> releases internal MediaPipe objects
"""

import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]



    def find_hands(self, frame, draw=True):

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks((frame), hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        
        return frame


    def find_position(self, frame, hand_no=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lm_list.append([id, cx, cy])

                if draw:
                    cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)
            
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(frame, (xmin-20, ymin-20), (xmax+20, ymax+20), (0,255,0), 2)

        return self.lm_list, bbox


    def fingers_up(self):
        fingers =[]
        # Thumb
        if self.lm_list[self.tipIds[0]][1] > self.lm_list[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Fingers
        for id in range(1,5):
            if self.lm_list[self.tipIds[id]][2] < self.lm_list[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers


    def find_distance(self, p1, p2, frame, draw=True, r=15, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]


    def close(self):
        # Release MediaPipe resources.
        self.hands.close()



def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = detector.find_hands(frame)
        lm_list = detector.find_position(frame)
        # if lm_list: 
        #     print(lm_list[4])

        cv2.imshow("Webcam hand tracking", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()