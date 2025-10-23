"""
Simple Hand Tracking module using MediaPipe.
Provides a HandDetector class with methods:
 - find_hands(frame, draw=True) -> (frame_with_drawings)
 - find_position(frame, hand_no=0, draw=False) -> lm_list (id, x, y)
 - close() -> releases internal MediaPipe objects
"""

import cv2
import mediapipe as mp

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
        # self.results = None



    def find_hands(self, frame, draw=True):

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks((frame), hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        
        return frame



    def find_position(self, frame, hand_no=0, draw=True):
        """
        Return list of (id, x_px, y_px) for a specific hand index (hand_no).
        Must call find_hands before (or find_position will internally use the last processed results).
        """
        lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

                if draw:
                    cv2.circle(frame, (cx,cy), 10, (255,0,255), cv2.FILLED)

        return lm_list



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