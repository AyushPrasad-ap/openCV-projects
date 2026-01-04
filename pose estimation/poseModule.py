import cv2
import mediapipe as mp




class PoseDetector:

    def __init__(self, staticMode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.staticMode = staticMode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

        self.pose = self.mpPose.Pose(
            static_image_mode=self.staticMode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if(self.results.pose_landmarks):
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return frame

    def findPosition(self, frame, draw=True):
        lmlist = []
        
        if(self.results.pose_landmarks):
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmlist.append([id, cx, cy])

                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)

        return lmlist



def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, frame = cap.read()

        frame = detector.findPose(frame)
        lmlist = detector.findPosition(frame)


        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == "__main__":
    main()