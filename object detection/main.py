from ultralytics import YOLO
import cv2
import time


#  load yolov8 model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")


prev_time = time.time()
fps = 0.0



while True:

    # read frames
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Empty frame, skipping...")
        break


    # detect and track objects
    results = model.track(frame, persist=True)


    # plot results
    frame_ = results[0].plot()



    # Calculate FPS
    cur_time = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, cur_time - prev_time))
    prev_time = cur_time
    cv2.putText(frame_, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)



    # visualize
    cv2.imshow("webcam object detection", frame_)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()