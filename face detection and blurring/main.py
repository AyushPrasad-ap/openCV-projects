import cv2
import argparse
import mediapipe as mp
import os


# this is the  core logic of the code where the faces are detected and blurred
def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_detection.process(img_rgb)

    H, W, _ = img.shape

    if result.detections is not None:
        for detection in result.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # blur faces
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (25,25))

    return img


def main():
    parser = argparse.ArgumentParser(description="Face detection + blurring")
    parser.add_argument("--mode", choices=["image", "video", "webcam"], required=True,
                        help="Operation mode: image, video or webcam")
    parser.add_argument("--filePath", default=None,
                        help="Path to image or video file (not needed for webcam)")
    parser.add_argument("--output_dir", default="output",
                        help="Directory to save outputs")
    args = parser.parse_args()

    # Validate input file when needed
    if args.mode in ("image", "video"):
        if not args.filePath:
            parser.error("--filePath must be provided for image/video modes")
        if not os.path.exists(args.filePath):
            parser.error(f"Provided --filePath does not exist: {args.filePath}")

    # create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

        
    # detect faces
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        
        if args.mode == "image":
            # read image
            img = cv2.imread(args.filePath)
            img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

            img = process_img(img, face_detection)

            # save image
            cv2.imwrite(os.path.join(args.output_dir, 'output.jpg'), img)


        elif args.mode == "video":
            cap = cv2.VideoCapture(args.filePath)
            ret, frame = cap.read()

            output_video = cv2.VideoWriter(os.path.join(args.output_dir, 'output.mp4'),
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        25,
                                        (frame.shape[1], frame.shape[0]))


            while ret:
                frame = process_img(frame, face_detection)
                output_video.write(frame)
                ret, frame = cap.read()

            cap.release()
            output_video.release()


        elif args.mode == "webcam":
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()

            while ret:
                frame = process_img(frame, face_detection)
                cv2.imshow("Webcam", frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                ret, frame = cap.read()

            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()










