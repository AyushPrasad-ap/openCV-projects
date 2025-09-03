import cv2
import matplotlib.pyplot as plt
import easyocr



# read image
img_path = "text detection/img.png"
img = cv2.imread(img_path)


# instance of text detector
reader = easyocr.Reader(['en'], gpu=True)



# detect text on the image
texts = reader.readtext(img)



# draw bounding box and text
threshold = 0.65
for t in texts:
    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0,255,0), 3)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)


# show image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
