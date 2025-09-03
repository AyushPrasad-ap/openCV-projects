import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# prepare data
inputDir = "image classifier/data"
categories = ['empty', 'not_empty']

data =[]
labels = []

for categoryIdx, category in enumerate(categories):
    for file in os.listdir(os.path.join(inputDir, category)):
        imgPath = os.path.join(inputDir, category, file)
        img = imread(imgPath)
        img = resize(img, (15,15))
        data.append(img.flatten())
        labels.append(categoryIdx)

data = np.asarray(data)
labels = np.asarray(labels)


# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)


# train classifier
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

gridSearch = GridSearchCV(classifier, parameters)

gridSearch.fit(x_train, y_train)


# test classifier performance

bestEstimator = gridSearch.best_estimator_

y_prediction = bestEstimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print("Accuracy: ", score*100, "%")

# save classifier
with open("./image_classifier.pkl", "wb") as f:
    pickle.dump(bestEstimator, f)