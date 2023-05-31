import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import cv2
import os


def detect_brain_tumour(path):

    # Prepare/collect data
    path = os.listdir('brain_tumor/Training')
    classes = {'no_tumor': 0, 'pituitary_tumor': 1}

    X = []
    Y = []

    for cls in classes:
        pth = 'brain_tumor/Training/' + cls
        for j in os.listdir(pth):
            img = cv2.imread(pth+'/'+j, 0)
            if img is not None:
                # Resize the image
                img = cv2.resize(img, (200, 200))
                X.append(img)
                Y.append(classes[cls])
                print("Successfully loaded:", j)
            else:
                print("Failed to load the image:", j)

    X = np.array(X)
    Y = np.array(Y)

    # Visualize data
    plt.imshow(X[0], cmap='gray')
    plt.show()

    # Prepare data
    X_updated = X.reshape(len(X), -1)

    # Split Data
    xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=.20)

    # Feature Scaling
    xtrain = xtrain / 255
    xtest = xtest / 255

    # Feature Selection: PCA
    pca = PCA(.98)
    pca_train = pca.fit_transform(xtrain)
    pca_test = pca.transform(xtest)

    # Train Model
    lg = LogisticRegression(C=0.1)
    lg.fit(pca_train, ytrain)

    sv = SVC()
    sv.fit(pca_train, ytrain)

    # Evaluation
    print("Logistic Regression:")
    print("Training Score:", lg.score(pca_train, ytrain))
    print("Testing Score:", lg.score(pca_test, ytest))
    print("\nSVM:")
    print("Training Score:", sv.score(pca_train, ytrain))
    print("Testing Score:", sv.score(pca_test, ytest))





    # Testing with a given image
    image_path = '/home/mohammadnoman/Freelancing_workspace/python_works/healthMonitorWebApplicationUsingMachinelearning/brain_tumor/Testing/pituitary_tumor/image(6).jpg'  # Replace with the actual image path
    img = cv2.imread(image_path, 0)
    resized_img = cv2.resize(img, (200, 200))
    input_img = resized_img.reshape(1, -1) / 255

    # Make prediction
    prediction = lg.predict(pca.transform(input_img))

    # Map the predicted label to the corresponding class
    class_mapping = {0: 'no_tumor', 1: 'pituitary_tumor'}
    predicted_class = class_mapping[prediction[0]]

    # Display the result
    plt.imshow(img, cmap='gray')
    plt.title(predicted_class)
    print(predicted_class)
    plt.axis('off')
    plt.show()


detect_brain_tumour("i")