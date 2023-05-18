from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from skimage.feature import hog

# Função para gerar os dados de treinamento e teste a partir das imagens
def generate_data(directory, target_size):
    datagen = ImageDataGenerator()

    generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        color_mode="rgb",
        class_mode="categorical"
    )
    
    X = []
    y = []
    
    for i in range(len(generator)):
        images, labels = generator[i]
        X.append(images)
        y.append(labels)
    
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    return X, y

# Função para extrair as características HOG das imagens
def extract_hog_features(X):
    hog_features = []
    for img in X:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features.append(features)
    
    return np.array(hog_features)