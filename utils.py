from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt

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

def plot_hog(img, hog_image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(img.astype('uint8'), cmap=plt.cm.gray)
    ax1.set_title('Imagem de entrada')

    ax2.axis('off')
    ax2.imshow(hog_image[1], cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients (HOG)')
    plt.show()


# Função para extrair as características HOG das imagens
def extract_hog_features(X, visualize=False, n_visualizations=0):
    hog_features = []
    for img in X:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Visualiza as características HOG de algumas imagens
        if visualize and n_visualizations > 0:
            hog_image_plt = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            plot_hog(img, hog_image_plt)
            n_visualizations -= 1
        
        hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

        hog_features.append(hog_image)
    
    return np.array(hog_features)