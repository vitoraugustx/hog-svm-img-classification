import numpy as np
from sklearn import svm
from utils import generate_data, extract_hog_features

# Define o tamanho da imagem
target_size = (256, 256)

# Gera os dados das imagens a partir do diretório "Train"
train_directory = './src/images/PandasBears/Train'
X_train, y_train = generate_data(train_directory, target_size)

# Gera os dados das imagens a partir do diretório "Test"
test_directory = './src/images/PandasBears/Test'
X_test, y_test = generate_data(test_directory, target_size)

# Converte y_train e y_test para um array 1D
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

# Extrai as características HOG dos conjuntos de treinamento e teste
# Selecione visualize=True para visualizar as características HOG de algumas (n_visualizations) imagens
X_train_hog = extract_hog_features(X_train, visualize=False, n_visualizations=4)
X_test_hog = extract_hog_features(X_test)

# Definição e treinamento do classificador SVM radial
classifier = svm.SVC(kernel='rbf', gamma=0.01)
classifier.fit(X_train_hog, y_train)

# Classificação do conjunto de teste
y_pred = classifier.predict(X_test_hog)

# Calcula a acurácia
accuracy = np.mean(y_pred == y_test)
print("Acurácia no conjunto de teste:", accuracy * 100, "%")