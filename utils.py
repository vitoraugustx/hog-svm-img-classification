from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image_data():
    # Create a data generator
    datagen_train = ImageDataGenerator()  
    datagen_valid = ImageDataGenerator() # No need to augment validation data

    # Utilizando o método flow_from_directory() da biblioteca ImageDataGenerator
    # Load and iterate training dataset
    train_data = datagen_train.flow_from_directory(
        #"/content/train_data/train/",
        "./src/images/PandasBears/Train/",
        target_size=(256, 256),
        color_mode="grayscale",
        class_mode="categorical",
    )

    # Load and iterate validation dataset
    valid_data = datagen_valid.flow_from_directory(
        "./src/images/PandasBears/Test/",
        target_size=(256, 256),
        color_mode="grayscale",
        class_mode="categorical",
    )

    return train_data, valid_data
