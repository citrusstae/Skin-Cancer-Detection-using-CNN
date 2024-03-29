from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from src.model import create_model

def train_model(data_dir, target_size, epochs=10):
  """
  Trains the CNN model for skin cancer classification.

  Args:
      data_dir: Path to the directory containing the training and validation datasets (subfolders for benign and malignant).
      target_size: A tuple representing the desired image size for training (e.g., (64, 64)).
      epochs: Number of training epochs (default: 10).
  """
  train_datagen = ImageDataGenerator(rescale=1/255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    vertical_flip = True,
                                    rotation_range=40,
                                    brightness_range = (0.5, 1.5),
                                    horizontal_flip = True)

  train_data = train_datagen.flow_from_directory(data_dir + "/train",
                                                target_size=target_size,
                                                class_mode='sparse',
                                                shuffle=True,
                                                seed=1)

  test_datagen = ImageDataGenerator(rescale=1/255)
  test_data = test_datagen.flow_from_directory(data_dir + "/validation",
                                              target_size=target_size,
                                              class_mode='sparse',
                                              shuffle=True,
                                              seed=1)

  class_names = train_data.class_indices.keys()

  model = create_model(train_data.image_shape)

  early_stopping = EarlyStopping(monitor='val_loss', patience=5)

  history = model.fit(train_data,
                      validation_data=test_data,
                      callbacks=[early_stopping],
                      epochs=epochs)

  # Plot training curves (optional)
  # ... (implementation in utils.py)

  model.save(data_dir + "/skin.h5")

if __name__ == "__main__":
  data_dir = "path/to/your/data"  # Replace with your data directory path
  target_size = (64, 64)
  epochs = 10
  train_model(data_dir, target_size, epochs)
