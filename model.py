from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def create_model(input_shape):
  """
  Defines the CNN architecture for skin cancer classification.

  Args:
      input_shape: A tuple representing the input image shape (e.g., (64, 64, 3)).

  Returns:
      A compiled Keras sequential model.
  """
  model = Sequential()
  model.add(BatchNormalization(input_shape=input_shape))
  model.add(Conv2D(32, 3, activation='relu'))
  model.add(MaxPooling2D())
  model.add(Conv2D(64, 3, activation='relu'))
  model.add(MaxPooling2D())
  model.add(Dropout(0.3))
  model.add(Conv2D(128, 3, activation='relu'))
  model.add(MaxPooling2D())
  model.add(Dropout(0.2))
  model.add(Conv2D(256, 3, activation='relu'))
  model.add(MaxPooling2D())
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.15))
  model.add(Dense(2, activation='softmax'))

  model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
  return model
