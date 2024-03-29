from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Define data paths (assuming the model and data are in the same directory)
model_path = "skin_cancer.h5"  # Replace with your model filename
data_dir = "path/to/your/skincancer"  # Replace with your data directory path
test_dir = os.path.join(data_dir, "test")

# Define test parameters
target_size = (64, 64)
batch_size = 32  # Adjust based on your hardware capabilities

# Test data generator (no augmentation)
test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False  # Maintain order for evaluation
)

# Load the pre-trained model
model = load_model(model_path)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("Accuracy:", test_acc)
print(classification_report(y_true, y_pred_classes, target_names=["Benign", "Malignant"]))
