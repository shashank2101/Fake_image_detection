import numpy as np
from classifiers import MesoInception4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pipeline import compute_accuracy
# 1 - Load the model and its pretrained weights
classifier = MesoInception4(learning_rate=0.001)
classifier.load('trained_model.h5')

# 2 - Minimal image generator
batch_size = 32
target_size = (256, 256)

dataGenerator = ImageDataGenerator(rescale=1./255)

combined_generator = dataGenerator.flow_from_directory(
    'deepfake_database/train_test',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = dataGenerator.flow_from_directory(
    'deepfake_database/validation',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Step 4: Model Training
epochs =1 # Adjust as needed

# Assuming your Classifier class has been updated to accept validation_data
classifier.fit(
    combined_generator,
    epochs=epochs,
    validation_data=validation_generator
)
classifier.model.save('trained_model.h5')
# # Evaluate on Validation Set
validation_loss, validation_accuracy = classifier.evaluate(validation_generator)
print(f'Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}')

# Assuming 'image_path' is the path to the image you want to predict

#
# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])