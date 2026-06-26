import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Dataset path
dataset_path = "dataset"


# Create Image Generator block
# Image preprocessing 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
# Create Training Dataset block
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Create Validation Dataset block

val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

## Improved CNN Model   # sequential model 
model = Sequential([
    # First Convolution Layer
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),        # Max Pooling Layer
    # Second Convolution Layer
    Conv2D(64, (3,3), activation='relu'),
     #  second Max Pooling Layer
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    #  third Max Pooling Layer
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256, activation='relu'),

    Dense(6, activation='softmax')  #Output Layer
])
# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# Save model
model.save("waste_model.h5")

print("Model training completed and saved!")