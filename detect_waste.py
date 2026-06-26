import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("waste_model.h5")

# Waste labels
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Open laptop webcam
cap = cv2.VideoCapture(0)

while True:

    # Read webcam frame
    ret, frame = cap.read()

    # Resize image
    img = cv2.resize(frame, (224, 224))

    # Normalize image
    img = img / 255.0

    # Convert into array
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    # Get highest prediction index
    index = np.argmax(prediction)

    # Get label
    label = labels[index]

    # Show detected label on webcam screen
    cv2.putText(
        frame,
        f"Detected: {label}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display webcam
    cv2.imshow("Waste Detection", frame)

    # Press ESC key to exit
    if cv2.waitKey(1) == 27:
        break

# Release webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()