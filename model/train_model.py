import os

train_script_content = """
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def train_model(processed_data_dir="brain_tumor_detection/data/processed",
                saved_model_dir="brain_tumor_detection/model/saved_model",
                epochs=15,
                batch_size=32):

    # Ensure directories exist
    os.makedirs(saved_model_dir, exist_ok=True)

    # Load processed features and labels
    features_path = os.path.join(processed_data_dir, 'features.npy')
    labels_path = os.path.join(processed_data_dir, 'labels.npy')

    try:
        X = np.load(features_path)
        y = np.load(labels_path)
    except FileNotFoundError:
        print(f"Error: Processed data files not found in {processed_data_dir}. Please run preprocessing step first.")
        return

    input_shape = X.shape[1:]
    num_classes = len(np.unique(y))

    print(f"Loaded features shape: {X.shape}")
    print(f"Loaded labels shape: {y.shape}")
    print(f"Model input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_val: {X_val.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_val: {y_val.shape}")

    # Define the CNN model
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("\nCNN Model Architecture:")
    model.summary()
    print("CNN model defined and compiled successfully.")

    # Train the model
    print(f"\nStarting model training for {epochs} epochs with a batch size of {batch_size}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )
    print("Model training completed. Training history stored.")

    # Save the trained model
    model_save_path = os.path.join(saved_model_dir, "model.h5")
    model.save(model_save_path)
    print(f"\nModel saved successfully to: {model_save_path}")

    return history, model

if __name__ == '__main__':
    print("Running train_model.py directly...")
    # You can customize parameters here if running as a script
    history, model = train_model()
"""

# Define the directory and file path for the script
script_dir = "brain_tumor_detection/model"
script_path = os.path.join(script_dir, "train_model.py")

# Create the directory if it doesn't exist
os.makedirs(script_dir, exist_ok=True)

# Write the content to the file
with open(script_path, "w") as f:
    f.write(train_script_content)

print(f"`train_model.py` script created successfully at: {script_path}")
