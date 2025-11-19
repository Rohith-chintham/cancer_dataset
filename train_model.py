import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def train_model(train_dir, val_dir, test_dir, epochs=15, img_size=(150, 150)):
    # Check if directories exist
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"❌ Directory not found: {d}. Please check your dataset structure.")

    # Data generators (with augmentation on training)
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='binary'
    )

    # CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

    # Save model
    model.save("cancer_model.h5")
    print("💾 Model saved as cancer_model.h5")

    return model, history, test_acc


if __name__ == "__main__":
    # Relative paths (since you run inside C:\Users\rohit\cancer_dataset)
    train_dir = "train"
    val_dir = "validation"
    test_dir = "test"

    train_model(train_dir, val_dir, test_dir, epochs=15)
