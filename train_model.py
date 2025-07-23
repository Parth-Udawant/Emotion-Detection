import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset/fer2013/fer2013.csv")

pixels = data['pixels'].tolist()
faces = np.array([np.fromstring(p, sep=' ').reshape(48, 48) for p in pixels])
faces = faces / 255.0
faces = np.expand_dims(faces, -1)

emotions = to_categorical(data['emotion'], num_classes=7)

x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test))

model.save("model/emotion_model.h5")