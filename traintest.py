import os
import cv2
import numpy as np

# Encoding and Split data into Train/Test Sets
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Tensorflow Keras CNN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Plot Images
import matplotlib.pyplot as plt

folder_dir = "C:\\Users\\EXCALIBUR\\OneDrive\\Masaüstü\\Çiçek Tanıma Kod\\archive\\flowers"

data = []
label = []

SIZE = 128  # Crop the image to 128x128

for folder in os.listdir(folder_dir):
    for file in os.listdir(os.path.join(folder_dir, folder)):
        if file.endswith("jpg"):
            label.append(folder)
            img = cv2.imread(os.path.join(folder_dir, folder, file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = cv2.resize(img_rgb, (SIZE, SIZE))
            data.append(im)
        else:
            continue

data_arr = np.array(data)
label_arr = np.array(label)

encoder = LabelEncoder()
y = encoder.fit_transform(label_arr)
y = to_categorical(y, 5)
X = data_arr / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(SIZE, SIZE, 3), kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu', kernel_regularizer=l2(0.01)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))

model.add(Dense(5, activation='softmax'))

datagen = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    brightness_range=[0.2, 1.0],
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(X_train)

model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 16
epochs = 100

history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    verbose=1)

categories = np.sort(os.listdir(folder_dir))
fig, ax = plt.subplots(6, 6, figsize=(25, 40))

for i in range(6):
    for j in range(6):
        k = int(np.random.random_sample() * len(X_test))
        prediction = model.predict(X_test, verbose=0)[k]
        true_label = categories[np.argmax(y_test[k])]
        predicted_label = categories[np.argmax(prediction)]
        color = 'green' if true_label == predicted_label else 'red'
        
        ax[i, j].set_title(f"TRUE: {true_label}", color=color)
        ax[i, j].set_xlabel(f"PREDICTED: {predicted_label}", color=color)
        ax[i, j].imshow(X_test[k].reshape(SIZE, SIZE, 3), cmap='gray')
# Eğitim ve doğrulama doğruluğunu görselleştir
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Eğitim ve doğrulama kaybını görselleştir
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()