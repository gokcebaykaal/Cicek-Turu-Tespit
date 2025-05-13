import os
import cv2
import numpy as np

# Encoding and Split data into Train/Test Sets
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Tensorflow Keras CNN Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Plot Images
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- DATA PREPARATION ---
folder_dir = "C:\\Users\\EXCALIBUR\\OneDrive\\Masaüstü\\Çiçek Tanıma Kod\\archive\\flowers"
data = []
label = []
SIZE = 128

for folder in os.listdir(folder_dir):
    for file in os.listdir(os.path.join(folder_dir, folder)):
        if file.endswith("jpg"):
            label.append(folder)
            img = cv2.imread(os.path.join(folder_dir, folder, file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = cv2.resize(img_rgb, (SIZE, SIZE))
            data.append(im)

data_arr = np.array(data)
label_arr = np.array(label)

categories = np.sort(os.listdir(folder_dir))  # klasör isimleri
num_classes = len(categories)
print("Eğitilen sınıflar:", categories)
print("Toplam veri sayısı:", len(data_arr))

encoder = LabelEncoder()
y = encoder.fit_transform(label_arr)
y = to_categorical(y, num_classes)
X = data_arr / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

# --- MODEL SECTION ---
model_path = "flower_model.h5"

if os.path.exists("flower_model.h5"):
    os.remove("flower_model.h5")
    print("Önceki model silindi. Yeni model eğitilecek.")

else:
    print("Kayıtlı model bulunamadı. Yeni model oluşturuluyor ve eğitiliyor...")
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu', input_shape=(SIZE, SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, activation='softmax'))

    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.20,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    )
    datagen.fit(X_train)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    batch_size = 32
    epochs = 64
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1)

    model.save(model_path)
    print("Model eğitildi ve kaydedildi!")

# --- PREDICTION VISUALIZATION ---

categories = np.sort(os.listdir(folder_dir))
fig, ax = plt.subplots(6, 6, figsize=(20, 20))
plt.subplots_adjust(wspace=0.6, hspace=0.8)

for i in range(6):
    for j in range(6):
        k = np.random.randint(0, len(X_test))
        img = np.array(X_test)[k].reshape(SIZE, SIZE, 3)

        prediction = model.predict(np.expand_dims(X_test[k], axis=0), verbose=0)
        pred_idx = np.argmax(prediction)
        true_idx = np.argmax(y_test[k])
        confidence = np.max(prediction) * 100

        true_label = categories[true_idx]
        pred_label = categories[pred_idx]

        ax[i, j].imshow(img)
        ax[i, j].axis('off')

        color = 'green' if pred_idx == true_idx else 'red'
        rect = patches.Rectangle((0, 0), SIZE, SIZE, linewidth=5, edgecolor=color, facecolor='none')
        ax[i, j].add_patch(rect)

        ax[i, j].set_title(f"True: {true_label}\nPred: {pred_label}\n{confidence:.1f}%", 
                           color=color, fontsize=10, pad=10)

plt.tight_layout()
plt.show()

# --- Interactive Prediction ---

def predict_flower_interactive():
    while True:
        image_path = input("Tahmin edilmesini istediğin resmin yolunu yaz (çıkmak için 'q' yaz): ")
        if image_path.lower() == 'q':
            break
        if not os.path.exists(image_path):
            print("Dosya bulunamadı, tekrar deneyin.")
            continue
        
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (SIZE, SIZE))
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        prediction = model.predict(img_input)
        predicted_class = categories[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        plt.imshow(img_rgb)

        if predicted_class.lower() == "unknown":
            plt.title(f"Tanımlanamayan Tür\nGüven: {confidence:.2f}%", color='red')
            print(f"⚠️ Bu çiçek türü modelde tanımlı değil. (Tahmin: UNKNOWN, Güven: {confidence:.2f}%)")
        else:
            plt.title(f"Tahmin: {predicted_class}\nGüven: {confidence:.2f}%", color='green')
            print(f"Modelin Tahmini: {predicted_class} ({confidence:.2f}%)")

        plt.axis('off')
        plt.show()

# --- Programı başlat ---
predict_flower_interactive()





