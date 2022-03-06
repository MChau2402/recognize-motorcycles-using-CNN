from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

EPOCH = 50
BATCH_SIZE = 32
raw_folder = "data/"


def save_data(raw_folder=raw_folder):
    dest_size = (224, 224)
    print("Bắt đầu xử lý ảnh...")
    pixels = []
    labels = []

    # Lặp qua các folder con trong thư mục data
    for folder in listdir(raw_folder):
        print("Folder=", folder)
        # Lặp qua các file trong từng thư mục
        for file in listdir(raw_folder + folder):
            print("File=", file)
            pixels.append(cv2.resize(cv2.imread(raw_folder + folder + "/" + file), dsize=dest_size))
            labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)

    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    # Tạo file ".data" để lưu trữ
    file = open('pix.data', 'wb')
    # Pickle dữ liệu
    pickle.dump((pixels, labels), file)
    # Đóng file
    file.close()

    return


def load_data():
    # Mở file ".data" chế độ đọc
    file = open('pix.data', 'rb')
    (pixels, labels) = pickle.load(file)

    # Đóng file
    file.close()

    print(pixels.shape)
    print(labels.shape)

    return pixels, labels


save_data()
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

print(X_train.shape)
print(y_train.shape)


def get_model():
    model_mbnetv2_conv = MobileNetV2(weights='imagenet', include_top=False)
    for layer in model_mbnetv2_conv.layers:
        layer.trainable = False

    # Tạo model
    input = Input(shape=(224, 224, 3), name='image_input')
    output_mbnetv2_conv = model_mbnetv2_conv(input)

    # Thêm các lớp
    x = Flatten(name='flatten')(output_mbnetv2_conv)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model


mbnetv2model = get_model()

filepath = "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Tăng cường ảnh dữ liệu cho tập train
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1, rescale=1./255, width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True, brightness_range=[0.2, 1.5], fill_mode="nearest")

aug_val = ImageDataGenerator(rescale=1./255)

mbnetv2_history = mbnetv2model.fit_generator(aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                 epochs=EPOCH,
                                 validation_data=aug.flow(X_test, y_test,
                                                          batch_size=BATCH_SIZE),
                                 callbacks=callbacks_list)

mbnetv2model.save("mbnetv2model.h5")

def plot_model():
    N = EPOCH
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), mbnetv2_history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), mbnetv2_history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), mbnetv2_history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), mbnetv2_history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

plot_model()

