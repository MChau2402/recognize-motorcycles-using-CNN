import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

camera_id = 0
cap = cv2.VideoCapture(camera_id)

# Định nghĩa class
class_name = ['không phải xe', 'xe số', 'xe ga']

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

# Load weights của model đã train
my_model = get_model()
my_model.load_weights("weights-16-0.97.hdf5")

while True:

    # Capture frame-by-frame
    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None, fx=1, fy=1)

    # Resize
    image = image_org.copy()
    image = cv2.resize(image, dsize=(224, 224))
    image = image.astype('float')*1./255

    # Chuyển thành tensor
    image = np.expand_dims(image, axis=0)

    # Dự đoán
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0], axis=0))
    if (np.max(predict)>=0.8) and (np.argmax(predict[0])!=0):

        # Show kết quả dự đoán
        font = cv2.FONT_HERSHEY_SIMPLEX # font chữ 
        org = (50, 50) # vị trí văn bản
        font_scale = 1.5 # cỡ chữ
        color = (0, 255, 0) # Màu BGR - màu hiển thị tương ứng là xanh lục
        thickness = 2 # Độ dày chữ trong văn bản

        cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                    font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

