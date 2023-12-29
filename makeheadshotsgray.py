import cv2
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data_train_folder = 'data/datatrain'
output_folder = 'data/facegray'

# dimension of images
image_width = 100
image_height = 100

# for detecting faces
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# set the directory containing the images
images_dir = os.path.join(".", data_train_folder)
print(images_dir)
# iterates through all the files in each subdirectories
for root, _, files in os.walk(images_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)

            # Lấy tên thư mục mẹ
            parent_folder = os.path.basename(root)

            # Đường dẫn đến thư mục mới trong Face_gray
            new_folder_path = os.path.join(output_folder, parent_folder)

            # get the label name (name of the person)
            label = parent_folder.replace(" ", ".").lower()

            # add the label (key) and its number (value)


            # load the image
            imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(imgtest, cv2.COLOR_BGR2GRAY)
            image_array = np.array(gray, "uint8")

            # get the faces detected in the image
            faces = facecascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

            # if not exactly 1 face is detected, skip this photo
            if len(faces) != 1:
                print(f'---Photo skipped---\n')
                continue

            # save the detected face(s) and associate them with the label
            for (x_, y_, w, h) in faces:

                # draw the face detected
                face_detect = cv2.rectangle(imgtest, (x_, y_), (x_ + w, y_ + h), (255, 0, 255), 2)
                #plt.imshow(face_detect)
                #plt.show()

                # resize the detected face to 224x224
                size = (image_width, image_height)

                # detected face region
                roi = image_array[y_: y_ + h, x_: x_ + w]

                # resize the detected head to target size
                resized_image = cv2.resize(roi, size)
                image_array = np.array(resized_image, "uint8")

                # Kiểm tra xem thư mục mới đã tồn tại hay chưa
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)

                # replace the image with only the face
                im = Image.fromarray(image_array)
                im.save(os.path.join(new_folder_path, file))
