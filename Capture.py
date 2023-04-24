import cv2
import sys
import numpy as np
import os
import time

count = 0
size = 4
fn_haar = 'data/haarcascade_frontalface_default.xml'
fn_dir = 'database'
fn_name = input("Enter the Person's Name: ")
path = os.path.join(fn_dir, fn_name)
if not os.path.isdir(path):
    os.mkdir(path)

(im_width, im_height) = (112, 92)
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)

print("-----------------------Taking pictures----------------------")
print("--------------------Give some expressions---------------------")

while count < 20:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (gray.shape[1] // size, gray.shape[0] // size))
    faces = haar_cascade.detectMultiScale(mini)
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Tính số thứ tự ảnh mới
        existing_pics = [int(n[:n.find('.')]) for n in os.listdir(path) if n[0] != '.']
        pin = 0
        if existing_pics:
            pin = max(existing_pics) + 1
            # Lưu ảnh
            cv2.imwrite(f"{path}/{pin}.png", face_resize)

            # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị tên
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0))

            # Dừng chương trình một chút để người dùng có thời gian thay đổi diện mạo
            time.sleep(1)

            count += 1
        else:
            pin = 1
            cv2.imwrite(f"{path}/{pin}.png", face_resize)

            # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị tên
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0))

            # Dừng chương trình một chút để người dùng có thời gian thay đổi diện mạo
            time.sleep(1)
        # Hiển thị hình ảnh lên màn hình và chờ phím ấn ESC
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
webcam.release()
cv2.destroyAllWindows()
print(f"{count} images taken and saved to {fn_name} folder in database")
