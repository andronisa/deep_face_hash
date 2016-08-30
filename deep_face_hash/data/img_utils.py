import cv2
import numpy as np

from scipy.misc import imread


def viola_jones(img):
    face_cascade = cv2.CascadeClassifier('/home/aandronis/projects/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    # Get the matrix and put zeros around
    for (x,y,w,h) in faces:
        return img[y:y+h,x:x+w]
        # # Debug
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return img


def preprocess_images(image_paths, img_size=(224, 224), crop_size=None, color_mode="rgb", img_options=None):
    if img_options:
        height = img_options['height']
        width = img_options['width']
        rgb = 3 if color_mode == 'rgb' else None
    else:
        height = 224
        width = 224
        rgb = 3

    images = np.zeros((len(image_paths), rgb, height, width), dtype=np.float32)

    counter = 0
    for image_path in image_paths:
        img = cv2.imread(image_path)
    
        img = viola_jones(img)

        if img_size:
            img = cv2.resize(img, img_size)

        img = img.astype('float32')
        # We permute the colors to get them in the BGR order
        if color_mode == "bgr":
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]

        # We normalize the colors with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939

        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
            , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]


        images[counter, ...] = img
        counter += 1


    return images
