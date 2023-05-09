from PIL import Image
import cv2

def show_cv_image(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)