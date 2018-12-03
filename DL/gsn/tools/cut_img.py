from PIL import Image
import face_recognition
import os

# for (int i = 1; i <= 10; i++)
list = os.listdir('./../datasets/img_align_celeba/')

WIDTH = 128
HIGHT = 128

for i in range(0, len(list)):
    imgName = os.path.join('./../datasets/img_align_celeba/', os.path.basename(list[i]))
    fileName = os.path.basename(list[i])
    # print imgName
    if (os.path.splitext(imgName)[1] != '.jpg'): continue

    image = face_recognition.load_image_file(imgName)

    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        x = (top + bottom) / 2
        y = (right + left) / 2

        top = x - HIGHT / 2
        bottom = x + HIGHT / 2
        left = y - WIDTH / 2
        right = y + WIDTH / 2

        if (top < 0) or (bottom > image.shape[0]) or (left < 0) or (right > image.shape[1]):
            top, right, bottom, left = face_location
            width = right - left
            height = bottom - top
            if (width > height):
                right -= (width - height)
            elif (height > width):
                bottom -= (height - width)


        face_image = image[top:bottom, left:right]

        pil_image = Image.fromarray(face_image)
        pil_image = pil_image.resize((128,128))
        pil_image.save('./../datasets/face/%s'%fileName)
    if i%100 == 0:
        print('the number of images is {}'.format(i))
