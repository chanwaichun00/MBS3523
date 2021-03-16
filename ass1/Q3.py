import cv2
import random
print(cv2.__version__)

car = cv2.CascadeClassifier('Resources/cars.xml')
hum = cv2.CascadeClassifier('Resources/hum.xml')

capture = cv2.VideoCapture('Resources/001..mp4')
capture.set(3, 640)
capture.set(4, 480)

while True:
    success, img = capture.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c1 = random.randint(0, 256)
    c2 = random.randint(0, 256)
    c3 = random.randint(0, 256)
    c4 = random.randint(0, 256)
    c5 = random.randint(0, 256)
    c6 = random.randint(0, 256)
    cars = car.detectMultiScale(imgGray, 1.5, 3)
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (c1, c2, c3), 2)
    human = hum.detectMultiScale(imgGray, 1.1, 2)
    for (x, y, w, h) in human:
        cv2.rectangle(img, (x, y), (x + w, y + h), (c4, c5, c6), 2)

    cv2.imshow('Frame', img)

    if cv2.waitKey(1) == ord(' '):
        break

capture.release()
cv2.destroyAllWindows()