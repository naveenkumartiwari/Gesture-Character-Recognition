from keras.models import load_model
from collections import deque
import numpy as np
import cv2
import pyttsx3

model = load_model('htr_model.h5')


letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

kernel = np.ones((5, 5), np.uint8)
background = np.zeros((480,640,3), dtype=np.uint8)
alphabet = np.zeros((200, 200, 3), dtype=np.uint8)
point_deq = deque(maxlen=512)
prediction = 26
string=''

b1 = np.array([100, 60, 60])
b2 = np.array([140, 255, 255])

i = 0
camera = cv2.VideoCapture(0)

alpha = ""


while True:
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, b1, b2)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    (_, contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
          cv2.CHAIN_APPROX_SIMPLE)
    g = None

    if len(contours) > 0: 
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        ((x, y), r) = cv2.minEnclosingCircle(contours)
        cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 255), 2)
        M = cv2.moments(contours)
        g = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        point_deq.appendleft(g)

    elif len(contours) == 0:
        if len(point_deq) != 0:
            background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(background_gray, 15)
            blur = cv2.GaussianBlur(blur, (5, 5), 0)
            threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            background_contours = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
            if len(background_contours) >= 1:
                contours = sorted(background_contours, key = cv2.contourArea, reverse = True)[0]

                if cv2.contourArea(contours) > 1000:
                    x, y, w, h = cv2.boundingRect(contours)
                    alphabet = background_gray[y-10:y + h + 10, x-10:x + w + 10]
                    img = cv2.resize(alphabet, (28, 28))
                    img = np.array(img)
                    img = img.astype('float32')/255
                    prediction = model.predict(img.reshape(1,28,28,1))[0]
                    prediction = np.argmax(prediction)
                    string =string + str(letters[int(prediction)+1])
                    result= str(letters[int(prediction)+1])
                    engine=pyttsx3.init()
                    engine.say(result)
                    engine.runAndWait()


            point_deq = deque(maxlen=512)
            background = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(1, len(point_deq)):
            if point_deq[i - 1] is None or point_deq[i] is None:
                    continue
            cv2.line(frame, point_deq[i - 1], point_deq[i], (0, 0, 0), 2)
            cv2.line(background, point_deq[i - 1], point_deq[i], (255, 255, 255), 8)


    cv2.putText(frame, "word:  " + string, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "prediction:  " + str(letters[int(prediction)+1]), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    


    cv2.imshow("alphabets Recognition Real Time", frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
print(string) 
engine=pyttsx3.init()
engine.say(string)
engine.runAndWait()
cv2.destroyAllWindows()
