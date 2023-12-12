import cv2
from keras.models import model_from_json
import numpy as np
import datetime
from deepface import DeepFace
import time 

json_file = open("C:/Users/90537/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("C:/Users/90537/facialemotionmodel.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

video = cv2.VideoCapture(0)
labels = {0: 'sinirli', 1: 'igrenme', 2: 'korku', 3: 'mutlu', 4: 'normal', 5: 'uzgun', 6: 'şaşırmış'}

# Initialize pTime outside the loop
pTime = time.time()

while True:
    ret, frame = video.read()
    if not ret:
        print("Açılamadı")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cv2.putText(frame, filename, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    try:
        for (x, y, w, h) in faces:
            yuz_image = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            yuz_image = cv2.resize(yuz_image, (48, 48))
            processed_image = preprocess_image(yuz_image)
            prediction = model.predict(processed_image)
            predicted_label = labels[np.argmax(prediction)]
            result = DeepFace.analyze(frame, actions=["age"], enforce_detection=False)
            yas = result[0]['age']
            cv2.putText(frame, f"Yas:{yas} {predicted_label}  {float(prediction[0][np.argmax(prediction)]) * 100:.2f}%", (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

            with open(fr"C:\Users\90537\fotograf\{filename}.jpg", "wb") as open_file:
                open_file.write(cv2.imencode('.jpg', frame)[1])

        # Calculate fps after displaying the frame
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("İfade ve Yaş Okuma", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except cv2.error:
        pass

video.release()
cv2.destroyAllWindows()
