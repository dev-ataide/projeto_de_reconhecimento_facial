import cv2
import numpy as np
import os
import pickle
import re
import face_recognition
import json
from datetime import datetime
import requests

# Carrega os códigos de codificação do arquivo pickle
pickle_name = "face_encodings_custom.pickle"
data_encoding = pickle.loads(open(pickle_name, "rb").read())
list_encodings = data_encoding["encodings"]
list_names = data_encoding["names"]

# URL do servidor Flask
flask_url = "http://localhost:5000/recognized_faces"

# Função para enviar os dados das pessoas identificadas para o servidor Flask
def send_recognized_faces(name):
    headers = {'Content-Type': 'application/json'}
    data = {
        "name": name,
        "recognized": True,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    response = requests.post(flask_url, json=data, headers=headers)
    if response.status_code == 200:
        print("Dados enviados com sucesso para o servidor Flask!")
        print(data)
    else:
        print("Falha ao enviar dados para o servidor Flask.")

# Reconhece os rostos em uma imagem dada (neste caso, o quadro do vídeo)
def recognize_faces(image, list_encodings, list_names, resizing=0.25, tolerance=0.6):
    image = cv2.resize(image, (0, 0), fx=resizing, fy=resizing)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

    face_names = []
    conf_values = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(list_encodings, encoding, tolerance=tolerance)
        name = "Não identificado"
        face_distances = face_recognition.face_distance(list_encodings, encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = list_names[best_match_index]
            # Envia os dados das pessoas identificadas para o servidor Flask
            send_recognized_faces(name)
        face_names.append(name)
        conf_values.append(face_distances[best_match_index])

    face_locations = np.array(face_locations)
    face_locations = face_locations / resizing
    return face_locations.astype(int), face_names, conf_values

# Mostra o reconhecimento sobre a imagem
def show_recognition(frame, face_locations, face_names, conf_values):
    for face_loc, name, conf in zip(face_locations, face_names, conf_values):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        conf = "{:.8f}".format(conf)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 255, 0), 4)
        if name != "Não identificado":
            cv2.putText(frame, conf, (x1, y2 + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20, 255, 0), 1, lineType=cv2.LINE_AA)

    return frame

cam = cv2.VideoCapture(0)
while (True):
    ret, frame = cam.read()

    face_locations, face_names, conf_values = recognize_faces(frame, list_encodings, list_names, 0.25)
    processed_frame = show_recognition(frame, face_locations, face_names, conf_values)

    cv2.imshow("Reconhecendo rostos", frame)
    cv2.waitKey(1)

print("Concluído!")
cam.release()
cv2.destroyAllWindows()
