import cv2
import numpy as np
import os
import pickle
import re
import time
import imutils
import dlib
import face_recognition

# Função para redimensionar o vídeo
def resize_video(width, height, max_width):
    if width > max_width:
        ratio = max_width / width
        width = int(max_width)
        height = int(height * ratio)
    return width, height

# Função para salvar o rosto capturado em uma pasta
def save_face(face_roi, person_name, folder_faces, folder_full, sample, starting_sample_number):
    sample += 1
    photo_sample = sample + starting_sample_number - 1 if starting_sample_number > 0 else sample
    image_name = person_name + "." + str(photo_sample) + ".jpg"
    cv2.imwrite(os.path.join(folder_faces, person_name, image_name), face_roi)  # Salva o rosto recortado (ROI)
    cv2.imwrite(os.path.join(folder_full, person_name, image_name), frame)  # Salva a imagem completa (não recortada)
    print("=> photo " + str(sample))
    print("Face captured and saved.")
    cv2.imshow("face", face_roi)
    cv2.waitKey(500)  # Espera 500 milissegundos

# Função para carregar os códigos de codificação do arquivo pickle
def load_encodings(path_dataset):
    list_encodings = []
    list_names = []
    subdirs = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset)]
    for subdir in subdirs:
        name = subdir.split(os.path.sep)[-1]
        images_list = [os.path.join(subdir, f) for f in os.listdir(subdir) if not os.path.basename(f).startswith(".")]
        for image_path in images_list:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_roi = face_recognition.face_locations(img, model="cnn")
            (start_y, end_x, end_y, start_x) = face_roi[0]
            roi = img[start_y:end_y, start_x:end_x]
            roi = imutils.resize(roi, width=100)
            cv2.imshow('face', cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
            img_encoding = face_recognition.face_encodings(img, face_roi)
            if (len(img_encoding) > 0):
                img_encoding = img_encoding[0]
                list_encodings.append(img_encoding)
                list_names.append(name)
            else:
                print("Couldn't encode face from image => {}".format(image_path))
    return list_encodings, list_names

# Reconhece os rostos em uma imagem dada
def recognize_faces(frame, list_encodings, list_names, resizing=0.25, tolerance=0.6):
    frame = cv2.resize(frame, (0, 0), fx=resizing, fy=resizing)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
    face_names = []
    conf_values = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(list_encodings, encoding, tolerance=tolerance)
        name = "Unknown"
        face_distances = face_recognition.face_distance(list_encodings, encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = list_names[best_match_index]
        face_names.append(name)
        conf_values.append(face_distances[best_match_index])
    face_locations = np.array(face_locations)
    face_locations = face_locations / resizing
    return face_locations.astype(int), face_names, conf_values

# Exibe o reconhecimento sobre a imagem
def show_recognition(frame, face_locations, face_names, conf_values):
    for face_loc, name, conf in zip(face_locations, face_names, conf_values):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        conf = "{:.8f}".format(conf)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 255, 0), 4)
        if name != "Unknown":
            cv2.putText(frame, conf, (x1, y2 + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20, 255, 0), 1, lineType=cv2.LINE_AA)
    return frame

# Define os parâmetros
pickle_name = "face_encodings_custom.pickle"
max_width = 800

# Carrega os códigos de codificação
data_encoding = pickle.loads(open(pickle_name, "rb").read())
list_encodings = data_encoding["encodings"]
list_names = data_encoding["names"]

# Objeto de captura de vídeo (webcam)
cam = cv2.VideoCapture(0)

# Loop sobre cada quadro do fluxo de vídeo
sample = 0
while (True):
    ret, frame = cam.read()

    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)
        frame = cv2.resize(frame, (video_width, video_height))

    face_locations, face_names, conf_values = recognize_faces(frame, list_encodings, list_names, 0.25)
    print(face_locations)
    processed_frame = show_recognition(frame, face_locations, face_names, conf_values)

    cv2.imshow("Reconhecendo rostos", frame)
    cv2.waitKey(1)

    if len(face_names) > 0 and "Unknown" not in face_names:
        # Salva o rosto capturado na pasta
        save_face(frame, face_names[0], "dataset", "dataset_full", sample, 0)

print("Concluído!")
cam.release()
cv2.destroyAllWindows()
