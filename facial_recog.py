import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='App de reconnaissance faciale')
parser.add_argument('-i', '--input', type=str, required=True, help='Répertoire des visages connus')

print('[INFO] Démarrage du système...')
print('[INFO] Importation des modèles pré entrainés..')

# Ces trois fichiers sont des modèles pré-entrainés
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("pretrained_model/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")


face_detector = dlib.get_frontal_face_detector()

print('[INFO] Importation des modèles pré entrainés...')


# Permet de dessiner les figures sur la photo pour délimiter le visage
def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces


def encode_face(image):
    # On va d'abord détecter les visage sur l'image grâce cette fonction disponible dans dlib.
    # Cette fonction retourne une liste de coordonnées des visages sur l'image.
    face_locations = face_detector(image, 1)

    face_encodings_list = []
    landmarks_list = []

    for face_location in face_locations:
        # DETECT FACES
        # Cette fonction sur chaque localisation sur le visage, place un point jusqu'à 68
        # et permet ainsi d'avoir une meilleure qualité de détection. On pouvait placer 5 points si on voulait
        # mais dans ce cas la qualité de détection serait faible
        shape = pose_predictor_68_point(image, face_location)

        # Génère un vecteur de dimension 128
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))

        # On récupère les 68 points qui permettent de décrire le visage sous forme de coordonnées
        # et pouvoir les afficher
        shape = face_utils.shape_to_np(shape)
        landmarks_list.append(shape)

    # Transformation des coordonnées de l'image
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list


def easy_face_reco(frame, known_face_encodings, known_face_names):
    rgb_small_frame = frame[:, :, ::-1]
    # ENCODING FACE
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # CHECK DISTANCE BETWEEN KNOWN FACES AND FACES DETECTED
        # Cette opération permet de faire la différence entre le vecteur de dimension 128 de
        # tous les visages de notre base avec celui détecté pour récupérer le visage le plus proche
        # du visage détecté. Le résultat de chaque différence est stocké dans vectors
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)

        # On a une valeur de tolérence. Donc si la différence est inférieure à 0.6 on considère que c'est le même visage
        # Plus c'est proche de 0 plus les visages sont similaires.
        tolerance = 0.6

        result = []
        # On parcourt le tableau vectors et pour chaque élément du tableau si c'est inférieur
        # à la tolérance on met vrai dans le tableau result sinon on met faux
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)

        if True in result:
            # On récupère l'index du nom dans le tableau de résultat
            first_match_index = result.index(True)

            # On récupère le nom qui correspond à cet index dans le tableau de noms des visages connus
            name = known_face_names[first_match_index]

        else:
            name = "Inconnu"
        face_names.append(name)

    # On dessine les rectangle sur le visage avec le nom en bas etc
    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)


# Fonction principale
if __name__ == '__main__':
    args = parser.parse_args()

    # On importe les visages qui sont dans notre base (dossier 'visage_connus')
    print('[INFO] Importation des visages...')
    face_to_encode_path = Path(args.input)

    # On crée une variable tableau qui va stocker tous les visages connus
    files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]

    for file_ in face_to_encode_path.rglob('*.png'):
        files.append(file_)

    # On crée une variable de type tableau qui va stocker les nom des personnes dont le visage est dans la base
    known_face_names = ['Amed', 'Kanoute', 'Diop Wade', 'paulWalker', 'Zuckerberg']

    # Ce tableau va stocker le des encodages de chaque visage
    known_face_encodings = []

    # On parcourt la liste des fichiers des visages pour ouvrir chacun d'eux
    for file_ in files:
        image = PIL.Image.open(file_)
        image = np.array(image)

        # Encodage de chaque fichier
        face_encoded = encode_face(image)[0][0]
        known_face_encodings.append(face_encoded)

    print('[INFO] Visages importés')
    print('[INFO] Démarrage Webcam...')
    video_capture = cv2.VideoCapture(0)
    print('[INFO] Webcam démarré')
    print('[INFO] Détection...')
    while True:
        ret, frame = video_capture.read()
        easy_face_reco(frame, known_face_encodings, known_face_names)
        cv2.imshow('App de reconnaissance faciale', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Arrêt du système...')
    video_capture.release()
    cv2.destroyAllWindows()
