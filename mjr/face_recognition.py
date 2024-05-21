import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder
import os

from sklearn.svm import SVC

from database import connect_to_database

def extract_face_from_path(filepath, required_size=(160, 160)):
    try:
        img = Image.open(filepath)
        img = img.convert("RGB")
        img = img.resize(required_size)
        pixels = np.asarray(img)
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        if len(results) == 0:
            return None
        x1, y1, width, height = results[0]["box"]
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        img = Image.fromarray(face)
        img = img.resize(required_size)
        face_array = np.asarray(img)
        return face_array
    except Exception as e:
        print("Error processing file:", e)
        return None

def load_faces_from_database():
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute("SELECT url, label FROM register")
    rows = cursor.fetchall()
    faces, labels = [], []
    for row in rows:
        filepath, label = row
        face = extract_face_from_path(filepath)
        if face is not None:
            faces.append(face)
            labels.append(label)
    cursor.close()
    conn.close()
    return np.asarray(faces), np.asarray(labels)

def get_embedding(model, face):
    sample = np.expand_dims(face, axis=0)
    yhat = model.embeddings(sample)
    return yhat[0]

def train_model():
    trainX, trainy = load_faces_from_database()
    facenet_model = FaceNet()
    emdTrainX = [get_embedding(facenet_model, face) for face in trainX]
    emdTrainX = np.asarray(emdTrainX)
    np.savez_compressed("D:/mcp/face_embeddings.npz", emdTrainX, trainy)

def find_match(image_path):
    face = extract_face_from_path(image_path)
    if face is None:
        return "No face detected", 0.0

    facenet_model = FaceNet()
    embedding = get_embedding(facenet_model, face)
    
    data = np.load("D:/mcp/face_embeddings.npz")
    emdTrainX, trainy = data['arr_0'], data['arr_1']

    in_encoder = Normalizer()
    emdTrainX_norm = in_encoder.transform(emdTrainX)

    model = GaussianNB()
    model.fit(emdTrainX_norm, trainy)

    # Train SVM model
    svm_model = SVC(probability=True)
    svm_model.fit(emdTrainX_norm, trainy)
    
    embedding_norm = in_encoder.transform(np.expand_dims(embedding, axis=0))
    yhat_prob = model.predict_proba(embedding_norm)
    
    class_index = np.argmax(yhat_prob)
    class_probability = yhat_prob[0, class_index] * 100
    # Predict using SVM
    yhat_prob_svm = svm_model.predict_proba(embedding_norm)
    class_index_svm = np.argmax(yhat_prob_svm)
    class_probability_svm = yhat_prob_svm[0, class_index_svm] * 100
    return int(class_index), float(class_probability)

"""def find_match(image_path):
    face = extract_face_from_path(image_path)
    if face is None:
        return "No face detected", 0.0

    facenet_model = FaceNet()
    embedding = get_embedding(facenet_model, face)
    
    data = np.load("D:/mcp/face_embeddings.npz")
    emdTrainX, trainy = data['arr_0'], data['arr_1']

    in_encoder = Normalizer()
    emdTrainX_norm = in_encoder.transform(emdTrainX)
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)

    model = GaussianNB()
    model.fit(emdTrainX_norm, trainy)
    
    embedding_norm = in_encoder.transform(np.expand_dims(embedding, axis=0))
    yhat_prob = model.predict_proba(embedding_norm)
    
    class_index = np.argmax(yhat_prob)
    class_probability = yhat_prob[0, class_index] * 100
    
    # Decode the predicted class label
    predicted_class_name = out_encoder.inverse_transform([class_index])[0]

    return predicted_class_name, class_probability"""

"""
def find_match(image_path):
    face = extract_face_from_path(image_path)
    if face is None:
        return "No face detected", 0.0
    facenet_model = FaceNet()
    embedding = get_embedding(facenet_model, face)
    data = np.load("D:/mcp/face_embeddings.npz")
    emdTrainX, trainy = data['arr_0'], data['arr_1']
    in_encoder = Normalizer()
    emdTrainX_norm = in_encoder.transform(emdTrainX)
    # Define and fit the LabelEncoder
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    model = GaussianNB()
    model.fit(emdTrainX_norm, trainy)
    embedding_norm = in_encoder.transform(np.expand_dims(embedding, axis=0))
    yhat_prob = model.predict_proba(embedding_norm)
    class_index = np.argmax(yhat_prob)
    class_probability = yhat_prob[0, class_index] * 100
    # Decode the predicted class label
    predicted_class_name = out_encoder.inverse_transform([class_index])[0]
    return predicted_class_name, class_probability"""


"""model = GaussianNB()
    model.fit(emdTrainX_norm, trainy)

    # Predict the class label and probability
    yhat_class = model.predict(embedding)
    yhat_prob = model.predict_proba(embedding)
    
    # Get the class index and probability
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    
    # Decode the predicted class label
    predicted_class_name = out_encoder.inverse_transform([yhat_class[0]])[0]"""

"""
# Retrieve additional information from the database based on the predicted class name
    conn = connect_to_database()
    cur = conn.cursor()
    cur.execute("SELECT place, date FROM register WHERE label = %s", (predicted_class_name,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        place, date = row
        title_x = f"{predicted_class_name} Place: {place} Date: {date}"
        print(title_x)
    else:
        title_x = f"Details not available for {predicted_class_name}"
        print(title_x)
    
    return int(class_index), float(class_probability)"""
    

"""def find_match(image_path):
    face = extract_face_from_path(image_path)
    if face is None:
        return "No face detected", 0.0

    facenet_model = FaceNet()
    embedding = get_embedding(facenet_model, face)
    
    data = np.load("D:/mcp/face_embeddings.npz")
    emdTrainX, trainy = data['arr_0'], data['arr_1']

    in_encoder = Normalizer()
    emdTrainX_norm = in_encoder.transform(emdTrainX)

    model = GaussianNB()
    model.fit(emdTrainX_norm, trainy)
    
    embedding_norm = in_encoder.transform(np.expand_dims(embedding, axis=0))
    yhat_class=model.predict(embedding_norm)
    yhat_prob = model.predict_proba(embedding_norm)
    print("Predicted class label:",yhat_class[0])
    print("Predicted class probabilities:", yhat_prob[0])
    print(in_encoder.inverse_transform([yhat_class[0]]))
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print("Predicted class index:", class_index)
    print("Predicted class probability:", class_probability)
    print("Predicted class names:", predict_names)
    return int(class_index), float(class_probability)"""


"""
def find_match(image_path):
    face = extract_face_from_path(image_path)
    if face is None:
        return "No face detected", 0.0

    facenet_model = FaceNet()
    embedding = get_embedding(facenet_model, face)
    
    data = np.load("D:/mcp/face_embeddings.npz")
    emdTrainX, trainy = data['arr_0'], data['arr_1']

    in_encoder = Normalizer()
    emdTrainX_norm = in_encoder.transform(emdTrainX)

    model = GaussianNB()
    model.fit(emdTrainX_norm, trainy)
    
    embedding_norm = in_encoder.transform(np.expand_dims(embedding, axis=0))
    yhat_class=model.predict(embedding_norm)
    yhat_prob = model.predict_proba(embedding_norm)
    print("Predicted class label:",yhat_class[0])

    class_index = np.argmax(yhat_prob)
    class_probability = yhat_prob[0, class_index] * 100
    return int(class_index), float(class_probability)
"""
