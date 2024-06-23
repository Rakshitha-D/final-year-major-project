from flask import Flask, render_template, request, redirect, send_file, url_for, jsonify,session
import os
from matplotlib import pyplot as plt
import numpy as np
from database import connect_to_database, insert_record, update_record, delete_record, get_all_records
from face_recognition import extract_face_from_path, train_model, find_match
import requests
from PIL import Image
from io import BytesIO
from flask_cors import CORS
from flask import send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
app = Flask(__name__)
# Update CORS configuration
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:3000"}})

app.config['UPLOAD_FOLDER'] = 'static/uploads'

app.secret_key = 'your_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)

users = {
    "admin": {"password": "adminpass", "role": "admin"},
    "user": {"password": "userpass", "role": "user"}
}

class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.role = users[username]["role"]

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if username in users and users[username]["password"] == password:
        user = User(username)
        login_user(user)
        session['role'] = user.role
        return jsonify({"message": "Login successful", "role": user.role}), 200
    return jsonify({"message": "Invalid credentials"}), 401

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out"})

@app.route('/records', methods=['GET'])
@login_required
def get_records():
    records = get_all_records()
    return jsonify({"records": records, "role": current_user.role})

@app.route('/delete', methods=['POST'])
@login_required
def delete():
    if current_user.role != 'admin':
        return jsonify({"message": "Unauthorized"}), 403
    data = request.json
    item_id = data.get('id')
    # Your deletion logic here
    delete_record(item_id)
    train_model()  # Train model after deleting data
    #return redirect(url_for('index'))
    return jsonify({"message": "Item deleted successfully"})


@app.route('/')
def index():
    records = get_all_records()
    return render_template('index.html', records=records)

@app.route('/submit', methods=['POST'])
@login_required
def submit():
    if current_user.role != 'admin':
        return jsonify({"message": "Unauthorized"}), 403
    label = request.form['label']
    place = request.form['place']
    date = request.form['date']
    image_file = request.files['image']

    if image_file.filename != '':
        folder_path = os.path.join('D:\\', 'mcp', 'dataset_train', 'train', label)
        os.makedirs(folder_path, exist_ok=True)
        image_path = os.path.join(folder_path, image_file.filename)
        image_file.save(image_path)
        insert_record(label, place, date, image_path)
        train_model()  # Train model after adding new data

    return redirect(url_for('index'))

@app.route('/update', methods=['POST'])
@login_required
def update():
    if current_user.role != 'admin':
        return jsonify({"message": "Unauthorized"}), 403
    record_id = request.form['id']
    label = request.form['label']
    place = request.form['place']
    date = request.form['date']
    image_file = request.files.get('image', None)

    new_image_path = None
    if image_file and image_file.filename != '':
        folder_path = os.path.join('D:\\', 'mcp', 'dataset_train', 'train', label)
        os.makedirs(folder_path, exist_ok=True)
        new_image_path = os.path.join(folder_path, image_file.filename)
        image_file.save(new_image_path)

    update_record(record_id, label, place, date, new_image_path)
    train_model()  # Train model after updating data

    return redirect(url_for('index'))
"""
@app.route('/delete', methods=['POST'])
def delete():
    record_id = request.json.get('id')
    #record_id = request.form['id']
    delete_record(record_id)
    train_model()  # Train model after deleting data
    return redirect(url_for('index'))"""


@app.route('/register-data', methods=['GET'])
def get_register_data():
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM register")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(data)

@app.route('/match', methods=['POST'])
def match():
    image_file = request.files['image']
    if image_file.filename != '':
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', image_file.filename)
        os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
        image_file.save(temp_image_path)
        
        match_label, match_probability = find_match(temp_image_path)
        matched_image_path = None
        
        if match_label != "No face detected":
            conn = connect_to_database()
            cursor = conn.cursor()
            cursor.execute("SELECT url FROM register WHERE id = %s", (str(match_label+1),))
            matched_image_path = cursor.fetchone()
            cursor.execute("SELECT label, place, date FROM register WHERE url = %s", (matched_image_path,))
            matched_image_details = cursor.fetchone()
            cursor.close()
            conn.close()
        
        os.remove(temp_image_path)  # Clean up temporary file

        if isinstance(matched_image_path, tuple):
            matched_image_path = matched_image_path[0]

        if matched_image_details:
            name, place, date = matched_image_details
        else:
            name, place, date = None, None, None

        try:
            img = Image.open(matched_image_path)
        except Exception as e:
            print(f"Error opening image: {e}")
            raise
        
        if match_probability > 80:
            status = "Exact match found"
        elif match_probability > 50:
            status = "Probable match found"
        else:
            status = "No match found"

        return jsonify({
            'name': name,
            'place': place,
            'date': date,
            'status': status,
            'matched_label': match_label,
            'matched_probability': match_probability,
            'matched_image_url': matched_image_path if match_probability > 50 else None
            
        })
    return jsonify({'error': 'No image provided'}), 400
"""
@app.route('/match', methods=['POST'])
def match():
    image_file = request.files['image']
    if image_file.filename != '':
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', image_file.filename)
        os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
        image_file.save(temp_image_path)
        
        match_label, match_probability = find_match(temp_image_path)
        matched_image_path = None
        
        if match_label != "No face detected":
            conn = connect_to_database()
            cursor = conn.cursor()
            cursor.execute("SELECT url FROM register WHERE id = %s", (str(match_label+1),))
            matched_image_path = cursor.fetchone()
            cursor.close()
            conn.close()
        
        os.remove(temp_image_path)  # Clean up temporary file

        # If matched_image_path is a tuple, extract the first element (the path)
        if isinstance(matched_image_path, tuple):
            matched_image_path = matched_image_path[0]

        # Open the image with PIL
        try:
            img = Image.open(matched_image_path)
        except Exception as e:
            print(f"Error opening image: {e}")
            raise
        
        return send_file(matched_image_path, mimetype='image/jpeg')
        return jsonify({
            'matched_label': match_label,
            'matched_probability': match_probability,
            'given_image_path': temp_image_path,
            'matched_image_path': matched_image_path
        })
    return jsonify({'error': 'No image provided'}), 400
"""

@app.route('/images/<path:image_path>')
def serve_image(image_path):
    # Assuming image_path is the path stored in the database
    return send_file(image_path, mimetype='image/jpeg' if image_path.endswith('.jpg') else 'image/png')

if __name__ == '__main__':
    app.run(debug=True)
