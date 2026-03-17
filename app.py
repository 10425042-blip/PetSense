import os
import json
import base64
import sqlite3
from io import BytesIO
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np

# ============================================================
# App setup
# ============================================================
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'petsense.db')
MODEL_PATH = os.path.join(BASE_DIR, 'pet_breed_model.tflite')
CLASSES_PATH = os.path.join(BASE_DIR, 'class_names.json')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# Load AI model + class names (TFLite version)
# ============================================================
interpreter = None
class_names = []

def load_model():
    global interpreter, class_names
    try:
        # Thu import tflite_runtime truoc (cho deploy)
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            # Fallback: dung tf.lite neu chay local co TensorFlow
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        with open(CLASSES_PATH, 'r') as f:
            class_names = json.load(f)
        print(f"[OK] TFLite model loaded - {len(class_names)} breeds")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")

load_model()

# ============================================================
# Database setup
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS pets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            breed TEXT,
            pet_type TEXT DEFAULT 'unknown',
            age_months INTEGER DEFAULT 0,
            weight_kg REAL DEFAULT 0,
            gender TEXT DEFAULT '',
            color TEXT DEFAULT '',
            image_data TEXT,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        )
    ''')
    conn.execute('''CREATE TABLE IF NOT EXISTS vaccines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pet_id INTEGER NOT NULL,
            vaccine_name TEXT NOT NULL,
            scheduled_date TEXT NOT NULL,
            status TEXT DEFAULT 'upcoming',
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (pet_id) REFERENCES pets(id) ON DELETE CASCADE
        )
    ''')
    conn.execute('''CREATE TABLE IF NOT EXISTS weight_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pet_id INTEGER NOT NULL,
            weight_kg REAL NOT NULL,
            recorded_date TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (pet_id) REFERENCES pets(id) ON DELETE CASCADE
        )
    ''')
    conn.execute('''CREATE TABLE IF NOT EXISTS health_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pet_id INTEGER NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (pet_id) REFERENCES pets(id) ON DELETE CASCADE
        )
    ''')
    # Add new columns to existing tables if they don't exist
    try:
        conn.execute('ALTER TABLE pets ADD COLUMN age_months INTEGER DEFAULT 0')
    except: pass
    try:
        conn.execute('ALTER TABLE pets ADD COLUMN weight_kg REAL DEFAULT 0')
    except: pass
    try:
        conn.execute('ALTER TABLE pets ADD COLUMN gender TEXT DEFAULT ""')
    except: pass
    try:
        conn.execute('ALTER TABLE pets ADD COLUMN color TEXT DEFAULT ""')
    except: pass
    conn.commit()
    conn.close()
    print("[OK] Database ready")

init_db()

# ============================================================
# Helper: classify cat vs dog breeds
# ============================================================
CAT_BREEDS = {
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx'
}

def get_pet_type(breed_name):
    return 'cat' if breed_name in CAT_BREEDS else 'dog'

def format_breed_name(name):
    """american_pit_bull_terrier -> American Pit Bull Terrier"""
    return name.replace('_', ' ').title()

# ============================================================
# Page routes
# ============================================================
@app.route('/')
def index():
    return send_from_directory('.', 'landingpage.html')

@app.route('/predict')
def predict_page():
    return send_from_directory('.', 'predict.html')

@app.route('/pets')
def pets_page():
    return send_from_directory('.', 'pets.html')

@app.route('/vaccines')
def vaccines_page():
    return send_from_directory('.', 'vaccines.html')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ============================================================
# API: Predict breed
# ============================================================
@app.route('/api/predict', methods=['POST'])
def predict_breed():
    if interpreter is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        image_data = None

        # Accept base64 image from JSON body
        if request.is_json:
            data = request.get_json()
            img_b64 = data.get('image', '')
            # Strip data URI prefix if present
            if ',' in img_b64:
                img_b64 = img_b64.split(',')[1]
            image_data = base64.b64decode(img_b64)

        # Accept file upload
        elif 'file' in request.files:
            file = request.files['file']
            image_data = file.read()

        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        # Save uploaded image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_filename = f'predict_{timestamp}.jpg'
        img_path = os.path.join(UPLOAD_FOLDER, img_filename)

        # Process image for model
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img.save(img_path, 'JPEG')
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run prediction with TFLite
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        top_indices = predictions.argsort()[-3:][::-1]

        results = []
        for idx in top_indices:
            breed = class_names[idx]
            results.append({
                'breed': format_breed_name(breed),
                'breed_raw': breed,
                'confidence': round(float(predictions[idx]) * 100, 2),
                'pet_type': get_pet_type(breed)
            })

        return jsonify({
            'success': True,
            'predictions': results,
            'image_url': f'/uploads/{img_filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# API: Pets CRUD
# ============================================================
@app.route('/api/pets', methods=['GET'])
def get_pets():
    conn = get_db()
    pets = conn.execute('SELECT * FROM pets ORDER BY created_at DESC').fetchall()
    conn.close()
    return jsonify([dict(p) for p in pets])

@app.route('/api/pets', methods=['POST'])
def add_pet():
    data = request.get_json()
    name = data.get('name', 'Unknown')
    breed = data.get('breed', '')
    pet_type = data.get('pet_type', get_pet_type(breed))
    image_data = data.get('image_data', '')
    age_months = data.get('age_months', 0)
    weight_kg = data.get('weight_kg', 0)
    gender = data.get('gender', '')
    color = data.get('color', '')

    conn = get_db()
    cursor = conn.execute(
        '''INSERT INTO pets (name, breed, pet_type, image_data, age_months, weight_kg, gender, color)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (name, breed, pet_type, image_data, age_months, weight_kg, gender, color)
    )
    pet_id = cursor.lastrowid
    # Add initial weight to history if provided
    if weight_kg and float(weight_kg) > 0:
        conn.execute('INSERT INTO weight_history (pet_id, weight_kg) VALUES (?, ?)', (pet_id, weight_kg))
    conn.commit()
    pet = conn.execute('SELECT * FROM pets WHERE id = ?', (pet_id,)).fetchone()
    conn.close()
    return jsonify(dict(pet)), 201

@app.route('/api/pets/<int:pet_id>', methods=['PUT'])
def update_pet(pet_id):
    data = request.get_json()
    conn = get_db()
    conn.execute(
        '''UPDATE pets SET name=?, breed=?, pet_type=?, age_months=?, weight_kg=?, gender=?, color=? WHERE id=?''',
        (data.get('name'), data.get('breed'), data.get('pet_type', 'unknown'),
         data.get('age_months', 0), data.get('weight_kg', 0),
         data.get('gender', ''), data.get('color', ''), pet_id)
    )
    # Track weight change
    new_weight = data.get('weight_kg', 0)
    if new_weight and float(new_weight) > 0:
        conn.execute('INSERT INTO weight_history (pet_id, weight_kg) VALUES (?, ?)', (pet_id, new_weight))
    conn.commit()
    pet = conn.execute('SELECT * FROM pets WHERE id = ?', (pet_id,)).fetchone()
    conn.close()
    if pet:
        return jsonify(dict(pet))
    return jsonify({'error': 'Pet not found'}), 404

@app.route('/api/pets/<int:pet_id>', methods=['DELETE'])
def delete_pet(pet_id):
    conn = get_db()
    conn.execute('DELETE FROM vaccines WHERE pet_id = ?', (pet_id,))
    conn.execute('DELETE FROM weight_history WHERE pet_id = ?', (pet_id,))
    conn.execute('DELETE FROM health_notes WHERE pet_id = ?', (pet_id,))
    conn.execute('DELETE FROM pets WHERE id = ?', (pet_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

# ============================================================
# API: Weight History
# ============================================================
@app.route('/api/pets/<int:pet_id>/weight-history', methods=['GET'])
def get_weight_history(pet_id):
    conn = get_db()
    rows = conn.execute('SELECT * FROM weight_history WHERE pet_id = ? ORDER BY recorded_date DESC', (pet_id,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/pets/<int:pet_id>/weight-history', methods=['POST'])
def add_weight_record(pet_id):
    data = request.get_json()
    conn = get_db()
    conn.execute('INSERT INTO weight_history (pet_id, weight_kg) VALUES (?, ?)', (pet_id, data['weight_kg']))
    conn.execute('UPDATE pets SET weight_kg = ? WHERE id = ?', (data['weight_kg'], pet_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True}), 201

# ============================================================
# API: Health Notes
# ============================================================
@app.route('/api/pets/<int:pet_id>/health-notes', methods=['GET'])
def get_health_notes(pet_id):
    conn = get_db()
    rows = conn.execute('SELECT * FROM health_notes WHERE pet_id = ? ORDER BY created_at DESC', (pet_id,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/pets/<int:pet_id>/health-notes', methods=['POST'])
def add_health_note(pet_id):
    data = request.get_json()
    conn = get_db()
    conn.execute('INSERT INTO health_notes (pet_id, note) VALUES (?, ?)', (pet_id, data['note']))
    conn.commit()
    conn.close()
    return jsonify({'success': True}), 201

@app.route('/api/pets/<int:pet_id>/health-notes/<int:note_id>', methods=['DELETE'])
def delete_health_note(pet_id, note_id):
    conn = get_db()
    conn.execute('DELETE FROM health_notes WHERE id = ? AND pet_id = ?', (note_id, pet_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

# ============================================================
# API: Vaccines CRUD
# ============================================================
@app.route('/api/vaccines', methods=['GET'])
def get_vaccines():
    conn = get_db()
    vaccines = conn.execute('''
        SELECT v.*, p.name as pet_name, p.breed as pet_breed
        FROM vaccines v
        JOIN pets p ON v.pet_id = p.id
        ORDER BY v.scheduled_date ASC
    ''').fetchall()
    conn.close()

    result = []
    today = datetime.now().strftime('%Y-%m-%d')
    for v in vaccines:
        vd = dict(v)
        # Auto-update status based on date
        if vd['status'] != 'done':
            vd['status'] = 'overdue' if vd['scheduled_date'] < today else 'upcoming'
        result.append(vd)
    return jsonify(result)

@app.route('/api/vaccines', methods=['POST'])
def add_vaccine():
    data = request.get_json()
    conn = get_db()
    cursor = conn.execute(
        'INSERT INTO vaccines (pet_id, vaccine_name, scheduled_date, notes) VALUES (?, ?, ?, ?)',
        (data['pet_id'], data['vaccine_name'], data['scheduled_date'], data.get('notes', ''))
    )
    vaccine_id = cursor.lastrowid
    conn.commit()
    vaccine = conn.execute('''
        SELECT v.*, p.name as pet_name FROM vaccines v
        JOIN pets p ON v.pet_id = p.id WHERE v.id = ?
    ''', (vaccine_id,)).fetchone()
    conn.close()
    return jsonify(dict(vaccine)), 201

@app.route('/api/vaccines/<int:vaccine_id>', methods=['PUT'])
def update_vaccine(vaccine_id):
    data = request.get_json()
    conn = get_db()
    conn.execute(
        'UPDATE vaccines SET vaccine_name=?, scheduled_date=?, status=?, notes=? WHERE id=?',
        (data.get('vaccine_name'), data.get('scheduled_date'),
         data.get('status', 'upcoming'), data.get('notes', ''), vaccine_id)
    )
    conn.commit()
    vaccine = conn.execute('SELECT * FROM vaccines WHERE id = ?', (vaccine_id,)).fetchone()
    conn.close()
    if vaccine:
        return jsonify(dict(vaccine))
    return jsonify({'error': 'Vaccine not found'}), 404

@app.route('/api/vaccines/<int:vaccine_id>', methods=['DELETE'])
def delete_vaccine(vaccine_id):
    conn = get_db()
    conn.execute('DELETE FROM vaccines WHERE id = ?', (vaccine_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

# ============================================================
# Run server
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n  PetSense Server Running!")
    print(f"  http://127.0.0.1:{port}\n")
    app.run(debug=True, host='0.0.0.0', port=port)
