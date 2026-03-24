# ============================================
# PetSense - Pet Breed Recognition Web App
# Backend server using Flask framework
# ============================================
# This file is the main backend of the PetSense app.
# It does 3 main things:
#   1. Serves the HTML pages (landing page, predict page, etc.)
#   2. Runs the AI model to predict pet breed from an image
#   3. Stores pet data, vaccines, weight history in a SQLite database
#
# How it works:
#   - Flask is a Python web framework that handles HTTP requests
#   - When user visits a URL (like /predict), Flask sends back the HTML page
#   - When user uploads a pet image, Flask runs the Keras model and returns the breed
#   - All pet data is saved in a local SQLite database file (petsense.db)
# ============================================

# --- Import libraries we need ---
import os          # for file paths
import json        # for reading JSON files
import base64      # for decoding base64 image data
import sqlite3     # for database operations
from io import BytesIO       # for handling image bytes in memory
from datetime import datetime # for timestamps

# Flask = the web framework we use to create the server
from flask import Flask, request, jsonify, send_from_directory
# CORS = allows our frontend to talk to the backend (Cross-Origin Resource Sharing)
from flask_cors import CORS
# PIL = Python Imaging Library, used to open and resize images
from PIL import Image
# numpy = math library, used to convert image to array of numbers for the AI model
import numpy as np
# tensorflow = Google's AI/ML library, we use it to load and run our trained model
import tensorflow as tf


# ============================================
# 1. CREATE THE FLASK APP
# ============================================
# static_folder='.' means Flask will serve files from the current folder
# This lets us serve HTML, CSS, JS files directly
app = Flask(__name__, static_folder='.', static_url_path='')

# Enable CORS so the frontend can make API calls to the backend
CORS(app)

# --- File paths ---
# BASE_DIR = the folder where this app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the SQLite database file
DB_PATH = os.path.join(BASE_DIR, 'petsense.db')
# Path to the trained AI model (.h5 format = Keras model file)
MODEL_PATH = os.path.join(BASE_DIR, 'pet_breed_model.h5')
# Path to the JSON file that lists all breed names the model can recognize
CLASSES_PATH = os.path.join(BASE_DIR, 'class_names.json')
# Folder to save uploaded images
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# Create the uploads folder if it doesn't exist yet
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================
# 2. LOAD THE AI MODEL
# ============================================
# The model is a .h5 file (Keras format) trained to recognize 37 pet breeds.
# We load it once when the server starts, so it's ready to use.
#
# How it works:
#   1. Load the model using tf.keras.models.load_model()
#   2. Give it an image (as an array of numbers)
#   3. It returns a list of confidence scores for each breed
#   4. We pick the breed with the highest score

# These variables are global so all functions can use them
model = None          # the trained Keras model
class_names = []      # list of breed names like ["Abyssinian", "beagle", ...]

def load_model():
    """Load the Keras model and class names from files"""
    global model, class_names
    try:
        # Load the trained model from the .h5 file
        model = tf.keras.models.load_model(MODEL_PATH)

        # Load the list of breed names from JSON file
        with open(CLASSES_PATH, 'r') as f:
            class_names = json.load(f)

        print(f"[OK] Model loaded - can recognize {len(class_names)} breeds")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")

# Load the model right when the server starts
load_model()


# ============================================
# 3. SET UP THE DATABASE
# ============================================
# We use SQLite which is a simple database stored in a single file (petsense.db)
# No need to install a database server - it's built into Python!

def get_db():
    """Connect to the SQLite database and return the connection"""
    conn = sqlite3.connect(DB_PATH)
    # Row factory lets us access columns by name (like row['name'])
    # instead of by index (like row[0])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create all the database tables if they don't exist yet"""
    conn = get_db()

    # --- Table: pets ---
    # Stores basic info about each pet
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

    # --- Table: vaccines ---
    # Stores vaccination records for each pet
    conn.execute('''
        CREATE TABLE IF NOT EXISTS vaccines (
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

    # --- Table: weight_history ---
    # Tracks weight changes over time for each pet
    conn.execute('''
        CREATE TABLE IF NOT EXISTS weight_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pet_id INTEGER NOT NULL,
            weight_kg REAL NOT NULL,
            recorded_date TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (pet_id) REFERENCES pets(id) ON DELETE CASCADE
        )
    ''')

    # --- Table: health_notes ---
    # Stores health-related notes for each pet
    conn.execute('''
        CREATE TABLE IF NOT EXISTS health_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pet_id INTEGER NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (pet_id) REFERENCES pets(id) ON DELETE CASCADE
        )
    ''')

    # Try to add new columns to existing pets table
    # (in case the database was created before we added these columns)
    # If the column already exists, it will throw an error which we just ignore
    try:
        conn.execute('ALTER TABLE pets ADD COLUMN age_months INTEGER DEFAULT 0')
    except:
        pass
    try:
        conn.execute('ALTER TABLE pets ADD COLUMN weight_kg REAL DEFAULT 0')
    except:
        pass
    try:
        conn.execute('ALTER TABLE pets ADD COLUMN gender TEXT DEFAULT ""')
    except:
        pass
    try:
        conn.execute('ALTER TABLE pets ADD COLUMN color TEXT DEFAULT ""')
    except:
        pass

    conn.commit()
    conn.close()
    print("[OK] Database ready")

# Create the tables when the server starts
init_db()


# ============================================
# 4. HELPER FUNCTIONS
# ============================================

# List of cat breed names (used to tell if a breed is a cat or dog)
CAT_BREEDS = {
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx'
}

def get_pet_type(breed_name):
    """Check if a breed is a cat or dog"""
    # If the breed is in our cat list, it's a cat. Otherwise it's a dog.
    if breed_name in CAT_BREEDS:
        return 'cat'
    else:
        return 'dog'

def format_breed_name(name):
    """Make breed names look nice
    Example: 'american_pit_bull_terrier' -> 'American Pit Bull Terrier'
    """
    return name.replace('_', ' ').title()


# ============================================
# 5. PAGE ROUTES
# ============================================
# These routes tell Flask which HTML file to show for each URL

@app.route('/')
def index():
    """Show the landing page when user visits the home URL"""
    return send_from_directory('.', 'landingpage.html')

@app.route('/predict')
def predict_page():
    """Show the breed prediction page"""
    return send_from_directory('.', 'predict.html')

@app.route('/pets')
def pets_page():
    """Show the pets management page"""
    return send_from_directory('.', 'pets.html')

@app.route('/vaccines')
def vaccines_page():
    """Show the vaccines page"""
    return send_from_directory('.', 'vaccines.html')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded images (like pet photos)"""
    return send_from_directory(UPLOAD_FOLDER, filename)


# ============================================
# 6. API: PREDICT BREED (the main AI feature!)
# ============================================
# This is the core feature of our app.
# When the user uploads a pet image, this function:
#   1. Receives the image
#   2. Resizes it to 224x224 pixels (what the model expects)
#   3. Converts it to numbers (numpy array)
#   4. Feeds it to the AI model
#   5. Returns the top 3 breed predictions with confidence scores

@app.route('/api/predict', methods=['POST'])
def predict_breed():
    """Predict the breed of a pet from an uploaded image"""
    # Check if the model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        image_data = None

        # --- Get the image from the request ---
        # The image can come in 2 ways:

        # Way 1: As base64 string in JSON body (from the webcam/drag-drop)
        if request.is_json:
            data = request.get_json()
            img_b64 = data.get('image', '')
            # Remove the "data:image/jpeg;base64," prefix if it exists
            if ',' in img_b64:
                img_b64 = img_b64.split(',')[1]
            # Decode base64 string to raw image bytes
            image_data = base64.b64decode(img_b64)

        # Way 2: As a file upload (from the file picker)
        elif 'file' in request.files:
            file = request.files['file']
            image_data = file.read()

        # If no image was provided, return an error
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        # --- Save the uploaded image to the uploads folder ---
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_filename = f'predict_{timestamp}.jpg'
        img_path = os.path.join(UPLOAD_FOLDER, img_filename)

        # --- Process the image for the AI model ---
        # Open the image using PIL
        img = Image.open(BytesIO(image_data)).convert('RGB')
        # Save it to disk
        img.save(img_path, 'JPEG')
        # Resize to 224x224 pixels (the size our model was trained on)
        img_resized = img.resize((224, 224))
        # Convert image to numpy array of numbers (0 to 1 range)
        # The model expects pixel values between 0 and 1, not 0 to 255
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        # Add batch dimension: shape goes from (224,224,3) to (1,224,224,3)
        # The model expects a "batch" of images, even if we only have 1
        img_array = np.expand_dims(img_array, axis=0)

        # --- Run the AI model prediction ---
        # model.predict() returns confidence scores for each breed
        predictions = model.predict(img_array)[0]

        # --- Get the top 3 predictions ---
        # argsort() sorts the indices by value, we take the last 3 (highest)
        # and reverse them so the best prediction comes first
        top_indices = predictions.argsort()[-3:][::-1]

        # Build the results list
        results = []
        for idx in top_indices:
            breed = class_names[idx]
            results.append({
                'breed': format_breed_name(breed),       # nice name like "Golden Retriever"
                'breed_raw': breed,                       # raw name like "golden_retriever"
                'confidence': round(float(predictions[idx]) * 100, 2),  # percentage like 95.23
                'pet_type': get_pet_type(breed)           # "cat" or "dog"
            })

        # Return the predictions as JSON
        return jsonify({
            'success': True,
            'predictions': results,
            'image_url': f'/uploads/{img_filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# 7. API: PETS CRUD (Create, Read, Update, Delete)
# ============================================
# These routes handle pet data management

@app.route('/api/pets', methods=['GET'])
def get_pets():
    """Get all pets from the database"""
    conn = get_db()
    pets = conn.execute('SELECT * FROM pets ORDER BY created_at DESC').fetchall()
    conn.close()
    # Convert each row to a dictionary so Flask can turn it into JSON
    return jsonify([dict(p) for p in pets])

@app.route('/api/pets', methods=['POST'])
def add_pet():
    """Add a new pet to the database"""
    data = request.get_json()

    # Get pet info from the request (with default values)
    name = data.get('name', 'Unknown')
    breed = data.get('breed', '')
    pet_type = data.get('pet_type', get_pet_type(breed))
    image_data = data.get('image_data', '')
    age_months = data.get('age_months', 0)
    weight_kg = data.get('weight_kg', 0)
    gender = data.get('gender', '')
    color = data.get('color', '')

    conn = get_db()
    # Insert the pet into the database
    cursor = conn.execute(
        '''INSERT INTO pets (name, breed, pet_type, image_data, age_months, weight_kg, gender, color)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (name, breed, pet_type, image_data, age_months, weight_kg, gender, color)
    )
    pet_id = cursor.lastrowid  # get the ID of the new pet

    # If weight was provided, also add it to weight history
    if weight_kg and float(weight_kg) > 0:
        conn.execute('INSERT INTO weight_history (pet_id, weight_kg) VALUES (?, ?)', (pet_id, weight_kg))

    conn.commit()
    # Return the newly created pet
    pet = conn.execute('SELECT * FROM pets WHERE id = ?', (pet_id,)).fetchone()
    conn.close()
    return jsonify(dict(pet)), 201

@app.route('/api/pets/<int:pet_id>', methods=['PUT'])
def update_pet(pet_id):
    """Update an existing pet's info"""
    data = request.get_json()
    conn = get_db()

    # Update all fields for this pet
    conn.execute(
        '''UPDATE pets SET name=?, breed=?, pet_type=?, age_months=?, weight_kg=?, gender=?, color=? WHERE id=?''',
        (data.get('name'), data.get('breed'), data.get('pet_type', 'unknown'),
         data.get('age_months', 0), data.get('weight_kg', 0),
         data.get('gender', ''), data.get('color', ''), pet_id)
    )

    # Track weight change in history
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
    """Delete a pet and all its related data"""
    conn = get_db()
    # Delete related data first (vaccines, weight history, health notes)
    conn.execute('DELETE FROM vaccines WHERE pet_id = ?', (pet_id,))
    conn.execute('DELETE FROM weight_history WHERE pet_id = ?', (pet_id,))
    conn.execute('DELETE FROM health_notes WHERE pet_id = ?', (pet_id,))
    # Then delete the pet itself
    conn.execute('DELETE FROM pets WHERE id = ?', (pet_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ============================================
# 8. API: WEIGHT HISTORY
# ============================================
# These routes track pet weight over time

@app.route('/api/pets/<int:pet_id>/weight-history', methods=['GET'])
def get_weight_history(pet_id):
    """Get all weight records for a specific pet"""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM weight_history WHERE pet_id = ? ORDER BY recorded_date DESC',
        (pet_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/pets/<int:pet_id>/weight-history', methods=['POST'])
def add_weight_record(pet_id):
    """Add a new weight record for a pet"""
    data = request.get_json()
    conn = get_db()
    # Add to weight history
    conn.execute('INSERT INTO weight_history (pet_id, weight_kg) VALUES (?, ?)',
                 (pet_id, data['weight_kg']))
    # Also update the pet's current weight
    conn.execute('UPDATE pets SET weight_kg = ? WHERE id = ?',
                 (data['weight_kg'], pet_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True}), 201


# ============================================
# 9. API: HEALTH NOTES
# ============================================
# These routes manage health notes for each pet

@app.route('/api/pets/<int:pet_id>/health-notes', methods=['GET'])
def get_health_notes(pet_id):
    """Get all health notes for a specific pet"""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM health_notes WHERE pet_id = ? ORDER BY created_at DESC',
        (pet_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/pets/<int:pet_id>/health-notes', methods=['POST'])
def add_health_note(pet_id):
    """Add a new health note for a pet"""
    data = request.get_json()
    conn = get_db()
    conn.execute('INSERT INTO health_notes (pet_id, note) VALUES (?, ?)',
                 (pet_id, data['note']))
    conn.commit()
    conn.close()
    return jsonify({'success': True}), 201

@app.route('/api/pets/<int:pet_id>/health-notes/<int:note_id>', methods=['DELETE'])
def delete_health_note(pet_id, note_id):
    """Delete a specific health note"""
    conn = get_db()
    conn.execute('DELETE FROM health_notes WHERE id = ? AND pet_id = ?',
                 (note_id, pet_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ============================================
# 10. API: VACCINES CRUD
# ============================================
# These routes manage vaccination records

@app.route('/api/vaccines', methods=['GET'])
def get_vaccines():
    """Get all vaccines with pet info"""
    conn = get_db()
    # JOIN with pets table to also get the pet's name and breed
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
        # Auto-update status: if the date has passed and it's not done, mark as overdue
        if vd['status'] != 'done':
            if vd['scheduled_date'] < today:
                vd['status'] = 'overdue'
            else:
                vd['status'] = 'upcoming'
        result.append(vd)
    return jsonify(result)

@app.route('/api/vaccines', methods=['POST'])
def add_vaccine():
    """Add a new vaccine record"""
    data = request.get_json()
    conn = get_db()
    cursor = conn.execute(
        'INSERT INTO vaccines (pet_id, vaccine_name, scheduled_date, notes) VALUES (?, ?, ?, ?)',
        (data['pet_id'], data['vaccine_name'], data['scheduled_date'], data.get('notes', ''))
    )
    vaccine_id = cursor.lastrowid
    conn.commit()

    # Return the new vaccine with pet name
    vaccine = conn.execute('''
        SELECT v.*, p.name as pet_name FROM vaccines v
        JOIN pets p ON v.pet_id = p.id WHERE v.id = ?
    ''', (vaccine_id,)).fetchone()
    conn.close()
    return jsonify(dict(vaccine)), 201

@app.route('/api/vaccines/<int:vaccine_id>', methods=['PUT'])
def update_vaccine(vaccine_id):
    """Update a vaccine record"""
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
    """Delete a vaccine record"""
    conn = get_db()
    conn.execute('DELETE FROM vaccines WHERE id = ?', (vaccine_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ============================================
# 11. START THE SERVER
# ============================================
# This runs the Flask development server on port 5000
# Visit http://127.0.0.1:5000/ in your browser to use the app

if __name__ == '__main__':
    print("\n  PetSense Server Running!")
    print("  http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
