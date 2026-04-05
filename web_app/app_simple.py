# web_app/app_simple.py
from flask import Flask, render_template, request, jsonify, session, redirect
from flask_cors import CORS
from datetime import datetime
import json
import os

app = Flask(__name__)
app.secret_key = 'yoga-prenatal-secret-key-2024'
CORS(app)

# Load the yoga poses angles data
json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yoga_pose_angles.json')
print(f"Looking for JSON at: {json_path}")

if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        POSE_ANGLES = json.load(f)
    print(f"Loaded {len(POSE_ANGLES)} poses")
else:
    print("JSON file not found, using sample data")
    POSE_ANGLES = {}

# Safety guidelines
SAFETY_GUIDELINES = {
    'FIRST_TRIMESTER': {
        'weeks': (1, 13),
        'tips': ['Avoid deep twists', 'Focus on gentle movements']
    },
    'SECOND_TRIMESTER': {
        'weeks': (14, 27),
        'tips': ['Avoid lying on back', 'Use props for support']
    },
    'THIRD_TRIMESTER': {
        'weeks': (28, 40),
        'tips': ['Avoid lying on back', 'Use wall support']
    }
}

def calculate_trimester(pregnancy_start_date):
    try:
        start_date = datetime.strptime(pregnancy_start_date, '%Y-%m-%d')
        today = datetime.now()
        weeks_pregnant = (today - start_date).days // 7
        
        if weeks_pregnant <= 13:
            return 'FIRST_TRIMESTER', weeks_pregnant
        elif weeks_pregnant <= 27:
            return 'SECOND_TRIMESTER', weeks_pregnant
        else:
            return 'THIRD_TRIMESTER', weeks_pregnant
    except:
        return 'FIRST_TRIMESTER', 0

def get_pose_safety(pose_name):
    pose_lower = pose_name.lower()
    
    restricted_keywords = ['camel', 'bow', 'plow', 'shoulderstand', 'headstand']
    for keyword in restricted_keywords:
        if keyword in pose_lower:
            return {'safety': 'COMPLETELY_RESTRICTED', 'color': 'danger', 
                   'message': 'Not recommended during pregnancy'}
    
    trimester_keywords = ['boat', 'downward', 'cobra']
    for keyword in trimester_keywords:
        if keyword in pose_lower:
            return {'safety': 'TRIMESTER_RESTRICTED', 'color': 'warning',
                   'message': 'Only in first trimester with modifications'}
    
    return {'safety': 'FULLY_ALLOWED', 'color': 'success',
           'message': 'Safe with proper modifications'}

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    trimester, weeks = calculate_trimester(data.get('pregnancy_date'))
    
    session['user'] = {
        'name': data.get('name'),
        'age': data.get('age'),
        'trimester': trimester,
        'weeks': weeks
    }
    
    return jsonify({'status': 'success', 'trimester': trimester, 'weeks': weeks})

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    
    poses = []
    for pose_name in list(POSE_ANGLES.keys())[:20]:  # First 20 poses
        safety = get_pose_safety(pose_name)
        poses.append({
            'name': pose_name,
            'safety': safety['safety'],
            'message': safety['message'],
            'color': safety['color']
        })
    
    return render_template('dashboard.html', 
                         user=session['user'],
                         poses=poses,
                         guidelines=SAFETY_GUIDELINES[session['user']['trimester']]['tips'])

@app.route('/pose/<pose_name>')
def pose_page(pose_name):
    if 'user' not in session:
        return redirect('/')
    
    safety = get_pose_safety(pose_name)
    reference_angles = POSE_ANGLES.get(pose_name, {}).get('angles', {})
    
    return render_template('pose_practice.html',
                         pose_name=pose_name,
                         safety=safety,
                         reference_angles=reference_angles)

@app.route('/api/pose/angles/<pose_name>')
def get_pose_angles(pose_name):
    angles = POSE_ANGLES.get(pose_name, {}).get('angles', {})
    return jsonify(angles)

@app.route('/api/pose/correct', methods=['POST'])
def correct_pose():
    data = request.json
    user_angles = data.get('angles', {})
    
    # Simple accuracy calculation
    accuracy = 75  # Placeholder
    corrections = ["Adjust your posture", "Keep your back straight"]
    
    return jsonify({
        'accuracy': accuracy,
        'corrections': corrections
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Prenatal Yoga Web App")
    print("=" * 50)
    print(f"Server running at: http://localhost:5000")
    print("Press CTRL+C to stop")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)