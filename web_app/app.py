# web_app/app.py
from flask import Flask, redirect, render_template, request, jsonify, session
from flask_cors import CORS
from datetime import datetime
import json
import os
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.secret_key = 'yoga-prenatal-secret-key-2024'
CORS(app)

# Load the yoga poses angles data
with open('yoga_pose_angles.json', 'r') as f:
    POSE_ANGLES = json.load(f)

# Define safety guidelines for each trimester
SAFETY_GUIDELINES = {
    'FIRST_TRIMESTER': {
        'weeks': (1, 13),
        'recommended': ['FULLY_ALLOWED', 'TRIMESTER_RESTRICTED'],
        'restricted': ['COMPLETELY_RESTRICTED'],
        'tips': [
            'Avoid deep twists and intense backbends',
            'Avoid lying flat on back for extended periods',
            'Focus on gentle movements and breath work',
            'Stay hydrated and avoid overheating'
        ]
    },
    'SECOND_TRIMESTER': {
        'weeks': (14, 27),
        'recommended': ['FULLY_ALLOWED'],
        'restricted': ['TRIMESTER_RESTRICTED', 'COMPLETELY_RESTRICTED'],
        'tips': [
            'Avoid lying flat on back (supine positions)',
            'Use props for support and balance',
            'Widen stance for better stability',
            'Avoid deep backbends and intense twists'
        ]
    },
    'THIRD_TRIMESTER': {
        'weeks': (28, 40),
        'recommended': ['FULLY_ALLOWED'],
        'restricted': ['TRIMESTER_RESTRICTED', 'COMPLETELY_RESTRICTED'],
        'tips': [
            'Avoid lying on back completely',
            'Use wall support for standing poses',
            'Focus on gentle hip openers and relaxation',
            'Listen to your body and avoid overexertion'
        ]
    }
}

def calculate_trimester(pregnancy_start_date):
    """Calculate trimester based on pregnancy start date."""
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
    """Get safety classification for a pose."""
    # This would come from your labeled dataset
    # For now, using a mapping based on pose name patterns
    pose_lower = pose_name.lower()
    
    # Completely restricted poses
    restricted_keywords = ['camel', 'bow', 'plow', 'shoulderstand', 'headstand', 
                          'scorpion', 'wheel', 'fish', 'peacock']
    for keyword in restricted_keywords:
        if keyword in pose_lower:
            return {
                'safety': 'COMPLETELY_RESTRICTED',
                'allowed_trimesters': [],
                'message': 'This pose is not recommended during pregnancy',
                'color': 'danger'
            }
    
    # Trimester restricted poses
    trimester_keywords = ['boat', 'downward', 'cobra', 'bridge', 'warrior_iii', 'half_moon']
    for keyword in trimester_keywords:
        if keyword in pose_lower:
            return {
                'safety': 'TRIMESTER_RESTRICTED',
                'allowed_trimesters': [1],
                'message': 'Only recommended in first trimester with modifications',
                'color': 'warning'
            }
    
    # Fully allowed poses
    return {
        'safety': 'FULLY_ALLOWED',
        'allowed_trimesters': [1, 2, 3],
        'message': 'Safe for all trimesters with proper modifications',
        'color': 'success'
    }

def get_recommended_poses(trimester):
    """Get list of recommended poses based on trimester."""
    recommended = []
    
    for pose_name, pose_data in POSE_ANGLES.items():
        if pose_data.get('status') != 'success':
            continue
        
        safety = get_pose_safety(pose_name)
        
        if trimester == 'FIRST_TRIMESTER':
            if safety['safety'] != 'COMPLETELY_RESTRICTED':
                recommended.append({
                    'name': pose_name,
                    'safety': safety['safety'],
                    'message': safety['message'],
                    'color': safety['color']
                })
        else:  # SECOND or THIRD trimester
            if safety['safety'] == 'FULLY_ALLOWED':
                recommended.append({
                    'name': pose_name,
                    'safety': safety['safety'],
                    'message': safety['message'],
                    'color': safety['color']
                })
    
    return recommended[:20]  # Return top 20 recommended poses

def calculate_angle_difference(reference_angles, user_angles):
    """Calculate difference between reference and user angles."""
    differences = {}
    corrections = []
    
    for joint, ref_angle in reference_angles.items():
        if joint in user_angles and ref_angle is not None:
            user_angle = user_angles[joint]
            diff = abs(ref_angle - user_angle)
            differences[joint] = diff
            
            # Generate correction suggestions
            if diff > 30:
                if 'elbow' in joint:
                    corrections.append(f"Adjust your {joint.replace('_', ' ')} - should be {ref_angle}°")
                elif 'knee' in joint:
                    corrections.append(f"Straighten or bend your {joint.replace('_', ' ')} to {ref_angle}°")
                elif 'hip' in joint:
                    corrections.append(f"Adjust your hip angle - target {ref_angle}°")
                elif 'shoulder' in joint:
                    corrections.append(f"Open or close your shoulders to {ref_angle}°")
    
    return differences, corrections

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    name = data.get('name')
    age = data.get('age')
    pregnancy_date = data.get('pregnancy_date')
    
    trimester, weeks = calculate_trimester(pregnancy_date)
    
    session['user'] = {
        'name': name,
        'age': age,
        'pregnancy_date': pregnancy_date,
        'trimester': trimester,
        'weeks': weeks
    }
    
    return jsonify({
        'status': 'success',
        'trimester': trimester,
        'weeks': weeks,
        'guidelines': SAFETY_GUIDELINES[trimester]['tips']
    })

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    
    trimester = session['user']['trimester']
    recommended_poses = get_recommended_poses(trimester)
    
    return render_template('dashboard.html', 
                         user=session['user'],
                         poses=recommended_poses,
                         guidelines=SAFETY_GUIDELINES[trimester]['tips'])

@app.route('/pose/<pose_name>')
def pose_page(pose_name):
    if 'user' not in session:
        return redirect('/')
    
    safety = get_pose_safety(pose_name)
    reference_angles = POSE_ANGLES.get(pose_name, {}).get('angles', {})
    
    # Ensure color is properly set
    if 'color' not in safety:
        if safety['safety'] == 'FULLY_ALLOWED':
            safety['color'] = 'success'
        elif safety['safety'] == 'TRIMESTER_RESTRICTED':
            safety['color'] = 'warning'
        else:
            safety['color'] = 'danger'
    
    return render_template('pose_practice.html',
                         pose_name=pose_name,
                         safety=safety,
                         reference_angles=reference_angles)

@app.route('/api/pose/angles/<pose_name>')
def get_pose_angles(pose_name):
    """API endpoint to get reference angles for a pose."""
    reference_angles = POSE_ANGLES.get(pose_name, {}).get('angles', {})
    return jsonify(reference_angles)

@app.route('/api/pose/correct', methods=['POST'])
def correct_pose():
    """API endpoint for real-time pose correction."""
    data = request.json
    pose_name = data.get('pose_name')
    user_angles = data.get('angles', {})
    
    reference_angles = POSE_ANGLES.get(pose_name, {}).get('angles', {})
    
    if not reference_angles:
        return jsonify({'error': 'Reference angles not found'})
    
    differences, corrections = calculate_angle_difference(reference_angles, user_angles)
    
    # Calculate overall accuracy
    total_diff = sum(differences.values())
    max_possible = len(differences) * 180
    accuracy = max(0, 100 - (total_diff / max_possible * 100)) if max_possible > 0 else 100
    
    return jsonify({
        'accuracy': round(accuracy, 1),
        'corrections': corrections[:5],  # Top 5 corrections
        'differences': differences
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)