import os
import json
import base64
import random
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request, session, redirect, send_from_directory
import cv2
import numpy as np

print("=" * 60)
print("🚀 Starting Prenatal Yoga Pose Correction System")
print("=" * 60)

app = Flask(__name__)
app.secret_key = 'prenatal-yoga-secret-2024'

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "train")
ANGLES_FILE = os.path.join(BASE_DIR, "data", "yoga_pose_angles.json")

# If angles file not in data folder, check root
if not os.path.exists(ANGLES_FILE):
    ANGLES_FILE = os.path.join(BASE_DIR, "yoga_pose_angles.json")

# ==================== LOAD REFERENCE ANGLES ====================
REFERENCE_ANGLES = {}
if os.path.exists(ANGLES_FILE):
    with open(ANGLES_FILE, 'r') as f:
        all_data = json.load(f)
        # Handle different JSON structures
        if isinstance(all_data, dict):
            for key, value in all_data.items():
                if isinstance(value, dict) and 'angles' in value:
                    REFERENCE_ANGLES[key] = value['angles']
                elif isinstance(value, dict):
                    REFERENCE_ANGLES[key] = value
    print(f"✅ Loaded reference angles for {len(REFERENCE_ANGLES)} poses")
else:
    print(f"⚠️ Angles file not found at {ANGLES_FILE}")
    # Create sample reference angles
    REFERENCE_ANGLES = {
        "Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_": {
            "left_elbow": 144.28, "right_elbow": 138.36, "left_shoulder": 128.12,
            "right_shoulder": 135.2, "left_hip": 12.75, "right_hip": 146.17,
            "left_knee": 169.21, "right_knee": 149.24, "left_ankle": 133.08,
            "right_ankle": 160.45, "neck": 6.7
        }
    }

# ==================== POSE SAFETY CLASSIFICATION ====================
# SAFE poses (Green) - All trimesters
SAFE_POSES = [
    "Cat_Cow_Pose", "Child_Pose", "Bound_Angle_Pose", "Corpse_Pose",
    "Tree_Pose", "Warrior_II", "Triangle_Pose", "Garland_Pose",
    "Happy_Baby_Pose", "Legs_Up_the_Wall", "Butterfly_Pose", "Easy_Pose",
    "Goddess_Pose", "Low_Lunge", "Seated_Forward_Bend", "Wide_Legged_Forward_Bend",
    "Cow_Face_Pose", "Eagle_Pose", "Gate_Pose", "Standing_Forward_Bend",
    "Staff_Pose", "Virasana", "Yogic_sleep_pose"
]

# ALLOWED WITH MODIFICATIONS poses (Orange) - Need caution
MODIFIED_POSES = [
    "Downward_Facing_Dog", "Cobra_Pose", "Bridge_Pose", "Boat_Pose",
    "Half_Moon_Pose", "Warrior_III", "Plank_Pose", "Side_Plank_Pose",
    "Pigeon_Pose", "Chair_Pose", "Warrior_I", "Intense_Side_Stretch",
    "Head_to_Knee_Forward_Bend", "Standing_Split", "Standing_Big_Toe_Hold",
    "Wide_Angle_Seated_Forward_Bend", "Viparita_Virabhadrasana",
    "Extended_Puppy_Pose", "Reclining_Hand_to_Big_Toe", "Half_Lord_of_the_Fishes"
]

def get_pose_safety(pose_name):
    """Determine if pose is SAFE (green) or MODIFIED (orange)"""
    pose_lower = pose_name.lower()
    for safe in SAFE_POSES:
        if safe.lower() in pose_lower:
            return {"status": "SAFE", "color": "green", "message": "Safe for all trimesters"}
    for modified in MODIFIED_POSES:
        if modified.lower() in pose_lower:
            return {"status": "ALLOWED WITH MODIFICATIONS", "color": "orange", "message": "Practice with caution, use props"}
    return {"status": "ALLOWED WITH MODIFICATIONS", "color": "orange", "message": "Consult your healthcare provider"}

# ==================== TRIMESTER CALCULATION ====================
def calculate_trimester(lmp_date):
    """Calculate trimester based on Last Menstrual Period (LMP)"""
    try:
        lmp = datetime.strptime(lmp_date, '%Y-%m-%d')
        today = datetime.now()
        days_pregnant = (today - lmp).days
        weeks_pregnant = max(4, days_pregnant // 7)
        
        if weeks_pregnant <= 12:
            return 'FIRST', weeks_pregnant
        elif weeks_pregnant <= 26:
            return 'SECOND', weeks_pregnant
        else:
            return 'THIRD', weeks_pregnant
    except Exception as e:
        print(f"Error calculating trimester: {e}")
        return 'FIRST', 12

def get_trimester_tips(trimester):
    """Get safety tips based on trimester"""
    tips = {
        'FIRST': [
            '✅ Gentle stretching is safe',
            '⚠️ Avoid deep twists and intense backbends',
            '✅ Stay hydrated and avoid overheating',
            '⚠️ Avoid lying flat on back for extended periods'
        ],
        'SECOND': [
            '⚠️ Avoid lying flat on your back',
            '✅ Use props (blocks, straps) for support',
            '✅ Widen your stance for better balance',
            '⚠️ Avoid deep backbends and intense core work'
        ],
        'THIRD': [
            '❌ Avoid lying on your back completely',
            '✅ Use wall support for standing poses',
            '✅ Focus on gentle hip openers',
            '❌ Avoid deep squats and intense core work'
        ]
    }
    return tips.get(trimester, tips['FIRST'])

# ==================== GET AVAILABLE POSES FROM DATASET ====================
def get_available_poses():
    """Get all poses from dataset folder"""
    poses = []
    if os.path.exists(DATASET_PATH):
        for folder in os.listdir(DATASET_PATH):
            folder_path = os.path.join(DATASET_PATH, folder)
            if os.path.isdir(folder_path):
                images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
                if images:
                    poses.append({
                        'key': folder,
                        'name': folder.replace('_', ' '),
                        'image': images[0]
                    })
    return poses

AVAILABLE_POSES = get_available_poses()
print(f"📸 Found {len(AVAILABLE_POSES)} poses in dataset")

# ==================== MEDIAPIPE INITIALIZATION ====================
MEDIAPIPE_AVAILABLE = False
pose_detector = None

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import Image as mpImage
    from mediapipe import ImageFormat
    import urllib.request
    
    model_path = os.path.join(BASE_DIR, "pose_landmarker.task")
    if not os.path.exists(model_path):
        print("📥 Downloading pose model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)
        print("✅ Model downloaded")
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose_detector = vision.PoseLandmarker.create_from_options(options)
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe initialized successfully")
except Exception as e:
    print(f"⚠️ MediaPipe not available: {e}")
    print("📌 Running in demo mode")

# ==================== CREATE STATIC IMAGES FOLDER ====================
os.makedirs('static_images', exist_ok=True)
for pose in AVAILABLE_POSES:
    pose_dir = os.path.join('static_images', pose['key'])
    os.makedirs(pose_dir, exist_ok=True)
    src = os.path.join(DATASET_PATH, pose['key'], pose['image'])
    dst = os.path.join(pose_dir, pose['image'])
    if not os.path.exists(dst) and os.path.exists(src):
        import shutil
        shutil.copy(src, dst)

# ==================== ANGLE EXTRACTION AND CALCULATION ====================
def extract_angles_from_landmarks(landmarks):
    """Extract joint angles from MediaPipe landmarks"""
    angles = {}
    
    def get_angle(a, b, c):
        if a is None or b is None or c is None:
            return 0
        ba_x = a.x - b.x
        ba_y = a.y - b.y
        bc_x = c.x - b.x
        bc_y = c.y - b.y
        dot = ba_x * bc_x + ba_y * bc_y
        norm_ba = (ba_x**2 + ba_y**2)**0.5
        norm_bc = (bc_x**2 + bc_y**2)**0.5
        if norm_ba == 0 or norm_bc == 0:
            return 0
        cosine = dot / (norm_ba * norm_bc)
        cosine = max(-1, min(1, cosine))
        return np.arccos(cosine) * 180.0 / np.pi
    
    if len(landmarks) > 28:
        # Elbow angles
        angles['left_elbow'] = round(get_angle(landmarks[11], landmarks[13], landmarks[15]), 1)
        angles['right_elbow'] = round(get_angle(landmarks[12], landmarks[14], landmarks[16]), 1)
        # Shoulder angles
        angles['left_shoulder'] = round(get_angle(landmarks[13], landmarks[11], landmarks[23]), 1)
        angles['right_shoulder'] = round(get_angle(landmarks[14], landmarks[12], landmarks[24]), 1)
        # Hip angles
        angles['left_hip'] = round(get_angle(landmarks[11], landmarks[23], landmarks[25]), 1)
        angles['right_hip'] = round(get_angle(landmarks[12], landmarks[24], landmarks[26]), 1)
        # Knee angles
        angles['left_knee'] = round(get_angle(landmarks[23], landmarks[25], landmarks[27]), 1)
        angles['right_knee'] = round(get_angle(landmarks[24], landmarks[26], landmarks[28]), 1)
        # Ankle angles
        angles['left_ankle'] = round(get_angle(landmarks[25], landmarks[27], landmarks[31]), 1)
        angles['right_ankle'] = round(get_angle(landmarks[26], landmarks[28], landmarks[32]), 1)
        # Neck angle
        angles['neck'] = round(get_angle(landmarks[11], landmarks[0], landmarks[12]), 1)
    
    return angles

def calculate_pose_accuracy(user_angles, reference_angles):
    """Calculate pose accuracy by comparing angles with reference"""
    if not reference_angles or not user_angles:
        return 50
    
    # Define joint weights (more important joints have higher weight)
    joint_weights = {
        'left_knee': 1.5, 'right_knee': 1.5,
        'left_hip': 1.3, 'right_hip': 1.3,
        'left_shoulder': 1.0, 'right_shoulder': 1.0,
        'left_elbow': 0.8, 'right_elbow': 0.8,
        'left_ankle': 0.6, 'right_ankle': 0.6,
        'neck': 0.5
    }
    
    total_weight = 0
    weighted_error = 0
    
    for joint, weight in joint_weights.items():
        if joint in user_angles and joint in reference_angles:
            if user_angles[joint] and reference_angles[joint]:
                # Calculate absolute difference
                diff = abs(user_angles[joint] - reference_angles[joint])
                # Normalize difference (max 90 degrees difference)
                normalized_diff = min(diff / 90.0, 1.0)
                weighted_error += normalized_diff * weight
                total_weight += weight
    
    if total_weight == 0:
        return 50
    
    # Calculate accuracy percentage
    accuracy = (1 - (weighted_error / total_weight)) * 100
    return round(max(0, min(100, accuracy)), 1)

def generate_corrections(user_angles, reference_angles):
    """Generate specific correction suggestions based on angle differences"""
    corrections = []
    
    if not reference_angles:
        return ["Position yourself properly in front of the camera"]
    
    # Check each joint for significant differences
    for joint, ref_val in reference_angles.items():
        if joint in user_angles and ref_val and user_angles[joint]:
            diff = user_angles[joint] - ref_val
            abs_diff = abs(diff)
            
            if abs_diff > 25:  # Only suggest corrections for significant differences
                if 'knee' in joint:
                    side = joint.split('_')[0]
                    if diff > 0:
                        corrections.append(f"Bend your {side} knee {abs_diff:.0f}° more")
                    else:
                        corrections.append(f"Straighten your {side} knee {abs_diff:.0f}°")
                elif 'hip' in joint:
                    side = joint.split('_')[0]
                    if diff > 0:
                        corrections.append(f"Tilt your {side} hip forward")
                    else:
                        corrections.append(f"Tilt your {side} hip backward")
                elif 'shoulder' in joint:
                    side = joint.split('_')[0]
                    if diff > 0:
                        corrections.append(f"Lower your {side} shoulder")
                    else:
                        corrections.append(f"Lift your {side} shoulder")
                elif 'elbow' in joint:
                    side = joint.split('_')[0]
                    if diff > 0:
                        corrections.append(f"Bend your {side} elbow more")
                    else:
                        corrections.append(f"Straighten your {side} elbow")
                elif 'ankle' in joint:
                    side = joint.split('_')[0]
                    corrections.append(f"Adjust your {side} ankle position")
                elif 'neck' in joint:
                    corrections.append(f"Align your head with your spine")
    
    if not corrections:
        corrections = ["✓ Perfect form! Keep practicing"]
    
    return corrections[:4]

# ==================== HTML TEMPLATES ====================
LOGIN_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Prenatal Yoga - Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 450px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        h1 { text-align: center; color: #333; font-size: 28px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #333; font-weight: 500; }
        input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
        }
        input:focus { outline: none; border-color: #667eea; }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
        }
        .info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 14px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧘 Prenatal Yoga</h1>
        <div class="subtitle">AI-powered safe yoga during pregnancy</div>
        <form id="loginForm">
            <div class="form-group">
                <label>Full Name</label>
                <input type="text" id="name" required placeholder="Enter your name">
            </div>
            <div class="form-group">
                <label>Age</label>
                <input type="number" id="age" required placeholder="Your age">
            </div>
            <div class="form-group">
                <label>First day of Last Menstrual Period (LMP)</label>
                <input type="date" id="lmpDate" required>
                <small style="color: #666; font-size: 12px;">Used to calculate your trimester</small>
            </div>
            <button type="submit">🌸 Start My Journey 🌸</button>
        </form>
        <div class="info">
            📋 Always consult your healthcare provider before starting any exercise routine.
        </div>
    </div>
    <script>
        document.getElementById('loginForm').onsubmit = async (e) => {
            e.preventDefault();
            const response = await fetch('/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    name: document.getElementById('name').value,
                    age: document.getElementById('age').value,
                    lmp_date: document.getElementById('lmpDate').value
                })
            });
            const data = await response.json();
            if (data.status === 'success') {
                window.location.href = '/dashboard';
            }
        };
    </script>
</body>
</html>
'''

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard - Prenatal Yoga</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; }
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        .navbar h2 { font-size: 24px; }
        .logout-btn {
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
        }
        .container { max-width: 1400px; margin: 30px auto; padding: 0 20px; }
        .trimester-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 30px;
        }
        .trimester-card h2 { font-size: 28px; margin-bottom: 10px; }
        .tips {
            background: white;
            padding: 25px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .tips h3 { color: #2e7d32; margin-bottom: 15px; font-size: 20px; }
        .tips ul { margin-left: 25px; }
        .tips li { margin: 12px 0; color: #555; }
        .section-title { font-size: 24px; margin: 30px 0 20px; color: #333; }
        .poses-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }
        .pose-card {
            background: white;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }
        .pose-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        .pose-image {
            height: 220px;
            overflow: hidden;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .pose-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .pose-content { padding: 20px; }
        .pose-name { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #333; }
        .safety-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 12px;
        }
        .safety-badge.safe { background: #e8f5e9; color: #2e7d32; }
        .safety-badge.modified { background: #fff3e0; color: #ef6c00; }
        .pose-reason { font-size: 12px; color: #666; margin-bottom: 15px; }
        .start-btn {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: opacity 0.3s;
        }
        .start-btn:hover { opacity: 0.9; }
        @media (max-width: 768px) {
            .navbar { flex-direction: column; text-align: center; }
            .poses-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h2>🧘 Prenatal Yoga Studio</h2>
        <div>Welcome, {{ user.name }} | Week {{ user.weeks }}</div>
        <button class="logout-btn" onclick="logout()">🚪 Logout</button>
    </div>
    <div class="container">
        <div class="trimester-card">
            <h2>🌸 {{ trimester_name }}</h2>
            <p>Week {{ user.weeks }} of pregnancy - You're doing great!</p>
        </div>
        <div class="tips">
            <h3>💡 Safety Tips for {{ trimester_name }}</h3>
            <ul>
                {% for tip in tips %}
                <li>{{ tip }}</li>
                {% endfor %}
            </ul>
        </div>
        <h2 class="section-title">✨ Recommended Yoga Poses ✨</h2>
        <div class="poses-grid">
            {% for pose in poses %}
            <div class="pose-card" onclick="selectPose('{{ pose.key }}')">
                <div class="pose-image">
                    {% if pose.image_url %}
                    <img src="{{ pose.image_url }}" alt="{{ pose.name }}">
                    {% else %}
                    <div style="font-size: 60px;">🧘</div>
                    {% endif %}
                </div>
                <div class="pose-content">
                    <div class="pose-name">{{ pose.name }}</div>
                    <span class="safety-badge {{ pose.safety_color }}">{{ pose.safety_status }}</span>
                    <div class="pose-reason">{{ pose.safety_reason }}</div>
                    <button class="start-btn">🧘 Start Practice →</button>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    <script>
        function selectPose(poseKey) {
            window.location.href = '/practice/' + encodeURIComponent(poseKey);
        }
        function logout() {
            fetch('/logout', {method: 'POST'}).then(() => {
                window.location.href = '/';
            });
        }
    </script>
</body>
</html>
'''

PRACTICE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Practice - Prenatal Yoga</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a2e; color: white; }
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .back-btn {
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            cursor: pointer;
        }
        .mode-selector {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            padding: 15px;
            background: #0f3460;
            border-radius: 15px;
        }
        .mode-btn {
            flex: 1;
            padding: 12px;
            background: rgba(255,255,255,0.1);
            border: 2px solid transparent;
            border-radius: 10px;
            cursor: pointer;
            text-align: center;
            font-weight: bold;
            transition: all 0.3s;
        }
        .mode-btn.active { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-color: white; }
        .mode-btn:hover { background: rgba(255,255,255,0.2); }
        .camera-container, .upload-container { padding: 20px; background: #0f0f1a; border-radius: 15px; }
        .camera-section { position: relative; background: #0f0f1a; border-radius: 15px; overflow: hidden; }
        #videoElement { width: 100%; border-radius: 10px; }
        #canvasOverlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover { background: rgba(102,126,234,0.1); }
        .upload-area input { display: none; }
        .preview-image { max-width: 100%; max-height: 400px; border-radius: 10px; margin-top: 15px; }
        .analyze-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 15px;
            font-weight: bold;
        }
        .feedback-section { background: #16213e; padding: 20px; border-radius: 15px; margin-top: 20px; }
        .accuracy-circle { text-align: center; margin: 20px 0; }
        .accuracy-value { font-size: 64px; font-weight: bold; }
        .accuracy-value.high { color: #4caf50; }
        .accuracy-value.medium { color: #ff9800; }
        .accuracy-value.low { color: #f44336; }
        .corrections-list { list-style: none; margin-top: 20px; }
        .corrections-list li { background: rgba(255,255,255,0.1); margin: 10px 0; padding: 12px; border-radius: 8px; border-left: 4px solid #ff9800; }
        .instruction { background: #0f3460; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
        .pose-status { text-align: center; padding: 10px; margin-top: 10px; border-radius: 8px; }
        .pose-status.detected { background: rgba(76,175,80,0.3); color: #4caf50; }
        .pose-status.not-detected { background: rgba(244,67,54,0.3); color: #f44336; }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
            font-weight: bold;
        }
        .container { display: flex; gap: 20px; padding: 20px; height: calc(100vh - 60px); }
        .left-panel { flex: 2; }
        .right-panel { flex: 1; }
        @media (max-width: 768px) { .container { flex-direction: column; } }
        .joints-details {
            margin-top: 15px;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            font-size: 11px;
            max-height: 200px;
            overflow-y: auto;
        }
        .joints-details span { color: #ff9800; }
    </style>
</head>
<body>
    <div class="navbar">
        <button class="back-btn" onclick="goBack()">← Back to Dashboard</button>
        <div>Practicing: {{ pose_name.replace('_', ' ') }}</div>
        <div style="width: 100px;"></div>
    </div>
    <div class="container">
        <div class="left-panel">
            <div class="mode-selector">
                <div class="mode-btn active" onclick="setMode('camera')">📷 Start Camera</div>
                <div class="mode-btn" onclick="setMode('upload')">📤 Upload Image</div>
            </div>
            <div id="cameraMode" class="camera-container">
                <div class="camera-section">
                    <video id="videoElement" autoplay playsinline style="width: 100%; border-radius: 10px;"></video>
                    <canvas id="canvasOverlay"></canvas>
                </div>
                <div id="cameraStatus" class="pose-status not-detected" style="margin-top: 10px;">⚠️ Click "Start Camera" to begin</div>
                <button onclick="startCamera()">🎥 Start Camera</button>
                <button onclick="stopCamera()">🛑 Stop Camera</button>
            </div>
            <div id="uploadMode" style="display: none;">
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div style="font-size: 48px;">📸</div>
                    <p>Click or drag to upload an image</p>
                    <p style="font-size: 12px;">Supports JPG, PNG, JPEG</p>
                    <input type="file" id="fileInput" accept="image/jpeg,image/png,image/jpg">
                </div>
                <div id="imagePreview"></div>
                <button class="analyze-btn" onclick="analyzeUploadedImage()" style="display: none;" id="analyzeBtn">🔍 Analyze Pose</button>
            </div>
        </div>
        <div class="right-panel">
            <div class="instruction">
                <strong>📋 How to practice:</strong><br>
                1️⃣ Choose Camera or Upload mode<br>
                2️⃣ Allow camera access or upload a photo<br>
                3️⃣ See green skeleton on your body<br>
                4️⃣ Follow AI corrections below
            </div>
            <div class="feedback-section">
                <div class="accuracy-circle">
                    <div class="accuracy-value" id="accuracyValue">0%</div>
                    <div>Pose Accuracy</div>
                </div>
                <div id="poseStatus" class="pose-status not-detected">⚠️ Waiting for analysis...</div>
                <div>
                    <strong>💡 Corrections:</strong>
                    <ul class="corrections-list" id="correctionsList">
                        <li>Start camera or upload an image</li>
                    </ul>
                </div>
                <div class="joints-details" id="jointsDetails">
                    <strong>📐 Joint Angles:</strong><br>
                    <span>Waiting for pose detection...</span>
                </div>
            </div>
        </div>
    </div>
    <script>
        const poseName = "{{ pose_name }}";
        let currentMode = 'camera';
        let stream = null;
        let interval = null;
        let uploadedImageData = null;
        
        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            if (mode === 'camera') {
                document.querySelector('.mode-btn:first-child').classList.add('active');
                document.getElementById('cameraMode').style.display = 'block';
                document.getElementById('uploadMode').style.display = 'none';
                stopCamera();
            } else {
                document.querySelector('.mode-btn:last-child').classList.add('active');
                document.getElementById('cameraMode').style.display = 'none';
                document.getElementById('uploadMode').style.display = 'block';
                if (interval) clearInterval(interval);
                if (stream) stream.getTracks().forEach(t => t.stop());
            }
        }
        
        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                const video = document.getElementById('videoElement');
                video.srcObject = stream;
                await video.play();
                document.getElementById('cameraStatus').innerHTML = '✅ Camera active - Green skeleton will appear';
                document.getElementById('cameraStatus').className = 'pose-status detected';
                if (interval) clearInterval(interval);
                interval = setInterval(sendFrame, 500);
            } catch(e) {
                document.getElementById('cameraStatus').innerHTML = '❌ Camera access denied';
            }
        }
        
        function stopCamera() {
            if (interval) clearInterval(interval);
            if (stream) stream.getTracks().forEach(t => t.stop());
            document.getElementById('cameraStatus').innerHTML = '⚠️ Camera stopped';
            document.getElementById('cameraStatus').className = 'pose-status not-detected';
        }
        
        async function sendFrame() {
            const video = document.getElementById('videoElement');
            if (!video || !video.videoWidth) return;
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            await analyzeImage(imageData, true);
        }
        
        document.getElementById('fileInput').onchange = function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    uploadedImageData = event.target.result;
                    const preview = document.getElementById('imagePreview');
                    preview.innerHTML = `<img src="${uploadedImageData}" class="preview-image" alt="Preview">`;
                    document.getElementById('analyzeBtn').style.display = 'block';
                    document.getElementById('poseStatus').innerHTML = '✅ Image uploaded. Click "Analyze Pose"';
                    document.getElementById('poseStatus').className = 'pose-status detected';
                };
                reader.readAsDataURL(file);
            }
        };
        
        async function analyzeUploadedImage() {
            if (uploadedImageData) await analyzeImage(uploadedImageData, false);
        }
        
        async function analyzeImage(imageData, drawSkeletonFlag) {
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: imageData, pose_name: poseName})
                });
                const result = await response.json();
                
                document.getElementById('accuracyValue').innerHTML = result.accuracy + '%';
                const accClass = result.accuracy > 70 ? 'high' : (result.accuracy > 40 ? 'medium' : 'low');
                document.getElementById('accuracyValue').className = 'accuracy-value ' + accClass;
                
                const correctionsList = document.getElementById('correctionsList');
                correctionsList.innerHTML = '';
                if (result.corrections && result.corrections.length > 0) {
                    result.corrections.forEach(c => {
                        const li = document.createElement('li');
                        li.innerHTML = c;
                        correctionsList.appendChild(li);
                    });
                } else {
                    correctionsList.innerHTML = '<li>Great form! Keep practicing</li>';
                }
                
                // Display joint angles
                if (result.user_angles) {
                    const jointsDiv = document.getElementById('jointsDetails');
                    jointsDiv.innerHTML = '<strong>📐 Your Joint Angles:</strong><br>';
                    for (const [joint, angle] of Object.entries(result.user_angles)) {
                        jointsDiv.innerHTML += `<span>${joint}: ${angle}°</span><br>`;
                    }
                }
                
                if (drawSkeletonFlag && result.landmarks && result.landmarks.length > 0) drawSkeleton(result.landmarks);
                document.getElementById('poseStatus').innerHTML = '✅ Analysis complete';
                document.getElementById('poseStatus').className = 'pose-status detected';
            } catch(e) { console.error(e); }
        }
        
        function drawSkeleton(landmarks) {
            const video = document.getElementById('videoElement');
            const canvas = document.getElementById('canvasOverlay');
            const ctx = canvas.getContext('2d');
            canvas.width = video.clientWidth;
            canvas.height = video.clientHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const scaleX = canvas.width / video.videoWidth;
            const scaleY = canvas.height / video.videoHeight;
            const connections = [[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[11,12]];
            ctx.beginPath();
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 3;
            for (const [i1, i2] of connections) {
                if (landmarks[i1] && landmarks[i2]) {
                    ctx.beginPath();
                    ctx.moveTo(landmarks[i1].x * scaleX, landmarks[i1].y * scaleY);
                    ctx.lineTo(landmarks[i2].x * scaleX, landmarks[i2].y * scaleY);
                    ctx.stroke();
                }
            }
            for (let i = 0; i < landmarks.length; i++) {
                if (landmarks[i]) {
                    ctx.beginPath();
                    ctx.arc(landmarks[i].x * scaleX, landmarks[i].y * scaleY, 5, 0, 2 * Math.PI);
                    ctx.fillStyle = '#ff0000';
                    ctx.fill();
                }
            }
        }
        
        function goBack() {
            if (interval) clearInterval(interval);
            if (stream) stream.getTracks().forEach(t => t.stop());
            window.location.href = '/dashboard';
        }
        
        setMode('camera');
    </script>
</body>
</html>
'''

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template_string(LOGIN_HTML)

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    trimester, weeks = calculate_trimester(data.get('lmp_date'))
    session['user'] = {
        'name': data.get('name'),
        'age': data.get('age'),
        'trimester': trimester,
        'weeks': weeks
    }
    return jsonify({'status': 'success'})

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    
    user = session['user']
    trimester = user['trimester']
    
    trimester_names = {
        'FIRST': 'First Trimester (Weeks 1-12)',
        'SECOND': 'Second Trimester (Weeks 13-26)',
        'THIRD': 'Third Trimester (Weeks 27-40)'
    }
    
    poses = []
    for pose in AVAILABLE_POSES:
        safety = get_pose_safety(pose['key'])
        poses.append({
            'key': pose['key'],
            'name': pose['name'],
            'image_url': f"/static_images/{pose['key']}/{pose['image']}",
            'safety_status': safety['status'],
            'safety_color': 'safe' if safety['color'] == 'green' else 'modified',
            'safety_reason': safety['message']
        })
    
    return render_template_string(DASHBOARD_HTML,
        user=user,
        trimester_name=trimester_names[trimester],
        tips=get_trimester_tips(trimester),
        poses=poses
    )

@app.route('/practice/<pose_name>')
def practice(pose_name):
    if 'user' not in session:
        return redirect('/')
    return render_template_string(PRACTICE_HTML, pose_name=pose_name)

@app.route('/static_images/<path:filename>')
def serve_static(filename):
    return send_from_directory('static_images', filename)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    pose_name = data.get('pose_name')
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'accuracy': 0, 'corrections': ['No image']})
    
    try:
        img_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'accuracy': 50, 'corrections': ['Processing...'], 'landmarks': []})
        
        if not MEDIAPIPE_AVAILABLE or pose_detector is None:
            # Demo mode
            accuracy = random.randint(60, 95)
            corrections = ["Keep your back straight"] if accuracy < 80 else ["Great form!"]
            return jsonify({'accuracy': accuracy, 'corrections': corrections, 'landmarks': [], 'user_angles': {}})
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mpImage(image_format=ImageFormat.SRGB, data=rgb)
        result = pose_detector.detect(mp_image)
        
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return jsonify({'accuracy': 0, 'corrections': ['No pose detected - Please stand in front of camera'], 'landmarks': [], 'user_angles': {}})
        
        landmarks = result.pose_landmarks[0]
        landmark_coords = [{'x': lm.x, 'y': lm.y} for lm in landmarks]
        
        # Extract user angles
        user_angles = extract_angles_from_landmarks(landmarks)
        
        # Get reference angles for this specific pose
        ref_angles = REFERENCE_ANGLES.get(pose_name, {})
        if not ref_angles:
            # Try to find by partial match
            for key, angles in REFERENCE_ANGLES.items():
                if pose_name.lower() in key.lower() or key.lower() in pose_name.lower():
                    ref_angles = angles
                    break
        
        # Calculate accuracy and corrections
        accuracy = calculate_pose_accuracy(user_angles, ref_angles)
        corrections = generate_corrections(user_angles, ref_angles)
        
        return jsonify({
            'accuracy': accuracy,
            'corrections': corrections,
            'landmarks': landmark_coords,
            'user_angles': user_angles
        })
        
    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'accuracy': 50, 'corrections': ['Processing...'], 'landmarks': [], 'user_angles': {}})

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    print("=" * 60)
    print("🧘 Prenatal Yoga Pose Correction System")
    print("=" * 60)
    print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"📁 Angles file: {ANGLES_FILE}")
    print(f"📸 Found {len(AVAILABLE_POSES)} poses")
    print(f"📊 Loaded {len(REFERENCE_ANGLES)} reference angle sets")
    print("=" * 60)
    print("✅ Server starting...")
    print("📱 Open your browser to: http://localhost:5000")
    print("🔴 Press Ctrl+C to stop")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
