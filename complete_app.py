from flask import Flask, render_template_string, jsonify, request, session, redirect
import json
import os
import random

app = Flask(__name__)
app.secret_key = 'yoga-prenatal-secret-key-2024'

# Load pose data
pose_data = {}
json_files = ['yoga_pose_angles.json', 'yoga_pose_angles_complete.json']
for json_file in json_files:
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            pose_data = json.load(f)
        print(f"Loaded {len(pose_data)} poses from {json_file}")
        break

# Pregnancy safety guidelines
SAFETY_GUIDELINES = {
    'FIRST': {'tips': ['Avoid deep twists', 'Focus on gentle movements', 'Stay hydrated']},
    'SECOND': {'tips': ['Avoid lying on back', 'Use props', 'Widen stance']},
    'THIRD': {'tips': ['Avoid lying on back', 'Use wall support', 'Gentle movements only']}
}

def calculate_trimester(weeks):
    if weeks <= 13:
        return 'FIRST'
    elif weeks <= 27:
        return 'SECOND'
    else:
        return 'THIRD'

def get_pose_safety(pose_name):
    pose_lower = pose_name.lower()
    restricted = ['camel', 'bow', 'plow', 'shoulderstand', 'headstand']
    for r in restricted:
        if r in pose_lower:
            return {'level': 'danger', 'message': '❌ Not recommended during pregnancy'}
    
    trimester_restricted = ['boat', 'downward', 'cobra']
    for r in trimester_restricted:
        if r in pose_lower:
            return {'level': 'warning', 'message': '⚠️ Only in first trimester'}
    
    return {'level': 'success', 'message': '✅ Safe with modifications'}

# HTML Templates
LOGIN_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Prenatal Yoga - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            width: 90%;
            max-width: 450px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        input {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
        }
        input:focus {
            outline: none;
            border-color: #667eea;
        }
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
        button:hover {
            transform: translateY(-2px);
        }
        .info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 14px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧘 Prenatal Yoga</h1>
        <div class="subtitle">Safe yoga practice during pregnancy</div>
        <form id="loginForm">
            <input type="text" id="name" placeholder="Full Name" required>
            <input type="number" id="age" placeholder="Age" required>
            <input type="number" id="weeks" placeholder="Weeks Pregnant" required min="1" max="40">
            <button type="submit">Start Journey</button>
        </form>
        <div class="info">
            <strong>📋 Note:</strong> Always consult your healthcare provider before starting any exercise routine.
        </div>
    </div>
    <script>
        document.getElementById('loginForm').addEventListener('submit', (e) => {
            e.preventDefault();
            const name = document.getElementById('name').value;
            const age = document.getElementById('age').value;
            const weeks = document.getElementById('weeks').value;
            localStorage.setItem('user', JSON.stringify({name, age, weeks}));
            window.location.href = '/dashboard';
        });
    </script>
</body>
</html>
'''

DASHBOARD_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard - Prenatal Yoga</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
        }
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .container {
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .trimester-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 30px;
        }
        .tips {
            background: #e8f5e9;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .tips h3 { color: #2e7d32; margin-bottom: 15px; }
        .tips li { margin: 10px 0; margin-left: 20px; }
        .poses-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .pose-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.3s;
        }
        .pose-card:hover { transform: translateY(-5px); }
        .pose-header {
            padding: 20px;
            border-bottom: 3px solid;
        }
        .pose-header.success { border-color: #4caf50; }
        .pose-header.warning { border-color: #ff9800; }
        .pose-header.danger { border-color: #f44336; }
        .pose-name { font-size: 18px; font-weight: bold; }
        .safety-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            margin-top: 10px;
        }
        .safety-badge.success { background: #4caf50; color: white; }
        .safety-badge.warning { background: #ff9800; color: white; }
        .safety-badge.danger { background: #f44336; color: white; }
        .pose-message { padding: 15px; font-size: 14px; color: #666; }
        .start-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px;
            width: 100%;
            cursor: pointer;
            font-weight: bold;
        }
        .logout-btn {
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h2>🧘 Prenatal Yoga</h2>
        <div>Welcome, {{ user.name }} | Week {{ user.weeks }} | {{ trimester }} Trimester</div>
        <button class="logout-btn" onclick="logout()">Logout</button>
    </div>
    <div class="container">
        <div class="trimester-card">
            <h2>🌸 {{ trimester }} Trimester</h2>
            <p>Week {{ user.weeks }} of pregnancy</p>
        </div>
        <div class="tips">
            <h3>💡 Tips for {{ trimester }} Trimester</h3>
            <ul>
                {% for tip in tips %}
                <li>{{ tip }}</li>
                {% endfor %}
            </ul>
        </div>
        <h2>Recommended Poses</h2>
        <div class="poses-grid">
            {% for pose in poses %}
            <div class="pose-card" onclick="selectPose('{{ pose.name }}')">
                <div class="pose-header {{ pose.safety.level }}">
                    <div class="pose-name">{{ pose.name.replace('_', ' ') }}</div>
                    <span class="safety-badge {{ pose.safety.level }}">{{ pose.safety.level.upper() }}</span>
                </div>
                <div class="pose-message">{{ pose.safety.message }}</div>
                <button class="start-btn">Start Practice →</button>
            </div>
            {% endfor %}
        </div>
    </div>
    <script>
        function selectPose(poseName) {
            window.location.href = `/pose/${encodeURIComponent(poseName)}`;
        }
        function logout() {
            localStorage.clear();
            window.location.href = '/';
        }
    </script>
</body>
</html>
'''

PRACTICE_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Practice - Prenatal Yoga</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: white;
        }
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .back-btn {
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
        }
        .container {
            display: flex;
            height: calc(100vh - 60px);
        }
        .camera-section {
            flex: 2;
            background: #0f0f1a;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .camera-placeholder {
            text-align: center;
            padding: 40px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
        }
        .feedback-section {
            flex: 1;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
        }
        .pose-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .accuracy-circle {
            text-align: center;
            margin: 30px 0;
        }
        .accuracy-value {
            font-size: 64px;
            font-weight: bold;
            color: #4caf50;
        }
        .corrections-list {
            list-style: none;
            margin-top: 20px;
        }
        .corrections-list li {
            background: rgba(255,255,255,0.1);
            margin: 10px 0;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
        }
        .instruction {
            background: #0f3460;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
            font-weight: bold;
        }
        @media (max-width: 768px) {
            .container { flex-direction: column; }
            .camera-section { height: 50vh; }
            .feedback-section { height: 50vh; }
        }
    </style>
</head>
<body>
    <div class="navbar">
        <button class="back-btn" onclick="goBack()">← Back</button>
        <div>{{ pose_name.replace('_', ' ') }}</div>
        <div style="width: 80px;"></div>
    </div>
    <div class="container">
        <div class="camera-section">
            <div class="camera-placeholder">
                <div style="font-size: 48px;">📷</div>
                <p>Camera simulation mode</p>
                <p style="font-size: 12px; margin-top: 10px;">(Demo mode - Camera would activate here)</p>
            </div>
        </div>
        <div class="feedback-section">
            <div class="pose-title">{{ pose_name.replace('_', ' ') }}</div>
            <div class="instruction">
                <strong>📋 Practice Instructions:</strong><br>
                1. Position yourself properly<br>
                2. Hold the pose<br>
                3. Follow the AI corrections below
            </div>
            <div class="accuracy-circle">
                <div class="accuracy-value" id="accuracy">0%</div>
                <div>Pose Accuracy</div>
            </div>
            <div>
                <strong>💡 AI Corrections:</strong>
                <ul class="corrections-list" id="corrections">
                    <li>Waiting for pose detection...</li>
                </ul>
            </div>
            <button onclick="stopPractice()">Stop Practice</button>
        </div>
    </div>
    <script>
        let interval;
        const referenceAngles = {{ reference_angles | tojson }};
        
        function startSimulation() {
            interval = setInterval(() => {
                const accuracy = Math.floor(Math.random() * 100);
                document.getElementById('accuracy').textContent = accuracy + '%';
                const correctionsList = document.getElementById('corrections');
                correctionsList.innerHTML = '';
                
                if (accuracy > 80) {
                    correctionsList.innerHTML = '<li>✓ Perfect pose! Keep it up!</li>';
                } else if (accuracy > 60) {
                    correctionsList.innerHTML = '<li>⚠️ Good start! Slight adjustments needed</li><li>Straighten your back more</li>';
                } else if (accuracy > 40) {
                    correctionsList.innerHTML = '<li>⚠️ Bend your knees slightly more</li><li>Align your shoulders with hips</li><li>Keep your core engaged</li>';
                } else {
                    correctionsList.innerHTML = '<li>❌ Major adjustments needed</li><li>Check your posture</li><li>Refer to the pose reference</li><li>Start again from the beginning</li>';
                }
            }, 3000);
        }
        
        function stopPractice() {
            if (interval) clearInterval(interval);
            window.location.href = '/dashboard';
        }
        
        function goBack() {
            stopPractice();
        }
        
        startSimulation();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(LOGIN_PAGE)

@app.route('/dashboard')
def dashboard():
    # Get user from localStorage via query parameter
    trimester = 'SECOND'  # Default
    tips = SAFETY_GUIDELINES['SECOND']['tips']
    
    poses = []
    for pose_name in list(pose_data.keys())[:20]:
        safety = get_pose_safety(pose_name)
        poses.append({
            'name': pose_name,
            'safety': safety
        })
    
    user = {'name': 'Guest', 'weeks': 20}
    
    return render_template_string(DASHBOARD_PAGE, user=user, trimester=trimester, tips=tips, poses=poses)

@app.route('/pose/<pose_name>')
def practice_pose(pose_name):
    safety = get_pose_safety(pose_name)
    reference_angles = pose_data.get(pose_name, {}).get('angles', {})
    
    return render_template_string(PRACTICE_PAGE, pose_name=pose_name, safety=safety, reference_angles=reference_angles)

@app.route('/api/poses')
def get_poses():
    return jsonify(list(pose_data.keys()))

@app.route('/api/pose/angles/<pose_name>')
def get_pose_angles(pose_name):
    angles = pose_data.get(pose_name, {}).get('angles', {})
    return jsonify(angles)

if __name__ == '__main__':
    print("=" * 50)
    print("🧘 Prenatal Yoga Pose Correction System")
    print("=" * 50)
    print("Server running at: http://localhost:5000")
    print("Open this URL in your browser")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
