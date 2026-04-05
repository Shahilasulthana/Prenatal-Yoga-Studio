from flask import Flask, render_template, jsonify, request, session, redirect
import json
import os

app = Flask(__name__)
app.secret_key = 'yoga-secret-key'

# Load pose data
pose_data = {}
json_files = ['yoga_pose_angles.json', 'yoga_pose_angles_complete.json']
for json_file in json_files:
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            pose_data = json.load(f)
        print(f"Loaded {len(pose_data)} poses from {json_file}")
        break

@app.route('/')
def index():
    return '''
    <html>
        <head><title>Prenatal Yoga</title></head>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>🧘 Prenatal Yoga Pose Correction</h1>
            <p>AI-powered pose correction for safe pregnancy yoga</p>
            <div style="margin-top: 30px;">
                <h3>Available Poses:</h3>
                <ul style="list-style: none; padding: 0;">
                    ''' + ''.join([f'<li style="margin: 10px;">✅ {pose}</li>' for pose in list(pose_data.keys())[:10]]) + '''
                </ul>
                <p style="margin-top: 30px;">Server is running successfully!</p>
            </div>
        </body>
    </html>
    '''

@app.route('/api/poses')
def get_poses():
    return jsonify(list(pose_data.keys()))

if __name__ == '__main__':
    print("=" * 50)
    print("Server starting...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
