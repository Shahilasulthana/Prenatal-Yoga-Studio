import cv2
import json
import numpy as np
from pathlib import Path
import urllib.request
import os
import sys

# MediaPipe imports
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import Image as mpImage
    from mediapipe import ImageFormat
    print("✓ MediaPipe Tasks API imported")
except ImportError as e:
    print(f"Error importing MediaPipe: {e}")
    sys.exit(1)

def download_pose_model():
    """Download the pose landmarker model if not exists."""
    model_path = "pose_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading pose model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"✓ Model downloaded")
    return model_path

class PoseVisualizer:
    def __init__(self):
        """Initialize the pose landmarker."""
        print("Initializing pose detector...")
        model_path = download_pose_model()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.detector = vision.PoseLandmarker.create_from_options(options)
        print("✓ Pose detector ready")
    
    # Landmark connections for drawing skeleton
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Face
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),  # Ears
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Hips
        (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
    ]
    
    # Landmark names for labeling
    LANDMARK_NAMES = {
        0: "Nose", 1: "LEye", 2: "REye", 3: "LEar", 4: "REar",
        5: "LMouth", 6: "RMouth", 7: "LShoulder", 8: "RShoulder",
        9: "LElbow", 10: "RElbow", 11: "LWrist", 12: "RWrist",
        13: "LPinky", 14: "RPinky", 15: "LIndex", 16: "RIndex",
        17: "LThumb", 18: "RThumb", 19: "LHip", 20: "RHip",
        21: "LKnee", 22: "RKnee", 23: "LAnkle", 24: "RAnkle",
        25: "LHeel", 26: "RHeel", 27: "LFoot", 28: "RFoot"
    }
    
    def visualize_pose(self, image_path, output_path, draw_labels=True):
        """Detect pose and draw landmarks on image."""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Error reading image: {image_path}")
            return False
        
        original_h, original_w = image.shape[:2]
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mpImage(image_format=ImageFormat.SRGB, data=rgb_image)
        
        # Detect pose landmarks
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.pose_landmarks:
            print(f"  No pose detected")
            return False
        
        landmarks = detection_result.pose_landmarks[0]
        
        # Create a copy for drawing
        output_image = image.copy()
        
        # Convert normalized coordinates to pixel coordinates
        points = []
        for i in range(len(landmarks)):
            x = int(landmarks[i].x * original_w)
            y = int(landmarks[i].y * original_h)
            points.append((x, y))
        
        # Draw connections (skeleton)
        for connection in self.POSE_CONNECTIONS:
            if connection[0] < len(points) and connection[1] < len(points):
                pt1 = points[connection[0]]
                pt2 = points[connection[1]]
                cv2.line(output_image, pt1, pt2, (0, 255, 0), 2)
        
        # Draw landmarks as circles
        for i, (x, y) in enumerate(points):
            # Different colors for different body parts
            if i in [7, 8, 9, 10, 11, 12]:  # Arms
                color = (255, 0, 0)  # Blue
            elif i in [19, 20, 21, 22, 23, 24]:  # Legs
                color = (0, 0, 255)  # Red
            else:  # Torso/Head
                color = (0, 255, 255)  # Yellow
            
            cv2.circle(output_image, (x, y), 5, color, -1)
            
            # Draw labels
            if draw_labels and i in self.LANDMARK_NAMES:
                label = self.LANDMARK_NAMES[i][:3]  # First 3 letters
                cv2.putText(output_image, label, (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add pose name as title
        pose_name = Path(image_path).parent.name
        cv2.putText(output_image, pose_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add landmark count
        cv2.putText(output_image, f"Landmarks: {len(points)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Save the image
        cv2.imwrite(output_path, output_image)
        return True

def main():
    """Main function to visualize all poses."""
    # Paths
    dataset_path = Path(r"C:\Users\SHAHILA SULTHANA\OneDrive\Documents\cv_project\dataset\train")
    output_dir = Path(r"C:\Users\SHAHILA SULTHANA\OneDrive\Documents\cv_project\labelled_images")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("POSE LANDMARK VISUALIZATION")
    print("=" * 60)
    print(f"📁 Dataset: {dataset_path}")
    print(f"📁 Output: {output_dir}")
    
    # Initialize visualizer
    visualizer = PoseVisualizer()
    
    # Find all pose folders
    pose_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    print(f"\n📊 Found {len(pose_folders)} pose categories\n")
    
    successful = 0
    failed = []
    
    for i, pose_folder in enumerate(pose_folders, 1):
        print(f"[{i}/{len(pose_folders)}] Processing: {pose_folder.name}")
        
        # Find images in folder
        images = list(pose_folder.glob("*.jpg")) + list(pose_folder.glob("*.png")) + list(pose_folder.glob("*.jpeg"))
        
        if not images:
            print(f"  ⚠️ No images found")
            failed.append(pose_folder.name)
            continue
        
        # Use first image (or find best one)
        img_path = images[0]
        
        # Output path
        output_path = output_dir / f"{pose_folder.name}_landmarked.jpg"
        
        # Visualize
        if visualizer.visualize_pose(str(img_path), str(output_path), draw_labels=True):
            print(f"  ✅ Saved to: {output_path.name}")
            successful += 1
        else:
            print(f"  ❌ No pose detected")
            failed.append(pose_folder.name)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print(f"   Successfully visualized: {successful}/{len(pose_folders)} poses")
    
    if failed:
        print(f"\n⚠️ Failed poses ({len(failed)}):")
        for f in failed[:10]:
            print(f"   - {f}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")
    
    # Create an HTML gallery
    create_gallery(output_dir)

def create_gallery(output_dir):
    """Create an HTML gallery of all visualized images."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Yoga Pose Landmark Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        h1 { color: #333; text-align: center; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
        .card { background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.3s; }
        .card:hover { transform: scale(1.05); }
        .card img { width: 100%; height: 250px; object-fit: cover; }
        .card h3 { padding: 10px; margin: 0; text-align: center; font-size: 14px; color: #555; }
        .stats { text-align: center; margin: 20px; padding: 10px; background: #333; color: white; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🧘 Yoga Pose Landmark Visualization</h1>
    <div class="stats">
        Total Poses: """ + str(len(list(output_dir.glob("*.jpg")))) + """
    </div>
    <div class="gallery">
"""
    
    # Add each image to gallery
    for img_path in sorted(output_dir.glob("*.jpg")):
        pose_name = img_path.stem.replace("_landmarked", "")
        html_content += f"""
        <div class="card">
            <img src="{img_path.name}" alt="{pose_name}">
            <h3>{pose_name}</h3>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Save HTML file
    html_path = output_dir / "gallery.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n📁 Gallery created: {html_path}")
    print(f"   Open this file in your browser to view all images")

if __name__ == "__main__":
    main()
