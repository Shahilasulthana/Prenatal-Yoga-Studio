# extract_angles_tasks_api.py - Works with MediaPipe 0.10.33 Tasks API
import cv2
import json
import numpy as np
from pathlib import Path
import urllib.request
import os

# MediaPipe Tasks API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def download_pose_model():
    """Download the pose landmarker model if not exists."""
    model_path = "pose_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading pose model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"Model downloaded to {model_path}")
    return model_path

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.arccos(cosine) * 180.0 / np.pi
    return round(angle, 2)

class YogaPoseExtractor:
    def __init__(self):
        """Initialize the pose landmarker."""
        model_path = download_pose_model()
        
        # Create options for the pose landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1
        )
        
        # Create the landmarker
        self.detector = vision.PoseLandmarker.create_from_options(options)
    
    def extract_angles(self, image_path):
        """Extract pose angles from an image."""
        # Read and convert image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect pose landmarks
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.pose_landmarks:
            return None
        
        # Get landmarks for the first pose
        landmarks = detection_result.pose_landmarks[0]
        
        # Extract key angles
        angles = {}
        
        # Helper to get coordinates
        def get_landmark(idx):
            return (landmarks[idx].x, landmarks[idx].y)
        
        # Elbow angles
        angles['left_elbow'] = calculate_angle(
            get_landmark(11), get_landmark(13), get_landmark(15)
        )
        angles['right_elbow'] = calculate_angle(
            get_landmark(12), get_landmark(14), get_landmark(16)
        )
        
        # Shoulder angles
        angles['left_shoulder'] = calculate_angle(
            get_landmark(13), get_landmark(11), get_landmark(23)
        )
        angles['right_shoulder'] = calculate_angle(
            get_landmark(14), get_landmark(12), get_landmark(24)
        )
        
        # Hip angles
        angles['left_hip'] = calculate_angle(
            get_landmark(11), get_landmark(23), get_landmark(25)
        )
        angles['right_hip'] = calculate_angle(
            get_landmark(12), get_landmark(24), get_landmark(26)
        )
        
        # Knee angles
        angles['left_knee'] = calculate_angle(
            get_landmark(23), get_landmark(25), get_landmark(27)
        )
        angles['right_knee'] = calculate_angle(
            get_landmark(24), get_landmark(26), get_landmark(28)
        )
        
        # Ankle angles
        angles['left_ankle'] = calculate_angle(
            get_landmark(25), get_landmark(27), get_landmark(31)
        )
        angles['right_ankle'] = calculate_angle(
            get_landmark(26), get_landmark(28), get_landmark(32)
        )
        
        # Neck angle
        angles['neck'] = calculate_angle(
            get_landmark(11), get_landmark(0), get_landmark(12)
        )
        
        return angles

def main():
    """Main function to process all yoga poses."""
    # Dataset path
    dataset_path = Path(r"C:\Users\SHAHILA SULTHANA\OneDrive\Documents\cv_project\dataset\train")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    print("=" * 60)
    print("YOGA POSE ANGLE EXTRACTION SYSTEM")
    print("=" * 60)
    print(f"📁 Dataset: {dataset_path}")
    
    # Initialize extractor
    print("\n🔧 Initializing pose detector...")
    extractor = YogaPoseExtractor()
    print("✅ Pose detector ready!\n")
    
    # Find all pose folders
    pose_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    print(f"Found {len(pose_folders)} pose categories\n")
    
    results = {}
    failed = []
    
    for i, pose_folder in enumerate(pose_folders, 1):
        print(f"[{i}/{len(pose_folders)}] Processing: {pose_folder.name}")
        
        # Find images in folder
        images = list(pose_folder.glob("*.jpg")) + list(pose_folder.glob("*.png")) + list(pose_folder.glob("*.jpeg"))
        
        if not images:
            print(f"  ⚠️ No images found")
            failed.append(pose_folder.name)
            continue
        
        # Try each image until we find one with a pose
        angles = None
        for img_path in images:
            angles = extractor.extract_angles(str(img_path))
            if angles:
                print(f"  ✅ Success using: {img_path.name}")
                break
        
        if angles:
            results[pose_folder.name] = {
                'angles': angles,
                'image_used': str(img_path),
                'status': 'success'
            }
        else:
            print(f"  ❌ No pose detected in any image")
            failed.append(pose_folder.name)
            results[pose_folder.name] = {'status': 'failed', 'error': 'No pose detected'}
    
    # Save results
    output_file = "yoga_pose_angles_complete.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📄 Results saved to: {output_file}")
    print(f"   Successfully processed: {len(results) - len(failed)} poses")
    print(f"   Failed: {len(failed)} poses")
    
    if failed:
        print(f"\n⚠️ Failed poses: {', '.join(failed[:10])}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")
    
    # Show sample angles
    print("\n📊 Sample angles from first successful pose:")
    for pose_name, data in results.items():
        if data.get('status') == 'success':
            print(f"\n  Pose: {pose_name}")
            angles = data['angles']
            for angle_name, value in list(angles.items())[:5]:
                print(f"    {angle_name}: {value}°")
            break

if __name__ == "__main__":
    main()