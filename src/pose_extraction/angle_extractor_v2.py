# src/pose_extraction/angle_extractor_v2.py
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import mediapipe correctly for newer version
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class YogaPoseAngleExtractor:
    """
    Extract joint angles from yoga pose images using MediaPipe.
    Compatible with MediaPipe 0.10.33+
    """
    
    def __init__(self, static_image_mode: bool = True, model_complexity: int = 2):
        """
        Initialize MediaPipe Pose solution.
        """
        # Initialize pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define angle pairs
        self.angle_definitions = {
            'left_elbow_angle': ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            'right_elbow_angle': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
            'left_shoulder_angle': ('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'),
            'right_shoulder_angle': ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP'),
            'left_hip_angle': ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
            'right_hip_angle': ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
            'left_knee_angle': ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
            'right_knee_angle': ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
            'left_ankle_angle': ('LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_FOOT_INDEX'),
            'right_ankle_angle': ('RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'),
            'left_arm_abduction': ('LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW'),
            'right_arm_abduction': ('RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            'neck_angle': ('LEFT_SHOULDER', 'NOSE', 'RIGHT_SHOULDER'),
            'spine_angle': ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
        }
        
        # Map landmark names to indices
        self.landmark_names = {
            'NOSE': 0,
            'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24,
            'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
            'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32,
            'LEFT_INDEX': 19, 'RIGHT_INDEX': 20,
        }
    
    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """Calculate angle between three points."""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle) * 180.0 / np.pi
        return round(angle, 2)
    
    def get_landmark_coordinates(self, landmarks, landmark_name: str) -> Optional[Tuple[float, float]]:
        """Extract coordinates for a specific landmark."""
        if landmark_name in self.landmark_names:
            idx = self.landmark_names[landmark_name]
            if idx < len(landmarks):
                landmark = landmarks[idx]
                return (landmark.x, landmark.y)
        return None
    
    def extract_angles_from_image(self, image_path: str) -> Optional[Dict]:
        """Process a single image and extract all joint angles."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print(f"Warning: No pose detected in {image_path}")
            return None
        
        landmarks = results.pose_landmarks.landmark
        angles = {}
        
        for angle_name, (p1_name, p2_name, p3_name) in self.angle_definitions.items():
            p1 = self.get_landmark_coordinates(landmarks, p1_name)
            p2 = self.get_landmark_coordinates(landmarks, p2_name)
            p3 = self.get_landmark_coordinates(landmarks, p3_name)
            
            if all([p1, p2, p3]):
                angle = self.calculate_angle(p1, p2, p3)
                angles[angle_name] = angle
            else:
                angles[angle_name] = None
        
        return {
            'angles': angles,
            'image_path': image_path,
            'image_name': Path(image_path).stem
        }
    
    def process_dataset(self, dataset_path: str, output_json_path: str):
        """Process all images in the dataset."""
        all_poses = []
        failed_images = []
        
        train_path = Path(dataset_path)
        if not train_path.exists():
            print(f"Error: Path {dataset_path} does not exist!")
            return None
        
        pose_folders = [f for f in train_path.iterdir() if f.is_dir()]
        print(f"Found {len(pose_folders)} pose categories...")
        
        for pose_folder in pose_folders:
            pose_name = pose_folder.name
            print(f"\n📁 Processing: {pose_name}")
            
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
                image_files.extend(pose_folder.glob(ext))
            
            print(f"   Found {len(image_files)} images")
            
            best_result = None
            best_visibility = 0
            
            for img_path in image_files:
                result = self.extract_angles_from_image(str(img_path))
                if result:
                    visibility = sum(1 for v in result['angles'].values() if v is not None)
                    if visibility > best_visibility:
                        best_visibility = visibility
                        best_result = result
            
            if best_result:
                best_result['pose_name'] = pose_name
                all_poses.append(best_result)
                print(f"   ✅ Extracted {best_visibility} angles")
            else:
                failed_images.append(str(pose_folder))
                print(f"   ❌ Failed")
        
        output_data = {
            'metadata': {
                'total_poses_processed': len(all_poses),
                'failed_poses': len(failed_images),
                'angle_definitions': list(self.angle_definitions.keys())
            },
            'poses': all_poses,
            'failed_poses_list': failed_images
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return output_data


def main():
    """Main execution function."""
    # Update this path to your actual dataset location
    DATASET_PATH = r"C:\Users\SHAHILA SULTHANA\OneDrive\Documents\cv_project\archive (10)\train"
    OUTPUT_DIR = Path(r"C:\Users\SHAHILA SULTHANA\OneDrive\Documents\cv_project\outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    OUTPUT_JSON = OUTPUT_DIR / "yoga_pose_angles.json"
    
    print("=" * 60)
    print("YOGA POSE ANGLE EXTRACTION SYSTEM")
    print("=" * 60)
    print(f"📁 Dataset: {DATASET_PATH}")
    print(f"📁 Output: {OUTPUT_JSON}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found!")
        return
    
    extractor = YogaPoseAngleExtractor()
    angles_data = extractor.process_dataset(DATASET_PATH, str(OUTPUT_JSON))
    
    if angles_data:
        print("\n" + "=" * 60)
        print("✅ PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"\n📄 Output saved to: {OUTPUT_JSON}")
        print(f"   Processed: {len(angles_data['poses'])} poses")
        print(f"   Failed: {len(angles_data['failed_poses_list'])} poses")

if __name__ == "__main__":
    main()