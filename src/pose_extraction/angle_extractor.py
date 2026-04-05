# src/pose_extraction/angle_extractor.py

import os
import json
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class YogaPoseAngleExtractor:
    """
    Extract joint angles from yoga pose images using MediaPipe.
    Creates a reference dataset for pose comparison.
    """
    
    def __init__(self, static_image_mode: bool = True, model_complexity: int = 2):
        """
        Initialize MediaPipe Pose solution.
        
        Args:
            static_image_mode: True for processing static images
            model_complexity: 0=Lite, 1=Full, 2=Heavy (most accurate)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose detection with high accuracy for reference poses
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define all relevant joint angle pairs
        # Format: (name, point1, point2, point3) where point2 is the joint vertex
        self.angle_definitions = {
            # Elbow angles
            'left_elbow_angle': ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            'right_elbow_angle': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
            
            # Shoulder angles
            'left_shoulder_angle': ('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'),
            'right_shoulder_angle': ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP'),
            
            # Hip angles
            'left_hip_angle': ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
            'right_hip_angle': ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
            
            # Knee angles
            'left_knee_angle': ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
            'right_knee_angle': ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
            
            # Ankle angles
            'left_ankle_angle': ('LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_FOOT_INDEX'),
            'right_ankle_angle': ('RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'),
            
            # Shoulder abduction (arm lift)
            'left_arm_abduction': ('LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW'),
            'right_arm_abduction': ('RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            
            # Neck/lateral flexion
            'neck_angle': ('LEFT_SHOULDER', 'NOSE', 'RIGHT_SHOULDER'),
            
            # Spine angle (torso lean)
            'spine_angle': ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
            
            # Wrist angles
            'left_wrist_angle': ('LEFT_ELBOW', 'LEFT_WRIST', 'LEFT_INDEX'),
            'right_wrist_angle': ('RIGHT_ELBOW', 'RIGHT_WRIST', 'RIGHT_INDEX'),
            
            # Additional angles for better pose detection
            'left_hip_knee_ankle': ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
            'right_hip_knee_ankle': ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
            'left_shoulder_elbow_wrist': ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            'right_shoulder_elbow_wrist': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
        }
        
        # Map landmark names to MediaPipe indices
        self.landmark_names = {
            'NOSE': 0,
            'LEFT_EYE_INNER': 1, 'LEFT_EYE': 2, 'LEFT_EYE_OUTER': 3,
            'RIGHT_EYE_INNER': 4, 'RIGHT_EYE': 5, 'RIGHT_EYE_OUTER': 6,
            'LEFT_EAR': 7, 'RIGHT_EAR': 8,
            'MOUTH_LEFT': 9, 'MOUTH_RIGHT': 10,
            'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
            'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
            'LEFT_INDEX': 19, 'RIGHT_INDEX': 20,
            'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24,
            'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
            'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
            'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32,
        }
    
    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """
        Calculate the angle between three points.
        
        Args:
            point1, point2, point3: (x, y) coordinates
            point2 is the vertex of the angle
            
        Returns:
            Angle in degrees (0-180)
        """
        # Convert to numpy arrays
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate cosine of angle
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Handle floating point errors
        
        # Return angle in degrees
        angle = np.arccos(cosine_angle) * 180.0 / np.pi
        return round(angle, 2)
    
    def get_landmark_coordinates(self, landmarks, landmark_name: str) -> Optional[Tuple[float, float]]:
        """
        Extract normalized coordinates for a specific landmark.
        
        Args:
            landmarks: MediaPipe landmarks object
            landmark_name: Name of the landmark to extract
            
        Returns:
            Tuple of (x, y) coordinates or None if not found
        """
        if landmark_name in self.landmark_names:
            idx = self.landmark_names[landmark_name]
            if idx < len(landmarks):
                landmark = landmarks[idx]
                return (landmark.x, landmark.y)
        return None
    
    def extract_angles_from_image(self, image_path: str) -> Optional[Dict]:
        """
        Process a single image and extract all joint angles.
        
        Args:
            image_path: Path to the yoga pose image
            
        Returns:
            Dictionary containing all extracted angles and metadata
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print(f"Warning: No pose detected in {image_path}")
            return None
        
        landmarks = results.pose_landmarks.landmark
        angles = {}
        
        # Calculate each defined angle
        for angle_name, (p1_name, p2_name, p3_name) in self.angle_definitions.items():
            p1 = self.get_landmark_coordinates(landmarks, p1_name)
            p2 = self.get_landmark_coordinates(landmarks, p2_name)
            p3 = self.get_landmark_coordinates(landmarks, p3_name)
            
            if all([p1, p2, p3]):
                angle = self.calculate_angle(p1, p2, p3)
                angles[angle_name] = angle
            else:
                angles[angle_name] = None
        
        # Add all raw landmark coordinates for reference
        landmarks_data = {}
        for name, idx in self.landmark_names.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                landmarks_data[name] = {
                    'x': round(landmark.x, 4),
                    'y': round(landmark.y, 4),
                    'z': round(landmark.z, 4),
                    'visibility': round(landmark.visibility, 4)
                }
        
        return {
            'angles': angles,
            'landmarks': landmarks_data,
            'image_path': image_path,
            'image_name': Path(image_path).stem
        }
    
    def process_dataset(self, dataset_path: str, output_json_path: str):
        """
        Process all images in the dataset and save angles to JSON.
        
        Args:
            dataset_path: Root directory containing pose folders
            output_json_path: Path to save the output JSON file
        """
        all_poses = []
        failed_images = []
        pose_folders = []
        
        # Find all pose folders (subdirectories in train folder)
        train_path = Path(dataset_path)
        
        # Get all subdirectories (each subdirectory is a yoga pose)
        if train_path.exists():
            pose_folders = [f for f in train_path.iterdir() if f.is_dir()]
        else:
            print(f"Error: Path {dataset_path} does not exist!")
            return None
        
        print(f"Found {len(pose_folders)} pose categories to process...")
        
        for pose_folder in pose_folders:
            pose_name = pose_folder.name
            print(f"\n📁 Processing pose: {pose_name}")
            
            # Find all images in this pose folder
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
                image_files.extend(pose_folder.glob(ext))
            
            print(f"   Found {len(image_files)} images")
            
            # Process each image (take the best one, or process all)
            best_result = None
            best_visibility_score = 0
            
            for img_path in image_files:
                result = self.extract_angles_from_image(str(img_path))
                
                if result:
                    # Calculate visibility score (sum of all landmark visibilities)
                    visibility_score = sum(
                        lm['visibility'] for lm in result['landmarks'].values()
                    )
                    
                    if visibility_score > best_visibility_score:
                        best_visibility_score = visibility_score
                        best_result = result
            
            if best_result:
                best_result['pose_name'] = pose_name
                best_result['folder_path'] = str(pose_folder)
                all_poses.append(best_result)
                print(f"   ✅ Successfully extracted angles (visibility: {best_visibility_score:.2f})")
            else:
                failed_images.append(str(pose_folder))
                print(f"   ❌ Failed to detect pose in any image")
        
        # Save results to JSON
        output_data = {
            'metadata': {
                'total_poses_processed': len(all_poses),
                'failed_poses': len(failed_images),
                'angle_definitions': list(self.angle_definitions.keys()),
                'total_angle_types': len(self.angle_definitions)
            },
            'poses': all_poses,
            'failed_poses_list': failed_images
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ Processing complete!")
        print(f"{'='*60}")
        print(f"   Successfully processed: {len(all_poses)} poses")
        print(f"   Failed: {len(failed_images)} poses")
        print(f"   Output saved to: {output_json_path}")
        
        return output_data


class PrenatalSafetyLabeler:
    """
    Label yoga poses based on safety during pregnancy trimesters.
    """
    
    # Safety level definitions
    SAFETY_LEVELS = {
        'FULLY_ALLOWED': 'Safe for all trimesters',
        'TRIMESTER_RESTRICTED': 'Allowed only in specific trimesters',
        'COMPLETELY_RESTRICTED': 'Not safe during pregnancy'
    }
    
    # Pose safety guidelines based on medical recommendations
    POSE_SAFETY = {
        # Fully allowed poses (Safe throughout pregnancy)
        'Akarna_Dhanurasana': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1], 'notes': 'Deep stretch, avoid in later trimesters'},
        'Bharadvajas_Twist_pose': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1,2], 'notes': 'Gentle twist, avoid compression'},
        'Boat_Pose': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1], 'notes': 'Core engagement, modify in 2nd/3rd trimester'},
        'Bound_Angle_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Excellent for hip opening'},
        'Bow_Pose': {'safety': 'COMPLETELY_RESTRICTED', 'allowed_trimesters': [], 'notes': 'Pressure on abdomen'},
        'Bridge_Pose': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1,2], 'notes': 'Use block for support'},
        'Camel_Pose': {'safety': 'COMPLETELY_RESTRICTED', 'allowed_trimesters': [], 'notes': 'Deep backbend compresses abdomen'},
        'Cat_Cow_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Excellent for spinal mobility'},
        'Chair_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Builds leg strength, use wall for balance'},
        'Child_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Widen knees in later trimesters'},
        'Cobra_Pose': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1], 'notes': 'Gentle backbend only in first trimester'},
        'Corpse_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Lie on left side after first trimester'},
        'Cow_Face_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Great for hip opening'},
        'Downward_Facing_Dog': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1,2], 'notes': 'Avoid in 3rd trimester, use wall/chair'},
        'Eagle_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Use wall for balance'},
        'Fish_Pose': {'safety': 'COMPLETELY_RESTRICTED', 'allowed_trimesters': [], 'notes': 'Backbend lying on floor'},
        'Garland_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Excellent squat position for birth prep'},
        'Half_Moon_Pose': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1,2], 'notes': 'Balance challenge, use wall support'},
        'Happy_Baby_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Gentle hip opener'},
        'Legs_Up_the_Wall': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Great for circulation'},
        'Pigeon_Pose': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1,2], 'notes': 'Deep hip opener, modify in 3rd trimester'},
        'Plank_Pose': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1,2], 'notes': 'Avoid full plank in 3rd trimester'},
        'Plow_Pose': {'safety': 'COMPLETELY_RESTRICTED', 'allowed_trimesters': [], 'notes': 'Neck strain and abdominal pressure'},
        'Tree_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Use wall for balance support'},
        'Triangle_Pose': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Great for side body stretch'},
        'Warrior_I': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Builds leg strength, widen stance'},
        'Warrior_II': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Builds leg strength, great for stamina'},
        'Warrior_III': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1], 'notes': 'Balance challenge, avoid in later trimesters'},
        'Wide_Legged_Forward_Bend': {'safety': 'FULLY_ALLOWED', 'allowed_trimesters': [1,2,3], 'notes': 'Keep back flat, avoid rounding'},
        'Wind_Relieving_Pose': {'safety': 'TRIMESTER_RESTRICTED', 'allowed_trimesters': [1,2], 'notes': 'Gentle, avoid compression in 3rd trimester'},
    }
    
    @classmethod
    def label_pose(cls, pose_name: str) -> Dict:
        """
        Get safety label for a specific pose.
        
        Args:
            pose_name: Name of the yoga pose
            
        Returns:
            Dictionary with safety classification
        """
        # Clean pose name for matching
        clean_name = pose_name.replace('_', ' ').strip()
        
        # Try exact match first
        for known_pose, safety_info in cls.POSE_SAFETY.items():
            if known_pose.lower() == clean_name.lower() or known_pose.lower() == pose_name.lower():
                return safety_info.copy()
        
        # Try partial match
        for known_pose, safety_info in cls.POSE_SAFETY.items():
            if known_pose.lower() in clean_name.lower() or clean_name.lower() in known_pose.lower():
                return safety_info.copy()
        
        # Default: Unknown pose - recommend caution
        return {
            'safety': 'TRIMESTER_RESTRICTED',
            'allowed_trimesters': [1],
            'notes': '⚠️ Unknown pose - consult healthcare provider before practicing'
        }
    
    @classmethod
    def create_labeled_dataset(cls, angles_data: Dict) -> Dict:
        """
        Add safety labels to the angle dataset.
        
        Args:
            angles_data: Dictionary containing extracted angles
            
        Returns:
            Dictionary with added safety labels
        """
        labeled_poses = []
        
        for pose_data in angles_data.get('poses', []):
            pose_name = pose_data.get('pose_name', 'Unknown')
            safety_info = cls.label_pose(pose_name)
            
            labeled_pose = pose_data.copy()
            labeled_pose['prenatal_safety'] = safety_info
            labeled_pose['original_pose_name'] = pose_name
            labeled_poses.append(labeled_pose)
        
        angles_data['poses'] = labeled_poses
        angles_data['metadata']['safety_levels'] = cls.SAFETY_LEVELS
        
        return angles_data


def create_angle_summary_csv(labeled_data: Dict, output_csv_path: str):
    """
    Create a CSV summary of all angles for easy viewing.
    
    Args:
        labeled_data: Labeled dataset dictionary
        output_csv_path: Path to save CSV file
    """
    import csv
    
    if not labeled_data or 'poses' not in labeled_data:
        print("No data to create CSV")
        return
    
    # Get all angle keys from first pose
    angle_keys = []
    for pose in labeled_data['poses']:
        if pose.get('angles'):
            angle_keys = list(pose['angles'].keys())
            break
    
    # Prepare CSV data
    csv_data = []
    for pose in labeled_data['poses']:
        row = {
            'pose_name': pose.get('pose_name', 'Unknown'),
            'safety_level': pose.get('prenatal_safety', {}).get('safety', 'UNKNOWN'),
            'allowed_trimesters': str(pose.get('prenatal_safety', {}).get('allowed_trimesters', [])),
            'notes': pose.get('prenatal_safety', {}).get('notes', '')
        }
        
        # Add all angles
        angles = pose.get('angles', {})
        for angle_key in angle_keys:
            row[angle_key] = angles.get(angle_key, 'N/A')
        
        csv_data.append(row)
    
    # Write CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        if csv_data:
            fieldnames = list(csv_data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
    
    print(f"✅ CSV summary saved to: {output_csv_path}")


def visualize_single_pose(image_path: str, output_path: str = None):
    """
    Test function to visualize angles on a single image.
    
    Args:
        image_path: Path to the image
        output_path: Path to save visualization
    """
    extractor = YogaPoseAngleExtractor()
    result = extractor.extract_angles_from_image(image_path)
    
    if result:
        print("\n📊 Extracted Angles:")
        print("-" * 40)
        for angle_name, angle_value in result['angles'].items():
            if angle_value:
                print(f"   {angle_name}: {angle_value}°")
        
        # Load and display image
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Draw landmarks on image
        for name, coords in result['landmarks'].items():
            x = int(coords['x'] * w)
            y = int(coords['y'] * h)
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(image, name.split('_')[-1][:3], (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Display angles on image
        y_offset = 30
        for angle_name, angle_value in list(result['angles'].items())[:8]:  # Show first 8 angles
            if angle_value:
                text = f"{angle_name.replace('_', ' ')[:20]}: {angle_value}°"
                cv2.putText(image, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_offset += 20
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"\n✅ Visualization saved to: {output_path}")
        else:
            cv2.imshow('Pose Visualization', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("❌ Failed to extract pose from image")


def main():
    """
    Main execution function with proper project paths.
    """
    # Get the project root (assuming this file is in src/pose_extraction/)
    project_root = Path(__file__).parent.parent.parent
    
    # Define paths relative to project root
    DATASET_PATH = project_root / "dataset" / "train"
    DATA_DIR = project_root / "data" / "reference_angles"
    OUTPUTS_DIR = project_root / "outputs"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Output file paths
    ANGLES_JSON = DATA_DIR / "yoga_pose_angles_dataset.json"
    LABELED_JSON = DATA_DIR / "yoga_pose_angles_labeled.json"
    CSV_SUMMARY = DATA_DIR / "yoga_pose_angles_summary.csv"
    
    # Check if dataset exists
    if not DATASET_PATH.exists():
        print(f"❌ ERROR: Dataset not found at {DATASET_PATH}")
        print("\nPlease make sure your dataset is in the correct location:")
        print(f"   Expected: {DATASET_PATH}")
        print("\nYour current dataset is at:")
        print(r"   C:\Users\SHAHILA SULTHANA\OneDrive\Documents\cv_project\archive (10)\train")
        print("\nOption 1: Move your dataset to the expected location")
        print("Option 2: Update the DATASET_PATH in this file")
        
        # Ask if user wants to use the existing location
        use_existing = input("\nUse your existing dataset location? (yes/no): ")
        if use_existing.lower() == 'yes':
            existing_path = r"C:\Users\SHAHILA SULTHANA\OneDrive\Documents\cv_project\archive (10)\train"
            DATASET_PATH = Path(existing_path)
            print(f"Using existing dataset at: {DATASET_PATH}")
        else:
            return
    
    print("=" * 60)
    print("YOGA POSE ANGLE EXTRACTION SYSTEM")
    print("=" * 60)
    print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"📁 Output path: {DATA_DIR}")
    print(f"📁 Visualizations: {OUTPUTS_DIR}")
    
    # Test with first image found
    first_pose_folder = next(DATASET_PATH.iterdir(), None)
    if first_pose_folder and first_pose_folder.is_dir():
        first_image = next(first_pose_folder.glob("*.jpg"), None)
        if first_image:
            print(f"\n🧪 Testing with: {first_image.name}")
            visualize_single_pose(str(first_image), str(OUTPUTS_DIR / "test_visualization.jpg"))
    
    # Ask user to continue
    print("\n" + "=" * 60)
    response = input("Process entire dataset? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Exiting. Only test image processed.")
        return
    
    # Initialize and process
    extractor = YogaPoseAngleExtractor(static_image_mode=True, model_complexity=2)
    
    print("\n" + "=" * 60)
    print("STEP 1: Extracting Angles from Dataset")
    print("=" * 60)
    
    angles_data = extractor.process_dataset(str(DATASET_PATH), str(ANGLES_JSON))
    
    if angles_data is None:
        print("Failed to process dataset.")
        return
    
    print("\n" + "=" * 60)
    print("STEP 2: Adding Prenatal Safety Labels")
    print("=" * 60)
    
    labeled_data = PrenatalSafetyLabeler.create_labeled_dataset(angles_data)
    
    # Save labeled dataset
    with open(LABELED_JSON, 'w') as f:
        json.dump(labeled_data, f, indent=2)
    
    # Create CSV summary
    create_angle_summary_csv(labeled_data, str(CSV_SUMMARY))
    
    # Save processing report
    report = {
        'processing_date': datetime.now().isoformat(),
        'total_poses_processed': len(labeled_data.get('poses', [])),
        'output_files': {
            'angles_json': str(ANGLES_JSON),
            'labeled_json': str(LABELED_JSON),
            'csv_summary': str(CSV_SUMMARY)
        }
    }
    
    with open(DATA_DIR / "processing_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\n📄 Output files created in: {DATA_DIR}")
    print(f"   1. yoga_pose_angles_dataset.json - Raw angles")
    print(f"   2. yoga_pose_angles_labeled.json - With safety labels")
    print(f"   3. yoga_pose_angles_summary.csv - Excel readable")
    print(f"   4. processing_report.json - Processing metadata")
    
    # Display summary
    safety_counts = {'FULLY_ALLOWED': 0, 'TRIMESTER_RESTRICTED': 0, 'COMPLETELY_RESTRICTED': 0}
    trimester_breakdown = {1: 0, 2: 0, 3: 0}
    
    for pose in labeled_data.get('poses', []):
        safety = pose.get('prenatal_safety', {}).get('safety', 'UNKNOWN')
        if safety in safety_counts:
            safety_counts[safety] += 1
        
        allowed = pose.get('prenatal_safety', {}).get('allowed_trimesters', [])
        for t in allowed:
            if t in trimester_breakdown:
                trimester_breakdown[t] += 1
    
    print(f"\n📊 Safety Summary:")
    print(f"   ✅ Fully Allowed: {safety_counts['FULLY_ALLOWED']} poses")
    print(f"   ⚠️ Trimester Restricted: {safety_counts['TRIMESTER_RESTRICTED']} poses")
    print(f"   ❌ Completely Restricted: {safety_counts['COMPLETELY_RESTRICTED']} poses")
    
    print(f"\n📅 Poses Allowed by Trimester:")
    for trimester in [1, 2, 3]:
        print(f"   Trimester {trimester}: {trimester_breakdown[trimester]} poses")


if __name__ == "__main__":
    main()