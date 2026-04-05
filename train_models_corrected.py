"""
Train and compare multiple ML models for pose correction - COMPLETE FIXED VERSION
"""

import json
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🧘 YOGA POSE CORRECTION MODEL TRAINING - MULTIPLE MODELS COMPARISON")
print("=" * 80)

# ==================== CONFIGURATION ====================
ANGLES_FILE = "data/yoga_pose_angles.json"

if not os.path.exists(ANGLES_FILE):
    print(f"❌ Error: File not found at {ANGLES_FILE}")
    exit(1)

print(f"\n✅ Found angles file at: {ANGLES_FILE}")

# Create directories
os.makedirs("models/best_models", exist_ok=True)
os.makedirs("models/comparison_results", exist_ok=True)

# Joint order for features
JOINT_ORDER = [
    'left_elbow', 'right_elbow',
    'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'neck'
]

# Joint weights
JOINT_WEIGHTS = {
    'left_knee': 1.5, 'right_knee': 1.5,
    'left_hip': 1.4, 'right_hip': 1.4,
    'left_shoulder': 1.2, 'right_shoulder': 1.2,
    'left_elbow': 0.9, 'right_elbow': 0.9,
    'left_ankle': 0.7, 'right_ankle': 0.7,
    'neck': 0.6
}

# ==================== LOAD DATA ====================
def load_pose_data():
    with open(ANGLES_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 Total poses in file: {len(data)}")
    poses = {}
    for pose_name, pose_info in data.items():
        if 'angles' in pose_info:
            poses[pose_name] = pose_info['angles']
    print(f"   ✓ Loaded {len(poses)} poses")
    return poses

def extract_features(angles):
    features = []
    for joint in JOINT_ORDER:
        val = angles.get(joint, 0)
        features.append(float(val if val else 0))
    return np.array(features)

def calculate_accuracy_score(user_angles, ref_angles):
    if not ref_angles:
        return 50
    total_weight = 0
    weighted_error = 0
    for joint, weight in JOINT_WEIGHTS.items():
        if joint in user_angles and joint in ref_angles:
            if user_angles[joint] and ref_angles[joint]:
                diff = abs(user_angles[joint] - ref_angles[joint])
                normalized_diff = min(diff / 90.0, 1.0)
                weighted_error += normalized_diff * weight
                total_weight += weight
    if total_weight == 0:
        return 50
    accuracy = (1 - (weighted_error / total_weight)) * 100
    return round(max(0, min(100, accuracy)), 2)

def generate_training_data(ref_angles):
    X, y_accuracy = [], []
    variations = list(range(-40, 41, 5))
    for variation in variations:
        varied_angles = {}
        for joint, ref_val in ref_angles.items():
            if ref_val and isinstance(ref_val, (int, float)):
                varied = ref_val + variation + np.random.normal(0, 3)
                varied_angles[joint] = max(0, min(180, varied))
        if varied_angles:
            accuracy = calculate_accuracy_score(varied_angles, ref_angles)
            features = extract_features(varied_angles)
            X.append(features)
            y_accuracy.append(accuracy)
    return np.array(X), np.array(y_accuracy)

# ==================== DEFINE MODELS TO TEST ====================
def get_models():
    """Return dictionary of models (create fresh instances each time)"""
    return {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=1.0, random_state=42),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }

# ==================== TRAIN AND COMPARE MODELS ====================
def train_and_compare_models(pose_name, ref_angles):
    """Train multiple models and return comparison results"""
    print(f"\n📊 Training models for: {pose_name[:50]}...")
    
    X, y = generate_training_data(ref_angles)
    
    if len(X) < 10:
        print(f"   ⚠️ Insufficient data, skipping...")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    best_model_name = None
    best_model_obj = None
    best_scaler_obj = None
    best_r2 = -np.inf
    
    models = get_models()
    
    for model_name, model in models.items():
        try:
            # Create a fresh copy of the model
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train_scaled, y_train)
            y_pred = model_copy.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Store only numeric values (JSON serializable)
            results[model_name] = {
                'r2': round(r2, 4),
                'mae': round(mae, 2),
                'rmse': round(rmse, 2)
            }
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
                best_model_obj = model_copy
                best_scaler_obj = scaler
                
        except Exception as e:
            results[model_name] = {'error': str(e)[:50]}
    
    # Save the best model for this pose
    if best_model_obj is not None:
        safe_name = pose_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
        model_path = f"models/best_models/{safe_name}_model.pkl"
        scaler_path = f"models/best_models/{safe_name}_scaler.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(best_model_obj, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(best_scaler_obj, f)
    
    return {
        'results': results,
        'best_model': best_model_name,
        'best_r2': round(best_r2, 4) if best_r2 != -np.inf else None
    }

# ==================== MAIN TRAINING LOOP ====================
def main():
    print("\n📁 Loading pose data...")
    poses = load_pose_data()
    print(f"\n✅ Loaded {len(poses)} poses")
    
    if len(poses) == 0:
        print("❌ No poses found!")
        return
    
    all_results = {}
    trained_count = 0
    overall_best = {}
    
    for pose_name, ref_angles in poses.items():
        result = train_and_compare_models(pose_name, ref_angles)
        
        if result and result['best_model'] is not None:
            all_results[pose_name] = {
                'best_model': result['best_model'],
                'best_r2': result['best_r2'],
                'all_models': result['results']
            }
            
            if result['best_model'] not in overall_best:
                overall_best[result['best_model']] = 0
            overall_best[result['best_model']] += 1
            
            print(f"   🏆 Best: {result['best_model']} (R²: {result['best_r2']:.4f})")
            trained_count += 1
    
    # Save results to JSON (this will work now since no model objects are stored)
    with open("models/comparison_results/all_models_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Calculate average performance
    print("\n📈 AVERAGE PERFORMANCE ACROSS ALL POSES:")
    print("-" * 80)
    print(f"{'Model':<20} | {'Avg R²':<10} | {'Avg MAE':<10} | {'Times Best':<12}")
    print("-" * 80)
    
    model_performance = {}
    model_names = list(get_models().keys())
    
    for model_name in model_names:
        r2_scores = []
        mae_scores = []
        for pose_name, result in all_results.items():
            if model_name in result['all_models'] and 'error' not in result['all_models'][model_name]:
                r2_scores.append(result['all_models'][model_name]['r2'])
                mae_scores.append(result['all_models'][model_name]['mae'])
        
        if r2_scores:
            avg_r2 = np.mean(r2_scores)
            avg_mae = np.mean(mae_scores)
            times_best = overall_best.get(model_name, 0)
            print(f"{model_name:<20} | {avg_r2:.4f}   | {avg_mae:.2f}%    | {times_best}/{len(all_results)}")
            model_performance[model_name] = {'avg_r2': avg_r2, 'avg_mae': avg_mae, 'times_best': times_best}
    
    print("-" * 80)
    
    # Find best overall model
    if model_performance:
        best_overall = max(model_performance.items(), key=lambda x: x[1]['avg_r2'])
        print(f"\n🏆 BEST OVERALL MODEL: {best_overall[0]}")
        print(f"   Average R²: {best_overall[1]['avg_r2']:.4f}")
        print(f"   Average MAE: {best_overall[1]['avg_mae']:.2f}%")
        print(f"   Best for {best_overall[1]['times_best']}/{len(all_results)} poses")
    
    # Save performance summary to CSV
    perf_data = []
    for model_name, perf in model_performance.items():
        perf_data.append({
            'Model': model_name,
            'Avg R² Score': perf['avg_r2'],
            'Avg MAE (%)': perf['avg_mae'],
            'Times Best': perf['times_best']
        })
    
    df_perf = pd.DataFrame(perf_data)
    df_perf.to_csv("models/comparison_results/performance_summary.csv", index=False)
    
    # Create detailed comparison CSV
    comparison_data = []
    for pose_name, result in all_results.items():
        row = {'Pose': pose_name[:40], 'Best_Model': result['best_model'], 'Best_R2': result['best_r2']}
        for model_name in model_names:
            if model_name in result['all_models'] and 'error' not in result['all_models'][model_name]:
                row[model_name] = result['all_models'][model_name]['r2']
            else:
                row[model_name] = None
        comparison_data.append(row)
    
    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv("models/comparison_results/model_comparison.csv", index=False)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING AND COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"📊 Total poses processed: {trained_count}")
    print(f"📁 Best models saved to: models/best_models/")
    print(f"📁 Results saved to: models/comparison_results/")
    print("\n📄 Files generated:")
    print("   - model_comparison.csv (Detailed per-pose model performance)")
    print("   - performance_summary.csv (Average performance per model)")
    print("   - all_models_comparison.json (Complete results)")
    print("=" * 80)

if __name__ == "__main__":
    main()
