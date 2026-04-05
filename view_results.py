"""
Model Performance Visualization and Analysis
Run this after training to see detailed results
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os

print("=" * 80)
print("📊 YOGA POSE MODEL PERFORMANCE ANALYSIS")
print("=" * 80)

# ==================== LOAD RESULTS ====================
def load_results():
    """Load all comparison results"""
    results_dir = "models/comparison_results"
    
    if not os.path.exists(results_dir):
        print("❌ No results found! Please run train_models_corrected.py first")
        return None, None, None
    
    # Load performance summary
    perf_file = os.path.join(results_dir, "performance_summary.csv")
    comp_file = os.path.join(results_dir, "model_comparison.csv")
    json_file = os.path.join(results_dir, "all_models_comparison.json")
    
    df_perf = pd.read_csv(perf_file) if os.path.exists(perf_file) else None
    df_comp = pd.read_csv(comp_file) if os.path.exists(comp_file) else None
    json_results = json.load(open(json_file)) if os.path.exists(json_file) else None
    
    return df_perf, df_comp, json_results

# ==================== DISPLAY SUMMARY ====================
def display_summary(df_perf, df_comp, json_results):
    """Display performance summary"""
    
    print("\n" + "=" * 80)
    print("📈 MODEL PERFORMANCE RANKING")
    print("=" * 80)
    
    if df_perf is not None:
        df_sorted = df_perf.sort_values('Avg R² Score', ascending=False)
        
        print(f"\n{'Rank':<6} {'Model':<22} {'Avg R²':<12} {'Avg MAE':<12} {'Times Best':<12}")
        print("-" * 70)
        
        for i, row in df_sorted.iterrows():
            rank = i + 1
            medal = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{rank}{medal:<4} {row['Model']:<22} {row['Avg R² Score']:.4f}     {row['Avg MAE (%)']:.2f}%      {row['Times Best']}")
        
        print("-" * 70)
        
        # Best model recommendation
        best_model = df_sorted.iloc[0]
        print(f"\n🏆 BEST OVERALL MODEL: {best_model['Model']}")
        print(f"   • Average R² Score: {best_model['Avg R² Score']:.4f}")
        print(f"   • Average MAE: {best_model['Avg MAE (%)']:.2f}%")
        print(f"   • Best for {best_model['Times Best']} poses")
        
        # Second best
        if len(df_sorted) > 1:
            second = df_sorted.iloc[1]
            print(f"\n📌 RECOMMENDED ALTERNATIVE: {second['Model']}")
            print(f"   • Average R² Score: {second['Avg R² Score']:.4f}")
            print(f"   • Average MAE: {second['Avg MAE (%)']:.2f}%")

# ==================== TOP PERFORMING POSES ====================
def display_top_poses(json_results):
    """Display top performing poses"""
    if json_results is None:
        return
    
    print("\n" + "=" * 80)
    print("🌟 TOP 15 BEST PERFORMING POSES (Highest R²)")
    print("=" * 80)
    
    pose_performance = []
    for pose, data in json_results.items():
        if data['best_r2'] is not None:
            pose_performance.append({
                'Pose': pose,
                'Best_Model': data['best_model'],
                'Best_R2': data['best_r2']
            })
    
    top_poses = sorted(pose_performance, key=lambda x: x['Best_R2'], reverse=True)[:15]
    
    print(f"\n{'Rank':<6} {'Pose':<45} {'Model':<20} {'R² Score':<10}")
    print("-" * 85)
    
    for i, pose in enumerate(top_poses, 1):
        pose_name = pose['Pose'][:42] + "..." if len(pose['Pose']) > 42 else pose['Pose']
        print(f"{i:<6} {pose_name:<45} {pose['Best_Model']:<20} {pose['Best_R2']:.4f}")

# ==================== BOTTOM PERFORMING POSES ====================
def display_bottom_poses(json_results):
    """Display poses that need improvement"""
    if json_results is None:
        return
    
    print("\n" + "=" * 80)
    print("⚠️ POSES NEEDING IMPROVEMENT (Lowest R²)")
    print("=" * 80)
    
    pose_performance = []
    for pose, data in json_results.items():
        if data['best_r2'] is not None:
            pose_performance.append({
                'Pose': pose,
                'Best_Model': data['best_model'],
                'Best_R2': data['best_r2']
            })
    
    bottom_poses = sorted(pose_performance, key=lambda x: x['Best_R2'])[:10]
    
    print(f"\n{'Rank':<6} {'Pose':<45} {'Model':<20} {'R² Score':<10}")
    print("-" * 85)
    
    for i, pose in enumerate(bottom_poses, 1):
        pose_name = pose['Pose'][:42] + "..." if len(pose['Pose']) > 42 else pose['Pose']
        print(f"{i:<6} {pose_name:<45} {pose['Best_Model']:<20} {pose['Best_R2']:.4f}")

# ==================== MODEL PERFORMANCE BY POSE ====================
def display_model_by_pose(df_comp):
    """Show which model works best for each pose category"""
    if df_comp is None:
        return
    
    print("\n" + "=" * 80)
    print("📊 MODEL DISTRIBUTION BY POSE")
    print("=" * 80)
    
    # Count best model per pose
    model_counts = df_comp['Best_Model'].value_counts()
    
    print(f"\n{'Model':<25} {'Number of Poses':<20} {'Percentage':<10}")
    print("-" * 55)
    
    total = len(df_comp)
    for model, count in model_counts.items():
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 2)
        print(f"{model:<25} {count:<20} {percentage:.1f}% {bar}")

# ==================== CREATE VISUALIZATION CHARTS ====================
def create_charts(df_perf, json_results):
    """Create visualization charts"""
    
    if df_perf is None:
        return
    
    print("\n" + "=" * 80)
    print("📊 CREATING VISUALIZATION CHARTS")
    print("=" * 80)
    
    # Create charts directory
    os.makedirs("models/comparison_results/charts", exist_ok=True)
    
    # Chart 1: Model Performance Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart for R² scores
    df_sorted = df_perf.sort_values('Avg R² Score', ascending=True)
    colors = ['#4CAF50' if i == len(df_sorted)-1 else '#2196F3' for i in range(len(df_sorted))]
    
    axes[0].barh(df_sorted['Model'], df_sorted['Avg R² Score'], color=colors)
    axes[0].set_xlabel('Average R² Score')
    axes[0].set_title('Model Performance by R² Score')
    axes[0].axvline(x=df_sorted['Avg R² Score'].mean(), color='red', linestyle='--', label='Average')
    axes[0].legend()
    
    # Bar chart for MAE
    df_sorted_mae = df_perf.sort_values('Avg MAE (%)', ascending=True)
    colors_mae = ['#4CAF50' if i == 0 else '#FF9800' for i in range(len(df_sorted_mae))]
    
    axes[1].barh(df_sorted_mae['Model'], df_sorted_mae['Avg MAE (%)'], color=colors_mae)
    axes[1].set_xlabel('Average MAE (%)')
    axes[1].set_title('Model Performance by MAE (lower is better)')
    axes[1].axvline(x=df_sorted_mae['Avg MAE (%)'].mean(), color='red', linestyle='--', label='Average')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('models/comparison_results/charts/model_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Chart 1: model_performance.png")
    
    # Chart 2: Times Best Model Pie Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    model_counts = df_perf[df_perf['Times Best'] > 0].sort_values('Times Best', ascending=False)
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(model_counts)))
    
    wedges, texts, autotexts = ax.pie(
        model_counts['Times Best'], 
        labels=model_counts['Model'], 
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90
    )
    ax.set_title('Models That Performed Best for Each Pose')
    
    plt.tight_layout()
    plt.savefig('models/comparison_results/charts/best_model_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Chart 2: best_model_distribution.png")
    
    # Chart 3: R² Distribution Histogram
    if json_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        r2_values = [data['best_r2'] for data in json_results.values() if data['best_r2'] is not None]
        
        ax.hist(r2_values, bins=20, color='#4CAF50', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(r2_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r2_values):.3f}')
        ax.axvline(x=np.median(r2_values), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(r2_values):.3f}')
        ax.set_xlabel('R² Score')
        ax.set_ylabel('Number of Poses')
        ax.set_title('Distribution of Best R² Scores Across All Poses')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('models/comparison_results/charts/r2_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Chart 3: r2_distribution.png")
    
    print("\n   📁 Charts saved to: models/comparison_results/charts/")

# ==================== GENERATE HTML REPORT ====================
def generate_html_report(df_perf, json_results):
    """Generate an HTML report"""
    
    if df_perf is None:
        return
    
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Yoga Pose Model Performance Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        h2 { color: #555; margin-top: 30px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        tr:hover { background: #f5f5f5; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 5px; font-size: 12px; font-weight: bold; }
        .badge-high { background: #4caf50; color: white; }
        .badge-medium { background: #ff9800; color: white; }
        .badge-low { background: #f44336; color: white; }
        .chart-container { text-align: center; margin: 30px 0; }
        .chart-container img { max-width: 100%; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #888; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧘 Yoga Pose Correction Model Performance Report</h1>
        
        <h2>📈 Model Performance Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Avg R² Score</th>
                    <th>Avg MAE (%)</th>
                    <th>Times Best</th>
                </tr>
            </thead>
            <tbody>
'''
    
    df_sorted = df_perf.sort_values('Avg R² Score', ascending=False)
    for i, row in df_sorted.iterrows():
        rank = i + 1
        r2_class = 'badge-high' if row['Avg R² Score'] > 0.7 else 'badge-medium' if row['Avg R² Score'] > 0.4 else 'badge-low'
        html_content += f'''
                <tr>
                    <td>{rank}</td>
                    <td><strong>{row['Model']}</strong></td>
                    <td><span class="badge {r2_class}">{row['Avg R² Score']:.4f}</span></td>
                    <td>{row['Avg MAE (%)']:.2f}%</td>
                    <td>{row['Times Best']}</td>
                </tr>
'''
    
    html_content += '''
            </tbody>
        </table>
        
        <div class="chart-container">
            <h2>📊 Visualization Charts</h2>
            <img src="charts/model_performance.png" alt="Model Performance">
            <img src="charts/best_model_distribution.png" alt="Best Model Distribution" style="margin-top: 20px;">
            <img src="charts/r2_distribution.png" alt="R² Distribution" style="margin-top: 20px;">
        </div>
'''
    
    # Add top poses
    if json_results:
        html_content += '''
        <h2>🌟 Top 10 Best Performing Poses</h2>
        <table>
            <thead>
                <tr><th>Rank</th><th>Pose Name</th><th>Best Model</th><th>R² Score</th></tr>
            </thead>
            <tbody>
'''
        pose_performance = []
        for pose, data in json_results.items():
            if data['best_r2'] is not None:
                pose_performance.append({'Pose': pose, 'Best_Model': data['best_model'], 'Best_R2': data['best_r2']})
        
        top_poses = sorted(pose_performance, key=lambda x: x['Best_R2'], reverse=True)[:10]
        for i, pose in enumerate(top_poses, 1):
            html_content += f'''
                <tr>
                    <td>{i}</td>
                    <td>{pose['Pose'][:50]}</td>
                    <td>{pose['Best_Model']}</td>
                    <td>{pose['Best_R2']:.4f}</td>
                </tr>
'''
        
        html_content += '''
            </tbody>
        </table>
'''
    
    html_content += '''
        <div class="footer">
            <p>Generated by Yoga Pose Correction Model Training System</p>
            <p>📁 Results saved in: models/comparison_results/</p>
        </div>
    </div>
</body>
</html>
'''
    
    # Save HTML report
    with open("models/comparison_results/model_performance_report.html", "w", encoding='utf-8') as f:
        f.write(html_content)
    
    print("\n   ✓ HTML Report: model_performance_report.html")

# ==================== MAIN ====================
def main():
    # Load results
    df_perf, df_comp, json_results = load_results()
    
    if df_perf is None:
        print("❌ No results found! Please run train_models_corrected.py first")
        return
    
    # Display summaries
    display_summary(df_perf, df_comp, json_results)
    display_top_poses(json_results)
    display_bottom_poses(json_results)
    display_model_by_pose(df_comp)
    
    # Create visualizations
    create_charts(df_perf, json_results)
    generate_html_report(df_perf, json_results)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n📁 Results saved in: models/comparison_results/")
    print("   - performance_summary.csv (Model performance data)")
    print("   - model_comparison.csv (Per-pose detailed results)")
    print("   - model_performance_report.html (Interactive HTML report)")
    print("   - charts/ (Visualization images)")
    print("=" * 80)

if __name__ == "__main__":
    main()
