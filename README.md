# 🧘 Prenatal Yoga Studio - AI-Powered Pose Correction System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org/)

## 📋 Overview

**Prenatal Yoga Studio** is an intelligent, AI-powered prenatal yoga assistant that helps pregnant women practice yoga safely throughout their pregnancy journey. The system uses computer vision and machine learning to analyze yoga poses in real-time, provide corrective feedback, and ensure exercises are appropriate for each trimester.

### 🎯 Problem Statement

Pregnancy requires special attention during physical exercise. Traditional yoga practice can be risky because:
- ❌ Certain poses can harm the mother or baby
- ❌ Women may not know which poses are safe for their trimester
- ❌ Without an instructor, it's difficult to know if poses are done correctly
- ❌ Prenatal yoga classes are expensive and not always accessible

### 💡 Solution

**Prenatal Yoga Studio** addresses these challenges by providing:
- ✅ **Trimester-based pose recommendations** - Automatically suggests safe poses based on pregnancy stage
- ✅ **Real-time pose correction** - Uses AI to analyze poses and provide instant feedback
- ✅ **Camera or upload mode** - Practice with live camera or upload photos
- ✅ **Safety-first approach** - Clear safety labels for every pose

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤰 **Trimester Detection** | Automatically calculates trimester from LMP date |
| ✅ **Smart Recommendations** | Shows only safe poses for your pregnancy stage |
| 🎥 **Real-time Pose Detection** | Live camera feed with skeleton overlay |
| 📸 **Upload Mode** | Analyze uploaded yoga pose images |
| 🤖 **AI Corrections** | Real-time accuracy percentage and correction tips |
| 🎨 **Beautiful UI** | Modern glass morphism design with animations |
| 📱 **Responsive** | Works on desktop, tablet, and mobile |

---

## 📊 Trimester Safety Guidelines

| Trimester | Weeks | Safe Poses | Modified Poses |
|-----------|-------|------------|----------------|
| **First** | 1-12 | Cat Cow, Child, Bound Angle, Tree, Warrior II, Garland, Happy Baby, Legs Up Wall | Downward Dog, Cobra, Bridge, Boat, Half Moon, Warrior III, Plank |
| **Second** | 13-26 | Cat Cow, Child, Bound Angle, Tree, Warrior II, Garland, Cow Face, Eagle | Downward Dog, Bridge, Plank, Pigeon, Chair, Warrior I |
| **Third** | 27-40 | Cat Cow, Child, Bound Angle, Tree, Garland, Cow Face, Gate, Virasana | Warrior II, Chair, Warrior I, Wide-Legged Forward Bend |

### Pose Safety Classification

| Safety Level | Color | Description |
|--------------|-------|-------------|
| **SAFE** | 🟢 Green | Completely safe for all trimesters |
| **ALLOWED WITH MODIFICATIONS** | 🟠 Orange | Practice with caution, use props |
| **RESTRICTED** | 🔴 Red | Not recommended during pregnancy |

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Flask** | Web framework for API and routing |
| **MediaPipe Tasks API** | Pose detection and landmark extraction |
| **OpenCV** | Image processing and frame manipulation |
| **Scikit-learn** | Machine learning models for pose correction |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5** | Structure and layout |
| **CSS3** | Styling, animations, glass morphism |
| **JavaScript** | Camera handling, API calls, real-time updates |

### Machine Learning Models
| Model | Purpose | Performance |
|-------|---------|-------------|
| **Random Forest** | Pose accuracy prediction | R²: 0.35-0.64 |
| **Gradient Boosting** | Accuracy regression | MAE: 5-8% |
| **SVR** | Support Vector Regression | Best for complex poses |
| **Lasso/Ridge** | Linear regression | Fast inference |

---

## 📁 Directory Structure Overview

Below is the layout of the project, including a short one-line description for each folder and its significant files.

### Folders

* **`data/`** - Stores structured JSON dataset files containing extracted keypoint angles from the raw images.
  * `yoga_pose_angles.json` - The master dataset of calculated angles for various yoga poses.
* **`dataset/`** - Contains the split and batched image data (train, test, valid folders) structured for model training.
* **`labelled_images/`** - Stores processed visual outputs of yoga poses with the skeleton layout explicitly drawn over them for verification.
* **`models/`** - Stores trained machine learning `.pkl` models, metric history, and the pose correction pipelines.
  * `all_models/` & `best_models/` - Archives containing various trained ML algorithms and their best iterations.
  * `comparison_results/` - Data regarding accuracy across different ML algorithmic approaches.
  * `pose_correction_model.py` - AI module script specifically created to calculate correction advice given user posture.
* **`src/`** - Core source components holding scripts to detect and extract joints on isolated scripts.
  * `pose_detection/` - Core computer vision scripts detecting initial joints.
  * `pose_extraction/` - Core extraction math implementations measuring specific body angles.
* **`static_images/`** - Master directory organizing the raw input image datasets categorized neatly into individual pose-named subdirectories.
* **`web_app/`** - Contains the Flask Python backend, HTML templates, CSS, and interactive application routing.
  * `app.py` / `app_with_model.py` - Flask routing scripts serving the web interface locally.
  * `templates/` & `static/` - Frontend HTML markup and CSS styling elements for the web dashboard.
  * `uploads/` - Temporary storage directory managing user-uploaded session images or videos.
* **`yoga_env/`** - The local Python virtual environment isolating all required pip packages and dependencies.

---

### Root Level Python Files

* **`app.py`** - A primary entry script intended for general application or Streamlit launching test.
* **`complete_app.py`** - An advanced frontend integration merging keypose extraction, AI checking, and the display logic into a single app.
* **`extract_angles_fixed.py`** - Robust script designed to parse through `static_images` to map landmarks and calculate joint angles precisely.
* **`get-pip.py`** - Standard utility script to install `pip` if unavailable.
* **`pose_landmarker.task`** - Machine learning weight/task file utilized by MediaPipe to track complex human body landmarks.
* **`run_project.py`** - Global wrapper script configured to start the primary project dashboard or pipeline sequence.
* **`simple_app.py`** - A lightweight minimal-viable-product version of the pose tracking application block.
* **`train_model.py`** - Original foundational script to train machine learning models atop the extracted datasets.
* **`train_models_corrected.py`** - Revised, comprehensive training script responsible for training optimal ML tracking classifiers.
* **`view_results.py`** - Visualizer module built to generate metrics and charts detailing classification performances.
* **`visualize_landmarks.py`** - Single-pass demonstration script to display an image with overlaying joint tracking dots perfectly rendered.
* **`working_app.py`** - A stable known-good interface runner representing one of the major tracking steps.
* **`yoga_app*.py`** *(e.g. `yoga_app_beautiful.py`)* - Various highly-styled, customized implementations and iterations of the tracking visualization app.

### Root Level Configuration Files

* **`requirements.txt`** - Defines the list of necessary Python libraries (like `opencv-python`, `mediapipe`, `flask`, `scikit-learn`) to replicate the environment.
* **`.gitignore`** - File ignoring local cache, binary caches, and overly large items like `yoga_env` and cached datasets during pushing.

---

## 🚀 Quick Setup

1. **Activate Virtual Environment:**
   ```bash
   yoga_env\Scripts\activate
   ```
2. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Web Interface Application:**
   ```bash
   python web_app/app.py
   ```
