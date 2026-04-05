# Prenatal Yoga Studio (Computer Vision Tracker)

This repository contains the source code, machine learning pipelines, and web applications for an AI-based **Prenatal Yoga Pose Tracker**. The project leverages MediaPipe and OpenCV to perform real-time body tracking, calculate joint angles, evaluate safe posture, and provide safety recommendations specialized for pregnant women.

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
