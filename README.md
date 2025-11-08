# Vanilla CNN Student Image Classifier

## Overview
Vanilla CNN Student Image Classifier is a small end-to-end project that trains a simple convolutional neural network to classify student face images and serves inference through a Streamlit web app. The app accepts an uploaded image, preprocesses it, runs the trained Keras model, displays the predicted class and confidence scores, and shows the uploaded image in the UI.

## Features
- Training script using Keras (TensorFlow backend) with ImageDataGenerator and directory-based train/validation data loading.
- Vanilla CNN architecture with several Conv2D + MaxPooling blocks, Dense layers and Dropout.
- Model export to a single HDF5 file (`Vanilla_CNN_Student_Image_Classifier.h5`) for serving.
- Streamlit app for uploading images, running inference, and showing per-class confidence scores.
- Simple, reproducible pipeline suitable for small classroom/demo datasets.

## Repo structure (key files)
- `app.py` — Streamlit application: loads the saved Keras model, provides the file uploader, performs preprocessing (convert to RGB → resize to 128×128 → scale by 1/255), runs prediction, and displays predicted class + confidence for each class. Class names are defined in the app (e.g., `['anu','bharti','deepak','manidhar','sudh']`).
- `train_cnn.py` — Training script: sets up ImageDataGenerator for training/validation directories, builds the CNN model, compiles, trains (epochs = 10), and saves the trained model as `Vanilla_CNN_Student_Image_Classifier.h5`.
- `Vanilla_CNN_Student_Image_Classifier.h5` — Trained Keras model (expected output of `train_cnn.py`) — place this file in the repo root for the Streamlit app to load.
- `requirements.txt` — (add necessary packages: see Requirements below).  
- `data/train/` and `data/val/` — expected folder structure for `train_cnn.py` ImageDataGenerator input.

## Model & training details
- Input image size: 128 × 128 RGB (IMG_SIZE = (128,128)).
- Batch size used during training: 16 (BATCH_SIZE = 16).
- Number of classes: inferred from training data directory structure (train generator `class_indices`).
- Architecture highlights: multiple Conv2D + MaxPooling2D blocks, Flatten → Dense(128, relu) → Dropout(0.3) → Dense(num_class, softmax).
- Compilation: categorical crossentropy loss, Adam optimizer, metric = accuracy.
- Training: `model.fit(..., epochs=10, validation_data=val_data)` and then `model.save("Vanilla_CNN_Student_Image_Classifier.h5")`.

## Inference (how the Streamlit app works)
- The app loads the model from `Vanilla_CNN_Student_Image_Classifier.h5` and defines the class labels in `class_names`.
- UI: sidebar file uploader accepts JPG/JPEG/PNG files. Uploaded image is converted to RGB, shown in the app, resized to 128×128, scaled (divided by 255), expanded to batch shape, and fed to `model.predict()`.
- Output: predicted class (highest softmax score) shown using `st.success()`, and confidence scores for all classes are displayed below.

## Quickstart - local

1. Clone the repository:
   ```bash
   git clone https://github.com/Anurag0798/Vanilla_CNN_Student_Image_Classifier
   ```
   ```bash
   cd Vanilla_CNN_Student_Image_Classifier
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   .venv\Scripts\Activate.ps1     # Windows (PowerShell)
   ```

3. Install required packages (example):
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not provided or missing some packages, install at least:
   ```bash
   pip install tensorflow streamlit pillow numpy
   ```

4. Prepare data for training (if you want to train):
   - Organize images into `data/train/<class_name>/...` and `data/val/<class_name>/...` folders. `train_cnn.py` uses `flow_from_directory()` to read these directories.

5. Train the model (optional):
   ```bash
   python train_cnn.py
   ```
   This will train for 10 epochs (as implemented) and save `Vanilla_CNN_Student_Image_Classifier.h5`.

6. Run the Streamlit app for inference:
   ```bash
   streamlit run app.py
   ```
   Upload an image via the sidebar to get a prediction and confidence scores.

## Requirements
Minimum recommended packages:
- Python 3.7+
- tensorflow (for Keras)
- streamlit
- pillow (PIL)
- numpy

Example install:
```bash
pip install tensorflow streamlit pillow numpy
```

Add any additional packages to `requirements.txt` as needed.

## Tips & troubleshooting
- Model file not found: ensure `Vanilla_CNN_Student_Image_Classifier.h5` is in the same directory as `app.py`, or update `MODEL_PATH` in `app.py`.
- Class mismatch: If you train a model with different class folders than the `class_names` defined in `app.py`, update `class_names` to match the training labels, or better, load class names dynamically from training artifacts. The current app hardcodes class labels for demo purposes.
- Training memory/GPUs: For larger datasets, prefer running training on a GPU-enabled machine; adjust batch size and other hyperparameters as needed.
- Image preprocessing must match training preprocessing: both training and inference resize to 128×128 and scale pixels by 1/255 - keep these consistent.

## License
MIT License included. Please read the License file for more details.