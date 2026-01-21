# Brain Tumor Detection using Deep Learning

[![Brain Tumor Detection CI](https://github.com/SameerRamzan/Brain-Tumor-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/SameerRamzan/Brain-Tumor-Detection/actions/workflows/ci.yml)

This project uses Convolutional Neural Networks (CNNs) to classify MRI scans of human brains as either containing a tumor ("Yes") or not ("No"). The project explores multiple state-of-the-art architectures via transfer learning, identifying **VGG16** and **ResNet50** as the most effective models for this specific dataset. An ensemble of these models is then used to achieve a more robust final prediction.

## Table of Contents
- [Brain Tumor Detection using Deep Learning](#brain-tumor-detection-using-deep-learning)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Methodology](#methodology)
    - [1. Data Preparation](#1-data-preparation)
    - [2. Data Augmentation](#2-data-augmentation)
    - [3. Model Architectures](#3-model-architectures)
    - [4. Training Strategy](#4-training-strategy)
    - [5. Model Interpretability (Grad-CAM)](#5-model-interpretability-grad-cam)
    - [6. Ensemble Learning](#6-ensemble-learning)
  - [Results \& Analysis](#results--analysis)
  - [Application Architecture](#application-architecture)
    - [1. Backend (FastAPI)](#1-backend-fastapi)
    - [2. Frontend (Streamlit)](#2-frontend-streamlit)
    - [3. Database (MongoDB)](#3-database-mongodb)
    - [4. Deployment (Docker)](#4-deployment-docker)
    - [5. CI/CD Pipeline (GitHub Actions)](#5-cicd-pipeline-github-actions)
  - [How to Run (Web Application)](#how-to-run-web-application)
  - [How to Run (Model Training)](#how-to-run-model-training)
  - [Dependencies](#dependencies)

## Project Overview

The goal of this project is to build a robust deep learning classifier for brain tumor detection. Given the limited size of the dataset, **Transfer Learning** is utilized. Several pre-trained models were experimented with to determine the best fit for the data:
- **VGG16**
- **ResNet50**
- **InceptionV3**
- **EfficientNetB0**
- **MobileNetV3**

The best & under performing models were then combined using an ensemble technique to improve overall accuracy and reliability.

## Dataset

The dataset is sourced from the Kaggle an online open source platform & Hub for Data Science and Machine Learning `path/brain_tumor_dataset` directory and contains two classes:
- **`yes`**: MRI images indicating the presence of a brain tumor.
- **`no`**: MRI images of healthy brains.

The data is split into three subsets:
- **Training Set (`TRAIN/`)**: 176 images
- **Validation Set (`VAL/`)**: 26 images
- **Test Set (`TEST/`)**: 51 images

## Methodology

### 1. Data Preparation

- **Loading Data**: Images are loaded using a custom `load_data` function from `data_utils.py`.
- `load_data` is designed to be robust enough to support the possible dimensions. You just need to pass the sizes (e.g. 224x224, 299x299, etc)  
- **Letterbox Resizing**: Unlike standard resizing which stretches images, this project uses **letterbox resizing** (preserving aspect ratio and padding with black) to maintain the structural integrity of the brain scans. 
> In the notebook you might see the grayscale because at first I attempted to show the images with grayscale which isn't robust enough, I know and therefore, I changed the code in `data_utils.py` to black scale. So, when you run the notebook it will show the images with black borders.
  - Target sizes used: 224x224 (for most models like VGG16, ResNet50, and EfficientNetB0) and 299x299 (for InceptionV3).
- **Preprocessing**:
  - BGR to RGB conversion (crucial for models pre-trained on ImageNet).
  - Model-specific preprocessing (e.g., `preprocess_input` for VGG/ResNet).

### 2. Data Augmentation

To combat the small dataset size (176 training images), aggressive data augmentation is applied during training using `ImageDataGenerator`:
- Rotation (up to 20 degrees)
- Width and height shifts
- Shearing
- Zooming
- Horizontal flipping

### 3. Model Architectures

We utilized **Transfer Learning** with weights pre-trained on ImageNet.

1.  **Base Model**: The convolutional base of the respective architecture (e.g., VGG16, ResNet50) is loaded without the top layer.
2.  **Freezing**: Initial training is done with the base layers frozen.
3.  **Custom Head**:
    - `GlobalAveragePooling2D` (or Flatten)
    - `Dense` (256 neurons, ReLU)
    - `Dropout` (0.5) to prevent overfitting
    - `Dense` (1 neuron, Sigmoid) for binary classification

### 4. Training Strategy

- **Optimizer**: `Adam` with a learning rate of `0.0001`.
- **Loss Function**: `binary_crossentropy`.
- **Callbacks**:
  - `EarlyStopping`: Monitors validation loss to stop training when the model stops improving (patience=4).
  - `ReduceLROnPlateau`: Reduces learning rate when validation loss plateaus to fine-tune the weights.

### 5. Model Interpretability (Grad-CAM)

**Grad-CAM (Gradient-weighted Class Activation Mapping)** was implemented to visualize which parts of an MRI scan the model focuses on to make its prediction. This technique produces a heatmap highlighting the "important" regions, providing insight into the model's decision-making process and helping to verify that it is focusing on relevant pathological areas.

### 6. Ensemble Learning

To improve predictive performance and robustness, an **averaging ensemble** was created. This method combines the prediction probabilities from all the models (VGG16 and ResNet50 etc.) and averages them to produce a final, more reliable classification.

## Results & Analysis

| Model | Performance | Observations |
| :--- | :--- | :--- |
| **VGG16** | **High** | Performed best. Its simple architecture generalizes well on small datasets. |
| **ResNet50** | **High** | Comparable to VGG16. Residual connections helped convergence. |
| **MobileNetV3** | Low | Less accurate than VGG/ResNet on this specific data. |
| **InceptionV3** | Low | Struggled with overfitting. The model complexity is too high for 176 images. |
| **EfficientNetB0** | Low | Unstable training. Batch Normalization layers struggle with small batch sizes and limited data diversity. |
| **Ensemble (VGG16 + ResNet50 + MobileNetV3 + InceptionV3 + EfficientNetB0)** | **High** | The ensemble model performed good but less than the best performing model (VGG16, ResNet50), demonstrating improved accuracy and generalization. |

**Conclusion**: For this specific "Micro-Dataset", simpler architectures like VGG16 and ResNet50 outperform complex modern architectures. Combining them in an ensemble provides the best overall result.

> You may only see the ResNet50 model as the best performing model in the `Model.ipynb` because I ran the model several times and sometimes the `VGG16` performs better, sometimes the `ResNet50` performs better. So I saved the best versions for both of the models. 

## Application Architecture

The project has been evolved from a Jupyter Notebook into a full-stack web application using a microservices approach containerized with Docker.

### 1. Backend (FastAPI)
- **Framework**: FastAPI for high-performance, asynchronous API endpoints.
- **Functionality**:
  - Loads the trained Keras models (VGG16 & ResNet50) into memory.
  - Exposes endpoints for prediction (`/predict`), authentication (`/token`, `/register`), and history management.
  - Implements **Grad-CAM** logic to generate heatmaps dynamically upon request.
  - Handles image processing (DICOM conversion, resizing) before inference.

### 2. Frontend (Streamlit)
- **Framework**: Streamlit for a responsive, Python-based UI.
- **Features**:
  - **Authentication**: Secure Login/Register system with JWT token management.
  - **Dashboard**: Visual analytics of scan history using Matplotlib/Seaborn.
  - **Analysis**: Upload interface for MRI scans (JPG, PNG, DICOM) with zoom and side-by-side heatmap comparison.
  - **Reporting**: Generates downloadable PDF medical reports containing the scan, prediction, and heatmaps.
  - **Admin Panel**: User management and system statistics for administrators.

### 3. Database (MongoDB)
- **Storage**: MongoDB is used for persistent data storage.
- **Collections**:
  - `users`: Stores hashed credentials and profile info.
  - `history`: Stores metadata of analyzed scans (diagnosis, confidence, timestamp).
  - **GridFS**: Stores the actual raw image files and generated heatmaps efficiently.

### 4. Deployment (Docker)
- The entire stack (Frontend, Backend, Database) is orchestrated using **Docker Compose**, ensuring a consistent environment across different machines.

### 5. CI/CD Pipeline (GitHub Actions)
- **Automation**: A Continuous Integration workflow is set up to automatically build the Docker containers and run integration tests on every push to the `main` branch.
- **Testing**: Integration tests verify that the API endpoints (`/docs`, `/token`) are reachable and functioning correctly within the containerized environment.

## How to Run (Web Application)

To run the complete web application using Docker:

1.  **Prerequisites**: Ensure **Docker Desktop** is installed and running on your machine.
2.  **Build and Run**: Open your terminal in the project root and execute:
    ```bash
    docker-compose up --build
    ```
3.  **Access the App**: Open your browser and navigate to `http://localhost:8501`.
4.  **Create Admin User**: To access admin features, run the creation script inside the container:
    ```bash
    docker exec -it brain_tumor_ui python create_admin.py
    ```
5.  **API Documentation**: You can view the backend API docs at `http://localhost:8000/docs`.

## How to Run (Model Training)

1.  Ensure the dataset is in `path/brain_tumor_dataset/` (it could be our path).
2.  Install dependencies.
3.  Run the Jupyter Notebook (`Model.ipynb`).
4.  The notebook contains cells for training and evaluating the different models, as well as Grad-CAM visualization and ensembling.

## Dependencies

- TensorFlow / Keras
- scikit-learn
- OpenCV-Python (`cv2`)
- NumPy
- Matplotlib
- Seaborn
- `data_utils.py` (local utility file for data loading)
