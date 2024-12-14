# A Spatial And Frequency Based Method For Micro Facial Expressions Recognition Using Color And Depth Images

### This code implements the following paper:
- A SPATIAL AND FREQUENCY BASED METHOD FOR MICRO FACIAL EXPRESSIONS RECOGNITION USING COLOR AND DEPTH IMAGESA SPATIAL AND FREQUENCY BASED METHOD FOR MICRO FACIAL EXPRESSIONS RECOGNITION USING COLOR AND DEPTH IMAGES, Journal of Software Engineering & Intelligent Systems 6 (1), 17,(2021).
 
### Link to the paper:
- https://www.researchgate.net/publication/351334905_A_SPATIAL_AND_FREQUENCY_BASED_METHOD_FOR_MICRO_FACIAL_EXPRESSIONS_RECOGNITION_USING_COLOR_AND_DEPTH_IMAGES_ISSN_2518-8739_Online_JSEIS_httpwwwjseisorgvolumevol6no1html
- Or
- https://figshare.com/articles/journal_contribution/V6N1-2_A_SPATIAL_AND_FREQUENCY_BASED_METHOD_FOR_MICRO_FACIAL_EXPRESSIONS_RECOGNITION_USING_COLOR_AND_DEPTH_IMAGES/14538405?file=27875415
- Or
- https://www.academia.edu/50348966/A_SPATIAL_AND_FREQUENCY_BASED_METHOD_FOR_MICRO_FACIAL_EXPRESSIONS_RECOGNITION_USING_COLOR_AND_DEPTH_IMAGES
- DOI: http://dx.doi.org/10.6084/m9.figshare.14538405
- https://doi.org/10.6084/m9.figshare.14538405.v1

### Please cite:
- MOUSAVI, SEYED MUHAMMAD HOSSEIN. "A SPATIAL AND FREQUENCY BASED METHOD FOR MICRO FACIAL EXPRESSIONS RECOGNITION USING COLOR AND DEPTH IMAGES." Journal of Software Engineering & Intelligent Systems 6.1 (2021): 17.
- Mousavi, Seyed Muhammad Hossein (2021). V6N1-2 A SPATIAL AND FREQUENCY BASED METHOD FOR MICRO FACIAL EXPRESSIONS RECOGNITION USING COLOR AND DEPTH IMAGES. figshare. Journal contribution. https://doi.org/10.6084/m9.figshare.14538405.v1

# Micro Facial Expressions Recognition Using Color and Depth Images

This repository contains the implementation of a **spatial and frequency-based method** for **Micro Facial Expressions Recognition (MFER)** using color and depth images captured with Kinect V2. The method leverages advanced feature extraction, dimensionality reduction, and classification techniques to achieve high accuracy in recognizing micro-expressions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Methodology](#methodology)
  - [Data Acquisition](#data-acquisition)
  - [Preprocessing](#preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Classification](#classification)


---

## Overview

Micro-expressions are subtle and rapid facial movements lasting between **0.1 to 0.5 seconds**. Detecting them requires precise and robust algorithms. This project uses **Kinect V2** to collect depth and color images and applies cutting-edge methods to enhance recognition accuracy.

Key features:
- Input: **Color (RGB)** and **Depth (2.5D)** images from Kinect V2.
- Focus on **spatial and frequency domain features**.
- Benchmark evaluation using six popular datasets: **Eurecom Kinect Face DB**, **VAP RGBD-T Face**, **JAFFE**, **Face Grabber DB**, **FEEDB**, and **CASME**.

---

## Features

- **Feature Extraction**: 
  - Histogram of Oriented Gradients (**HOG**)
  - Gabor Filters
  - Speeded Up Robust Features (**SURF**)
  - Local Binary Patterns (**LBP**)
  - Local Phase Quantization (**LPQ**)

- **Dimensionality Reduction**: 
  - **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** for evolutionary feature selection.

- **Classifiers**:
  - **Artificial Neural Networks (ANN)**
  - **Neuro-Fuzzy Classifier**

- **Comparative Analysis**:
  - Evaluated against state-of-the-art methods, including **CNN** and **SVM**, for both **Facial Expression Recognition (FER)** and **Micro Facial Expressions Recognition (FMER)**.

---

## Methodology

### Data Acquisition
- **Sensor**: Kinect V2 capturing color and depth data at **30 FPS**.
- **Datasets**: Eurecom Kinect Face DB, VAP RGBD-T Face, JAFFE, Face Grabber DB, FEEDB, and CASME.
- Data split into **70% training** and **30% testing**.

### Preprocessing
- Facial regions split into **mouth**, **eyes**, and **nose**.
- Applied filters:
  - **Median low-pass filter** for noise reduction.
  - **Unsharp masking** for edge enhancement.
  - **Histogram Equalization** for brightness normalization.
  - **Canny Edge Detection** for feature refinement.

### Feature Extraction
Features extracted from **spatial** and **frequency domains**:
- **HOG**: Captures edge orientations and gradients.
- **Gabor Filters**: Effective for edge and wrinkle detection.
- **SURF**: Fast feature detection and robust against rotation.
- **LBP**: Texture-based feature robust to illumination changes.
- **LPQ**: Fourier-based features for combating blurring effects.

### Dimensionality Reduction
- **NSGA-II**: Optimized feature selection for better classification performance and reduced runtime.

### Classification
- **Artificial Neural Networks (ANN)**:
  - 50 hidden layers.
  - Conjugate gradient backpropagation for training.
- **Neuro-Fuzzy Classifier**:
  - Combines neural networks with fuzzy logic.
  - Utilizes fuzzy IF-THEN rules for non-linear approximation.



