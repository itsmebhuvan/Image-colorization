# 🎨 Image Colorization using CNN

This project implements an **Image Colorization Model** using a **Convolutional Neural Network (CNN)** in Python.  
Given a **grayscale image**, the model predicts its **color channels** and reconstructs the full-color version.

---

## 🚀 Project Overview

- Converts **grayscale images (L-channel)** to **colored images (a,b channels)** in **LAB color space**.  
- Built using **TensorFlow/Keras**.  
- Trains a CNN to learn colorization patterns from sample color images.

---

## 🧠 Model Architecture

The CNN consists of:
- **Encoder:** Several convolutional layers with ReLU activation to extract grayscale features.  
- **Decoder:** Upsampling and convolution layers to generate color channels.  
- **Output:** Predicts 2 channels (a, b) in the LAB color space.

