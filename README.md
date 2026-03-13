<h1 align="center">
Enhancing Event-Based Vehicle Detection: A Hybrid Spiking Neural Network and Vision Transformer Architecture
</h1>

<p align="center">
<b>Degree Project in Computer Science and Engineering (DA150X)</b>



KTH Royal Institute of Technology

Authors: Viggo Jahr & Axel Prander
</p>

## 📌 Project Overview
This repository contains the code for a Bachelor's thesis investigating a hybrid neuromorphic computer vision architecture. Traditional computer vision relies heavily on frame-based cameras and compute-heavy Convolutional Neural Networks (CNNs). To address energy and latency limitations, this project utilizes Event-based cameras and Spiking Neural Networks (SNNs).

To overcome the limitations of pure SNN architectures in capturing long-range spatial dependencies (such as distinguishing between visually similar vehicle classes like buses and trucks), this project introduces a Vision Transformer (ViT) as a global aggregation head. The detection task is framed as a heatmap regression problem, outputting a spatial map of Gaussian blobs corresponding to vehicle centers.

## 🏗️ Technical Foundation & Acknowledgements
This project is a direct continuation and expansion of previous research conducted at KTH Royal Institute of Technology. It builds upon the technical pipeline established by:

1. Emma Hagrot (2025): Original raw event data collection, cleaning, and formatting. (Original Repo)

2. Olof Eliasson & Tobias Persson (2025): Baseline pure-SNN architecture and the multi-class data preprocessing pipeline for traffic monitoring.

3. Nora Hulth (2026): New algoirthm for the preprocessing of the data - resulting in better data.

The data, the SNN backbone and data loading utilities in this repository are heavily based on their foundational work. Our novel contribution focuses on substituting the final fully-connected SNN layers with a ViT attention mechanism.

## ⚙️ Data Preprocessing Pipeline
updated soon with explanations.

## 🧠 Model Architecture (Hybrid SNN-ViT)
The system is divided into two primary components:

* The Backbone (Feature Extractor): A Spiking Neural Network (or Convolutional) backbone that processes the 10ms event-frame bins to extract low-level spatial and temporal features efficiently.

* The Classification Head (Global Aggregation): A Vision Transformer (ViT) layer that replaces the standard SNN readout. It applies self-attention mechanisms to the feature maps to capture global context, outputting class-specific heatmaps.

These heatmaps are post-processed using peak extraction (local maxima) to map predicted object centers back to bounding boxes for final evaluation.

## 📦 Requirements
The main external packages and libraries required for this project include:

* PyTorch: Core deep learning framework for the Vision Transformer and model training.

* snnTorch / Norse: For compiling and simulating the Spiking Neural Network components.

(Note: A complete requirements.txt file will be provided as the training environment is finalized.)

## 🚀 Usage & Installation
Setup instructions, training scripts, and evaluation commands will be added here as the codebase is actively developed during the DA150X course.
