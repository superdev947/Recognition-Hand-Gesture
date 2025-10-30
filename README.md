# Hand Gesture Recognition using Deep Learning

A machine learning project that uses Convolutional Neural Networks (CNNs) to classify different hand gestures with over 95% accuracy.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/filipefborba/HandRecognition/blob/master/project3/project3.ipynb)

## 📖 Overview

This project implements a Deep Learning model capable of classifying images of different hand gestures, including fist, palm, thumbs up/down, and other common hand signs. The model uses Convolutional Neural Networks (CNNs) built with TensorFlow and Keras to achieve high-confidence (>95%) classification accuracy.

**Practical Applications:**
- Gesture Navigation Systems
- Human-Computer Interaction
- Assistive Technology
- Touchless Control Interfaces

## 🎯 Project Goals

- Train a Machine Learning algorithm capable of classifying 10 different hand gestures
- Understand and implement Deep Learning techniques using CNNs
- Achieve high accuracy in gesture recognition
- Create a reusable model for real-world applications

## 📊 Dataset

This project uses the [Hand Gesture Recognition Database](https://www.kaggle.com/gti-upm/leapgestrecog/version/1) from Kaggle, which contains:
- **20,000 images** of hand gestures
- **10 different gestures** from **10 different people** (5 male, 5 female)
- Images captured using the **Leap Motion** hand tracking device

### Gesture Classes

| Gesture | Label |
|---------|-------|
| Thumb down | 0 |
| Palm (Horizontal) | 1 |
| L | 2 |
| Fist (Horizontal) | 3 |
| Fist (Vertical) | 4 |
| Thumbs up | 5 |
| Index | 6 |
| OK | 7 |
| Palm (Vertical) | 8 |
| C | 9 |

**Citation:**
> T. Mantecón, C.R. del Blanco, F. Jaureguizar, N. García, "Hand Gesture Recognition using Infrared Imagery Provided by Leap Motion Controller", Int. Conf. on Advanced Concepts for Intelligent Vision Systems, ACIVS 2016, Lecce, Italy, pp. 47-57, 24-27 Oct. 2016. (doi: 10.1007/978-3-319-48680-2_5)

## 🏗️ Model Architecture

The CNN model consists of:
- **3 Convolutional Layers** with ReLU activation (32, 64, 64 filters)
- **3 MaxPooling Layers** for dimensionality reduction
- **Flatten Layer** to convert 2D features to 1D
- **Dense Hidden Layer** with 128 neurons and ReLU activation
- **Output Layer** with 10 neurons and Softmax activation

### Key Components:
- **Conv2D Layers**: Extract features from images using convolution filters
- **MaxPooling Layers**: Downsample feature maps to reduce processing time
- **Dense Layers**: Perform final classification based on extracted features
- **ReLU Activation**: Introduces non-linearity and faster convergence
- **Softmax Output**: Provides probability distribution across all gesture classes

## 🛠️ Technologies Used

- **Python** - Programming language
- **TensorFlow & Keras** - Deep Learning framework
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **OpenCV** - Image processing
- **scikit-learn** - Model evaluation and data splitting
- **Google Colab** - Development environment

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn pandas
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/superdev947/Recognition-Hand-Gesture.git
cd Recognition-Hand-Gesture
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog/version/1)

3. Extract the dataset to the project directory

### Usage

#### Training the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook project3/project3.ipynb
```

Or use the provided Google Colab link for cloud-based training.

#### Using Pre-trained Model

The repository includes a pre-trained model (`handrecognition_model.h5`). To load it:

```python
from keras.models import load_model

model = load_model('project3/handrecognition_model.h5')
```

## 📈 Results

- **Training Accuracy**: >95%
- **Validation Accuracy**: >95%
- Successfully classifies all 10 hand gesture types with high confidence

The model includes:
- Confusion matrix analysis
- Visual prediction examples
- Performance metrics and evaluation

## 📁 Project Structure

```
Recognition-Hand-Gesture/
├── project3/
│   ├── project3.ipynb           # Main Jupyter notebook
│   ├── handrecognition_model.h5 # Pre-trained model
│   └── abstract3.md             # Project abstract
├── proposal3.md                  # Project proposal
├── report3.md                    # Project report
└── README.md                     # This file
```

## 📝 Documentation

- [Medium Tutorial](https://medium.com/@filipefborba/tutorial-using-deep-learning-and-cnns-to-make-a-hand-gesture-recognition-model-371770b63a51) - Detailed walkthrough of the project (88 views and 6 claps in less than 2 days!)
- [Project Proposal](proposal3.md) - Initial project planning
- [Project Abstract](project3/abstract3.md) - Summary of the work
- [Full Report](report3.md) - Complete documentation
