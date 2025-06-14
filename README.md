
# Real-Time Mood Detection

## Overview

This application provides real-time facial mood detection using a webcam feed. It analyzes facial expressions to classify seven basic emotions: neutral, happiness, surprise, sadness, anger, disgust, and fear, displaying the results with both text and emoji representations.

## Features

- **Real-time emotion detection** from webcam feed
- **Clean, modern UI** with intuitive controls
- **Multi-camera support** for systems with multiple webcams
- **Emotion visualization** with both text and emoji representations
- **Color-coded results** for different emotions

## Technologies Used

- **Python 3.7+**
- **TensorFlow/Keras** for deep learning model
- **OpenCV** for image processing and face detection
- **PyQt5** for the graphical user interface

## Installation

### Prerequisites

- Python 3.7 or higher
- Pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hridayanshu236/MoodDetection.git
cd MoodDetection
```

2. Create and activate a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the model file:
The application expects a trained model file named `BestModel.keras` in the root directory. If you don't have this file, you can:
   - Download it from the releases section
   - Train your own model using the notebook at [Kaggle Notebook](https://www.kaggle.com/code/hridayanshu23/mood-detection/)

## Usage

1. Run the application:
```bash
python mood_detector.py
```

2. Use the interface controls:
   - Click "Start Camera" to begin webcam capture
   - Use "Prev Camera" and "Next Camera" to switch between webcams if you have multiple
   - Position your face in the frame for emotion detection
   - The detected emotion will be displayed with an emoji and confidence score

## Project Structure

```
MoodDetection/
├── mood_detector.py        # Main application file
├── BestModel.keras         # Trained emotion detection model
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Model Information

The emotion detection model:

- Architecture: Convolutional Neural Network (CNN)
- Input: Grayscale face images of size 48x48 pixels
- Output: Probabilities for 7 emotion classes
- Training Dataset: FER-2013 (Facial Emotion Recognition)
- Training Notebook: Available on [Kaggle](https://www.kaggle.com/code/hridayanshu23/mood-detection/)

## Troubleshooting

- **Camera not found**: Ensure your webcam is properly connected and not in use by another application
- **Model loading error**: Verify that the model file exists in the root directory with the correct name
- **No face detected**: Adjust lighting conditions and ensure your face is clearly visible

## Requirements

The following packages are required:
```
tensorflow>=2.4.0
opencv-python>=4.5.0
PyQt5>=5.15.0
numpy>=1.19.0
```

## Contributing

Contributions to improve the Mood Detection application are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Created by [Hridayanshu](https://github.com/hridayanshu236) - feel free to contact me!

Last updated: 2023-06-14
