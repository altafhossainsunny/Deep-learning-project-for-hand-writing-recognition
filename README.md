# Deep Learning Project for Handwriting Recognition

## Project Overview
This project implements a handwriting writer identification system using two approaches:
1. **Transfer Learning with MobileNetV2** - Deep learning classification approach
2. **ORB Feature Extraction** - Computer vision keypoint matching approach

The system analyzes handwriting samples and classifies them by writer identity, working with PNG images from a dataset of 70 different writers.

## Features
- 🎯 Writer identification using MobileNetV2 transfer learning
- 🔍 ORB keypoint extraction for handwriting fingerprinting  
- 📊 Performance evaluation with multiple metrics
- 🖼️ Processes PNG handwriting samples
- ⚙️ Automated preprocessing with Otsu's thresholding
- 📈 Results tracking and CSV export

## Dataset Structure
- **Training Data**: 70 writers (01-70), 1 sample each in `train/` folder
- **Test Data**: 140 test samples, 2 samples per writer in `test/` folder
- **Image Format**: PNG files with naming pattern `{writer_id}_{session}_{image_id}.png`

## File Structure
```
├── train.py           # MobileNetV2 transfer learning training script
├── run.py            # ORB-based evaluation and testing script
├── model.h5          # Trained MobileNetV2 model
├── result.csv        # Performance results output
├── requirements.txt  # Python dependencies
├── train/           # Training images (70 PNG files)
├── test/            # Test images (140 PNG files)
├── .gitignore       # Git ignore file
├── LICENSE          # Project license
└── README.md        # This documentation file
```

## Requirements
```
numpy
opencv-python
tensorflow
pandas
matplotlib
seaborn
scikit-learn
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/altafhossainsunny/Deep-learning-project-for-hand-writing-recognition.git
   cd Deep-learning-project-for-hand-writing-recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Method 1: MobileNetV2 Transfer Learning
Train a deep learning model for writer classification:

```bash
python train.py
```

**What it does:**
- Loads training images from `train/` folder and resizes to 128x128
- Uses MobileNetV2 as feature extractor (frozen)
- Adds classification layers for 70 writers
- Trains for 30 epochs with data augmentation
- Saves trained model as `model.h5`

### Method 2: ORB Feature Matching
Run the ORB-based handwriting fingerprinting:

```bash
python run.py
```

**What it does:**
- Extracts ORB keypoints from handwriting samples in both `train/` and `test/` folders
- Uses FLANN matcher for feature comparison
- Evaluates performance on test set
- Saves results to `result.csv`

## Technical Details

### Preprocessing Pipeline
1. **Grayscale Conversion**: Focus on pen stroke intensity
2. **Otsu's Thresholding**: Separates ink from paper texture
3. **Binary Inversion**: White ink on black background for feature extraction

### MobileNetV2 Architecture (train.py)
```
Input (128x128x3)
       ↓
MobileNetV2 (frozen)
       ↓
GlobalAveragePooling2D
       ↓
Dense(256, relu)
       ↓
Dropout(0.5)
       ↓
Dense(70, softmax)
```

### ORB Feature Extraction (run.py)
- **Features**: 500 ORB keypoints per image
- **Matching**: FLANN-based matcher with ratio test
- **Similarity**: Good matches count as similarity score

## Performance Metrics
The system evaluates using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores  
- **F1-Score**: Balanced precision-recall measure
- **ROC-AUC**: Area under ROC curve

## Example Output
```
📦 Loading Data for Transfer Learning...
🔥 Training on 70 Writers...
Epoch 30/30
9/9 ━━━━━━━━━━━━━━━━━━━━ 2s 156ms/step - loss: 0.5432 - accuracy: 0.8571
✅ New model.h5 saved.
```

## Dataset Information
- **Writers**: 70 unique individuals (labeled 01-70)
- **Training**: 1 sample per writer (70 total)
- **Testing**: 2 samples per writer (140 total)
- **Format**: PNG images of handwritten text
- **Resolution**: Variable, resized to 128x128 for deep learning

## Troubleshooting

### Common Issues:
1. **Missing images**: Ensure PNG files are in `train/` and `test/` folders
2. **Memory errors**: Reduce batch size in train.py (currently set to 8)
3. **OpenCV errors**: Check image file integrity and paths

### Performance Tips:
- 🚀 GPU acceleration recommended for training
- 💾 Sufficient RAM needed for loading all images
- 📏 Consistent image quality improves results

## Results
The project achieves writer identification through two complementary approaches:
- **Deep Learning**: Uses learned features from MobileNetV2
- **Traditional CV**: Uses handcrafted ORB features

Results are automatically saved to `result.csv` for analysis.

## License
This project is available for educational and research purposes.

## Author
**MD Altaf Hossain**
- GitHub: [@altafhossainsunny](https://github.com/altafhossainsunny)

## Contributing
1. Fork the repository
2. Create a feature branch
3. Test your changes
4. Submit a pull request

---
**Last Updated**: December 22, 2025