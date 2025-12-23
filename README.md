# Deep Learning Project for Handwriting Recognition

## Project Overview
This project implements a handwriting writer identification system using two approaches:
1. **Custom CNN Architecture** - Deep learning classification approach with forensic handwriting analysis
2. **ORB Feature Extraction** - Computer vision keypoint matching approach

The system analyzes handwriting texture patterns using a custom CNN that extracts features from 64x64 image patches, classifying them by writer identity. The model works with PNG images from a dataset of 70 different writers.

## Features
- 🎯 Writer identification using custom CNN forensic architecture
- 🔍 ORB keypoint extraction for handwriting fingerprinting  
- 📊 Performance evaluation with multiple metrics
- 🖼️ Processes PNG handwriting samples with texture patch analysis
- ⚙️ Automated preprocessing with Otsu's thresholding
- 📈 Results tracking and CSV export
- 🧩 Image patch extraction for detailed texture analysis

## Dataset Structure
- **Training Data**: 70 writers (01-70), 1 sample each in `train/` folder
- **Test Data**: 140 test samples, 2 samples per writer in `test/` folder
- **Image Format**: PNG files with naming pattern `{writer_id}_{session}_{image_id}.png`

## File Structure
```
├── train.py           # Custom CNN forensic training script
├── run.py            # ORB-based evaluation and testing script
├── writer_model.keras # Trained custom CNN model
├── labels.pkl        # Encoded writer labels
├── result.csv        # Performance results output
├── requirements.txt  # Python dependencies
├── train/           # Training images (70 PNG files)
├── test/            # Test images (140 PNG files)
├── uploads/         # Upload directory
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

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/altafhossainsunny/Deep-learning-project-for-hand-writing-recognition.git
   cd Deep-learning-project-for-hand-writing-recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project:**
   ```bash
   # Option 1: Train Custom CNN model (takes ~5-10 minutes)
   python train.py

   # Option 2: Run ORB feature matching (faster, ~1-2 minutes)  
   python run.py
   ```

## Usage

### Method 1: Custom CNN Forensic Architecture
Train a deep learning model for writer classification:

```bash
python train.py
```

**What it does:**
- Loads training images from `train/` folder and processes with Otsu's thresholding
- Extracts overlapping 64x64 texture patches from handwriting samples
- Trains a custom CNN with forensic-focused architecture for 26 epochs
- Uses texture pattern analysis for writer identification
- Saves trained model as `writer_model.keras` and labels as `labels.pkl`

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
2. **Otsu's Thresholding**: Separates ink from paper texture automatically
3. **Binary Inversion**: White ink on black background for feature extraction
4. **Patch Extraction**: Overlapping 64x64 patches with 32-pixel stride for texture analysis

### Custom CNN Forensic Architecture (train.py)

Our custom CNN is specifically designed for handwriting forensics and texture pattern recognition:

```
Input Layer (64×64×3)
       ↓
Conv2D Layer 1: 32 filters, 3×3 kernel, ReLU
    • Input: 64×64×3 → Output: 62×62×32
    • Parameters: (3×3×3×32) + 32 = 896 parameters
    • Neurons: 62×62×32 = 123,008 neurons
    • Function: Detects basic edges and strokes
       ↓
MaxPooling2D (2×2)
    • Input: 62×62×32 → Output: 31×31×32
    • Function: Reduces spatial dimensions, retains important features
       ↓
Conv2D Layer 2: 64 filters, 3×3 kernel, ReLU  
    • Input: 31×31×32 → Output: 29×29×64
    • Parameters: (3×3×32×64) + 64 = 18,496 parameters
    • Neurons: 29×29×64 = 53,824 neurons
    • Function: Detects complex patterns like loops, curves, pressure variations
       ↓
MaxPooling2D (2×2)
    • Input: 29×29×64 → Output: 14×14×64
    • Function: Further dimensionality reduction
       ↓
Flatten Layer
    • Input: 14×14×64 → Output: 12,544 features
    • Function: Converts 2D feature maps to 1D vector
       ↓
Dense Layer: 128 neurons, ReLU
    • Input: 12,544 → Output: 128
    • Parameters: (12,544×128) + 128 = 1,605,760 parameters
    • Function: High-level feature combination and writer-specific pattern learning
       ↓
Dropout Layer (0.5)
    • Function: Prevents overfitting by randomly setting 50% of inputs to 0
       ↓
Output Dense Layer: 70 neurons, Softmax
    • Input: 128 → Output: 70 (number of writers)
    • Parameters: (128×70) + 70 = 9,030 parameters
    • Function: Writer classification probabilities

Total Parameters: 896 + 18,496 + 1,605,760 + 9,030 = 1,634,182 parameters
```

#### Layer-by-Layer Analysis:

**🔍 Convolutional Layers:**
- **Conv2D Layer 1**: Detects basic handwriting features like edges, horizontal/vertical strokes
- **Conv2D Layer 2**: Identifies more complex patterns like letter shapes, writing angles, pen pressure variations

**📉 Pooling Layers:**
- Reduce computational load while preserving essential features
- Help achieve translation invariance for different writing positions

**🧠 Dense Layers:**
- **Hidden Dense (128 neurons)**: Combines features to learn writer-specific patterns
- **Output Dense (70 neurons)**: Final classification for each of the 70 writers

**⚙️ Key Architectural Decisions:**
- **64×64 input size**: Optimal for capturing fine handwriting details without excessive computation
- **32-pixel stride**: Overlapping patches ensure no texture information is lost
- **ReLU activation**: Prevents vanishing gradients and enables faster training
- **Dropout (0.5)**: Regularization to prevent overfitting on limited training data

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
📦 Extracting Handwriting Textures...
🔥 Training on [X] handwriting texture samples...
Epoch 26/26
[X]/[X] ━━━━━━━━━━━━━━━━━━━━ 2s - loss: 0.2145 - accuracy: 0.9234
✅ Forensic Model Saved
```

## Dataset Information
- **Writers**: 70 unique individuals (labeled 01-70)
- **Training**: 1 sample per writer (70 total)
- **Testing**: 2 samples per writer (140 total)
- **Format**: PNG images of handwritten text
- **Patch Size**: 64×64 pixels extracted with 32-pixel overlapping stride
- **Preprocessing**: Otsu's thresholding + binary inversion for texture analysis

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
- **Custom CNN**: Uses learned texture features from handwriting patches with forensic analysis
- **Traditional CV**: Uses handcrafted ORB features

The CNN model with **1.63M parameters** focuses on fine-grained texture analysis, making it particularly effective for forensic handwriting analysis. Results are automatically saved to `result.csv` for analysis.

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
**Last Updated**: December 23, 2025