# Custom Food Segmentation with EfficientNet

## Overview
This project involves building a custom food segmentation model using EfficientNet-B0 as the backbone. The model is fine-tuned for pixel-wise segmentation of specific food categories, leveraging a custom dataset with both real and AI-generated images. This segmentation task can be used in various real-world applications like recipe generation, nutritional analysis, and more.

---

## Dataset

### Description
- The dataset is custom-built and contains real and AI-generated images annotated for segmentation tasks.
- Focus is on segmenting food items into four distinct classes:
  - **Grilled Chicken** (Red: RGB (255, 0, 0))
  - **Paneer** (Green: RGB (0, 255, 0))
  - **Eggplant** (Blue: RGB (0, 0, 255))
  - **Background** (Black: RGB (0, 0, 0))

### Dataset Statistics
- **Number of Samples**: 200 images with corresponding annotated masks
- **Image Size**: Resized to 512x512 pixels during preprocessing
- **Partitioning**:
  - Training Set: 80% (160 images)
  - Validation Set: 20% (40 images)

### Data Normalization
- **Mean**: [0.485, 0.456, 0.406]
- **Standard Deviation**: [0.229, 0.224, 0.225]

### Data Augmentation Techniques
- **Pre-Training**:
  - Resize to 512x512
  - Convert to tensor
- **Post-Training**:
  - Random horizontal flip
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation, hue)

### Dataset Link
The dataset is sourced from  AI-generated images using Stable Diffusion.  [here](https://drive.google.com/drive/folders/1sXK6uv-XI-iL3fJ-dB6TQKmeuDH1Vm6n?usp=sharing).

---

## Neural Network Architecture

### EfficientNet-B0 for Food Segmentation
1. **Backbone**:
   - EfficientNet-B0 pre-trained on ImageNet, used as a feature extractor.
   - Frozen backbone layers to retain learned features.
2. **Upsampling Layer**:
   - Restores spatial resolution to 512x512.
   - Outputs a pixel-wise classification map.
3. **Output Layer**:
   - Produces segmentation maps with class probabilities for each pixel.

---

## Training and Optimization

### Loss Function
- **Cross-Entropy Loss**:
  - Handles class imbalance by assigning higher weights to underrepresented classes.
  - Class Weights:
    - Background: 0.1
    - Grilled Chicken, Paneer, Eggplant: 1.0

### Hyperparameters
- **Mini-batch Size**: 8
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Scheduler**: StepLR (Step Size: 5, Gamma: 0.5)
- **Epochs**: 20

### Results
- **Training Loss**: 0.34
- **Validation Loss**: 0.50
- **Intersection over Union (IoU)**: 0.45

---

## Visualization

### Pre-Training Performance
- Basic image resizing and tensor conversion.
- Training Loss: 1.49
- Validation Loss: 1.50
- IoU: Not calculated.

### Post-Training Performance
- Enhanced data augmentations and normalization.
- Fine-tuned EfficientNet-B0 with weighted loss.
- Improved IoU and reduced loss.

---

## Tools and Frameworks
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Annotation Tool**: LabelMe

---

## Future Improvements
- Explore advanced segmentation architectures like U-Net and DeepLab.
- Experiment with more extensive data augmentation to improve model robustness.
- Incorporate additional food categories to enhance versatility.

---

## Authors
- **Shaagun Suresh**
