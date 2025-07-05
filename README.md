# FaceFeel 😊

A real-time facial emotion recognition system using PyTorch and deep learning.

## Description

FaceFeel is a facial emotion classification project that uses a Convolutional Neural Network (CNN) to detect and classify 8 different emotions from facial images using the FER+ dataset:

- **Anger** 
- **Contempt** 
- **Disgust** 
- **Fear** 
- **Happiness** 
- **Neutral** 
- **Sadness** 
- **Surprise** 

## Model Architecture

The model uses a custom CNN architecture with:
- 4 convolutional layers (32 → 64 → 128 → 256 filters)
- Batch normalization and dropout for regularization
- 3 fully connected layers (512 → 256 → 8 neurons)
- Data augmentation to improve generalization

## Project Structure

```
FaceFeel/
│
├── README.md
├── requirement.txt
│
├── archive/                    # FER+ Dataset
│   ├── train/
│   │   ├── anger/
│   │   ├── contempt/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happiness/
│   │   ├── neutral/
│   │   ├── sadness/
│   │   └── surprise/
│   └── val/
│       ├── anger/
│       ├── contempt/
│       ├── disgust/
│       ├── fear/
│       ├── happiness/
│       ├── neutral/
│       ├── sadness/
│       └── surprise/
│
├── experiments/
│   └── checkpoints/           # Saved models
│       └── best_model.pt
│
└── src/
    ├── dataloader.py         # Data loading and preprocessing
    ├── model.py              # Neural network architecture
    ├── train.py              # Training script
    └── predict.py            # Prediction script
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA compatible GPU (recommended)

### 1. Clone the repository
```bash
git clone git@github.com:JulienExr/FaceFeel.git
cd FaceFeel
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirement.txt
```

## Dataset (if you want to train your own model)

### Download FER+ Dataset

1. **Download the FER+ dataset** from [Microsoft FER+ GitHub](https://github.com/Microsoft/FERPlus) or [Kaggle FER+ Dataset](https://www.kaggle.com/datasets/mahmoudima/ferplus)

2. **Organize the data** according to the following structure in the `archive/` folder:

```
archive/
├── train/
│   ├── anger/          # Anger images
│   ├── contempt/       # Contempt images  
│   ├── disgust/        # Disgust images
│   ├── fear/           # Fear images
│   ├── happiness/      # Happiness images
│   ├── neutral/        # Neutral images
│   ├── sadness/        # Sadness images
│   └── surprise/       # Surprise images
└── val/
    ├── anger/
    ├── contempt/
    ├── disgust/
    ├── fear/
    ├── happiness/
    ├── neutral/
    ├── sadness/
    └── surprise/
```

### Dataset Specifications
- **Format**: 48x48 pixel grayscale images
- **Classes**: 8 emotions (including contempt)
- **Train**: ~28,709 images
- **Validation**: ~7,178 images
- **Improvement over FER2013**: Cleaner labels and additional contempt class

## Usage

### Train the model (skip if you just want to test my model)

```bash
cd src
python train.py
```

**Default training parameters:**
- Batch size: 32
- Learning rate: 1e-4
- Epochs: 120
- Optimizer: Adam with weight decay
- Scheduler: CosineAnnealingLR

### Predict on new images

```bash
cd src
python predict.py --mode image --image /path/to/your/image.jpg
```

### Real-time prediction (webcam)

```bash
cd src
python predict.py --mode webcam
```

## Performance

### Current Results
- **Training Accuracy**: ~75-80%
- **Validation Accuracy**: ~68-72%
- **Training Time**: 10 mins (RTX 4070)

### Optimization Techniques Used
- **Data Augmentation**: rotation, flip, translation, color jitter
- **Dropout**: regularization to prevent overfitting
- **Batch Normalization**: training stabilization
- **Early Stopping**: automatic stop if no improvement
- **Learning Rate Scheduling**: cosine annealing LR decay

## Configuration

### Modify hyperparameters

Edit the `src/train.py` file:

```python
batch_size = 32          # Batch size
num_epochs = 120         # Number of epochs
learning_rate = 1e-4     # Learning rate
num_classes = 8          # Number of classes (emotions)
```

### Modify model architecture

Edit the `src/model.py` file to adjust:
- Number of convolutional layers
- Filter sizes
- Dropout rates
- Fully connected layer architecture

## Available Scripts

| Script | Description |
|--------|-------------|
| `train.py` | Trains the model on the dataset |
| `predict.py` | Makes predictions on images |
| `dataloader.py` | Handles data loading and preprocessing |
| `model.py` | Defines the CNN architecture |

**Note**: This project is for educational purposes. Performance may vary depending on usage conditions and input image quality.