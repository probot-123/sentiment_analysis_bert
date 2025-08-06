# Sentiment Analysis with BERT

A deep learning project that performs sentiment analysis on Reddit comments using BERT (Bidirectional Encoder Representations from Transformers). This project demonstrates how to fine-tune a pre-trained BERT model for multi-class sentiment classification.

## 📋 Project Overview

This project implements a sentiment analysis system that can classify Reddit comments into three sentiment categories:

- **Negative (-1)**: Negative sentiment
- **Neutral (0)**: Neutral sentiment
- **Positive (1)**: Positive sentiment

## 🚀 Features

- **BERT-based Model**: Uses the pre-trained `bert-base-uncased` model fine-tuned for sentiment analysis
- **Multi-class Classification**: Handles three sentiment classes (negative, neutral, positive)
- **Comprehensive Training**: Includes training/validation split, checkpointing, and performance metrics
- **Performance Tracking**: Monitors training/validation loss, accuracy, and F1 scores
- **Model Persistence**: Saves best model checkpoints and final trained model

## 📊 Dataset

The project uses the Reddit Data dataset containing cleaned Reddit comments with sentiment labels. The dataset is automatically downloaded from:

```
https://raw.githubusercontent.com/campusx-team/Text-Datasets/refs/heads/main/Reddit_Data.csv
```

### Dataset Statistics

- **Total samples**: ~37,000 comments
- **Training set**: 85% of data
- **Validation set**: 15% of data
- **Max sequence length**: 256 tokens
- **Batch size**: 16 (training), 32 (validation)

## 🛠️ Requirements

### Python Packages

```bash
torch>=2.4.0
transformers
pandas
numpy
scikit-learn
tqdm
```

### Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (CUDA 12.3+)
- **RAM**: Minimum 8GB RAM
- **Storage**: ~2GB for model and checkpoints

## 📦 Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd sentiment_analysis_bert
```

2. **Install dependencies**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers pandas numpy scikit-learn tqdm
```

3. **Run the notebook**:

```bash
jupyter notebook sentiment-analysis-bert-reddit-data.ipynb
```

## 🎯 Usage

### Training the Model

1. **Open the Jupyter notebook**:

   ```bash
   jupyter notebook sentiment-analysis-bert-reddit-data.ipynb
   ```

2. **Run all cells** to:
   - Load and preprocess the Reddit dataset
   - Split data into training/validation sets
   - Tokenize text using BERT tokenizer
   - Initialize BERT model for sequence classification
   - Train the model for 10 epochs
   - Evaluate performance metrics

### Model Training Process

The training includes:

- **Optimizer**: AdamW with learning rate 1e-5
- **Scheduler**: Linear warmup schedule
- **Loss Function**: Cross-entropy loss
- **Gradient Clipping**: Norm of 1.0
- **Checkpointing**: Saves best model based on validation loss

## 📈 Results

### Training Performance

- **Final Training Accuracy**: 99.6%
- **Final Validation Accuracy**: 95.6%
- **Best F1 Score (weighted)**: 95.6%

### Per-Class Performance

| Class         | Precision | Recall | F1-Score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Negative (-1) | 0.93      | 0.92   | 0.92     | 1,242   |
| Neutral (0)   | 0.97      | 0.98   | 0.98     | 1,956   |
| Positive (1)  | 0.96      | 0.96   | 0.96     | 2,375   |

### Model Convergence

- **Best validation loss**: Achieved at epoch 3
- **Training epochs**: 10
- **Final model saved**: `./final_model.pth`

## 🔧 Model Architecture

- **Base Model**: `bert-base-uncased`
- **Classification Head**: Linear layer for 3-class classification
- **Input Processing**: Tokenization with max length 256
- **Special Tokens**: [CLS], [SEP] tokens added automatically

## 📁 Project Structure

```
sentiment_analysis_bert/
├── sentiment-analysis-bert-reddit-data.ipynb  # Main training notebook
├── README.md                                  # This file
├── final_model.pth                           # Trained model (generated)
└── checkpoints/                              # Model checkpoints (generated)
    ├── checkpoint_epoch_1.pth
    ├── checkpoint_epoch_2.pth
    └── ...
```

## 🎛️ Configuration

### Key Parameters

- **Learning Rate**: 1e-5
- **Batch Size**: 16 (training), 32 (validation)
- **Max Sequence Length**: 256
- **Epochs**: 10
- **Warmup Steps**: 0
- **Random Seed**: 8

### Model Hyperparameters

- **Optimizer**: AdamW
- **Scheduler**: Linear warmup
- **Loss Function**: CrossEntropyLoss
- **Gradient Clipping**: 1.0

## 🔍 Evaluation Metrics

The model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 score across all classes
- **Per-class Accuracy**: Individual class performance
- **Classification Report**: Detailed precision, recall, F1-score

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:

   - Reduce batch size from 16 to 8
   - Reduce max sequence length from 256 to 128

2. **Installation Issues**:

   - Ensure PyTorch version matches your CUDA version
   - Use conda for environment management if needed

3. **Dataset Download Issues**:
   - Check internet connection
   - Verify the dataset URL is accessible

