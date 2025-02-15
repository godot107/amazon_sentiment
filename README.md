# Sentiment Analysis Exploration

This repository contains an exploration of different sentiment analysis approaches, from traditional methods to modern deep learning techniques. The project implements and compares various algorithms for analyzing sentiment in product reviews.

## 🎯 Project Overview

The project explores sentiment analysis through multiple implementations:
1. GloVe Embeddings + Logistic Regression
2. BERT (Bidirectional Encoder Representations from Transformers)
3. (More approaches to be added...)

## 🚀 Features

- Text preprocessing and cleaning
- Multiple sentiment classification approaches
- GPU acceleration support
- Performance metrics and visualization
- Easy-to-use interface for predictions
- Support for both binary and multi-class sentiment classification

## 📊 Dataset

The project uses the Amazon Product Reviews dataset, which contains:
- Product reviews
- Rating scores
- Additional metadata

Source: [Amazon Product Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset/data)

## 🛠️ Installation

1. Clone the repository:

bash
git clone https://github.com/yourusername/sentiment-analysis-exploration.git
cd sentiment-analysis-exploration


2. Download GloVe embeddings:


wget https://nlp.stanford.edu/data/glove.6B.zip

unzip glove.6B.zip -d ./assets/

3. Create and activate the environment:

bash
conda env create -f environment.yml
conda activate sentiment_env

## 📁 Project Structure

sentiment-analysis-exploration/
├── data/
├── notebooks/
├── src/
├── README.md
└── requirements.txt

## 📦 Dependencies

- Python 3.8+
- PyTorch

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NVIDIA GPU (optional, but recommended for BERT)
- See `environment.yml` for complete list

## 📈 Performance

Different models achieve different performance metrics:

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| GloVe + LogReg | 85% | 0.84 | Fast |
| BERT | 92% | 0.91 | Slower (GPU recommended) |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GloVe embeddings from Stanford NLP
- Hugging Face for BERT implementation
- Kaggle for the dataset

## 📧 Contact

For questions or feedback, please open an issue or contact [williemaize828@gmail.com].
