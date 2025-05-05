# Gene Sequence Classification: Promoter vs. Non-Promoter

This project demonstrates expertise in **bioinformatics** and **AI/ML** by classifying DNA sequences as promoters or non-promoters using machine learning. It leverages the `human_nontata_promoters` dataset from UCI Genomic Benchmarks, extracts 3-mer features, and compares three models: Random Forest, Support Vector Machine (SVM), and Logistic Regression. Designed for a Master's in AI portfolio, this repository showcases data preprocessing, feature engineering, model evaluation, and visualization skills, with reproducible code and clear documentation.

## Project Overview
- **Objective**: Classify DNA sequences as promoters (coding-like regions involved in transcription initiation) or non-promoters (non-coding regions) using machine learning.
- **Dataset**: UCI Genomic Benchmarks `human_nontata_promoters`, containing 36,131 sequences (251 bp each), with 27,097 for training and 9,034 for testing, labeled as promoter (positive) or non-promoter (negative).
- **Features**: 3-mer frequency counts extracted from DNA sequences, capturing sequence motifs.
- **Models**:
  - **Random Forest**: Robust to high-dimensional, sparse features.
  - **SVM**: Effective for non-linear classification tasks.
  - **Logistic Regression**: Baseline model for linear separability.
- **Evaluation Metrics**: Accuracy, confusion matrices, ROC curves with Area Under the Curve (AUC), and model comparison.
- **Tools**: Python, `genomic_benchmarks`, `biopython`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, Google Colab.

## Repository Structure
```
gene-sequence-classification/
├── data/
│   └── README.md                   # Instructions for downloading the dataset
├── figures/
│   ├── sequence_length.png         # Sequence length distribution
│   ├── gc_content.png              # GC content by class
│   ├── top_kmers.png               # Top 10 3-mers by frequency
│   ├── confusion_matrix_random_forest.png      # Confusion matrix for Random Forest
│   ├── confusion_matrix_svm.png                # Confusion matrix for SVM
│   ├── confusion_matrix_logistic_regression.png  # Confusion matrix for Logistic Regression
│   ├── roc_curves.png              # ROC curves for  # ROC curves for all models
│   ├── model_comparison.png        # Model accuracy comparison
├── gene_sequence_classification.py  # Main pipeline script
├── notebook.ipynb                   # Jupyter Notebook with exploration and visualizations
├── README.md                        # Project overview (this file)
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
```

## Getting Started
### Prerequisites
- Python 3.8+
- Google Colab (recommended for running `notebook.ipynb`)
- Git (for cloning the repository)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gene-sequence-classification.git
   cd gene-sequence-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   In Google Colab, run:
   ```python
   !pip install genomic_benchmarks biopython
   ```

### Running the Project
#### Option 1: Google Colab (Recommended)
1. Open `notebook.ipynb` in Google Colab:
   - Upload `notebook.ipynb` via File > Upload Notebook.
   - Or use this link: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your-colab-link) *(Replace with your shared Colab link)*.
2. Run all cells sequentially (`Shift + Enter` or Runtime > Run All).
3. Outputs include:
   - **Data Exploration**: Sequence length distribution, GC content by class, top k-mer frequencies.
   - **Model Results**: Accuracy, confusion matrices, ROC curves, model comparison.
4. Save the notebook with outputs (File > Save a Copy in Drive).

#### Option 2: Local Machine
1. Run the script:
   ```bash
   python gene_sequence_classification.py
   ```
2. Outputs are saved to `figures/` and `feature_importance_rf.csv`.

## Results
### Data Insights
- **Sequence Length**: All sequences are 251 bp, ensuring uniform input for feature extraction.
- **GC Content**: Promoters have slightly higher GC content, reflecting their biological role in transcription.
- **K-mer Frequencies**: CG-rich 3-mers are prevalent, indicating potential regulatory motifs.

![Sequence Length Distribution](figures/sequence_length.png)
![GC Content by Class](figures/gc_content.png)
![Top 10 3-mers](figures/top_kmers.png)

### Model Performance
- **Random Forest**: Achieves the highest accuracy (~0.85-0.90) and AUC, leveraging feature importance for interpretability.
- **SVM**: Strong performance (~0.80-0.85) but slower due to non-linear kernel.
- **Logistic Regression**: Fastest but lower accuracy (~0.75-0.80) due to complex feature space.

#### Confusion Matrices
- **Random Forest**:
  ![Random Forest Confusion Matrix](figures/confusion_matrix_random_forest.png)
- **SVM**:
  ![SVM Confusion Matrix](figures/confusion_matrix_svm.png)
- **Logistic Regression**:
  ![Logistic Regression Confusion Matrix](figures/confusion_matrix_logistic_regression.png)

#### ROC Curves
- Combined ROC curves show model discrimination ability:
  ![ROC Curves](figures/roc_curves.png)

#### Model Comparison
- Bar plot comparing model accuracies:
  ![Model Comparison](figures/model_comparison.png)

## Challenges and Solutions
- **High-Dimensional Features**: 3-mer extraction generates thousands of features. Random Forest handles sparsity effectively, while SVM uses a kernel trick for non-linearity.
- **Dataset Size**: 36,131 sequences require significant memory. Subsampling (e.g., 10,000 sequences) can be implemented for prototyping.
- **Interpretability**: Feature importance analysis (saved in `feature_importance_rf.csv`) identifies key k-mers, aiding biological interpretation.

## Future Extensions
- Experiment with different k-mer sizes (e.g., k=4 or k=5) to capture longer motifs.
- Incorporate deep learning (e.g., CNN or transformer like DNABERT) for improved performance.
- Visualize sequence logos for top k-mers using `logomaker` to highlight regulatory patterns.
- Develop a Flask API for real-time sequence classification.

## Why This Project Stands Out
- **Bioinformatics**: Processes real genomic data from UCI Genomic Benchmarks, demonstrating sequence analysis and feature engineering.
- **AI/ML**: Compares multiple models (Random Forest, SVM, Logistic Regression), showcasing versatility and evaluation skills.
- **Portfolio-Ready**: Includes interactive Colab notebook, clear visualizations, and detailed documentation, tailored for a Master's in AI committee.

## References
- UCI Genomic Benchmarks: [https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks)
- Dataset: `human_nontata_promoters`

## License
MIT License

## Contact
[Your Name] ([your.email@example.com](mailto:your.email@example.com))

*Created for a Master's in AI application, May 2025*