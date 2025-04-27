# Breast Cancer Classification Project

This project demonstrates the classification of breast cancer tumors as either benign or malignant using machine learning models. The dataset used is the Breast Cancer dataset from scikit-learn. The project includes data preprocessing, noise simulation, model training, evaluation, and visualization.

## Project Overview

1. **Dataset**: The Breast Cancer dataset from scikit-learn.
2. **Goal**: Predict whether a tumor is benign or malignant.
3. **Models**: 
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
4. **Key Steps**:
   - Data standardization.
   - Adding Gaussian noise to simulate measurement errors.
   - Training and evaluating models on noisy data.
   - Visualizing results using histograms, scatterplots, and confusion matrices.

## Features

- **Data Preprocessing**: Standardizes the dataset and adds Gaussian noise to simulate real-world measurement errors.
- **Model Training**: Implements KNN and SVM classifiers.
- **Evaluation Metrics**: Includes accuracy, classification reports, and confusion matrices.
- **Visualization**: Provides histograms, scatterplots, and heatmaps for better understanding of the data and model performance.

## How to Run

1. Clone the repository and navigate to the project directory.
2. Install the required Python libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
3. Open the Jupyter Notebook file `Breast_Cancer_Classification.ipynb`.
4. Run the notebook cells sequentially to execute the project.

## Results

- The project evaluates the performance of KNN and SVM models on noisy data.
- It provides insights into the impact of noise on model performance and the importance of evaluation metrics.

## Visualizations

- Histograms and scatterplots to compare original and noisy features.
- Confusion matrices to analyze model predictions.

## Conclusion

This project is not about finding the best classifier but about understanding and interpreting evaluation metrics in the context of a real-world problem. It provides valuable intuition for working with noisy data and evaluating machine learning models.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
