# Shopping Behavior Prediction using k-NN Classifier

## Overview
This project uses a k-Nearest Neighbors (k-NN) classifier to predict whether a user will make a purchase on an online shopping platform. The dataset is processed to extract features and target variables, and the model is evaluated based on its sensitivity and specificity.

## File Structure
- **shopping.ipynb**: Main Python script containing the data processing, model training, evaluation, and visualization.
- **shopping.csv**: The dataset used for training and testing the model.
- **README.md**: Documentation for the project (this file).

## Requirements
- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

Install the required libraries using the following command: pip install pandas numpy scikit-learn matplotlib seaborn


## Dataset
The dataset (shopping.csv) contains the following features:
- **Administrative, Administrative_Duration**: Number and time spent on administrative pages.
- **Informational, Informational_Duration**: Number and time spent on informational pages.
- **ProductRelated, ProductRelated_Duration**: Number and time spent on product-related pages.
- **BounceRates, ExitRates, PageValues, SpecialDay**: Metrics related to user engagement and behavior.
- **Month**: Month of the session (categorical).
- **OperatingSystems, Browser, Region, TrafficType**: Categorical features related to user demographics and access.
- **VisitorType**: Indicates if the visitor is a returning visitor.
- **Weekend**: Indicates if the session occurred on a weekend.
- **Revenue**: Target variable (1 if purchase was made, 0 otherwise).

## Workflow
1. **Load Data**:
   - Data is loaded from a CSV file and preprocessed.
   - Categorical features like `Month` and `VisitorType` are converted to numerical values.

2. **Split Data**:
   - The data is split into training and testing sets (80% train, 20% test).

3. **Train Model**:
   - A k-NN classifier with `k=1` is trained on the training data.

4. **Evaluate Model**:
   - Sensitivity (True Positive Rate) and Specificity (True Negative Rate) are calculated.
   - A confusion matrix is plotted for better visualization.

5. **Testing Functions**:
   - Functions are tested individually to ensure proper operation.

## Code
### Running the Main Script
The `main` function handles the entire process of data loading, model training, evaluation, and visualization. Run the script using: shopping.ipynb


### Functions
- load_data(filename)`: Reads the CSV file and converts the data into numerical format.
- train_model(evidence, labels)`: Trains a k-NN classifier on the given data.
- evaluate(labels, predictions)`: Calculates sensitivity and specificity.
- plot_confusion_matrix(y_test, predictions)`: Visualizes the confusion matrix using Seaborn.
- test_shopping()`: Tests the main functions for correctness.

## Output
**Metrics**:
  - Sensitivity (True Positive Rate): Indicates the model's ability to correctly identify positive cases (purchases).
  - Specificity (True Negative Rate): Indicates the model's ability to correctly identify negative cases (no purchases).

  **Confusion Matrix**: Provides a detailed view of true positives, true negatives, false positives, and false negatives.
    A confusion matrix is also displayed graphically.

## Notes
- Update the `FILENAME` variable with the correct path to your `shopping.csv` file.
- Adjust KNN_neighbors` in the `KNeighborsClassifier` if needed to improve performance.
- Ensure the dataset is properly formatted for successful loading and processing.

## Troubleshooting
- If the script fails to run, ensure all dependencies are installed.
- Verify the path to the `shopping.csv` file.
- Check the dataset for missing or incorrect values.

