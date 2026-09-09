<<<<<<< HEAD
# Heart Disease Risk Prediction

A machine learning project that predicts the presence of heart disease using clinical patient data, comparing Logistic Regression and Random Forest models.

## Dataset
[Heart Disease UCI Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) — combines data from Cleveland, Hungary, Switzerland, and VA Long Beach medical centers. ~14 clinical features including age, cholesterol, chest pain type, and resting blood pressure.

## Approach
1. **Data Cleaning**: Handled missing values and discovered disguised missing data — cholesterol values of `0` (medically impossible) were skewing results and were treated as missing and imputed with the median.
2. **Exploratory Data Analysis**: Compared average age and cholesterol between patients with and without heart disease to validate the data made clinical sense before modeling.
3. **Modeling**: Trained and compared Logistic Regression and Random Forest classifiers.
4. **Evaluation**: Prioritized recall alongside accuracy, since missing a true heart disease case (false negative) carries higher real-world cost than a false alarm.

## Results

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 84.8% | 88.6% | 85.3% |
| **Random Forest** | **87.5%** | **89.8%** | **89.0%** |

Random Forest was selected as the final model due to its higher overall performance and better recall, meaning it misses fewer true heart disease cases. It also proved more robust to the cholesterol data quality issue than Logistic Regression, whose metrics improved noticeably after the cleaning fix.

## Key Finding
Initial EDA showed an unexpected *inverse* relationship between cholesterol and heart disease. Investigation revealed this was caused by missing values encoded as `0` rather than `NaN`, concentrated in the disease-positive group. After correcting for this, model recall improved for both models — a reminder that anomalous patterns should be investigated before being treated as real signal.

## Tools Used
Python, pandas, scikit-learn, matplotlib, Google Colab

## Future Improvements
- Test additional models (e.g. XGBoost, SVM)
- Hyperparameter tuning via GridSearchCV
- Deploy as an interactive Streamlit web app
- Check other columns (e.g. resting blood pressure) for similar disguised missing values

## How to Run
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Open `heart_disease_notebook.ipynb` in Jupyter or Google Colab
4. Run all cells
=======
# Machine-Learning
>>>>>>> b221b96a434980f2d591704a0c2cac4d94b1c781
