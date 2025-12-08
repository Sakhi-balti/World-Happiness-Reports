# World Happiness Score Prediction Project Report

## 1. Project Overview

The objective of this project is to predict the World Happiness Score for a given country based on socio-economic indicators using Machine Learning. The key features include:

Year

GDP per Capita

Social Support

Life Expectancy

Freedom

Generosity

Corruption Perception

This project was developed using a CI/CD (Continuous Integration/Continuous Deployment) pipeline approach. The focus was on automating model training, evaluation, and testing stages (Continuous Integration), while deployment to production (Continuous Deployment) is not included in this phase.

## 2. Project Architecture

The project is organized in a modular way with the following components:
| Component | Description |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `data_ingestion.py` | Handles reading raw data files and splitting into train/test datasets. |
| `data_transformation.py` | Cleans the data, performs feature engineering, handles missing values, and applies normalization/scaling. |
| `model_trainer.py` | Trains multiple models, evaluates them, performs hyperparameter tuning, and selects the best model. |
| `predict_pipeline.py` | Defines a prediction pipeline for user input data and converts input into a suitable format for the model. |
| `app.py` | Flask web application providing a GUI for users to enter input and get happiness score predictions. |

Component Description
data_ingestion.py Handles reading raw data files and splitting into train/test datasets.
data_transformation.py Cleans the data, performs feature engineering, handles missing values, and applies normalization/scaling.
model_trainer.py Trains multiple models, evaluates them, performs hyperparameter tuning, and selects the best model.
predict_pipeline.py Defines a prediction pipeline for user input data and converts input into a suitable format for the model.
app.py Flask web application providing a GUI for users to enter input and get happiness score predictions.

## 3. Data Preprocessing Techniques

### Handling Missing Values

Missing data was handled by:

Dropping columns or rows with excessive missing values.

Imputation with statistical measures (mean, median) for numeric columns.

Ensuring consistent input for ML models.

### Feature Selection

Selected features based on domain knowledge from the World Happiness Report.

Removed irrelevant or redundant columns to reduce noise.

Focused on numeric columns (GDP per Capita, Social Support, etc.) for regression modeling.

### Normalization/Scaling

Applied StandardScaler to numeric features to normalize them.

This ensures that features like GDP per Capita and Social Support are on a comparable scale.

Scaling improves model convergence and performance, especially for gradient-based models.

### Data Transformation

Converted categorical features (if any) using One-Hot Encoding.

Ensured all features matched the training dataset schema before passing to the model.

## 4. Machine Learning Models & Techniques

_Candidate Models_

_RandomForestRegressor_

_GradientBoostingRegressor_

_AdaBoostRegressor_

_XGBRegressor_

_LinearRegression_

### Hyperparameter Tuning

Used GridSearchCV / RandomizedSearchCV to optimize parameters like:

Number of estimators (n_estimators)

Maximum depth (max_depth)

Learning rate (learning_rate)

Goal: Improve R² score and reduce prediction error.

### Model Evaluation

Metrics used:

R² Score (Train/Test)

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

The best model is selected based on the highest test R² score.

Continuous Integration (CI)

Automated scripts for:

Running data ingestion and preprocessing

Model training and evaluation

Saving trained models and preprocessor objects

Ensures that any change in code or data triggers a new training pipeline.

Logging is implemented to track pipeline execution and errors.

## 1. Prediction Pipeline

Users can enter the features via a Flask web interface.

Input is validated with proper ranges for each feature based on historical World Happiness Report data:

Social Support: 0.23 – 0.99

Corruption: 0.04 – 0.98

Life Expectancy: 25 – 80 years

GDP per Capita (log scale): 5.6 – 11.9

Freedom: 0.23 – 0.99

Generosity: -0.34 – 0.69

Input is converted to a pandas DataFrame.

Data is scaled using the saved preprocessor before prediction.

The prediction result is returned to the user interface.

## 6. CI/CD Implementation

Continuous Integration (CI):

The project automates testing of new code commits.

Any change in data ingestion, transformation, or model training triggers a new pipeline.

Automated saving of preprocessor and trained models in the artifacts/ folder.

### Continuous Deployment (CD):

Not implemented in this project. Deployment could involve:

Hosting Flask app on a cloud server.

Integrating with Docker or Kubernetes for scalable production deployment.

## 7. Key Achievements

Successfully automated the entire ML pipeline from data ingestion to model evaluation.

Built a user-friendly web interface using Flask.

Implemented proper data validation for user inputs.

Applied feature scaling, missing value handling, and model selection systematically.

Enabled reproducibility by saving preprocessing and model objects.

## 8. Limitations & Future Work

Currently, deployment is local only. Future work can involve:

Deploying on a cloud server (AWS/GCP/Heroku).

Adding real-time updates for new World Happiness datasets.

Integrating CI/CD with Docker for containerized deployment.

Expanding the input features to include more socio-economic indicators.

## 9. Conclusion

This project demonstrates a complete machine learning lifecycle using a CI approach:

From raw data to feature engineering

Automated training and evaluation of multiple models

Building a robust prediction pipeline with data validation

Providing a professional web interface for predictions

The project emphasizes best practices in ML engineering, ensuring reproducibility, automation, and clean data handling.
