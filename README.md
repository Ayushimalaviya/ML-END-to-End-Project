# Student Performance Prediction - Ayushi Malaviya
________________________________________________________________________

### Introduction about Variable:

__The dataset__ goal is to predict the maths score for the student (Regression Analysis). 

__There are 7 independent variables__: 

  - __Gender__: gender of the students.
  - __Race/ethnicity__: race/ethnicity of the students.
  - __Parental level of education__:  parents' educational background
  - __lunch__:  Type of lunch students consume.
  - __test preparation course__: Have students take the preparation course or not?
  - __reading score__: marks obtained in reading out of 100 marks.
  - __writing score__: marks obtained in writing out of 100 marks.

### Target Variable:
__Maths score__: marks obtained in maths out of 100 marks.

__Dataset Source Link__: [https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977]

It is observed that the categorical variables 'gender', 'race/ethnicity, 'lunch', 'test preparation course', and 'Parental level of education' are ordinal in nature.

### AZURE Deployment LINK:
_____________________________________________________________________________

##### AZURE WEB APP LINK: [https://studentperformancepredicition.azurewebsites.net/]

### ScreenShots of UI:
______________________________________________________________________________

<img width="1420" alt="image" src="https://github.com/Ayushimalaviya/ML-END-to-End-Project/assets/61611744/2303fc19-bfbf-4fd6-bbd0-d8c84ab97271">

<img width="1420" alt="image" src="https://github.com/Ayushimalaviya/ML-END-to-End-Project/assets/61611744/2451d921-a06c-4969-9553-1b2fa3d8a8a5">

### Approach for the project
______________________________________________________________________________

#### 1. Data Ingestion

- In the Data Ingestion phase, the data is first read as a CSV file.
- Then, the data is split into training and testing sets and saved as separate CSV files.

#### 2. Data Transformation

- In this phase, a ColumnTransformer Pipeline is created.
- For numeric variables, SimpleImputer is applied with a strategy of median, followed by Standard Scaling.
- For categorical variables, SimpleImputer is applied with the most frequent strategy, followed by one-hot encoding. After this, the data is scaled with Standard Scaler.
- The preprocessor is saved as a pickle file.

#### 3. Model Training

- In this phase, the base model is tested, and the best-performing model found was the Ridge Regressor.
- Hyperparameter tuning is performed on the Adaboost Regressor and XGBoost Regressor models.
- Model selection is based on the R2 score.
- The selected model is saved as a pickle file.

#### 4. Prediction Pipeline

- This pipeline converts given data into a dataframe and contains various functions to load pickle files and predict the final results in Python.

#### 5. Flask App Creation

- A Flask app is created with a user interface to predict student math scores.

#### 6. Exploratory Data Analysis Notebook

- Link : [EDA Notebook](https://github.com/Ayushimalaviya/ML-END-to-End-Project/blob/main/notebook/EDA_Student_Performance.ipynb)

#### 7. Model Training Approach Notebook

- Link : [Model Training](https://github.com/Ayushimalaviya/ML-END-to-End-Project/blob/main/notebook/Model_training.ipynb)




