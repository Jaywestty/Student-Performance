# Import Necessary Libraries

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

#Load data
df = pd.read_csv('student_habits_performance.csv')

#Select columns needed
num_cols = ['age','study_hours_per_day','social_media_hours','netflix_hours','attendance_percentage','sleep_hours','exercise_frequency','mental_health_rating']
cat_cols = ['gender','part_time_job','diet_quality','internet_quality','extracurricular_participation']

parental_education_level = ['None', 'High School', 'Bachelor', 'Master']

#create a new multi-class target feature
def performance(exam_score):
    if exam_score < 60:
        return 'Fair'
    elif exam_score < 80:
        return 'Good'
    else:
        return 'Excellent'
    
df['Perfomance'] = df['exam_score'].apply(performance)

#Split data into features and target variables
X = df.drop(columns=['student_id', 'exam_score', 'Perfomance'], axis=1)
y = df['Perfomance']

#Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Build preprocessor pipeline
numeric_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')), #to handle missing values in numerical columns
    ('scaler', StandardScaler()) #scale the numerical columns
])

categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')), #to handle missing values in catgorical columns
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) #encoding categorical columns
])

ordinal_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')), #to handlle missing values in the ordinal column
    ('ordinal', OrdinalEncoder(categories=[parental_education_level])) #ordinal encoding
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols), #for numerical columns
    ('cat', categorical_transformer, cat_cols), #for categorical columns
    ('ord', ordinal_transformer, ['parental_education_level']) #for ordinal column
])

#Build model
model = LogisticRegression()

#pipline
pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

pipeline.fit(X_train, y_train)

#Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print('Model saved succesfully')

sample_data = {
    'age': 21,
    'study_hours_per_day': 1.0,
    'social_media_hours': 4.5,
    'netflix_hours': 3.0,
    'attendance_percentage': 40.0,
    'sleep_hours': 9.0,
    'exercise_frequency': 3,
    'mental_health_rating': 6,
    'gender': 'Male',
    'part_time_job': 'Yes',
    'diet_quality': 'Excellent',
    'internet_quality': 'Good',
    'extracurricular_participation': 'Yes',
    'parental_education_level': 'Bachelor'
    }

#create a dataframe

sample_df = pd.DataFrame([sample_data])

#Predict
pipeline.predict(sample_df)





