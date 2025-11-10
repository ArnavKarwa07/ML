#!/usr/bin/env python
# coding: utf-8

# Name: Arnav Karwa
# PRN: 1032232194

# ## Describe various data sources. Perform EDA. Perform Data visualization to gain insights and feature engineering

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# Create your own binary class dataset with 5 features (Exploratory Variable) and atleast 30 observations. Numerical or categorical attributes in data values. CSV file.

# In[3]:


# data = {
#     "StudentID": list(range(1, 31)),
#     "Gender": [
#         "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female",
#         "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female",
#         "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"
#     ],
#     "Age": [
#         18, 19, 20, 18, 21, 19, 22, 20, 18, 19,
#         20, 21, 18, 19, 22, 20, 18, 19, 21, 20,
#         18, 19, 22, 20, 21, 18, 19, 20, 18, 21
#     ],
#     "GPA": [
#         3.2, 3.8, 2.9, 3.5, 2.1, 3.7, 2.5, 3.6, 3.1, 3.9,
#         3.3, 3.4, 2.3, 3.8, 3.0, 3.9, 2.7, 3.5, 2.4, 3.2,
#         3.1, 3.7, 2.6, 3.6, 3.4, 3.3, 2.8, 3.8, 3.0, 3.5
#     ],
#     "TestScore": [
#         85, 92, 78, 88, 65, 90, 72, 89, 83, 95,
#         87, 86, 70, 93, 82, 96, 75, 88, 68, 84,
#         81, 91, 73, 89, 85, 87, 76, 94, 80, 90
#     ],
#     "ExtraCurricular": [
#         "Yes", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes",
#         "Yes", "No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes",
#         "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "Yes", "Yes"
#     ],
#     "Admitted": [
#         1, 1, 0, 1, 0, 1, 0, 1, 1, 1,
#         1, 1, 0, 1, 1, 1, 0, 1, 0, 1,
#         0, 1, 0, 1, 1, 1, 0, 1, 1, 1
#     ]
# }


# In[4]:


# df = pd.DataFrame(data)
# df.to_csv("student_admission_dataset.csv", index=False)
# print("Dataset created and saved as student_admission_dataset.csv")
# print(f"Shape: {df.shape}")
# print("\nFirst 5 rows:")
# print(df.head())
# print("\nDataset Info:")
# print(df.info())
# print("\nTarget Variable Distribution:")
# print(df['Admitted'].value_counts())


# In[5]:


# Load dataset
df = pd.read_csv("diabetes.csv")
student_admission_dataset = pd.read_csv("student_admission_dataset.csv")


# In[ ]:


# Display dataset:
print("\nFirst 5 rows:")
df.head()


# In[7]:


# Display rows from row 4 to 12
df.iloc[3:12]


# In[8]:


# Display 1st 19 records
df.head(19)


# In[9]:


# Shape
df.shape


# In[10]:


#Info
df.info()


# In[11]:


# Describe
df.describe()


# Data cleaning

# In[12]:


# Identify missing values - Display the total count of null values
missing_values = df.isnull().sum()
missing_values


# In[13]:


# Display any one numeric attribute in the dataset
df['Age'].head()


# In[14]:


# Handle duplicates
df.drop_duplicates(inplace=False)


# In[15]:


# Replace missing values with meaningful values
# Zero imputation	

# df.fillna(0, inplace=False)  # Example of filling missing values with zero

# Identify the appropriate value using the central tendency

# df.fillna(df.mean(), inplace=False)  # Example of filling missing values with mean
# df.fillna(df.median(), inplace=False)  # Example of filling missing values with median
# df.fillna(df.mode(), inplace=False)  # Example of filling missing values with mode

# Simple imputor
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])


# In[16]:


# Split the dataset into independent and dependent variables
y = df['Outcome']  # Dependent variable
X = df.drop('Outcome', axis=1)  # Independent variables

X.head()
y.head()


# Data encoding

# In[17]:


# using student_admission_dataset
# One-hot encoding fo gender column
one_hot_encoder = OneHotEncoder(sparse_output=False)
gender_encoded = one_hot_encoder.fit_transform(student_admission_dataset[["Gender"]])
gender_encoded_df = pd.DataFrame(gender_encoded, columns=one_hot_encoder.get_feature_names_out(["Gender"]))
student_admission_encoded = pd.concat([student_admission_dataset.drop("Gender", axis=1), gender_encoded_df], axis=1)
student_admission_encoded

# Label encoding for extra-curricular activities
label_encoder = LabelEncoder()
extra_encoded = one_hot_encoder.fit_transform(student_admission_dataset[["ExtraCurricular"]])
extra_encoded_df = pd.DataFrame(extra_encoded, columns=one_hot_encoder.get_feature_names_out(["ExtraCurricular"]))
student_admission_encoded = pd.concat([student_admission_encoded.drop("ExtraCurricular", axis=1), extra_encoded_df], axis=1)
student_admission_encoded


# Data Transformation

# In[18]:


# Normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
# Convert the scaled data back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.select_dtypes(include=[np.number]).columns)

X_scaled_df.head()


# EDA

# In[19]:


# Pie chart
plt.figure(figsize=(8, 6))
plt.pie(y.value_counts(), labels=['Yes', 'No'], autopct='%1.1f%%')
plt.title('Distribution of Outcomes')
plt.show()


# **Inference:**
# The pie chart shows the distribution of the target variable (Outcome). It helps visualize the proportion of positive and negative cases in the dataset, indicating any class imbalance.

# In[20]:


# KDE
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Age', hue='Outcome', fill=True)
plt.title('KDE Plot of Age by Outcome')
plt.show()


# **Inference:**
# The KDE plot illustrates the distribution of 'Age' for each outcome class. It helps identify if age is a distinguishing factor between the classes.

# In[21]:


# Violin
plt.figure(figsize=(10, 6))
sns.violinplot(x='Outcome', y='BMI', data=df)
plt.title('Violin Plot of Age by Outcome')
plt.show()


# **Inference:**
# The violin plot shows the distribution and density of 'BMI' for each outcome class, highlighting differences in spread and central tendency.

# In[22]:


# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Glucose', y='BMI', hue='Outcome')
plt.title('Scatter Plot of Glucose vs BMI by Outcome')
plt.show()


# **Inference:**
# The scatter plot visualizes the relationship between 'Glucose' and 'BMI' colored by outcome, helping to spot trends or clusters that may separate the classes.

# In[23]:


# Univariate 
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')


# **Inference:**
# The histogram displays the distribution of 'Age' in the dataset, revealing skewness, modality, and potential outliers.

# In[24]:


# Boxplot for BMI
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['BMI'])
plt.title('Boxplot of BMI')
plt.xlabel('BMI')
plt.show()


# **Inference:**
# The boxplot for 'BMI' highlights the median, quartiles, and potential outliers, providing insight into the spread and central tendency of BMI values.

# In[25]:


# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')


# **Inference:**
# The correlation matrix visualizes the relationships between all numerical features. Strong positive or negative correlations with the target variable ('Outcome') highlight which features are most influential for prediction, while high correlations between features may indicate multicollinearity.

# Feature importance

# In[26]:


# Display scores of every feature with custom color
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(correlation_matrix['Outcome'].drop('Outcome')).sort_values(ascending=False)
feature_importance.plot(kind='bar', color='orange')
plt.title('Feature Importance')
plt.ylabel('Correlation with Outcome')
plt.xlabel('Features')
plt.show()


# In[29]:


# SelectKBest for feature extraction/importance

from sklearn.feature_selection import SelectKBest, f_classif

# Select the top k features based on univariate statistical tests
k = 5  # Number of top features to select
selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_indices = selector.get_support(indices=True)

# Print the selected feature names
print("Selected features:", X.columns[selected_indices].tolist())

