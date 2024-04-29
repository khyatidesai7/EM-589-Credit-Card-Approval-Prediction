
#pip install --upgrade scikit-learn imbalanced-learn


# DATA READING, CLEANING, PREPROCESSING AND MERGING


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier


df = pd.read_csv("application_record.csv")
df.head()


df.shape


df.info()


df['ID'].duplicated().sum()


# Drop duplicates
df = df.drop_duplicates(subset='ID',keep='first')


df.shape

df.columns[1:]


df.isnull().sum()


#Selecting only the uniques values of categoricals
df['CODE_GENDER'].unique()
df['FLAG_OWN_CAR'].unique()
df['FLAG_OWN_REALTY'].unique()
df['NAME_INCOME_TYPE'].unique()
df['NAME_EDUCATION_TYPE'].unique()
df['NAME_FAMILY_STATUS'].unique()
df['NAME_HOUSING_TYPE'].unique()


df['FLAG_MOBIL'].value_counts()


df['FLAG_WORK_PHONE'].unique()

df['FLAG_PHONE'].unique()

df['FLAG_EMAIL'].unique()

#Find the null values
df['OCCUPATION_TYPE'].value_counts(dropna=False)

#Fill the null values with the term not specified
df['OCCUPATION_TYPE'].fillna('not_specified',inplace=True)

df['OCCUPATION_TYPE'].value_counts(dropna=False)

df1 = pd.read_csv("credit_record.csv")
df1.head()

df1.shape

df1.info

df1.duplicated().sum()

df1['MONTHS_BALANCE'].unique()

df1['STATUS'].unique()

df1[df1['STATUS'].isin(['X','C'])]

df1['ID'].nunique()

#Assign 1 to X and C and assign 0 to the remaining values for classification
df1['target'] = df1['STATUS'].replace({'C': 1, 'X': 1})
df1['target'] = df1['target'].astype(int)
df1.loc[df1['target']==0,'target']=0
df1.loc[df1['target']>=2,'target']=0
df1

df2=pd.DataFrame(df1.groupby(['ID'])['target'].agg("max")).reset_index()
df2["target"].value_counts()
df2
#this is the cleaned credit dataset

df[df['DAYS_EMPLOYED']>=0]['DAYS_EMPLOYED'].value_counts()

df['DAYS_EMPLOYED'].replace(365243,0,inplace=True)

df[df['DAYS_EMPLOYED']>=0]['DAYS_EMPLOYED'].value_counts()

df['AGE_YEARS']=round(-df['DAYS_BIRTH']/365.2425,0)

df['YEARS_EMPLOYED']=round(-df['DAYS_EMPLOYED']/365.2425)
df.loc[df['YEARS_EMPLOYED']<0,'YEARS_EMPLOYED']=0

df.drop(columns=["DAYS_BIRTH","DAYS_EMPLOYED"],inplace=True)

df['ID'].duplicated().sum()
df.shape

df[df['AMT_INCOME_TOTAL']>540000]

df.drop(columns=["FLAG_MOBIL"],inplace=True)
df.drop(columns=["FLAG_WORK_PHONE"],inplace=True)
df.drop(columns=["FLAG_EMAIL"],inplace=True)
df.drop(columns=["FLAG_PHONE"],inplace=True)

df.head()

# Define a dictionary mapping the values to be replaced to the replacement value
replacement = {'Lower secondary': 'NO GED',
                   'Secondary / secondary special': 'NO GED',
                   'Incomplete higher': 'NO GED'}

# Replace the values in the 'NAME_EDUCATION_TYPE' column using the dictionary
df['NAME_EDUCATION_TYPE'].replace(replacement, inplace=True)

df

# Define a dictionary mapping the values to be replaced to the replacement value
replacement_value = {'Civil marriage': 'Married'}

# Replace the values in the 'NAME_EDUCATION_TYPE' column using the dictionary
df['NAME_FAMILY_STATUS'].replace(replacement_value, inplace=True)

df

df = df[df['CNT_FAM_MEMBERS'] <= 9]

df.shape

df.info()

df3=pd.merge(df2,df,how='inner',on=['ID'])

df3

start_df = pd.DataFrame(df1.groupby(['ID'])['MONTHS_BALANCE'].agg(min)).reset_index()

start_df.rename(columns={'MONTHS_BALANCE': 'ACCOUNT_LENGTH'}, inplace=True)

start_df['ACCOUNT_LENGTH'] = -start_df['ACCOUNT_LENGTH']

start_df

df3=pd.merge(df3,start_df,how='inner', on=['ID'])

df3

df3.describe()

# Specify the file path where you want to save the CSV file
file_path = 'merged_file.csv'

# Save the DataFrame to a CSV file
df3.to_csv(file_path, index=False)  # Set index=False to exclude row numbers in the output

print("DataFrame successfully saved to", file_path)

#Label encoding for all the categorical variables
y = df3["target"]
X = df3.drop(["target","ID"], axis=1)
le = LabelEncoder()
X['CODE_GENDER'] = le.fit_transform(X['CODE_GENDER'])
X['FLAG_OWN_CAR'] = le.fit_transform(X['FLAG_OWN_CAR'])
X['FLAG_OWN_REALTY'] = le.fit_transform(X['FLAG_OWN_REALTY'])
X['NAME_INCOME_TYPE'] = le.fit_transform(X['NAME_INCOME_TYPE'])
X['NAME_EDUCATION_TYPE'] = le.fit_transform(X['NAME_EDUCATION_TYPE'])
X['NAME_FAMILY_STATUS'] = le.fit_transform(X['NAME_FAMILY_STATUS'])
X['NAME_HOUSING_TYPE'] = le.fit_transform(X['NAME_HOUSING_TYPE'])
X['OCCUPATION_TYPE'] = le.fit_transform(X['OCCUPATION_TYPE'])
X.head()

# SPLITTING THE DATASET

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)

# STANDARDISING THE DATA

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# SMOTE TO REMOVE SKEWNESS

# Importing the 'SMOTE' (Synthetic Minority Over-sampling Technique) from 'imblearn.over_sampling'
from imblearn.over_sampling import SMOTE

# Creating an instance of the 'SMOTE' object with a random state of 42
smote = SMOTE(random_state=42)

# Resampling the training data (X_train and y_train) using the 'SMOTE' technique to balance the classes
# The number of samples in the minority class will be increased by generating synthetic samples
X_train, y_train = smote.fit_resample(X_train_std, y_train)

# RECURVISE FEATURE ELIMINATION USING RANDOM FOREST CLASSIFIER
# Setting the number of features to select using RFE to 8
n_features_to_select = 8

# Creating an instance of 'RandomForestClassifier' with a random state of 123
rf_classifier = RandomForestClassifier(n_estimators=500 , random_state=123)

# Creating an instance of 'RFE' with the RandomForestClassifier as the estimator and the specified number of features to select
rf = RFE(estimator=rf_classifier, n_features_to_select=n_features_to_select)

# Fitting the RFE to the resampled training data (X_train_resampled, y_train_resampled) to select the best features
rf.fit(X_train, y_train)


# GRAPH SHOWING FEATURE IMPORTANCES

# Getting the names of the selected features based on RFE support
selected_features = X.columns[rf.support_]

# Getting the feature importances from the RandomForestClassifier
feature_importances = rf.estimator_.feature_importances_

# Sorting the feature indices based on their importances in descending order
sorted_indices = np.argsort(feature_importances)[::-1]

# Sorting the feature names and importances based on the sorted indices
sorted_features = selected_features[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Creating a bar plot to visualize the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importances)), sorted_importances)
plt.xticks(range(len(sorted_importances)), sorted_features, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances with RFE and Random Forest")
plt.show()

# GRADIENT BOOSTING
# Instantiate the Gradient Boosting classifier
gb_model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=8, random_state=123)
gb_model.fit(X_train,y_train)
y_predict_gb = gb_model.predict(X_test_std)
accuracy_gb = accuracy_score(y_test, y_predict_gb)
y_pred_train = gb_model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("Gradient Boosting Training Accuracy Score before feature selection:", training_accuracy)
print('Gradient Boosting Test Accuracy Score before feature selection: {:.5f}'.format(accuracy_gb))

# Perform feature selection using RFE
rfe = make_pipeline(rf, gb_model)
rfe.fit(X_train, y_train)
y_predict_gb = rfe.predict(X_test_std)
accuracy_gb = accuracy_score(y_test, y_predict_gb)
y_pred_train = rfe.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("Gradient Boosting Training Accuracy Score before feature selection:", training_accuracy)
print('Gradient Boosting Test Accuracy Score after feature selection: {:.5f}'.format(accuracy_gb))

# Confusion matrix
conf_matrix_gb = confusion_matrix(y_test, y_predict_gb)
print('Confusion Matrix for Gradient Boosting:')
print(pd.DataFrame(conf_matrix_gb))

# Plot confusion matrix
disp_gb = ConfusionMatrixDisplay(conf_matrix_gb)
disp_gb.plot()
plt.title('Confusion Matrix: Gradient Boosting')
plt.show()


print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict_gb)))
print('F1 Score is {:.5}'.format(f1_score(y_test, y_predict_gb)))
print('Precision Score is {:.5}'.format(precision_score(y_test, y_predict_gb)))
print('Recall Score is {:.5}'.format(recall_score(y_test, y_predict_gb)))

# XGBOOSTING CLASSIFIER
# Instantiate the XGBoost classifier
xgb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=8, random_state=123)
xgb_model.fit(X_train,y_train)
y_predict_xgb = xgb_model.predict(X_test_std)
accuracy_xgb = accuracy_score(y_test, y_predict_xgb)
y_pred_train = xgb_model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("XG Boost Training Accuracy Score before feature selection:", training_accuracy)
print('XG Boost Test Accuracy Score before feature selection: {:.5f}'.format(accuracy_xgb))

rfe = make_pipeline(rf, xgb_model)
rfe.fit(X_train, y_train)
y_predict_xgb = rfe.predict(X_test_std)
accuracy_xgb = accuracy_score(y_test, y_predict_xgb)
y_pred_train = rfe.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("XG Boost Training Accuracy Score before feature selection:", training_accuracy)
print('XG Boost Test Accuracy Score after feature selection: {:.5f}'.format(accuracy_xgb))

# Confusion matrix
conf_matrix_xgb = confusion_matrix(y_test, y_predict_xgb)
print('Confusion Matrix for XG Boost:')
print(pd.DataFrame(conf_matrix_xgb))

# Plot confusion matrix
disp_xgb = ConfusionMatrixDisplay(conf_matrix_xgb)
disp_xgb.plot()
plt.title('Confusion Matrix: XG Boost')
plt.show()


print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict_xgb)))
print('F1 Score is {:.5}'.format(f1_score(y_test, y_predict_xgb)))
print('Precision Score is {:.5}'.format(precision_score(y_test, y_predict_xgb)))
print('Recall Score is {:.5}'.format(recall_score(y_test, y_predict_xgb)))

# ADABOOST CLASSIFIER
# Instantiate the Ada Boost classifier
ada_model = AdaBoostClassifier(n_estimators=500,  random_state=123, algorithm='SAMME')
ada_model.fit(X_train,y_train)
y_predict_ada = ada_model.predict(X_test_std)
accuracy_ada = accuracy_score(y_test, y_predict_ada)
y_pred_train = ada_model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("Ada Boost Training Accuracy Score before feature selection:", training_accuracy)
print('Ada Boost Test Accuracy Score before feature selection: {:.5f}'.format(accuracy_ada))


rfe = make_pipeline(rf, ada_model)
rfe.fit(X_train, y_train)
y_predict_ada = rfe.predict(X_test_std)
accuracy_ada = accuracy_score(y_test, y_predict_ada)
y_pred_train = rfe.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("Ada Boost Training Accuracy Score before feature selection:", training_accuracy)
print('Ada Boost Test Accuracy Score after feature selection: {:.5f}'.format(accuracy_ada))

# Confusion matrix
conf_matrix_ada = confusion_matrix(y_test, y_predict_ada)
print('Confusion Matrix for Ada Boost:')
print(pd.DataFrame(conf_matrix_ada))

# Plot confusion matrix
disp_ada = ConfusionMatrixDisplay(conf_matrix_ada)
disp_ada.plot()
plt.title('Confusion Matrix: Ada Boost')
plt.show()

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict_ada)))
print('F1 Score is {:.5}'.format(f1_score(y_test, y_predict_ada)))
print('Precision Score is {:.5}'.format(precision_score(y_test, y_predict_ada)))
print('Recall Score is {:.5}'.format(recall_score(y_test, y_predict_ada)))

# RANDOM FOREST CLASSIFIER

# Instantiate the Random Forest classifier

model=RandomForestClassifier(n_estimators=500, random_state=123)
model.fit(X_train,y_train)
y_predict = model.predict(X_test_std)
accuracy = accuracy_score(y_test, y_predict)
y_pred_train = model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("Random Forest Training Accuracy Score before feature selection:", training_accuracy)
print('Random Forest Test Accuracy Score before feature selection: {:.5f}'.format(accuracy))


rfe = make_pipeline(rf, model)
rfe.fit(X_train, y_train)
y_predict = rfe.predict(X_test_std)
accuracy = accuracy_score(y_test, y_predict)
y_pred_train = rfe.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("Random Forest Training Accuracy Score before feature selection:", training_accuracy)
print('Random Forest Test Accuracy Score after feature selection: {:.5f}'.format(accuracy))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
print('Confusion Matrix for Random Forest:')
print(pd.DataFrame(conf_matrix))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.title('Confusion Matrix: Random Forest')
plt.show()

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print('F1 Score is {:.5}'.format(f1_score(y_test, y_predict)))
print('Precision Score is {:.5}'.format(precision_score(y_test, y_predict)))
print('Recall Score is {:.5}'.format(recall_score(y_test, y_predict)))

# DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=12,
                               min_samples_split=8,
                               random_state=123)
dt_model.fit(X_train,y_train)
y_predict_dt = dt_model.predict(X_test_std)
accuracy_dt = accuracy_score(y_test, y_predict_dt)
y_pred_train = dt_model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("Decision Tree Training Accuracy Score before feature selection:", training_accuracy)
print('Decision Tree Test Accuracy Score before feature selection: {:.5f}'.format(accuracy_dt))


rfe = make_pipeline(rf, dt_model)
rfe.fit(X_train, y_train)
y_predict_dt = rfe.predict(X_test_std)
accuracy_dt = accuracy_score(y_test, y_predict_dt)
y_pred_train = rfe.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("Decision Tree Training Accuracy Score after feature selection:", training_accuracy)
print('Decision Tree Test Accuracy Score after feature selection: {:.5f}'.format(accuracy_dt))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict_dt)
print('Confusion Matrix for Decision Tree:')
print(pd.DataFrame(conf_matrix))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.title('Confusion Matrix: Decision Tree')
plt.show()

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict_dt)))
print('F1 Score is {:.5}'.format(f1_score(y_test, y_predict_dt)))
print('Precision Score is {:.5}'.format(precision_score(y_test, y_predict_dt)))
print('Recall Score is {:.5}'.format(recall_score(y_test, y_predict_dt)))

# LOGISTIC REGRESSION

# Instantiate the Logistic Regression

lg_model = LogisticRegression(C=0.1,
                           solver='liblinear',
                           multi_class='ovr',
                           random_state=123,
                           penalty='l1')
lg_model.fit(X_train, y_train)
y_predict_lg = lg_model.predict(X_test_std)
accuracy_lg = accuracy_score(y_test, y_predict_lg)
y_pred_train = lg_model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("Logistic Regression Training Accuracy Score before feature selection:", training_accuracy)
print('Logistic Regression Test Accuracy Score before feature selection: {:.5f}'.format(accuracy_lg))


rfe = make_pipeline(rf, lg_model)
rfe.fit(X_train, y_train)
y_predict_lg = rfe.predict(X_test_std)
accuracy_lg = accuracy_score(y_test, y_predict_lg)
y_pred_train = rfe.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred_train)
print("Logistic Regression Training Accuracy Score before feature selection:", training_accuracy)
print('Logistic Regression Test Accuracy Score after feature selection: {:.5f}'.format(accuracy_lg))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict_lg)
print('Confusion Matrix for Logistic Regression:')
print(pd.DataFrame(conf_matrix))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.title('Confusion Matrix: Logistic Regression')
plt.show()

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict_lg)))
print('F1 Score is {:.5}'.format(f1_score(y_test, y_predict_lg)))
print('Precision Score is {:.5}'.format(precision_score(y_test, y_predict_lg)))
print('Recall Score is {:.5}'.format(recall_score(y_test, y_predict_lg)))

# DATA ANALYSIS AND VISUALIZATION

# Load the dataset
df4 = pd.read_csv('application_record.csv')

# 1. Histogram of Age
plt.figure(figsize=(10, 6))
sns.histplot(df3['AGE_YEARS'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.show()

# 2. Bar plot of Gender Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='CODE_GENDER', data=df3)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 3. Box plot of Income Distribution by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='CODE_GENDER', y='AMT_INCOME_TOTAL', data=df3)
plt.title('Income Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Income')
plt.show()

# 4. Bar plot of Education Level
plt.figure(figsize=(10, 6))
sns.countplot(x='NAME_EDUCATION_TYPE', data=df3)
plt.title('Education Level Distribution')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 5. Bar plot of Marital Status
plt.figure(figsize=(10, 6))
sns.countplot(x='NAME_FAMILY_STATUS', data=df3)
plt.title('Marital Status Distribution')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 6. Pair plot of Income, Age, and Family Size
sns.pairplot(df3[['AMT_INCOME_TOTAL', 'AGE_YEARS', 'CNT_FAM_MEMBERS']])
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df3, x='AGE_YEARS', hue='CODE_GENDER', bins=20, kde=True)
plt.title('Distribution of Ages by Gender')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

categorical_vars = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
for var in categorical_vars:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df3, x=var)
    plt.title(f'Count of {var}')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


# Numeric variables
numeric_vars = ['AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']

# Categorical variables
categorical_vars = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 
                    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
                        'OCCUPATION_TYPE']

# Visualize numeric variables
for var in numeric_vars:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='target', y=var, data=df3)
    plt.title(f'{var} vs. Approval Status')
    plt.xlabel('Approval Status')
    plt.ylabel(var)
    plt.xticks([0, 1], ['Not Approved', 'Approved'])
    plt.show()

# Visualize categorical variables
for var in categorical_vars:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=var, hue='target', data=df3)
    plt.title(f'{var} vs. Approval Status')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Approval Status', loc='upper right', labels=['Not Approved', 'Approved'])
    plt.show()



