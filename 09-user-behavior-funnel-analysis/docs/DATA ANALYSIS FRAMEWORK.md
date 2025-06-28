# Comprehensive Data Analysis Framework with Python

This framework provides a structured workflow for data analysis projects using Python. It covers each stage of the analysis process with key techniques, libraries, and best practices.

## 1. Project Setup and Data Collection

### 1.1 Environment Setup

```python
# Create a dedicated environment for each project
# conda create -n project_name python=3.10
# conda activate project_name

# Essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure visualization settings
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.2)
%matplotlib inline

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
```

### 1.2 Data Collection

```python
# Local files
df = pd.read_csv('data.csv')
df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Database connections
from sqlalchemy import create_engine
engine = create_engine('sqlite:///database.db')
df_sql = pd.read_sql('SELECT * FROM table_name', engine)

# APIs
import requests
response = requests.get('https://api.example.com/data')
data = response.json()
df_api = pd.DataFrame(data)

# Web scraping
import requests
from bs4 import BeautifulSoup
response = requests.get('https://example.com')
soup = BeautifulSoup(response.text, 'html.parser')
# Extract data from soup
```

## 2. Data Understanding and Cleaning

### 2.1 Initial Data Exploration

```python
# Basic information
print(f"Dataset shape: {df.shape}")
print(f"\nDataset info:")
df.info()
print(f"\nDescriptive statistics:")
display(df.describe(include='all').T)

# Check first and last rows
display(df.head())
display(df.tail())

# Column types
print(df.dtypes)

# Unique values in categorical columns
for col in df.select_dtypes(include=['object', 'category']).columns:
    print(f"\n{col} unique values ({df[col].nunique()}):")
    print(df[col].value_counts().head())
```

### 2.2 Missing Values

```python
# Identify missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})
missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)
display(missing_df)

# Check rows with any missing values
display(df[df.isnull().any(axis=1)].head())

# Summary of missing values
print("\nTotal number of missing values:", df.isnull().sum().sum())
print("Percentage of missing values in entire dataset:", 
      (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, "%")

# Visualize missing values
import missingno as msno
msno.matrix(df)
plt.show()
msno.heatmap(df)
plt.show()

# Handle missing values
# 1. Remove columns with too many missing values
df = df.drop(columns=['col_with_many_missing'])

# 2. Drop rows with missing values
df_cleaned = df.dropna()  # Be careful with this approach

# 3. Fill missing values
# Numerical: mean, median, mode
df['numeric_col'] = df['numeric_col'].fillna(df['numeric_col'].median())

# Check how many rows were dropped 
print(f"Original rows: {len(df)}") 
print(f"Rows after dropping NA: {len(df_cleaned)}") 
print(f"Rows dropped: {len(df) - len(df_cleaned)}")

# Categorical: mode or a new category
df['cat_col'] = df['cat_col'].fillna(df['cat_col'].mode()[0])
# or
df['cat_col'] = df['cat_col'].fillna('Unknown')

# 4. Advanced imputation
from sklearn.impute import SimpleImputer, KNNImputer
# Simple imputation
imputer = SimpleImputer(strategy='median')
df[['col1', 'col2']] = imputer.fit_transform(df[['col1', 'col2']])

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
df[['col1', 'col2']] = knn_imputer.fit_transform(df[['col1', 'col2']])
```

### 2.3 Duplicates

```python
# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Display the duplicated rows
if duplicate_count > 0:
    print("\nFirst 5 duplicate rows:")
    display(df[df.duplicated(keep='first')].head())
    
    print("\nDuplicate rows by DeviceIDHash and EventTimestamp:")
    device_time_dupes = df.duplicated(subset=['DeviceIDHash', 'EventTimestamp'], keep=False)
    display(df[device_time_dupes].sort_values(['DeviceIDHash', 'EventTimestamp']).head())
else:
    print("No duplicates to display")

# Remove duplicates if necessary
# Drop duplicate rows and reset index
df = df.drop_duplicates().reset_index(drop=True)

print(f"\nShape of dataset after removing duplicates: {df.shape}")
print(f"Number of rows removed: {duplicate_rows}")

```

### 2.4 Data Type Conversion

```python
# Convert data types
df['date_col'] = pd.to_datetime(df['date_col'])
df['numeric_col'] = pd.to_numeric(df['numeric_col'], errors='coerce')
df['category_col'] = df['category_col'].astype('category')

# Extract datetime components
df['year'] = df['date_col'].dt.year
df['month'] = df['date_col'].dt.month
df['day'] = df['date_col'].dt.day
df['day_of_week'] = df['date_col'].dt.day_name()
```

### 2.5 Outlier Detection and Handling

```python
# Z-score method
from scipy import stats
z_scores = stats.zscore(df['numeric_col'])
abs_z_scores = np.abs(z_scores)
outliers_z = (abs_z_scores > 3)
print(f"Outliers (Z-score): {outliers_z.sum()}")

# IQR method
Q1 = df['numeric_col'].quantile(0.25)
Q3 = df['numeric_col'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = ((df['numeric_col'] < lower_bound) | (df['numeric_col'] > upper_bound))
print(f"Outliers (IQR): {outliers_iqr.sum()}")

# Visualize outliers with boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['numeric_col'])
plt.title('Boxplot to Identify Outliers')
plt.show()

# Handle outliers
# 1. Remove
df_no_outliers = df[~outliers_iqr]

# 2. Cap/Floor
df['numeric_col_capped'] = df['numeric_col'].clip(lower_bound, upper_bound)

# 3. Transform (e.g., log transformation)
import numpy as np
df['numeric_col_log'] = np.log1p(df['numeric_col'])  # log(1+x) to handle zeros
```

## 3. Exploratory Data Analysis (EDA)

### 3.1 Univariate Analysis

```python
# Numerical variables
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    
    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print(f"\nStatistics for {col}:")
    print(df[col].describe())

# Categorical variables
for col in df.select_dtypes(include=['object', 'category']).columns:
    plt.figure(figsize=(12, 6))
    
    # Count plot
    count_order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=count_order)
    plt.title(f'Count of {col}')
    plt.tight_layout()
    plt.show()
    
    # Frequency table
    print(f"\nFrequency table for {col}:")
    freq = df[col].value_counts(normalize=True).reset_index()
    freq.columns = [col, 'Frequency (%)']
    freq['Frequency (%)'] = freq['Frequency (%)'] * 100
    display(freq)
```

### 3.2 Bivariate Analysis

```python
# Numerical vs Numerical
plt.figure(figsize=(10, 6))
sns.scatterplot(x='numeric_col1', y='numeric_col2', data=df)
plt.title('Relationship between numeric_col1 and numeric_col2')
plt.show()

# Calculate correlation
correlation = df['numeric_col1'].corr(df['numeric_col2'])
print(f"Correlation coefficient: {correlation:.2f}")

# Categorical vs Numerical
plt.figure(figsize=(12, 6))
sns.boxplot(x='cat_col', y='numeric_col', data=df)
plt.title('numeric_col by cat_col')
plt.xticks(rotation=45)
plt.show()

# ANOVA to test for significant differences
from scipy import stats
categories = df['cat_col'].unique()
anova_data = [df[df['cat_col'] == cat]['numeric_col'].dropna() for cat in categories]
f_stat, p_value = stats.f_oneway(*anova_data)
print(f"ANOVA results: F-statistic = {f_stat:.2f}, p-value = {p_value:.4f}")

# Categorical vs Categorical
pd.crosstab(df['cat_col1'], df['cat_col2'], normalize='index') * 100
```

### 3.3 Multivariate Analysis

```python
# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df[['numeric_col1', 'numeric_col2', 'numeric_col3', 'cat_col']], 
             hue='cat_col', diag_kind='kde')
plt.show()

# 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['numeric_col1'], df['numeric_col2'], df['numeric_col3'])
ax.set_xlabel('numeric_col1')
ax.set_ylabel('numeric_col2')
ax.set_zlabel('numeric_col3')
plt.title('3D Relationship')
plt.show()
```

### 3.4 Time Series Analysis (if applicable)

```python
# Set date as index
df_ts = df.copy()
df_ts.set_index('date_col', inplace=True)

# Resample to get monthly data
monthly_data = df_ts['numeric_col'].resample('M').mean()

# Plot time series
plt.figure(figsize=(14, 6))
monthly_data.plot()
plt.title('Monthly Trend')
plt.xlabel('Date')
plt.ylabel('Average Value')
plt.grid(True)
plt.show()

# Decompose time series
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(monthly_data, model='additive', period=12)
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.tight_layout()
plt.show()

# Check stationarity
from statsmodels.tsa.stattools import adfuller
result = adfuller(monthly_data.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
```

## 4. Feature Engineering

### 4.1 Creating New Features

```python
# Mathematical transformations
df['log_col'] = np.log1p(df['numeric_col'])
df['squared_col'] = df['numeric_col'] ** 2
df['sqrt_col'] = np.sqrt(df['numeric_col'])

# Interaction features
df['interaction'] = df['numeric_col1'] * df['numeric_col2']
df['ratio'] = df['numeric_col1'] / (df['numeric_col2'] + 1e-8)  # Avoid division by zero

# Binning
df['binned_col'] = pd.cut(df['numeric_col'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Date-based features
df['is_weekend'] = df['date_col'].dt.dayofweek >= 5
df['quarter'] = df['date_col'].dt.quarter
df['days_since_start'] = (df['date_col'] - df['date_col'].min()).dt.days

# Text-based features
df['text_length'] = df['text_col'].str.len()
df['word_count'] = df['text_col'].str.split().str.len()
df['contains_keyword'] = df['text_col'].str.contains('keyword', case=False).astype(int)
```

### 4.2 Encoding Categorical Variables

```python
# One-Hot Encoding
cat_cols = ['cat_col1', 'cat_col2']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Or using scikit-learn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_data = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))
df = pd.concat([df.drop(cat_cols, axis=1), encoded_df], axis=1)

# Label Encoding (for ordinal variables)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['encoded_cat'] = label_encoder.fit_transform(df['cat_col'])

# Target Encoding
target_means = df.groupby('cat_col')['target'].mean()
df['target_encoded'] = df['cat_col'].map(target_means)
```

### 4.3 Scaling and Normalization

```python
# Standard Scaling (Z-score normalization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['scaled_col1', 'scaled_col2']] = scaler.fit_transform(df[['numeric_col1', 'numeric_col2']])

# Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
df[['normalized_col1', 'normalized_col2']] = min_max_scaler.fit_transform(df[['numeric_col1', 'numeric_col2']])

# Robust Scaling (handles outliers better)
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
df[['robust_col1', 'robust_col2']] = robust_scaler.fit_transform(df[['numeric_col1', 'numeric_col2']])
```

### 4.4 Dimensionality Reduction

```python
# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
numeric_features = ['numeric_col1', 'numeric_col2', 'numeric_col3', 'numeric_col4']
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[numeric_features])
df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {pca.explained_variance_ratio_.sum()}")

# t-SNE for visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
tsne_result = tsne.fit_transform(df[numeric_features])
df['tsne_1'] = tsne_result[:, 0]
df['tsne_2'] = tsne_result[:, 1]

# Visualize the results
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
sns.scatterplot(x='pca_1', y='pca_2', hue='target', data=df)
plt.title('PCA')

plt.subplot(1, 2, 2)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='target', data=df)
plt.title('t-SNE')

plt.tight_layout()
plt.show()
```

### 4.5 Feature Selection

```python
# Correlation-based selection
corr_with_target = df.corr()['target'].sort_values(ascending=False)
print(corr_with_target)

# Feature importance from tree-based models
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=RANDOM_SEED)
model.fit(df[numeric_features], df['target'])

feature_importance = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Recursive Feature Elimination
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
selector = RFECV(estimator, step=1, cv=5, scoring='accuracy')
selector = selector.fit(df[numeric_features], df['target'])

print(f"Optimal number of features: {selector.n_features_}")
print(f"Selected features: {np.array(numeric_features)[selector.support_]}")
```

## 5. Modeling

### 5.1 Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
```

### 5.2 Model Selection and Training

```python
# Initialize models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Dictionary of models to evaluate
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_SEED),
    'SVM': SVC(random_state=RANDOM_SEED, probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(random_state=RANDOM_SEED),
    'LightGBM': LGBMClassifier(random_state=RANDOM_SEED),
}

# Train and evaluate each model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

results = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # For binary classification
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_prob
    }
    
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Compare model performance
model_comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results],
    'ROC AUC': [results[model]['roc_auc'] for model in results]
})

model_comparison = model_comparison.sort_values('ROC AUC', ascending=False)
display(model_comparison)

# Visualize comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='ROC AUC', data=model_comparison)
plt.title('Model Comparison - ROC AUC')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 5.3 Hyperparameter Tuning

```python
# Grid Search for the best model (e.g., Random Forest)
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
best_model = results['Random Forest']['model']
grid_search = GridSearchCV(
    estimator=best_model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### 5.4 Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# Set up cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Perform cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")
```

### 5.5 Model Interpretation

```python
# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# SHAP values
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.show()

# Detailed SHAP values
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test)
plt.show()

# Individual prediction explanation
sample_idx = 0  # Index of a sample to explain
plt.figure(figsize=(14, 6))
shap.force_plot(explainer.expected_value, shap_values[sample_idx,:], X_test.iloc[sample_idx,:], matplotlib=True)
plt.title('SHAP Force Plot for Sample')
plt.tight_layout()
plt.show()
```

## 6. Model Deployment and Reporting

### 6.1 Save the Model

```python
import joblib

# Save the trained model
joblib.dump(best_model, 'best_model.pkl')

# Save the preprocessing pipeline (if used)
joblib.dump(scaler, 'scaler.pkl')

# Later, load the model
loaded_model = joblib.load('best_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
```

### 6.2 Create a Prediction Function

```python
def make_prediction(data, model=loaded_model, scaler=loaded_scaler):
    """
    Make predictions on new data.
    
    Parameters:
    -----------
    data : DataFrame
        The data to make predictions on
    model : trained model
        The trained model
    scaler : trained scaler
        The trained scaler
    
    Returns:
    --------
    DataFrame with predictions
    """
    # Ensure data is in the correct format
    data_processed = data.copy()
    
    # Apply the same preprocessing steps
    numeric_cols = ['numeric_col1', 'numeric_col2']
    data_processed[numeric_cols] = scaler.transform(data_processed[numeric_cols])
    
    # Make predictions
    predictions = model.predict(data_processed)
    probabilities = model.predict_proba(data_processed)[:, 1]
    
    # Create a results DataFrame
    results = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })
    
    return results

# Example usage
new_data = pd.DataFrame({
    'numeric_col1': [1.5, 2.3, 0.7],
    'numeric_col2': [3.2, 1.1, 4.7],
    # Add other required columns
})

results = make_prediction(new_data)
print(results)
```

### 6.3 Create a Simple API (Optional)

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    
    # Convert to DataFrame
    input_data = pd.DataFrame(data)
    
    # Make prediction
    results = make_prediction(input_data)
    
    # Return results
    return jsonify({
        'predictions': results['prediction'].tolist(),
        'probabilities': results['probability'].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### 6.4 Create Interactive Visualizations (Optional)

```python
import plotly.express as px
import plotly.graph_objects as go

# Interactive scatter plot
fig = px.scatter(df, x='numeric_col1', y='numeric_col2', color='target',
                 hover_data=['numeric_col3'], title='Interactive Scatter Plot')
fig.show()

# Interactive dashboard with Dash (example)
# pip install dash
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Model Dashboard"),
    
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in X.columns],
        value=X.columns[0]
    ),
    
    dcc.Graph(id='feature-importance-graph')
])

@app.callback(
    Output('feature-importance-graph', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_graph(selected_feature):
    # Create custom visualization based on selected feature
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[selected_feature], nbinsx=30))
    fig.update_layout(title=f'Distribution of {selected_feature}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
```

## 7. Best Practices and Tips

1. **Notebook Organization**
    
    - Use clear section headers
    - Add markdown cells explaining your reasoning
    - Create a table of contents
2. **Performance Optimization**
    
    - Use `%time` and `%%timeit` to profile code
    - Vectorize operations instead of using loops
    - Use appropriate data types (e.g., categories for categorical data)
3. **Version Control**
    
    - Use Git to track changes in your notebooks
    - Consider using nbdime for notebook diff and merge
4. **Reproducibility**
    
    - Set random seeds for all operations
    - Use environment management tools (conda, venv)
    - Document all data sources and preprocessing steps
5. **Error Handling**
    
    - Add try-except blocks for data loading and processing
    - Validate inputs and outputs at each step
6. **Documentation**
    
    - Document functions with docstrings
    - Include a README with project overview
    - Add comments for complex code sections
7. **Final Checklist**
    
    - Review all visualizations for clarity
    - Check for data leakage in preprocessing steps
    - Validate model performance on fresh data
    - Clean up and organize final notebook

## 8. Resources and References

1. **Python Data Science Libraries**
    
    - Pandas: https://pandas.pydata.org/docs/
    - NumPy: https://numpy.org/doc/
    - Matplotlib: https://matplotlib.org/stable/contents.html
    - Seaborn: https://seaborn.pydata.org/
    - Scikit-learn: https://scikit-learn.org/stable/documentation.html
2. **Books**
    
    - Python for Data Analysis by Wes McKinney
    - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron
    - Data Science from Scratch by Joel Grus
3. **Courses and Tutorials**
    
    - Kaggle Learn: https://www.kaggle.com/learn
    - DataCamp Python Data Science Track
    - Coursera Data Science Specialization