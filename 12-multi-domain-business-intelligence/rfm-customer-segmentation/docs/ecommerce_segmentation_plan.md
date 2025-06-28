# E-Commerce Customer Segmentation Pandas Project Plan

## Project Overview
**Objective**: Segment customers of Everything Plus (online household items store) using pandas-based analysis to enable personalized marketing strategies.

## 1. Environment Setup & Data Loading

### 1.1 Notebook Setup
- [ ] Import required libraries:
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  import plotly.express as px
  import plotly.graph_objects as go
  from datetime import datetime, timedelta
  import warnings
  warnings.filterwarnings('ignore')
  ```

- [ ] Set display options and style:
  ```python
  pd.set_option('display.max_columns', None)
  plt.style.use('seaborn-v0_8')
  sns.set_palette("husl")
  ```

### 1.2 Data Loading & Initial Inspection
- [ ] Load the dataset:
  ```python
  df = pd.read_csv('ecommerce_dataset_us.csv')
  df.head()
  df.info()
  df.describe()
  ```

- [ ] Basic data exploration:
  ```python
  print(f"Dataset shape: {df.shape}")
  print(f"Unique customers: {df['CustomerID'].nunique()}")
  print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
  ```

### 1.3 Data Quality Assessment
- [ ] **Missing values analysis**:
  ```python
  missing_data = df.isnull().sum()
  missing_percentage = (missing_data / len(df)) * 100
  pd.DataFrame({'Missing Count': missing_data, 'Percentage': missing_percentage})
  ```

- [ ] **Data type conversion**:
  ```python
  df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
  df['CustomerID'] = df['CustomerID'].astype('Int64')  # Handle NaN in integers
  ```

- [ ] **Anomaly detection**:
  ```python
  # Check for negative quantities and prices
  negative_qty = df[df['Quantity'] < 0]
  negative_price = df[df['UnitPrice'] < 0]
  ```

## 2. Data Cleaning & Preprocessing

### 2.1 Data Cleaning Operations
- [ ] **Remove invalid records**:
  ```python
  # Remove rows with missing CustomerID
  df_clean = df.dropna(subset=['CustomerID'])
  
  # Remove cancelled orders (negative quantities)
  df_clean = df_clean[df_clean['Quantity'] > 0]
  
  # Remove non-product items (administrative codes)
  df_clean = df_clean[~df_clean['StockCode'].str.contains('POST|DOT|M|BANK|TEST', na=False)]
  
  # Remove zero or negative prices
  df_clean = df_clean[df_clean['UnitPrice'] > 0]
  ```

- [ ] **Create calculated fields**:
  ```python
  df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
  df_clean['Year'] = df_clean['InvoiceDate'].dt.year
  df_clean['Month'] = df_clean['InvoiceDate'].dt.month
  df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()
  df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
  ```

### 2.2 Feature Engineering
- [ ] **Customer-level aggregations**:
  ```python
  customer_summary = df_clean.groupby('CustomerID').agg({
      'InvoiceNo': 'nunique',
      'TotalAmount': ['sum', 'mean', 'count'],
      'InvoiceDate': ['min', 'max'],
      'Quantity': 'sum'
  }).round(2)
  
  # Flatten column names
  customer_summary.columns = ['TotalOrders', 'TotalSpent', 'AvgOrderValue', 
                             'TotalTransactions', 'FirstPurchase', 'LastPurchase', 'TotalItems']
  ```

- [ ] **Product categorization**:
  ```python
  def categorize_product(description):
      if pd.isna(description):
          return 'Unknown'
      desc_lower = description.lower()
      if any(word in desc_lower for word in ['kitchen', 'cook', 'dinner', 'lunch']):
          return 'Kitchen & Dining'
      elif any(word in desc_lower for word in ['decor', 'ornament', 'decoration']):
          return 'Home Decor'
      elif any(word in desc_lower for word in ['gift', 'card', 'wrap']):
          return 'Gifts & Cards'
      elif any(word in desc_lower for word in ['storage', 'box', 'holder']):
          return 'Storage & Organization'
      else:
          return 'Other'
  
  df_clean['Category'] = df_clean['Description'].apply(categorize_product)
  ```

### 2.3 Advanced Feature Engineering
- [ ] **Temporal features**:
  ```python
  # Customer lifespan and recency
  reference_date = df_clean['InvoiceDate'].max()
  customer_summary['CustomerLifespan'] = (customer_summary['LastPurchase'] - 
                                         customer_summary['FirstPurchase']).dt.days
  customer_summary['Recency'] = (reference_date - customer_summary['LastPurchase']).dt.days
  ```

- [ ] **Customer category preferences**:
  ```python
  category_preferences = df_clean.groupby(['CustomerID', 'Category']).agg({
      'TotalAmount': 'sum',
      'Quantity': 'sum'
  }).reset_index()
  
  # Get primary category for each customer
  primary_categories = category_preferences.loc[
      category_preferences.groupby('CustomerID')['TotalAmount'].idxmax()
  ][['CustomerID', 'Category']].rename(columns={'Category': 'PrimaryCategory'})
  ```

## 3. Exploratory Data Analysis (EDA)

### 3.1 Univariate Analysis
- [ ] **Distribution of key metrics**:
  ```python
  fig, axes = plt.subplots(2, 2, figsize=(15, 10))
  
  # Total spent distribution
  customer_summary['TotalSpent'].hist(bins=50, ax=axes[0,0])
  axes[0,0].set_title('Distribution of Total Spent')
  
  # Order frequency distribution
  customer_summary['TotalOrders'].hist(bins=30, ax=axes[0,1])
  axes[0,1].set_title('Distribution of Order Frequency')
  
  # Average order value distribution
  customer_summary['AvgOrderValue'].hist(bins=50, ax=axes[1,0])
  axes[1,0].set_title('Distribution of Average Order Value')
  
  # Recency distribution
  customer_summary['Recency'].hist(bins=50, ax=axes[1,1])
  axes[1,1].set_title('Distribution of Recency (Days)')
  
  plt.tight_layout()
  plt.show()
  ```

- [ ] **Product and category analysis**:
  ```python
  # Top products by revenue
  top_products = df_clean.groupby('Description')['TotalAmount'].sum().sort_values(ascending=False).head(20)
  
  # Category performance
  category_performance = df_clean.groupby('Category').agg({
      'TotalAmount': 'sum',
      'CustomerID': 'nunique',
      'Quantity': 'sum'
  }).sort_values('TotalAmount', ascending=False)
  ```

### 3.2 Bivariate Analysis
- [ ] **RFM correlation analysis**:
  ```python
  rfm_correlation = customer_summary[['Recency', 'TotalOrders', 'TotalSpent']].corr()
  sns.heatmap(rfm_correlation, annot=True, cmap='coolwarm', center=0)
  plt.title('RFM Metrics Correlation')
  plt.show()
  ```

- [ ] **Customer spending vs frequency**:
  ```python
  plt.figure(figsize=(12, 8))
  plt.scatter(customer_summary['TotalOrders'], customer_summary['TotalSpent'], alpha=0.6)
  plt.xlabel('Total Orders')
  plt.ylabel('Total Spent')
  plt.title('Customer Spending vs Order Frequency')
  plt.show()
  ```

### 3.3 Temporal Analysis
- [ ] **Monthly sales trends**:
  ```python
  monthly_trends = df_clean.groupby(['Year', 'Month']).agg({
      'TotalAmount': 'sum',
      'CustomerID': 'nunique',
      'InvoiceNo': 'nunique'
  }).reset_index()
  
  monthly_trends['YearMonth'] = pd.to_datetime(monthly_trends[['Year', 'Month']].assign(day=1))
  
  fig, axes = plt.subplots(3, 1, figsize=(15, 12))
  
  axes[0].plot(monthly_trends['YearMonth'], monthly_trends['TotalAmount'])
  axes[0].set_title('Monthly Revenue Trend')
  
  axes[1].plot(monthly_trends['YearMonth'], monthly_trends['CustomerID'])
  axes[1].set_title('Monthly Active Customers')
  
  axes[2].plot(monthly_trends['YearMonth'], monthly_trends['InvoiceNo'])
  axes[2].set_title('Monthly Number of Orders')
  
  plt.tight_layout()
  plt.show()
  ```

## 4. RFM Analysis Implementation

### 4.1 RFM Metrics Calculation
- [ ] **Calculate RFM values**:
  ```python
  # Prepare RFM dataframe
  rfm_df = customer_summary[['TotalOrders', 'TotalSpent', 'Recency']].copy()
  rfm_df.columns = ['Frequency', 'Monetary', 'Recency']
  
  # Add customer ID back
  rfm_df['CustomerID'] = customer_summary.index
  ```

### 4.2 RFM Scoring (Quintile-based)
- [ ] **Create RFM scores**:
  ```python
  # Calculate quintiles for scoring
  rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), 
                             q=5, labels=[5,4,3,2,1])  # Lower recency = higher score
  rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 
                             q=5, labels=[1,2,3,4,5])
  rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 
                             q=5, labels=[1,2,3,4,5])
  
  # Create RFM segment string
  rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + \
                       rfm_df['F_Score'].astype(str) + \
                       rfm_df['M_Score'].astype(str)
  ```

### 4.3 Customer Segmentation
- [ ] **Define segments based on RFM scores**:
  ```python
  def segment_customers(row):
      if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
          return 'Champions'
      elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
          return 'Loyal Customers'
      elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
          return 'Potential Loyalists'
      elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
          return 'New Customers'
      elif row['RFM_Score'] in ['512', '511', '412', '411']:
          return 'Promising'
      elif row['RFM_Score'] in ['155', '154', '245', '244', '253', '252', '243']:
          return 'Need Attention'
      elif row['RFM_Score'] in ['155', '254', '144']:
          return 'About to Sleep'
      elif row['RFM_Score'] in ['124', '123', '122', '212', '213']:
          return 'At Risk'
      elif row['RFM_Score'] in ['125', '124', '123', '122', '221', '213', '231']:
          return 'Cannot Lose Them'
      elif row['RFM_Score'] in ['155', '132', '231', '241', '221']:
          return 'Hibernating'
      else:
          return 'Lost'
  
  rfm_df['Segment'] = rfm_df.apply(segment_customers, axis=1)
  ```

## 5. Advanced Clustering Analysis

### 5.1 K-Means Clustering
- [ ] **Prepare data for clustering**:
  ```python
  from sklearn.preprocessing import StandardScaler
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_score
  
  # Select features for clustering
  clustering_features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
  
  # Standardize the features
  scaler = StandardScaler()
  clustering_features_scaled = scaler.fit_transform(clustering_features)
  ```

- [ ] **Determine optimal number of clusters**:
  ```python
  # Elbow method
  sse = []
  k_range = range(2, 11)
  
  for k in k_range:
      kmeans = KMeans(n_clusters=k, random_state=42)
      kmeans.fit(clustering_features_scaled)
      sse.append(kmeans.inertia_)
  
  plt.figure(figsize=(10, 6))
  plt.plot(k_range, sse, 'bo-')
  plt.xlabel('Number of Clusters')
  plt.ylabel('Sum of Squared Errors')
  plt.title('Elbow Method for Optimal K')
  plt.show()
  
  # Silhouette analysis
  silhouette_scores = []
  for k in k_range:
      kmeans = KMeans(n_clusters=k, random_state=42)
      cluster_labels = kmeans.fit_predict(clustering_features_scaled)
      silhouette_avg = silhouette_score(clustering_features_scaled, cluster_labels)
      silhouette_scores.append(silhouette_avg)
  
  plt.figure(figsize=(10, 6))
  plt.plot(k_range, silhouette_scores, 'ro-')
  plt.xlabel('Number of Clusters')
  plt.ylabel('Silhouette Score')
  plt.title('Silhouette Analysis')
  plt.show()
  ```

### 5.2 Hierarchical Clustering
- [ ] **Implement hierarchical clustering**:
  ```python
  from scipy.cluster.hierarchy import dendrogram, linkage
  from sklearn.cluster import AgglomerativeClustering
  
  # Create linkage matrix
  linkage_matrix = linkage(clustering_features_scaled, method='ward')
  
  # Plot dendrogram
  plt.figure(figsize=(15, 8))
  dendrogram(linkage_matrix, truncate_mode='level', p=5)
  plt.title('Hierarchical Clustering Dendrogram')
  plt.xlabel('Sample Index')
  plt.ylabel('Distance')
  plt.show()
  ```

### 5.3 Product Category-Based Clustering
- [ ] **Customer-product category matrix**:
  ```python
  # Create customer-category spending matrix
  category_matrix = df_clean.groupby(['CustomerID', 'Category'])['TotalAmount'].sum().unstack(fill_value=0)
  
  # Normalize by customer total spending
  category_matrix_normalized = category_matrix.div(category_matrix.sum(axis=1), axis=0)
  
  # Apply K-means to category preferences
  category_kmeans = KMeans(n_clusters=5, random_state=42)
  category_clusters = category_kmeans.fit_predict(category_matrix_normalized.fillna(0))
  ```

## 6. Statistical Hypothesis Testing

### 6.1 Import Statistical Libraries
- [ ] **Setup statistical testing**:
  ```python
  from scipy import stats
  from scipy.stats import chi2_contingency, f_oneway, kruskal
  import itertools
  ```

### 6.2 Segment Comparison Tests
- [ ] **H1: Different segments have significantly different average order values**:
  ```python
  # Merge segment data with customer summary
  segment_analysis = rfm_df.merge(customer_summary, left_on='CustomerID', right_index=True)
  
  # Group AOV by segment
  aov_by_segment = [group['AvgOrderValue'].values for name, group in segment_analysis.groupby('Segment')]
  
  # Perform ANOVA
  f_stat, p_value = f_oneway(*aov_by_segment)
  print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
  
  # If ANOVA is significant, perform post-hoc tests
  if p_value < 0.05:
      from itertools import combinations
      segments = segment_analysis['Segment'].unique()
      
      for seg1, seg2 in combinations(segments, 2):
          group1 = segment_analysis[segment_analysis['Segment'] == seg1]['AvgOrderValue']
          group2 = segment_analysis[segment_analysis['Segment'] == seg2]['AvgOrderValue']
          
          t_stat, p_val = stats.ttest_ind(group1, group2)
          print(f"{seg1} vs {seg2}: t-stat = {t_stat:.4f}, p-value = {p_val:.4f}")
  ```

- [ ] **H2: Customer segments show different purchase frequencies**:
  ```python
  # Kruskal-Wallis test for frequency differences
  freq_by_segment = [group['Frequency'].values for name, group in segment_analysis.groupby('Segment')]
  h_stat, p_value = kruskal(*freq_by_segment)
  print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {p_value:.4f}")
  ```

- [ ] **H3: Product category preferences vary by segment**:
  ```python
  # Create contingency table
  segment_category = pd.crosstab(
      df_clean.merge(rfm_df[['CustomerID', 'Segment']], on='CustomerID')['Segment'],
      df_clean['Category']
  )
  
  # Chi-square test
  chi2, p_value, dof, expected = chi2_contingency(segment_category)
  print(f"Chi-square statistic: {chi2:.4f}, p-value: {p_value:.4f}")
  
  # Calculate Cramér's V (effect size)
  n = segment_category.sum().sum()
  cramers_v = np.sqrt(chi2 / (n * (min(segment_category.shape) - 1)))
  print(f"Cramér's V: {cramers_v:.4f}")
  ```

## 7. Visualization & Insights

### 7.1 Segment Visualization
- [ ] **RFM 3D scatter plot**:
  ```python
  fig = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary', 
                      color='Segment', hover_data=['CustomerID'],
                      title='Customer Segments in RFM Space')
  fig.show()
  ```

- [ ] **Segment characteristics heatmap**:
  ```python
  segment_summary = segment_analysis.groupby('Segment').agg({
      'Recency': 'mean',
      'Frequency': 'mean',
      'Monetary': 'mean',
      'TotalOrders': 'mean',
      'AvgOrderValue': 'mean'
  }).round(2)
  
  plt.figure(figsize=(12, 8))
  sns.heatmap(segment_summary.T, annot=True, cmap='RdYlBu_r', center=0, 
              cbar_kws={'label': 'Normalized Values'})
  plt.title('Customer Segment Characteristics')
  plt.show()
  ```

### 7.2 Business Impact Visualizations
- [ ] **Revenue contribution by segment**:
  ```python
  segment_revenue = segment_analysis.groupby('Segment').agg({
      'TotalSpent': 'sum',
      'CustomerID': 'count'
  }).rename(columns={'CustomerID': 'CustomerCount'})
  
  segment_revenue['RevenuePerCustomer'] = segment_revenue['TotalSpent'] / segment_revenue['CustomerCount']
  
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))
  
  # Total revenue by segment
  segment_revenue['TotalSpent'].plot(kind='bar', ax=axes[0])
  axes[0].set_title('Total Revenue by Segment')
  axes[0].set_ylabel('Total Revenue')
  
  # Customer count by segment
  segment_revenue['CustomerCount'].plot(kind='bar', ax=axes[1])
  axes[1].set_title('Customer Count by Segment')
  axes[1].set_ylabel('Number of Customers')
  
  # Revenue per customer by segment
  segment_revenue['RevenuePerCustomer'].plot(kind='bar', ax=axes[2])
  axes[2].set_title('Revenue per Customer by Segment')
  axes[2].set_ylabel('Revenue per Customer')
  
  for ax in axes:
      ax.tick_params(axis='x', rotation=45)
  
  plt.tight_layout()
  plt.show()
  ```

## 8. Business Recommendations

### 8.1 Segment-Specific Strategies
- [ ] **Create recommendation framework**:
  ```python
  recommendations = {
      'Champions': {
          'Strategy': 'Reward and retain',
          'Actions': ['VIP treatment', 'Early access to new products', 'Loyalty rewards'],
          'Expected_Outcome': 'Maintain high value and advocacy'
      },
      'Loyal Customers': {
          'Strategy': 'Nurture and upsell',
          'Actions': ['Cross-sell complementary products', 'Personalized recommendations'],
          'Expected_Outcome': 'Increase order value and frequency'
      },
      'Potential Loyalists': {
          'Strategy': 'Develop loyalty',
          'Actions': ['Engagement campaigns', 'Product education', 'Special offers'],
          'Expected_Outcome': 'Convert to loyal segment'
      },
      'At Risk': {
          'Strategy': 'Win back',
          'Actions': ['Targeted discounts', 'Reactivation campaigns', 'Feedback surveys'],
          'Expected_Outcome': 'Prevent churn and reactivate'
      },
      'Cannot Lose Them': {
          'Strategy': 'Aggressive retention',
          'Actions': ['Personal outreach', 'Exclusive offers', 'Premium support'],
          'Expected_Outcome': 'Retain high-value customers'
      }
  }
  
  recommendations_df = pd.DataFrame(recommendations).T
  print(recommendations_df)
  ```

### 8.2 Product Strategy Insights
- [ ] **Category performance by segment**:
  ```python
  category_segment_analysis = df_clean.merge(
      rfm_df[['CustomerID', 'Segment']], on='CustomerID'
  ).groupby(['Segment', 'Category']).agg({
      'TotalAmount': 'sum',
      'Quantity': 'sum'
  }).reset_index()
  
  # Pivot for heatmap
  category_heatmap = category_segment_analysis.pivot(
      index='Segment', columns='Category', values='TotalAmount'
  ).fillna(0)
  
  plt.figure(figsize=(12, 8))
  sns.heatmap(category_heatmap, annot=True, fmt='.0f', cmap='Blues')
  plt.title('Revenue by Segment and Product Category')
  plt.show()
  ```

## 9. Documentation & Sources

### 9.1 Research Sources Framework
- [ ] **Source 1: RFM Analysis Methodology**
  - *Question*: How to properly calculate and interpret RFM scores for customer segmentation?
  - *Application*: Used for implementing quintile-based RFM scoring system

- [ ] **Source 2: Customer Segmentation in E-commerce**
  - *Question*: What are the best practices for e-commerce customer segmentation?
  - *Application*: Informed segment naming and business strategy development

- [ ] **Source 3: Statistical Methods for Business Analytics**
  - *Question*: Which statistical tests are appropriate for comparing customer segments?
  - *Application*: Guided hypothesis testing approach and significance testing

- [ ] **Source 4: Clustering Algorithms for Customer Data**
  - *Question*: How do different clustering methods compare for customer segmentation?
  - *Application*: Informed choice of K-means and hierarchical clustering approaches

- [ ] **Source 5: Pandas Documentation and Best Practices**
  - *Question*: How to efficiently manipulate large datasets with pandas?
  - *Application*: Optimized data processing and feature engineering workflows

### 9.2 Additional Recommended Sources
- [ ] **Source 6**: Customer Lifetime Value calculation methods
- [ ] **Source 7**: Behavioral economics in online shopping
- [ ] **Source 8**: Personalization strategies in e-commerce
- [ ] **Source 9**: Machine learning for customer retention
- [ ] **Source 10**: A/B testing for marketing campaigns

## 10. Presentation Preparation

### 10.1 Key Visualizations for Presentation
- [ ] Executive dashboard with key metrics
- [ ] Customer segment distribution and characteristics
- [ ] RFM analysis results and interpretation
- [ ] Statistical testing results summary
- [ ] Revenue impact and business recommendations
- [ ] Implementation roadmap

### 10.2 Notebook Organization
- [ ] **Section 1**: Data Loading and Cleaning
- [ ] **Section 2**: Exploratory Data Analysis
- [ ] **Section 3**: RFM Analysis and Segmentation
- [ ] **Section 4**: Advanced Clustering
- [ ] **Section 5**: Statistical Validation
- [ ] **Section 6**: Business Insights and Recommendations
- [ ] **Section 7**: Conclusions and Next Steps

## 11. Quality Assurance Checklist

### 11.1 Code Quality
- [ ] All code cells run without errors
- [ ] Appropriate comments and documentation
- [ ] Consistent variable naming conventions
- [ ] Efficient pandas operations (avoid loops where possible)
- [ ] Memory usage optimization for large datasets

### 11.2 Analysis Validation
- [ ] Cross-validate segmentation results
- [ ] Verify statistical test assumptions
- [ ] Check for data leakage in feature engineering
- [ ] Validate business logic of segments
- [ ] Ensure reproducibility with random seeds

## Expected Deliverables

1. **Complete Jupyter Notebook**: Well-organized analysis with code, visualizations, and markdown explanations
2. **PDF Presentation**: Executive summary with key findings and recommendations
3. **Source Documentation**: 5-10 referenced sources with explanations of their application
4. **Segment Profiles**: Detailed characteristics and strategies for each customer segment
5. **Statistical Validation Report**: Results of hypothesis testing with interpretation
6. **Business Recommendations**: Actionable strategies for each segment with expected outcomes