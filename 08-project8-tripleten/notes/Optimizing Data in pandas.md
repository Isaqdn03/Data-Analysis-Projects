---
aliases:
  - Study - Optimizing Data in pandas
tags:
  - type/study
  - context/studies
  - status/in-progress
  - review/pending
  - theme/data-analysis
date:
  "{ date }": 
last_updated:
  "{ date }": 
MOC: "[[Sprint 8 - Business Analysis]]"
---
# Data Optimization in Pandas

## 1. Introduction to Optimization

### 1.1 Importance of Optimization

- Working with large data volumes
- Prevention of memory errors
- Performance improvement
- Computational efficiency

> **Important Note**: Optimization is crucial when working with large datasets, especially in environments with limited resources.

## 2. Initial Data Analysis

### 2.1 Basic Data Loading

```python
import pandas as pd

# Loading sample data
df = pd.read_csv('data/visits.csv', nrows=500)

# Memory analysis
df.info(memory_usage='deep')
```

### 2.2 DataFrame Structure

|Column|Original Type|Description|
|---|---|---|
|Device|object|Device type|
|End Ts|object|End timestamp|
|Source Id|int64|Source ID|
|Start Ts|object|Start timestamp|
|Uid|uint64|User ID|

**Initial memory usage**: 109.7 KB

## 3. Optimization Techniques

### 3.1 Conversion to Categories

```python
# Analysis of unique values
df['Device'].value_counts()
# Result:
# desktop    362
# touch      138

# Conversion to category
df['Device'] = df['Device'].astype('category')
```

**Benefits of Category Conversion:**

- Significant reduction in memory usage
- Maintains data functionality
- Ideal for columns with repeated values

### 3.2 Timestamp Optimization

```python
# Converting strings to datetime
df['Start Ts'] = pd.to_datetime(df['Start Ts'], format="%Y-%m-%d %H:%M:%S")
df['End Ts'] = pd.to_datetime(df['End Ts'], format="%Y-%m-%d %H:%M:%S")
```

**Optimization Impact:**

1. Initial memory usage: 109.7 KB
2. After category conversion: 79.4 KB
3. After timestamp conversion: 16.4 KB

### 3.3 Optimization Results

|Stage|Memory Usage|Reduction|
|---|---|---|
|Initial|109.7 KB|-|
|Categories|79.4 KB|27.6%|
|Timestamps|16.4 KB|85.1%|

## 4. Optimized Loading

### 4.1 Type Definition During Reading

```python
# Optimized loading from the beginning
dd = pd.read_csv(
    'data/visits.csv',
    nrows=500,
    dtype={'Device': 'category'},
    parse_dates=['Start Ts', 'End Ts']
)
```

### 4.2 Optimization Parameters

1. **dtype**:
    
    - Defines specific data types
    - Prevents unnecessary conversions
    - Reduces memory usage from the start
2. **parse_dates**:
    
    - Converts strings to datetime
    - Processes dates efficiently
    - Improves temporal manipulation

## 5. Best Practices

### 5.1 General Recommendations

1. **Prior Analysis**:
    
    - Examine data types
    - Identify unique values
    - Evaluate precision needs
2. **Type Selection**:
    
    - Use categories for repetitive data
    - Convert timestamps appropriately
    - Choose appropriate numeric types
3. **Monitoring**:
    
    - Track memory usage
    - Verify performance
    - Validate optimizations

### 5.2 When to Optimize

- **Large Volumes**: Datasets with millions of rows
- **Limited Resources**: Environments with memory constraints
- **Critical Performance**: Applications requiring fast response

## 6. Practical Applications

### 6.1 Use Case Scenarios

1. **Log Analysis**:
    
    - Event processing
    - Session analysis
    - User monitoring
2. **Business Data**:
    
    - Financial reports
    - Sales analysis
    - User metrics

### 6.2 Impact on Real Projects

- Reduction of computational costs
- Improvement in processing time
- Greater efficiency in analyses

> **Professional Tip**: Even though the difference may seem small in samples (500 rows), the impact is significant in complete datasets with millions of records.

## 7. Key Points to Remember

1. **Progressive Optimization**:
    
    - Start with type analysis
    - Implement optimizations gradually
    - Monitor results
2. **Efficient Types**:
    
    - Categories for repetitive data
    - Datetime for timestamps
    - Appropriate numeric types
3. **Smart Loading**:
    
    - Use optimization parameters
    - Define types during reading
    - Avoid subsequent conversions 