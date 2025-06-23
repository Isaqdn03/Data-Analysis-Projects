# 🛒 Instacart Market Basket Analysis

## 📋 Project Overview

This project analyzes Instacart's grocery delivery data to understand customer behavior, product popularity, and shopping patterns. The analysis provides insights into when customers shop, what they buy, and how their purchasing behavior changes over time.

## 🎯 Project Objectives

- Analyze customer ordering patterns and behaviors
- Identify popular products and departments
- Understand reorder patterns and customer loyalty
- Examine shopping trends by day of week and time of day
- Provide actionable insights for business optimization

## 📊 Dataset Description

The project uses five interconnected datasets:

### 📁 Data Files
- **`instacart_orders.csv`** - Order-level information (478,967 orders)
- **`products.csv`** - Product catalog (49,694 products)
- **`order_products.csv`** - Order-product relationships (4.5M records)
- **`aisles.csv`** - Store aisle information (134 aisles)
- **`departments.csv`** - Department categories (21 departments)

### 🔍 Key Variables
- **Order timing**: Day of week, hour of day, days since previous order
- **Product details**: Names, departments, aisles
- **Customer behavior**: Reorder patterns, cart sequences
- **Shopping patterns**: Items per order, frequency analysis

## 🔧 Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive analysis environment

## 📈 Key Findings

### 🕐 Temporal Patterns
- **Peak shopping hours**: 9 AM - 4 PM
- **Busiest days**: Saturday and Sunday
- **Average reorder interval**: ~15 days
- **Quiet periods**: Late night and early morning hours

### 🛍️ Customer Behavior
- **Average items per order**: 10 products
- **Order distribution**: Most customers place 3-8 orders
- **High reorder rates**: Fresh produce and dairy products
- **Cart composition**: Essential items added first

### 🥕 Product Insights
- **Top categories**: Fresh produce, dairy, pantry staples
- **Most popular items**: 
  - Bananas (organic and conventional)
  - Organic strawberries
  - Organic baby spinach
  - Organic avocados
- **Reorder champions**: Basic groceries and perishables

### 📊 Shopping Patterns
- **Weekend vs. Weekday**: Different timing preferences
  - Weekdays: Peak before/after work hours
  - Weekends: More distributed throughout the day
- **Seasonal consistency**: Organic products show high loyalty
- **Cart behavior**: Essential items typically added first

## 📁 Project Structure

```
03 - project3-tripleten/
├── datasets/
│   ├── instacart_orders.csv
│   ├── products.csv
│   ├── order_products.csv
│   ├── aisles.csv
│   └── departments.csv
├── Final Project Notebook.ipynb
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas matplotlib jupyter
```

### Running the Analysis
1. Clone or download the project directory
2. Navigate to the project folder
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook "Final Project Notebook.ipynb"
   ```
4. Run cells sequentially to reproduce the analysis

## 🔍 Analysis Framework

### Phase 1: Data Overview
- Dataset exploration and structure analysis
- Data type verification and basic statistics
- Initial quality assessment

### Phase 2: Data Preparation
- Missing value identification and handling
- Duplicate detection and removal
- Data type corrections and validation

### Phase 3: Exploratory Data Analysis
- **Basic Analysis**:
  - Order timing patterns
  - Product popularity rankings
  - Customer ordering frequency
  
- **Intermediate Analysis**:
  - Day-of-week shopping patterns
  - Customer order distribution
  - Top 20 most popular products
  
- **Advanced Analysis**:
  - Items per order distribution
  - Reorder pattern analysis
  - Product loyalty metrics
  - Shopping cart sequence analysis

## 📊 Visualizations

The project includes various charts and graphs:
- **Histograms**: Order timing, items per order, customer frequency
- **Bar charts**: Day of week patterns, product rankings
- **Distribution plots**: Reorder intervals, shopping behaviors
- **Comparison charts**: Weekend vs. weekday patterns

## 💡 Business Insights

### 🎯 Marketing Recommendations
- **Prime Shopping Hours**: Focus promotions during 9 AM - 4 PM
- **Weekend Strategy**: Implement different campaigns for Saturday vs. Sunday
- **Product Placement**: Highlight organic and fresh produce
- **Loyalty Programs**: Target high-reorder items like bananas and dairy

### 📈 Operational Insights
- **Inventory Management**: Stock fresh produce heavily
- **Delivery Optimization**: Prepare for peak weekend demand
- **Customer Segmentation**: Identify loyal vs. occasional shoppers
- **Product Recommendations**: Leverage reorder patterns for suggestions

## 🔮 Future Enhancements

- **Seasonal Analysis**: Examine yearly trends and holiday patterns
- **Customer Segmentation**: Develop detailed customer personas
- **Market Basket Analysis**: Identify product association rules
- **Predictive Modeling**: Forecast customer lifetime value
- **Geographic Analysis**: Explore regional shopping differences

## 👥 Contributors

- **Data Analysis**: Comprehensive market basket analysis
- **Visualization**: Clear and informative charts
- **Documentation**: Detailed findings and recommendations

## 📝 Notes

- Dataset has been modified for educational purposes
- Original data distributions preserved during modifications
- Analysis conducted in Portuguese (Brazilian Portuguese)
- Results are based on historical data patterns

---