# Video Game Market Analysis

## 🎮 Project Overview

This project provides a comprehensive analysis of the global video game market using a dataset of 16,715 games spanning multiple platforms, genres, and regions. The analysis examines sales patterns, platform lifecycles, regional preferences, and user ratings to uncover valuable insights for game publishers, developers, and market analysts.

## 🎯 Objectives

The project aims to:
- Analyze global video game sales trends and patterns
- Identify the most successful gaming platforms and their lifecycles
- Examine regional market differences (North America, Europe, Japan, Other)
- Investigate the relationship between critic/user scores and commercial success
- Test statistical hypotheses about platform and genre performance
- Provide data-driven recommendations for the gaming industry

## 📊 Key Analyses

### 1. **Platform Analysis**
- Platform lifecycle and longevity analysis
- Sales performance comparison across different gaming platforms
- Historical trends and platform dominance periods

### 2. **Regional Market Analysis**
- North America (NA), Europe (EU), Japan (JP), and Other regions
- Regional preferences for platforms and genres
- Market size and growth patterns by region

### 3. **Genre Performance**
- Most popular and profitable game genres
- Genre distribution and market saturation
- Regional preferences for different game types

### 4. **Statistical Hypothesis Testing**
- Comparison of user ratings between Xbox One and PC platforms
- Statistical analysis of Action vs Sports genre ratings
- Significance testing with proper statistical methodology

## 📈 Key Findings

### Platform Performance:
- **PlayStation 2 (PS2)** dominates total sales with the longest lifecycle (12+ years)
- **Xbox 360**, **PS3**, and **Wii** were the leading platforms during 2007-2012
- Console lifecycles typically span 7-8 years with clear generational transitions

### Regional Market Insights:
- **North America** is the largest gaming market by sales volume
- **Japan** shows distinct preferences for RPGs and portable gaming platforms
- **Europe** follows similar patterns to North America but with lower overall volumes

### Genre Analysis:
- **Action games** represent the largest category with 1,031 titles
- **Role-Playing** (370 titles) and **Adventure** (302 titles) are significant genres
- Genre preferences vary significantly between regions

### Statistical Testing Results:
- **Xbox One vs PC**: No significant difference in user ratings (p-value = 0.5490)
- **Action vs Sports**: Significant difference in user ratings (p-value < 0.001)
  - Action games: 6.83 average rating
  - Sports games: 5.46 average rating

## 🛠️ Technology Stack

- **Python 3**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualizations
- **SciPy** - Statistical testing

## 📁 Project Structure

```
gaming-industry-market-analysis/
├── games-analysis-notebook.ipynb          # Main analysis notebook
├── games-analysis-notebook-corrections.ipynb  # Corrected version
├── datasets/
│   └── games.csv                          # Game sales dataset (1.1MB)
└── README.md                             # Project documentation
```

## 📊 Dataset Information

### Games Dataset (`games.csv`)
- **Size**: 16,715 game records (1.1MB)
- **Time Period**: Historical data through 2016
- **Columns**:
  - `name`: Game title
  - `platform`: Gaming platform (PS4, Xbox One, PC, etc.)
  - `year_of_release`: Release year
  - `genre`: Game genre (Action, Sports, RPG, etc.)
  - `na_sales`, `eu_sales`, `jp_sales`, `other_sales`: Regional sales (millions USD)
  - `critic_score`: Professional critic ratings (0-100)
  - `user_score`: User ratings (0-10)
  - `rating`: ESRB age rating

### Data Quality:
- **Missing Values**: Handled systematically with appropriate imputation strategies
- **Data Types**: Properly converted and validated
- **Derived Variables**: Total sales calculated from regional data

## 🚀 How to Run

### Prerequisites
- Python 3.7 or higher
- Required packages: pandas, numpy, matplotlib, seaborn, scipy

### Installation and Execution
1. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   ```

2. **Run the analysis**
   ```bash
   jupyter notebook games-analysis-notebook.ipynb
   ```

3. **Follow the analysis sections:**
   - Data Import and Cleaning
   - Platform Analysis
   - Regional Market Analysis
   - Genre Performance Analysis
   - Statistical Hypothesis Testing

## 📈 Visualizations

The project includes comprehensive visualizations:
- **Bar charts** showing platform and genre performance
- **Line plots** displaying temporal trends and platform lifecycles
- **Scatter plots** examining score vs. sales relationships
- **Distribution plots** for regional market analysis
- **Box plots** for statistical comparisons

## 🔍 Statistical Methods

### Data Preprocessing:
- Missing value imputation using appropriate strategies
- Data type conversions and validation
- Creation of derived variables (total_sales)

### Hypothesis Testing Framework:
- **Test 1**: Xbox One vs PC user ratings
  - Method: Independent t-test
  - Result: No significant difference (p = 0.5490)
  
- **Test 2**: Action vs Sports genre ratings
  - Method: Independent t-test with Levene's test for equal variances
  - Result: Significant difference (p < 0.001)

## 💡 Business Insights

### For Game Publishers:
- Focus on Action and Role-Playing genres for broad market appeal
- Consider regional preferences when localizing games
- Platform timing is crucial - align with console lifecycle peaks

### For Platform Holders:
- Platform lifecycles follow predictable 7-8 year patterns
- Regional market penetration strategies should differ significantly
- User ratings don't always correlate with commercial success

### For Developers:
- Action games have both high volume and good ratings
- Sports games show lower user satisfaction despite commercial viability
- Regional preferences should influence game design decisions

## 📅 Time Period Analysis

The analysis identifies **2012-2016** as the most relevant period for:
- Current generation platform performance
- Recent market trends and preferences
- Predictive modeling for future releases

## 📚 Skills Demonstrated

This project showcases the following data analysis capabilities:
- **Comprehensive EDA**: Systematic exploration of large datasets
- **Statistical Analysis**: Proper hypothesis testing methodology
- **Data Visualization**: Professional charts and graphs for business insights
- **Market Research**: Industry-specific analysis and recommendations
- **Data Cleaning**: Handling missing values and data quality issues

## 🔄 Future Enhancements

Potential areas for expansion:
- **Predictive Modeling**: Sales forecasting for new releases
- **Sentiment Analysis**: Text analysis of user reviews
- **Market Segmentation**: Advanced clustering of games and users
- **Real-time Data Integration**: Current market data incorporation
- **Machine Learning**: Advanced algorithms for pattern recognition
- **Interactive Dashboards**: Web-based visualization tools

## 📋 Project Deliverables

- Complete data analysis with statistical validation
- Business recommendations based on quantitative findings
- Reproducible methodology for ongoing market analysis
- Professional visualizations suitable for stakeholder presentations
 