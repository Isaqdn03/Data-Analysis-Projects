# Customer Behavior Analysis & A/B Testing

## 📱 Project Overview

This project provides a comprehensive analysis of customer behavior in a mobile application, focusing on conversion funnel optimization and A/B testing evaluation. The study examines user interactions, conversion patterns, and the statistical impact of font changes on user engagement and purchase completion rates.

## 🎯 Project Objectives

The analysis aims to:
- **Analyze user behavior patterns** throughout the mobile app conversion funnel
- **Evaluate conversion rates** at each stage of the customer journey
- **Conduct rigorous A/B testing** to assess the impact of font changes on user behavior
- **Apply statistical methods** with proper corrections for multiple testing scenarios
- **Provide data-driven recommendations** for UI/UX improvements
- **Identify optimization opportunities** in the customer conversion process

## 🔍 Key Business Questions Answered

### User Behavior Analysis
- How do users progress through the conversion funnel?
- What are the conversion rates at each stage of the customer journey?
- Where do we lose the most users in the funnel?

### A/B Testing Evaluation
- Does changing the font significantly impact user behavior?
- Which experimental group shows better conversion performance?
- Are the observed differences statistically significant?

### Statistical Validation
- How do we account for multiple testing scenarios?
- What is the appropriate significance level after Bonferroni correction?
- Can we confidently recommend implementing the font changes?

## 📁 Project Structure

```
user-behavior-funnel-analysis/
├── costumer-behavior-analysis.ipynb    # Main analysis notebook (690KB)
├── datasets/
│   └── logs_exp_us.csv                 # User event logs (13MB, 244K records)
├── docs/
│   └── DATA ANALYSIS FRAMEWORK.md      # Comprehensive analysis methodology
└── README.md                           # Project documentation
```

## 📊 Dataset Description

### User Event Logs (`logs_exp_us.csv`)
- **Size**: 243,713 events (after deduplication)
- **Time Period**: August 1-7, 2019 (7 days of complete data)
- **Users**: 7,534 unique users
- **Columns**:
  - `EventName`: Type of user action (MainScreenAppear, OffersScreenAppear, etc.)
  - `DeviceIDHash`: Unique user identifier
  - `EventTimestamp`: Unix timestamp of the event
  - `ExpId`: Experimental group identifier (246, 247, 248)
  - `DateTime`: Converted timestamp
  - `Date`: Date of the event

### Data Quality Metrics:
- **Completeness**: 98.8% of events retained after filtering incomplete data
- **User Retention**: 99.8% of users retained in analysis
- **Duplicate Removal**: 413 duplicate events removed (0.17%)
- **Missing Values**: Zero missing values in the dataset

## 📈 Key Findings & Results

### Conversion Funnel Analysis
#### **Funnel Performance**:
- **MainScreenAppear**: 7,439 users (100% - funnel entry point)
- **OffersScreenAppear**: 4,613 users (62.0% conversion from main screen)
- **CartScreenAppear**: 3,749 users (50.4% overall conversion)
- **PaymentScreenSuccessful**: 3,547 users (47.7% final conversion rate)

#### **User Loss Analysis**:
- **Biggest Drop**: MainScreen → OffersScreen (2,826 users lost, 38.0%)
- **Secondary Loss**: OffersScreen → CartScreen (864 users lost, 18.7%)
- **Final Stage Loss**: CartScreen → PaymentScreen (202 users lost, 5.4%)

### A/B Testing Results
#### **Experimental Groups**:
- **Group 246 (Control A)**: 2,489 users (32.96%)
- **Group 247 (Control B)**: 2,520 users (33.37%)
- **Group 248 (Font Change)**: 2,542 users (33.66%)

#### **Statistical Analysis**:
- **A/A Testing**: No significant difference between control groups (p = 0.8692)
- **Font Impact**: Group 248 showed improved conversion trends
- **Bonferroni Correction**: Applied adjusted significance level (α = 0.0167)

### Key Performance Metrics
#### **Average User Engagement**:
- **Events per User**: 32.28 average actions per user
- **Session Distribution**: Well-balanced across experimental groups
- **Time Period**: 7 days of high-quality behavioral data

#### **Conversion Rate Insights**:
- **Overall Conversion**: 47.7% from entry to purchase completion
- **Critical Bottleneck**: Initial engagement (MainScreen → OffersScreen)
- **Strong Final Intent**: 94.6% cart-to-payment conversion rate

## 🛠️ Technology Stack

- **Python 3** - Core analysis environment
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib & Seaborn** - Data visualization
- **SciPy** - Statistical testing and analysis
- **Jupyter Notebook** - Interactive analysis environment

## 🚀 How to Run

### Prerequisites
- Python 3.7 or higher
- Required packages: pandas, numpy, matplotlib, seaborn, scipy

### Installation and Execution
1. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scipy jupyter
   ```

2. **Launch the analysis**
   ```bash
   jupyter notebook costumer-behavior-analysis.ipynb
   ```

3. **Follow the analysis sections:**
   - Project Setup and Data Collection
   - Data Understanding and Cleaning
   - Exploratory Data Analysis
   - Conversion Funnel Analysis
   - A/B Testing Statistical Analysis
   - Results and Conclusions

## 📊 Analysis Methodology

### 1. **Data Preparation & Quality Assessment**
- Comprehensive data cleaning and validation
- Duplicate removal and missing value analysis
- Data type optimization and datetime conversion
- Filtering for complete data periods only

### 2. **Behavioral Analysis**
- User journey mapping through conversion funnel
- Event frequency and distribution analysis
- User engagement pattern identification
- Time-series analysis of user actions

### 3. **Statistical Testing Framework**
- A/A testing for baseline validation
- Proportion testing for conversion rates
- Mann-Whitney U tests for non-parametric comparisons
- Bonferroni correction for multiple testing scenarios

### 4. **Conversion Funnel Optimization**
- Stage-by-stage conversion rate calculation
- User loss identification and quantification
- Bottleneck analysis and improvement opportunities
- Performance benchmarking across experimental groups

## 📚 Statistical Methods Applied

### Hypothesis Testing:
- **Mann-Whitney U Test**: Non-parametric comparison of user behavior
- **Chi-Square Test**: Proportion comparisons between experimental groups
- **Z-Test for Proportions**: Conversion rate statistical significance
- **Bonferroni Correction**: Multiple testing adjustment (α = 0.0167)

### A/B Testing Best Practices:
- **Randomization Validation**: Ensuring balanced group allocation
- **Statistical Power**: Adequate sample sizes for meaningful results
- **Effect Size**: Practical significance beyond statistical significance
- **Multiple Testing Control**: Conservative approach to prevent false discoveries

## 💡 Business Recommendations

### 🎨 Font Change Implementation
#### **Strategic Decision**: ✅ **Implement the Font Changes**
- **Evidence**: Group 248 showed consistent improvement trends
- **User Experience**: Positive impact on conversion funnel progression
- **Risk Assessment**: Low risk, high potential reward for user engagement

### 📈 Conversion Optimization Priorities
1. **Critical Focus**: MainScreen → OffersScreen transition (38% user loss)
2. **Secondary Priority**: OffersScreen → CartScreen optimization (18.7% loss)
3. **Maintain Performance**: CartScreen → PaymentScreen already performing well (94.6%)

### 🔍 Future Testing Recommendations
- **Iterative Testing**: Continue A/B testing for other UI elements
- **Extended Analysis**: Longer time periods for seasonal effects
- **Segmentation Studies**: User behavior by demographics or device types
- **Multivariate Testing**: Combined effect of multiple UI changes

## 📚 Skills Demonstrated

This project showcases the following data analysis capabilities:

### **Advanced Analytics Skills**
- **Behavioral Data Analysis**: Complex user journey mapping and analysis
- **Statistical Rigor**: Proper A/B testing methodology with multiple testing corrections
- **Business Intelligence**: Translation of statistical results into actionable recommendations
- **Data Quality Management**: Comprehensive data cleaning and validation processes

### **Technical Proficiency**
- **Large Dataset Handling**: Efficient processing of 240K+ event records
- **Time Series Analysis**: Temporal pattern recognition and filtering
- **Statistical Computing**: Advanced hypothesis testing and significance calculations
- **Visualization Design**: Clear communication of complex analytical results

## 📋 Key Deliverables

### **Analysis Components**:
- ✅ Complete user behavior analysis with 7,534 unique users
- ✅ Comprehensive conversion funnel analysis (47.7% overall conversion)
- ✅ Rigorous A/B testing with statistical significance validation
- ✅ Bonferroni-corrected results for multiple testing scenarios
- ✅ Data-driven font change recommendation with supporting evidence

### **Technical Outputs**:
- ✅ Clean, well-documented analysis notebook
- ✅ Statistical framework for future experiments
- ✅ Comprehensive data analysis methodology guide
- ✅ Business-ready recommendations with confidence intervals

## 🏆 Business Impact

This analysis provides a solid foundation for:
- **User Experience Optimization**: Evidence-based UI/UX improvements
- **Conversion Rate Optimization**: Targeted funnel improvement strategies
- **Statistical Decision Making**: Rigorous framework for future A/B tests
- **Product Development**: Data-driven feature prioritization

### **Measurable Outcomes**:
- **Conversion Insights**: 47.7% baseline conversion rate established
- **Optimization Opportunities**: 38% user loss reduction potential identified
- **Statistical Confidence**: Bonferroni-corrected significance testing applied
- **Implementation Roadmap**: Clear font change recommendation with supporting data

The font change analysis demonstrates positive user behavior impact, providing confidence for implementation while establishing a robust framework for future experimentation and optimization efforts.

## 📖 Documentation Reference

The project includes a comprehensive **DATA ANALYSIS FRAMEWORK.md** document that serves as a complete guide for:
- Python-based data analysis best practices
- Statistical testing methodologies
- Data visualization techniques
- Feature engineering approaches
- Machine learning preparation workflows

This framework can be used as a reference for future data analysis projects and provides a structured approach to comprehensive data science workflows. 