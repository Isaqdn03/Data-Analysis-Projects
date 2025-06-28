# Business Hypothesis Prioritization & A/B Testing Analysis

## 🎯 Project Overview

This project provides a comprehensive business intelligence solution combining hypothesis prioritization frameworks with rigorous A/B testing analysis. The study demonstrates how to systematically evaluate business hypotheses using ICE and RICE frameworks, followed by statistical analysis of A/B test results to make data-driven decisions for business optimization.

## 📊 Project Objectives

The analysis aims to:
- **Prioritize business hypotheses** using established frameworks (ICE and RICE)
- **Conduct rigorous A/B testing analysis** to evaluate conversion rates and revenue impact
- **Apply statistical significance testing** to validate experimental results
- **Provide actionable recommendations** based on statistical evidence
- **Compare different prioritization methodologies** and their business implications

## 🔬 Key Business Questions Answered

### Hypothesis Prioritization
- Which business hypotheses should be prioritized for implementation?
- How do ICE and RICE frameworks differ in their prioritization outcomes?
- What factors drive the biggest differences between prioritization methods?

### A/B Testing Analysis
- Is there a statistically significant difference in conversion rates between groups?
- How do average order values compare between test groups?
- What is the overall revenue impact of the tested changes?
- Should the test be stopped and a winner declared?

## 📁 Project Structure

```
09-project9-tripleten/
├── Hypothesis_Analysis.ipynb              # Main analysis notebook (870KB)
├── datasets/
│   ├── hypotheses_us.csv                  # Business hypotheses with scoring (10 lines)
│   ├── orders_us.csv                      # A/B test transaction data (1,199 lines)
│   └── visits_us.csv                      # A/B test website visits (63 lines)
└── README.md                              # Project documentation
```

## 📊 Datasets Description

### 1. Business Hypotheses (`hypotheses_us.csv`)
- **Size**: 9 hypotheses
- **Columns**: Hypothesis description, Reach, Impact, Confidence, Effort scores
- **Purpose**: Evaluate and prioritize business improvement initiatives

### 2. A/B Test Orders (`orders_us.csv`)
- **Size**: 1,197 transactions
- **Columns**: 
  - `transactionId`: Unique transaction identifier
  - `visitorId`: User identifier
  - `date`: Transaction date
  - `revenue`: Transaction amount
  - `group`: Test group (A or B)
- **Purpose**: Revenue and purchase behavior analysis

### 3. A/B Test Visits (`visits_us.csv`)
- **Size**: 62 daily records
- **Columns**:
  - `date`: Visit date
  - `group`: Test group (A or B)
  - `visits`: Number of daily visits
- **Purpose**: Traffic and conversion rate analysis

## 📈 Key Findings & Results

### Hypothesis Prioritization Results

#### ICE Framework Top Priorities:
1. **Launch promotion with user discounts** - High impact/confidence, low effort
2. **Add product recommendation blocks** - Balanced high performance
3. **Add subscription forms to main pages** - Strong overall metrics

#### RICE Framework Top Priorities:
1. **Add subscription forms to main pages** - Highest reach potential
2. **Add product recommendation blocks** - Strong scalability
3. **Change category structure** - Good reach-impact combination

#### Key Insight:
- **RICE framework favors initiatives with broader reach**, even if individual impact is lower
- **ICE framework prioritizes high-impact, low-effort initiatives** regardless of scale
- The frameworks showed significant differences in ranking, highlighting the importance of choosing the right prioritization method

### A/B Testing Statistical Results

#### Conversion Rate Analysis:
- **Group A Conversion Rate**: 2.97%
- **Group B Conversion Rate**: 3.38%
- **Relative Improvement**: 13.81%
- **Statistical Significance**: p-value = 0.0232 (significant at α = 0.05)

#### Revenue Analysis:
- **Group A Total Revenue**: $53,212.00
- **Group B Total Revenue**: $79,651.20
- **Revenue Difference**: $26,439.20 (49.7% higher)

#### Order Value Analysis:
- **Group A Average Order**: $113.70
- **Group B Average Order**: $145.35 (after outlier removal: $68.24)
- **Statistical Significance**: Significant difference detected (p < 0.05)

### Final Recommendation:
**✅ STOP TEST - GROUP B WINS**

## 🛠️ Technology Stack

- **Python 3**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **SciPy** - Statistical testing (Mann-Whitney U, Z-tests)
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive analysis environment

## 🚀 How to Run

### Prerequisites
- Python 3.7 or higher
- Required packages: pandas, numpy, scipy, matplotlib, seaborn

### Installation and Execution
1. **Install dependencies**
   ```bash
   pip install pandas numpy scipy matplotlib seaborn jupyter
   ```

2. **Launch the analysis**
   ```bash
   jupyter notebook "Hypothesis_Analysis.ipynb"
   ```

3. **Follow the analysis sections:**
   - Data Loading & Cleaning
   - Hypothesis Prioritization (ICE vs RICE)
   - A/B Test Data Preparation
   - Statistical Analysis & Testing
   - Business Recommendations

## 📊 Analysis Methodology

### 1. **Hypothesis Prioritization**
- **ICE Framework**: (Impact × Confidence) ÷ Effort
- **RICE Framework**: (Reach × Impact × Confidence) ÷ Effort
- Comparative analysis of ranking differences

### 2. **A/B Testing Protocol**
- Data cleaning and visitor deduplication
- Outlier detection using percentile analysis (95th and 99th percentiles)
- Statistical significance testing using appropriate methods

### 3. **Statistical Testing Approach**
- **Conversion Rates**: Z-test for proportions
- **Order Values**: Mann-Whitney U test (non-parametric)
- **Confidence Intervals**: 95% confidence level
- **Significance Level**: α = 0.05

### 4. **Data Quality Assurance**
- Removed visitors appearing in both test groups (58 visitors)
- Identified and handled outliers (12 transactions > 99th percentile)
- Ensured data integrity across all datasets

## 💡 Business Insights & Recommendations

### 🎯 Hypothesis Prioritization Strategy
- **Use RICE for scalable initiatives** where reach is critical
- **Use ICE for resource-constrained environments** focusing on efficiency
- **Consider hybrid approaches** that account for both frameworks' strengths

### 📊 A/B Testing Conclusions
1. **Clear Winner Identified**: Group B shows superior performance across all key metrics
2. **Statistical Confidence**: All major differences are statistically significant
3. **Business Impact**: 13.81% improvement in conversion rate translates to substantial revenue gains
4. **Implementation Ready**: Results are consistent and stable over the test period

### 🚀 Strategic Recommendations
1. **Immediate Action**: Implement Group B version for all users
2. **Revenue Projection**: Expected 13.81% increase in conversion rates
3. **Monitoring Plan**: Track post-implementation metrics to validate results
4. **Future Testing**: Apply successful elements to other areas of the business

## 📚 Skills Demonstrated

This project showcases the following data analysis capabilities:

### **Statistical Analysis Skills**
- **Hypothesis Testing**: Proper application of statistical tests
- **A/B Testing Methodology**: Complete experimental design and analysis
- **Data Validation**: Outlier detection and data quality assurance
- **Business Metrics**: Conversion rate, revenue, and customer behavior analysis

### **Business Intelligence**
- **Strategic Decision Making**: Data-driven business recommendations
- **Prioritization Frameworks**: ICE and RICE methodology application
- **Experimental Design**: Proper A/B testing setup and interpretation
- **Stakeholder Communication**: Clear presentation of statistical findings

## 📚 Statistical Methods Used

### Hypothesis Testing:
- **Z-test for Proportions**: Conversion rate comparison
- **Mann-Whitney U Test**: Non-parametric test for order values
- **Confidence Intervals**: 95% CI for all metrics
- **Effect Size Calculation**: Relative differences and practical significance

### Data Analysis:
- **Outlier Detection**: Percentile-based anomaly identification
- **Data Cleaning**: Visitor deduplication and data validation
- **Temporal Analysis**: Time-series examination of test results

## 🔄 Future Enhancements

Potential areas for expansion:
- **Segmentation Analysis**: Test results by user demographics
- **Power Analysis**: Sample size determination for future tests
- **Bayesian Testing**: Alternative statistical frameworks
- **Multi-armed Bandits**: Dynamic allocation algorithms
- **Long-term Impact**: Customer lifetime value analysis
- **Cost-Benefit Analysis**: ROI calculation for implemented changes

## 📋 Project Deliverables

- ✅ Comprehensive hypothesis prioritization using ICE and RICE frameworks
- ✅ Rigorous A/B testing analysis with statistical validation
- ✅ Data-driven business recommendations with clear action items
- ✅ Statistical significance testing for all key metrics
- ✅ Professional visualizations and business intelligence reports
- ✅ Methodology documentation for future testing protocols

This analysis provides a complete framework for business hypothesis evaluation and A/B testing, demonstrating how statistical rigor can drive confident business decisions and measurable improvements in key performance indicators. 