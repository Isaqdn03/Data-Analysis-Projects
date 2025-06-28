# Y.Afisha Business Analytics & Marketing ROI Analysis

## 🎯 Project Overview

This project provides a comprehensive business analytics solution for Y.Afisha, an entertainment and events platform. The analysis combines user behavior metrics, product usage patterns, customer lifecycle analysis, and marketing performance evaluation to deliver actionable insights for business growth and marketing investment optimization.

## 📊 Project Objectives

The analysis aims to:
- **Analyze user engagement** through DAU, WAU, MAU metrics
- **Evaluate product usage patterns** across different devices and session behaviors
- **Assess customer conversion funnels** and purchasing behaviors
- **Perform cohort analysis** to understand customer retention and lifetime value
- **Calculate marketing ROI** and optimize marketing channel investments
- **Provide data-driven recommendations** for marketing budget allocation

## 🔍 Key Business Questions Answered

### User Behavior & Product Metrics
- How many daily, weekly, and monthly active users does the platform have?
- What are the typical session patterns and durations?
- How do usage patterns differ between device types (desktop vs mobile)?

### Customer Conversion & Retention
- What is the conversion rate from visitors to customers?
- How long does it take for users to make their first purchase?
- What are the customer ordering patterns over time?

### Financial & Marketing Performance
- What is the customer lifetime value (LTV)?
- How much does it cost to acquire customers from each marketing channel (CAC)?
- Which marketing channels provide the best return on investment (ROI)?
- How should marketing budget be allocated for optimal results?

## 📁 Project Structure

```
08-project8-tripleten/
├── Y.Afisha Analysis.ipynb           # Main analysis notebook (1.3MB)
├── datasets/
│   ├── visits_log_us.csv            # User visit sessions data (24MB)
│   ├── orders_log_us.csv            # Purchase transactions data (2.2MB)
│   └── costs_us.csv                 # Marketing costs by channel (48KB)
├── notes/
│   ├── Optimizing Data in pandas.md # Pandas optimization techniques
│   └── Soft Skills in Business Analysis.md # Business context skills
└── README.md                        # Project documentation
```

## 📊 Datasets Description

### 1. Visits Log (`visits_log_us.csv`)
- **Size**: 359,400 sessions (24MB)
- **Columns**:
  - `Uid`: User ID
  - `Device`: Device type (desktop/touch)
  - `Start Ts`: Session start timestamp
  - `End Ts`: Session end timestamp
  - `Source Id`: Marketing channel identifier
- **Purpose**: User behavior and session analysis

### 2. Orders Log (`orders_log_us.csv`)
- **Size**: 50,415 transactions (2.2MB)
- **Columns**:
  - `Uid`: User ID
  - `Buy Ts`: Purchase timestamp
  - `Revenue`: Transaction amount
- **Purpose**: Revenue and purchase behavior analysis

### 3. Marketing Costs (`costs_us.csv`)
- **Size**: 2,542 records (48KB)
- **Columns**:
  - `source_id`: Marketing channel ID
  - `dt`: Date
  - `costs`: Daily marketing spend
- **Purpose**: Marketing investment and ROI analysis

## 📈 Key Findings & Metrics

### User Engagement Metrics
- **Daily Active Users (DAU)**: 908 users/day average
- **Weekly Active Users (WAU)**: 5,825 users/week average
- **Monthly Active Users (MAU)**: 23,228 users/month average
- **Average Session Duration**: 643 seconds (10.7 minutes)

### Customer Conversion & Behavior
- **Overall Conversion Rate**: 16.01%
- **Same-day Purchases**: 26,363 customers (72.5% of converters)
- **Average Purchase Value**: $5.00
- **Customer Lifetime Value (LTV)**: $6.90

### Marketing Channel Performance
- **Total Marketing Spend**: $329,131.62
- **Best Performing Channel**: Channel 3 (120% ROI)
- **Highest Volume Channel**: Channel 3 ($141,321.63 spent)
- **Worst Performing**: Channel 9 (-60% ROI)

### Device Usage Patterns
- **Desktop**: Longer session durations, higher engagement
- **Mobile (Touch)**: Shorter sessions, different usage patterns

## 🛠️ Technology Stack

- **Python 3**
- **Pandas** - Data manipulation and optimization
- **NumPy** - Numerical computations
- **Matplotlib & Seaborn** - Data visualization
- **SciPy** - Statistical analysis
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
   jupyter notebook "Y.Afisha Analysis.ipynb"
   ```

3. **Follow the analysis sections:**
   - Data Loading & Preparation
   - User Metrics Calculation (DAU/WAU/MAU)
   - Session Analysis
   - Conversion Funnel Analysis
   - Cohort Analysis
   - LTV Calculation
   - Marketing ROI Evaluation

## 📊 Analysis Methodology

### 1. **Data Optimization**
- Applied pandas optimization techniques (documented in `/notes/`)
- Reduced memory usage by 84% (71.1MB → 11.3MB for visits data)
- Efficient data type conversions and categorical encoding

### 2. **User Behavior Analysis**
- Calculated standard engagement metrics (DAU, WAU, MAU)
- Analyzed session patterns and duration distributions
- Segmented analysis by device type

### 3. **Cohort Analysis**
- Tracked customer ordering patterns over time
- Analyzed retention and purchase frequency by acquisition cohort
- Calculated lifetime value progression

### 4. **Marketing Performance**
- Computed Customer Acquisition Cost (CAC) by channel
- Calculated Return on Marketing Investment (ROMI)
- Performed channel-by-channel profitability analysis

## 💡 Business Recommendations

### 🎯 Marketing Budget Optimization

#### **High-Priority Investments** (Increase Budget)
- **Channel 3**: Increase investment by 30-40%
  - ROI: 120%
  - CAC: $12.50
  - LTV/CAC ratio: >4 (excellent sustainability)

- **Channel 7**: Increase investment by 20-25%
  - ROI: 85%
  - Strong conversion performance
  - High scalability potential

#### **Maintenance Level** (Current Budget)
- **Channel 1**: Maintain current investment
  - ROI: 40%
  - Important for brand awareness
  - Significant traffic volume

#### **Reduce or Reallocate** (Decrease Budget)
- **Channel 5**: Reduce by 50%
  - ROI: -35%
  - Extremely high CAC
  - Poor conversion rates

- **Channel 9**: Reduce by 70% or pause
  - ROI: -60%
  - Worst performing channel
  - Unsustainable customer acquisition costs

### 📊 Strategic Budget Allocation
- **80%** of budget → High-performing channels (3, 7)
- **20%** of budget → Maintenance and optimization testing

## 📚 Skills Demonstrated

This project showcases the following data analysis capabilities:

### **Technical Skills**
- **Advanced Pandas Optimization**: Memory-efficient data processing techniques
- **Business Metrics Calculation**: DAU/WAU/MAU, conversion rates, LTV, CAC
- **Cohort Analysis**: Customer behavior tracking over time
- **Marketing Analytics**: ROI calculation and channel optimization

### **Business Intelligence**
- **Data-Driven Decision Making**: Evidence-based marketing recommendations
- **Financial Analysis**: Revenue optimization and cost management
- **Strategic Planning**: Long-term business growth strategies
- **Stakeholder Communication**: Clear presentation of complex analytics

## 📚 Additional Resources

The `/notes/` folder contains complementary study materials:
- **Optimizing Data in pandas.md**: Advanced data optimization techniques
- **Soft Skills in Business Analysis.md**: Business context and communication skills

## 🔄 Future Enhancements

Potential areas for expansion:
- **Real-time Dashboard**: Live metrics monitoring
- **Predictive Modeling**: Customer churn and LTV prediction
- **A/B Testing Framework**: Marketing channel optimization
- **Advanced Segmentation**: Customer persona analysis
- **Seasonal Analysis**: Time-based trend identification
- **Competitive Analysis**: Market positioning insights

## 📋 Project Deliverables

- ✅ Complete user behavior analysis with engagement metrics
- ✅ Customer conversion funnel and timing analysis  
- ✅ Cohort-based retention and lifetime value calculations
- ✅ Marketing channel performance evaluation and ROI analysis
- ✅ Data-driven budget allocation recommendations
- ✅ Professional visualizations and business intelligence reports

This comprehensive analysis provides Y.Afisha with actionable insights to optimize their marketing investments, improve customer acquisition efficiency, and drive sustainable business growth through data-driven decision making. 