# Megaline Telecommunications Plan Analysis

## 📊 Project Overview

This project analyzes data from a fictional telecommunications company called Megaline to determine which of their two prepaid plans (Surf and Ultimate) generates more revenue. The analysis is based on data from 500 customers collected during 2018.

## 🎯 Objective

Perform a comparative analysis of the Surf and Ultimate plans to:
- Understand customer usage behavior
- Determine which plan generates more revenue
- Provide data-driven recommendations for advertising budget optimization

## 📁 Data Structure

The project uses 5 datasets:

- **`megaline_calls.csv`** - Phone call data
- **`megaline_messages.csv`** - Text message data
- **`megaline_internet.csv`** - Internet usage data
- **`megaline_users.csv`** - User information
- **`megaline_plans.csv`** - Available plan details

## 🔍 Methodology

1. **Data Preparation**
   - Data cleaning and type correction
   - Date conversion to datetime format
   - Missing value treatment

2. **Behavioral Analysis**
   - Call, message, and internet usage patterns by plan
   - Monthly temporal analysis
   - Identification of differences between user groups

3. **Revenue Analysis**
   - Monthly revenue calculation per user
   - Revenue comparison between plans
   - Variability analysis

4. **Statistical Testing**
   - Student's t-test for comparing average revenues between plans
   - Hypothesis testing for regional differences (NY-NJ vs other regions)

## 📈 Key Findings

- **Ultimate Plan** generates significantly more revenue per user
- **Ultimate Users** have more variable and intensive usage patterns
- **Surf Users** show gradual growth in usage throughout the year
- **No significant difference** in revenue between geographical regions

## 🛠️ Technologies Used

- **Python 3**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Data visualization
- **SciPy** - Statistical testing

## 📋 How to Run

1. Ensure all dependencies are installed
2. Place CSV files in the `datasets/` folder
3. Run the notebook `Final Project Sprint 4.ipynb`

## 💡 Business Recommendations

1. **Focus marketing on Ultimate plan** for high-consumption customers
2. **Optimize Surf plan** to attract more moderate-consumption users
3. **Develop specific strategies** for each customer profile
4. **Consider price adjustments** based on identified usage patterns

## 👨‍💼 Project Context

This project demonstrates the application of:
- Exploratory data analysis
- Inferential statistics
- Data visualization
- Data-driven decision making 