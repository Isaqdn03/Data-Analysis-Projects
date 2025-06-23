# Chicago Taxi Data Analysis

## 🚕 Project Overview

This project analyzes Chicago taxi ride data to understand ride patterns, identify top-performing taxi companies, analyze popular destinations, and test hypotheses about how weather conditions affect ride duration. The analysis combines SQL query results with Python data analysis to provide comprehensive insights into the Chicago taxi market.

## 🎯 Objectives

The project aims to:
- Identify the top 10 taxi companies by number of rides
- Analyze the most popular destination neighborhoods
- Examine ride patterns and distributions
- Test the hypothesis: **"Average trip duration from Loop to O'Hare International Airport changes on rainy Saturdays"**

## 📊 Key Analyses

### 1. **Taxi Company Performance Analysis**
- Ranking of taxi companies by total number of trips
- Market share analysis across different operators

### 2. **Destination Analysis**
- Top 10 neighborhoods by average number of trips as destinations
- Geographic distribution of taxi demand

### 3. **Statistical Hypothesis Testing**
- Weather impact on trip duration (Loop to O'Hare route)
- Comparison between rainy and non-rainy Saturday trips
- Statistical significance testing using t-tests

## 📈 Key Findings

### Top Taxi Companies (by trip volume):
1. **Flash Cab** - 19,558 trips
2. **Taxi Affiliation Services** - 11,422 trips  
3. **Medallion Leasin** - 10,367 trips
4. **Yellow Cab** - 9,888 trips
5. **Taxi Affiliation Service Yellow** - 9,299 trips

### Most Popular Destinations:
1. **Loop** - 10,727 average trips
2. **River North** - 9,523 average trips
3. **Streeterville** - 6,664 average trips
4. **West Loop** - 5,163 average trips
5. **O'Hare** - 2,546 average trips

### Hypothesis Testing Results:
- **Statistical Significance**: p-value < 0.001 (highly significant)
- **Rainy Saturday Duration**: 2,427 seconds (40.5 minutes)
- **Non-Rainy Saturday Duration**: 1,999 seconds (33.3 minutes)
- **Conclusion**: Trip duration from Loop to O'Hare is significantly longer on rainy Saturdays

## 🛠️ Technology Stack

- **Python 3**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Advanced statistical visualizations
- **SciPy** - Statistical testing (t-tests)

## 📁 Project Structure

```
07-project7-tripleten/
├── project7-tripleten.ipynb           # Main analysis notebook
├── datasets/
│   ├── moved_project_sql_result_01.csv # Taxi company trip data (66 lines)
│   ├── moved_project_sql_result_04.csv # Destination analysis data (96 lines)
│   └── moved_project_sql_result_07.csv # Loop-to-O'Hare trip data (1,070 lines)
└── README.md                          # Project documentation
```

## 📊 Dataset Information

### File 1: Taxi Company Data (`moved_project_sql_result_01.csv`)
- **Size**: 66 records
- **Columns**: `company_name`, `trips_amount`
- **Purpose**: Analysis of taxi company performance

### File 2: Destination Data (`moved_project_sql_result_04.csv`) 
- **Size**: 96 records
- **Columns**: `dropoff_location_name`, `average_trips`
- **Purpose**: Neighborhood destination analysis

### File 3: Weather Impact Data (`moved_project_sql_result_07.csv`)
- **Size**: 1,070 records
- **Columns**: `start_ts`, `weather_conditions`, `duration_seconds`
- **Purpose**: Hypothesis testing for weather impact on trip duration

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
   jupyter notebook project7-tripleten.ipynb
   ```

3. **Follow the notebook sections:**
   - Data Import and Exploration
   - Company Performance Analysis
   - Destination Analysis
   - Hypothesis Testing

## 📈 Visualizations

The project includes several key visualizations:
- **Bar charts** showing top taxi companies and destinations
- **Box plots** comparing trip durations by weather conditions
- **Histograms** displaying distribution of trip durations
- **Statistical plots** for hypothesis testing results

## 🔍 Statistical Methods

### Hypothesis Testing Framework:
- **Null Hypothesis (H₀)**: No difference in trip duration between rainy and non-rainy Saturdays
- **Alternative Hypothesis (H₁)**: Significant difference exists
- **Test Used**: Welch's t-test (independent samples, unequal variances)
- **Significance Level**: α = 0.05
- **Result**: Rejected null hypothesis (p < 0.001)

## 💡 Business Insights

### Market Concentration:
- The taxi market shows significant concentration with top 5 companies handling majority of trips
- Flash Cab dominates with nearly double the trips of the second-largest operator

### Geographic Demand:
- Downtown areas (Loop, River North) receive the highest taxi traffic
- Commercial and business districts are primary destinations
- Airport connections represent significant but specialized demand

### Weather Impact:
- Rainy conditions significantly increase travel time to airports
- Trip duration increases by approximately 21% (7+ minutes) on rainy Saturdays
- Weather-related delays should be factored into scheduling and pricing

## 🎓 Learning Context

This project was developed as part of the TripleTen Data Analysis program, demonstrating:
- **SQL to Python Integration**: Working with SQL query results in Python
- **Statistical Analysis**: Hypothesis testing with real-world data
- **Data Visualization**: Creating meaningful charts for business insights
- **Business Intelligence**: Translating data analysis into actionable insights

## 🔄 Future Enhancements

Potential areas for expansion:
- Seasonal analysis across different months
- Route optimization based on traffic patterns
- Pricing strategy recommendations
- Weather prediction integration
- Real-time demand forecasting
- Customer satisfaction correlation analysis 