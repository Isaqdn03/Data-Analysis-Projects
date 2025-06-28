# E-commerce Customer Segmentation: RFM Analysis

Strategic customer segmentation using RFM (Recency, Frequency, Monetary) analysis to optimize marketing strategies and maximize revenue for e-commerce platform.

## 🎯 Overview

**Objective**: Segment 4,338 customers to enable personalized marketing strategies  
**Dataset**: 541K transactions, 2-year period (2018-2019)  
**Result**: Identified 8 strategic segments with actionable business recommendations

## 📊 Key Findings

### Customer Segments Identified

| Segment | Customers | Revenue Share | Avg Revenue | Priority |
|---------|-----------|---------------|-------------|----------|
| **Champions** | 947 | 64.6% | £6,077 | 🔥 HIGH |
| **Loyal Customers** | 995 | 16.9% | £1,510 | ⬆️ MEDIUM |
| **At Risk** | 363 | 9.3% | £2,277 | 🚨 URGENT |
| **About to Sleep** | 1,074 | 5.9% | £489 | ⬇️ LOW |
| **New Customers** | 126 | 0.6% | £403 | 🌱 MEDIUM |

**Business Impact**: Champions (22% of customers) drive 65% of total revenue

## 🔬 Technical Implementation

- **Framework**: RFM scoring with quintile-based segmentation
- **Tools**: Python (pandas, scikit-learn), Tableau, statistical testing
- **Validation**: K-means clustering, chi-square tests, ANOVA
- **Data Processing**: 541K→398K records after cleaning (returns, missing data)

## 📈 Strategic Recommendations

### Champions (£5.7M Revenue - 64.6%)
- **VIP program** with dedicated account management
- **Exclusive previews** and early product access
- **Referral bonuses** to leverage advocacy potential

### At Risk (£826K Revenue - 9.3%)
- **Immediate win-back** campaigns (149 days since last purchase)
- **Personalized offers** and feedback surveys
- **Phone outreach** for high-value customers

### Loyal Customers (£1.5M Revenue - 16.9%)
- **Upgrade campaigns** to increase frequency from 3.8 to 6+ orders
- **Cross-selling** to boost average revenue from £1,510 to £3,000+
- **Bundle offers** and volume discounts

## 📁 Files

- `customer_segmentation.ipynb` - Complete RFM analysis (821KB)
- `docs/ecommerce_segmentation_plan.md` - Methodology framework
- `Visualizations/` - Business intelligence charts
- `datasets/` - Processed data for Tableau integration

## 🎯 Achievements

- Identified Champions segment driving 65% of revenue with only 22% of customers
- Created data-driven resource allocation strategy (40% budget to Champions)
- Developed reusable RFM framework with statistical validation
- Generated actionable insights preventing customer churn worth £826K

## 📊 Live Dashboard

**[Interactive Tableau Dashboard →](https://public.tableau.com/app/profile/isaque.nascimento/viz/EcommerceFinalSprint/Dashboard1?publish=yes)**

---

*Comprehensive RFM segmentation framework demonstrating advanced analytics, statistical validation, and strategic business impact for e-commerce optimization.* 