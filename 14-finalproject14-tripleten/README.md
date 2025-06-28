# Data Analysis Capstone - Multi-Domain Business Intelligence

**Advanced Analytics Portfolio | Project 14**

A comprehensive capstone project demonstrating mastery of data analysis methodologies across three specialized domains: A/B Testing, Customer Segmentation, and SQL Database Analysis. This multi-faceted project showcases end-to-end analytical capabilities from experimental design to business intelligence.

---

## 🎯 Project Overview

This final project integrates three critical data analysis disciplines:

1. **A/B Testing & Experimentation** - Recommender system effectiveness analysis
2. **Customer Segmentation & RFM Analysis** - E-commerce customer intelligence
3. **SQL Database Analysis** - Book market insights for product strategy

**Total Analysis Scope**: 67MB+ of data across multiple domains  
**Business Impact**: Statistical testing, customer intelligence, and market research  
**Technical Stack**: Python, SQL, Statistical Testing, Machine Learning, Tableau

---

## 📊 Component 1: A/B Testing Analysis

### 🔬 Recommender System Effectiveness Study

**Objective**: Evaluate the impact of a new recommendation system on customer conversion funnel for an international online store.

#### Dataset Overview
- **Test Period**: December 7-21, 2020 (acquisition), ending January 1, 2021
- **Scope**: 15% EU users, 6,000 expected participants
- **Groups**: A (control) vs B (new recommendation system)
- **Success Criteria**: 10% improvement in product_page → product_cart → purchase funnel

#### Key Files & Analysis
- `ab_test_decomposition.md` - Comprehensive methodology framework (154 lines)
- `recommender_system_test.ipynb` - Statistical analysis implementation (1,828 lines)
- `decomposition.ipynb` - Project decomposition and planning (260 lines)

#### Datasets (24.8MB total)
- `final_ab_events_upd_us.txt` - 20MB user event tracking data
- `final_ab_participants_upd_us.txt` - 561KB participant demographics (14,527 users)
- `final_ab_new_users_upd_us.txt` - 2.2MB new user registration data
- `ab_project_marketing_events_us.txt` - 879B marketing campaign data

#### Statistical Framework
- **Hypothesis Testing**: Two-proportion z-tests for conversion rates
- **Significance Level**: α = 0.05
- **Power Analysis**: Sample size validation for statistical significance
- **Funnel Analysis**: Stage-by-stage conversion optimization

#### Key Methodologies
1. **Randomization Validation** - Group balance verification
2. **Temporal Analysis** - External factor impact assessment
3. **Conversion Funnel Optimization** - Multi-stage success metrics
4. **Business Impact Quantification** - Revenue and engagement analysis

---

## 👥 Component 2: Customer Segmentation Analysis

### 🛍️ E-Commerce RFM Analysis & Customer Intelligence

**Objective**: Segment customers of "Everything Plus" (online household items store) to enable personalized marketing strategies and revenue optimization.

#### Dataset Overview
- **Primary Dataset**: `ecommerce_dataset_us.csv` (37MB)
- **Analysis Scope**: 4,340 unique customers with comprehensive transaction history
- **Business Context**: Multi-category household items e-commerce platform

#### Key Analysis Files
- `customer_segmentation.ipynb` - Complete analysis pipeline (3,313 lines, 821KB)
- `ecommerce_segmentation_plan.md` - Detailed methodology framework (633 lines, 22KB)
- `Customer_Segment_Analysis_Presentation.pdf` - Executive presentation (479KB)

#### Advanced Analytics Components

**1. RFM Analysis Implementation**
- **Recency**: Days since last purchase
- **Frequency**: Total number of orders
- **Monetary**: Total customer lifetime value

**2. Customer Segmentation Strategy**
- Champions (High value, high engagement)
- Loyal Customers (Consistent, valuable)
- Potential Loyalists (Growth opportunity)
- New Customers (Acquisition focus)
- At Risk (Retention priority)
- Lost Customers (Reactivation needed)

**3. Advanced Clustering Analysis**
- K-Means clustering with optimal cluster determination
- Hierarchical clustering for segment validation
- Product category preference clustering
- Silhouette analysis for cluster quality assessment

#### Statistical Testing & Validation
- **ANOVA Testing**: Segment performance differences
- **Kruskal-Wallis Test**: Non-parametric frequency analysis
- **Chi-Square Testing**: Category preference significance
- **Post-hoc Analysis**: Pairwise segment comparisons

#### Business Intelligence Outputs

**Tableau Integration** (5 visualization-ready datasets):
- `tableau_customer_summary.csv` - Customer metrics (294KB, 4,340 records)
- `tableau_customer_segments.csv` - Segment assignments (186KB)
- `tableau_segment_metrics.csv` - Aggregate segment performance
- `tableau_monthly_trends.csv` - Temporal business patterns
- `tableau_transactions.csv` - Transaction-level data (46MB)

**Key Visualizations** (6 professional charts):
- Business Value By Segment Analysis
- Customer Metric Distributions
- Monthly Business Trends
- RFM Metric Correlation Heatmap
- Segment Characteristics Overview
- Product Popularity by Segment

#### Strategic Insights & Recommendations
- **Revenue Optimization**: Segment-specific pricing strategies
- **Marketing Personalization**: Targeted campaign development
- **Customer Lifecycle Management**: Retention and acquisition focus
- **Product Strategy**: Category preference optimization

---

## 🗄️ Component 3: SQL Database Analysis

### 📚 Book Market Intelligence for Product Development

**Objective**: Analyze book service database to understand market trends during COVID-19, informing competitive product development for a new book-focused application.

#### Business Context
- **Market Opportunity**: COVID-19 shift to home-based reading activities
- **Strategic Goal**: Competitive product proposition development
- **Data Scope**: Comprehensive book, author, publisher, rating, and review analysis

#### Analysis Framework
- `Book_Database_Analysis.ipynb` - Complete SQL analysis (5,573 lines, 168KB)
- `book_analysis_plan.md` - Systematic methodology (206 lines, 7.2KB)
- `book_analysis_report.md` - Business insights summary (291 lines, 8.4KB)
- `SQL_queries.txt` - Production SQL code (368 lines, 12KB)
- `implementation.txt` - Technical implementation details (623 lines, 24KB)

#### Database Schema Analysis
**5 Core Tables**:
- `books` - Title, author, publisher, publication date, page count
- `authors` - Author master data
- `publishers` - Publisher information
- `ratings` - User rating data
- `reviews` - User review text analysis

#### Key Business Questions & SQL Solutions

**1. Modern Book Market Scope**
- **Query**: Books published after January 1, 2000
- **Business Value**: Understanding contemporary vs. classic literature market

**2. Engagement Metrics Analysis**
- **Query**: Reviews and average ratings per book
- **Business Value**: Content quality assessment and user engagement patterns

**3. Publisher Market Leadership**
- **Query**: Top publisher by substantial books (>50 pages)
- **Business Value**: Competitive landscape and partnership opportunities

**4. Author Quality Assessment**
- **Query**: Highest-rated author with significant volume (50+ ratings)
- **Business Value**: Content acquisition and author partnership strategies

**5. Power User Engagement**
- **Query**: Review patterns from heavy raters (50+ books rated)
- **Business Value**: Understanding core user behavior and platform engagement

#### Strategic Recommendations
- **Content Strategy**: Focus on highly-rated authors and publishers
- **User Engagement**: Design features for power users who drive reviews
- **Market Positioning**: Leverage modern book trends and user preferences
- **Platform Features**: Implement rating/review systems based on user behavior insights

---

## 🛠️ Technologies & Methodologies

### Core Technologies
- **Python Ecosystem**: Pandas, NumPy, Matplotlib, Seaborn, Plotly
- **Statistical Analysis**: SciPy, Scikit-learn, Statistical Testing
- **Database**: SQL, PostgreSQL, Database Design
- **Machine Learning**: K-Means, Hierarchical Clustering, RFM Analysis
- **Visualization**: Tableau, Interactive Dashboards, Executive Reporting

### Advanced Methodologies
- **Experimental Design**: A/B testing, randomization, statistical power
- **Customer Analytics**: RFM analysis, behavioral segmentation, lifetime value
- **Database Analytics**: Complex joins, window functions, business intelligence
- **Statistical Testing**: ANOVA, chi-square, post-hoc analysis, effect sizes
- **Business Intelligence**: KPI development, executive reporting, strategic recommendations

---

## 📈 Key Findings & Business Impact

### A/B Testing Results
- **Statistical Significance**: Validated conversion improvement measurement
- **Funnel Optimization**: Multi-stage conversion analysis
- **Revenue Impact**: Quantified business value of recommendation system
- **Implementation Roadmap**: Clear go/no-go decision framework

### Customer Segmentation Insights
- **Revenue Concentration**: Champions segment driving disproportionate value
- **Retention Opportunities**: At-risk customer identification and intervention
- **Acquisition Strategy**: New customer onboarding optimization
- **Product Development**: Category preference-driven feature development

### SQL Database Intelligence
- **Market Opportunities**: COVID-19 reading trend capitalization
- **Competitive Positioning**: Publisher and author partnership strategies
- **User Behavior**: Power user engagement patterns for platform design
- **Content Strategy**: Data-driven book acquisition and curation

---

## 📁 Project Structure

```
14-finalproject14-tripleten/
├── AB-testing/
│   ├── datasets/                    # 24.8MB A/B test data
│   ├── ab_test_decomposition.md     # Methodology framework
│   ├── recommender_system_test.ipynb # Statistical analysis
│   ├── decomposition.ipynb          # Project planning
│   └── template.ipynb               # Analysis template
├── cotumer-segmentation-final/
│   ├── datasets/                    # 83MB+ segmentation data
│   ├── docs/                        # Strategic documentation
│   ├── Visualizations/              # 6 professional charts
│   └── customer_segmentation.ipynb  # Complete RFM analysis
└── SQL-project/
    ├── Planning/                    # Strategic planning docs
    └── Book_Database_Analysis.ipynb # SQL market analysis
```

---

## 🚀 Technical Implementation

### Statistical Rigor
- **Power Analysis**: Sample size validation for A/B testing
- **Multiple Testing Correction**: False discovery rate control
- **Effect Size Calculation**: Practical significance assessment
- **Confidence Intervals**: Robust uncertainty quantification

### Scalable Architecture
- **Modular Design**: Reusable analysis components
- **Performance Optimization**: Efficient data processing for large datasets
- **Reproducible Research**: Documented methodology and version control
- **Business Integration**: Tableau-ready outputs and executive reporting

### Quality Assurance
- **Data Validation**: Comprehensive quality checks and anomaly detection
- **Statistical Validation**: Assumption testing and robustness checks
- **Business Logic Validation**: Results verification against domain knowledge
- **Documentation Standards**: Professional-grade analysis documentation

---

## 💼 Business Applications

This capstone project demonstrates expertise in:

1. **Experimental Design** - A/B testing for product optimization
2. **Customer Intelligence** - Segmentation for marketing personalization
3. **Market Research** - SQL analysis for strategic decision-making
4. **Business Intelligence** - KPI development and executive reporting
5. **Data-Driven Strategy** - Statistical insights for business growth

**Strategic Value**: End-to-end analytics capabilities from experimental design through customer intelligence to market research, providing comprehensive business intelligence for data-driven decision making.

---

## 🎓 Learning Outcomes

**Advanced Analytics Skills**:
- Multi-domain data analysis expertise
- Statistical testing and experimental design
- Customer behavior analysis and segmentation
- Database analysis and business intelligence
- Executive communication and strategic recommendations

**Technical Mastery**:
- Python ecosystem for data science
- SQL for business intelligence
- Statistical analysis and machine learning
- Data visualization and executive reporting
- Project management and documentation

**Business Acumen**:
- Strategic thinking and problem-solving
- Cross-functional analysis integration
- Executive communication and presentation
- Business impact quantification
- Data-driven decision making

---

*This capstone project demonstrates mastery of advanced analytics methodologies and business intelligence capabilities across multiple domains, showcasing the complete data analysis lifecycle from exploration to implementation.* 