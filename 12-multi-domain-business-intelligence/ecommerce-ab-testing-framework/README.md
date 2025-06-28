# E-commerce A/B Testing: Recommendation System Analysis

Statistical analysis of recommendation system effectiveness for international e-commerce platform using rigorous experimental design and hypothesis testing.

## 🎯 Overview

**Objective**: Evaluate new recommendation system impact on conversion rates  
**Scale**: 11,498 participants, 15-day test period  
**Result**: **DO NOT IMPLEMENT** - 5.44% purchase conversion decline

## 📊 Key Findings

| Metric | Group A | Group B | Change | Target | Result |
|--------|---------|---------|--------|--------|--------|
| Product Page | 61.24% | 59.35% | -3.10% | +10% | ❌ |
| Add to Cart | 16.36% | 16.74% | +2.32% | +10% | ❌ |
| Purchase | 33.99% | 32.14% | -5.44% | +10% | ❌ |

**Business Impact**: $2,215.75 revenue loss, 93 fewer conversions

## 🔬 Technical Implementation

- **Framework**: Two-proportion z-tests (α = 0.05)
- **Tools**: Python (pandas, scipy.stats), matplotlib, seaborn
- **Data**: 423K+ events across 4 datasets
- **Validation**: Statistical significance testing, external factor analysis

## 📁 Files

- `recommender_system_test.ipynb` - Main analysis (170KB)
- `ab_test_decomposition.md` - Methodology breakdown
- `template.ipynb` - Reusable framework
- `datasets/` - Raw data files

## 🎯 Achievements

- Prevented $2,215+ revenue loss through data-driven decision
- Created reusable A/B testing framework
- Applied rigorous statistical methodology with proper validation
- Delivered executive-ready recommendations

---

*Comprehensive A/B testing framework demonstrating statistical rigor and business impact analysis for data-driven decision making.* 