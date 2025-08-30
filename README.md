# Customer Churn Prediction and AI-Powered Retention Email System

**DEPT Case Assignment - Data Scientist/AI Engineer Role**  
*Vodafone Telecommunications Customer Retention Solution*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)
[![AI](https://img.shields.io/badge/AI-Llama%203.1-green.svg)](https://huggingface.co)

## Project Overview

This comprehensive solution addresses customer retention challenges in the telecommunications industry through an integrated approach combining machine learning churn prediction with AI-powered personalized communication. The system identifies at-risk customers using advanced ML algorithms and automatically generates brand-compliant retention emails tailored to individual customer profiles and risk levels.

The project demonstrates end-to-end implementation from raw data analysis through automated customer outreach, providing measurable business value through improved retention rates and operational efficiency. Built for a telecommunications client experiencing customer churn challenges, the solution scales across different customer segments and journey stages while maintaining strict brand compliance standards.

## Business Objectives and Value Proposition

The primary business challenge addressed is the increasing customer churn rate leading to lost revenue and higher acquisition costs. This solution provides immediate value through accurate churn prediction enabling proactive retention efforts, personalized communication that improves customer engagement, automated content generation reducing operational costs, and scalable architecture supporting enterprise-wide deployment.

The system transforms reactive customer service into proactive retention management, allowing the business to intervene before customers decide to leave rather than attempting costly win-back campaigns after churn occurs. This shift in approach demonstrates significant ROI potential through improved customer lifetime value and reduced acquisition spending.

## Solution Architecture

The solution follows a multi-stage pipeline architecture beginning with comprehensive data preprocessing and exploratory analysis to understand churn patterns and customer behavior. The machine learning component trains and evaluates multiple classification algorithms to identify the optimal churn prediction model, while the AI email generation system leverages Large Language Models to create personalized, brand-compliant communications.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Raw Data      │───▶│  ML Pipeline     │───▶│  Email Generation   │
│   Processing    │    │  & Prediction    │    │  & Compliance       │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
   ┌──────────┐           ┌──────────────┐         ┌─────────────┐
   │   EDA    │           │ Churn Scores │         │ Personalized│
   │ Insights │           │ Risk Levels  │         │   Emails    │
   └──────────┘           └──────────────┘         └─────────────┘
```

Data flows seamlessly from the initial customer dataset through feature engineering and model training to generate churn probability scores, which then inform the email personalization engine. The system maintains strict brand compliance through automated validation against Vodafone tone-of-voice guidelines while supporting fallback mechanisms for reliability.

## Part I: Machine Learning Churn Prediction Results

The churn prediction component demonstrates exceptional performance through comprehensive model evaluation and selection. Seven different machine learning algorithms were trained and evaluated using cross-validation, with Gradient Boosting Classifier emerging as the top performer achieving a ROC AUC score of 0.847, representing 84.7% classification accuracy.

**Algorithm Performance Comparison**
The systematic evaluation revealed distinct performance characteristics across different model types. Gradient Boosting Classifier achieved the highest ROC AUC of 0.847, followed by Random Forest at 0.831, AdaBoost at 0.825, and Logistic Regression at 0.819. Support Vector Machine and Decision Tree algorithms showed competitive but lower performance, while Bagging Classifier provided robust baseline results.

**Feature Importance and Business Insights**
The analysis identified five critical factors driving customer churn behavior. Contract type emerged as the strongest predictor, with month-to-month customers showing significantly higher churn rates compared to annual contract holders. Customer tenure proved highly predictive, with customers in their first 12 months demonstrating elevated churn risk. Monthly charge levels correlate directly with churn probability, suggesting price sensitivity as a key factor. Internet service type impacts retention, particularly among fiber optic customers. Payment method preferences, especially electronic check users, indicate behavioral patterns associated with churn likelihood.

## Part II: AI-Powered Email Generation System

The email generation component represents an advanced application of Large Language Model technology specifically engineered for telecommunications customer retention. The system utilizes the Llama 3.1-8B model with custom prompt engineering to generate personalized, brand-compliant emails that adapt to individual customer risk profiles and demographic segments.

**Advanced Natural Language Processing**
The implementation demonstrates sophisticated prompt engineering techniques utilizing the Hugging Face Llama 3.1-8B model running locally (no external API dependencies). The system embeds Vodafone brand guidelines directly into the model's generation process through carefully crafted system prompts that include comprehensive tone requirements, mandatory structural elements, and content constraints ensuring consistency across all generated communications. User prompts dynamically incorporate customer-specific data including tenure, service details, churn probability, and identified risk factors.

**Brand Compliance and Quality Assurance**
Automated compliance checking validates every generated email against Vodafone brand standards through multi-dimensional analysis. The system evaluates mandatory elements including personalization, professional tone, content structure, and required messaging components. Quality scoring mechanisms ensure minimum compliance thresholds are met, with fallback template systems activating when LLM-generated content fails validation standards.

**Risk-Based Content Personalization**
The system implements sophisticated customer segmentation driving personalized content generation. Ultra-high risk customers (90th percentile and above) receive urgent messaging with premium offers including 30% discounts and exclusive services. High-risk customers receive appreciative messaging with valuable benefits, while medium-risk customers receive loyalty-focused communications celebrating their tenure with Vodafone.

## Technical Stack and Dependencies

The solution utilizes a carefully selected technology stack optimized for both machine learning performance and AI content generation. The implementation requires Python 3.10.18 with specific version dependencies ensuring compatibility across the AI model pipeline.

### Core Dependencies
```python
# Python Version
Python==3.10.18

# Data Science & ML
pandas==2.3.1
numpy==1.26.4
scikit-learn==1.7.1
matplotlib==3.10.5
seaborn==0.13.2

# AI & LLM
transformers==4.56.0
torch==2.6.0+cu124
huggingface-hub==0.34.4

# Jupyter Environment
jupyter==1.1.1
ipython==8.37.0

# Utilities (Built-in Python modules)
warnings
datetime
json
```

Core data science dependencies include pandas for data manipulation, numpy for numerical computing, scikit-learn for machine learning algorithms, matplotlib and seaborn for data visualization. The AI components leverage transformers library for local LLM integration with Hugging Face Llama 3.1-8B model, PyTorch for deep learning frameworks with CUDA 12.4 support, and huggingface-hub for model management without requiring external API access.

## Project Structure and Code Organization

The codebase follows enterprise-grade organization principles with clear separation of concerns across functional modules. The main analysis script orchestrates the complete workflow from data loading through results presentation, while specialized modules handle specific system components.

### Project Structure
```
├── DEPT.ipynb                  # Main analysis Jupyter notebook
├── email_generator.py          # Email generation orchestrator
├── model_loader.py            # LLM model management
├── prompt_templates.py        # Prompt engineering templates
├── compliance_checker.py      # Brand compliance validation
├── config.py                  # Configuration & brand guidelines
├── Vodafone_Customer_Churn_Sample_Dataset.csv
├── Vodafone_Tone_of_Voice_Guidelines.pdf
└── README.md
```

The main analysis notebook (DEPT.ipynb) orchestrates the complete workflow from data loading through results presentation. Specialized modules handle email generation orchestration (email_generator.py), LLM model management (model_loader.py), prompt template engineering (prompt_templates.py), brand compliance validation (compliance_checker.py), and system configuration (config.py).

## Quick Start Guide

The system can be deployed and executed through a straightforward setup process that accommodates both complete analysis workflows and individual component usage.

### Installation and Setup

**1. Clone Repository**
```bash
git clone https://github.com/Bakikhan/DEPT
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Run Complete Analysis**
```bash
jupyter notebook DEPT.ipynb
```

Installation involves standard Python package management through pip, with additional considerations for GPU acceleration when available. The modular architecture allows selective deployment of components based on specific business requirements.
