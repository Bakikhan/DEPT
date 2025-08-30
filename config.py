"""
Configuration file for DEPT Case Assignment - Email Generation System
Contains all constants, model settings, and brand guidelines
"""

import os
from datetime import datetime

# ========================================================================
# MODEL CONFIGURATION
# ========================================================================

# Hugging Face Configuration
HF_TOKEN = "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"  # Replace with your token
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Local Model Storage
MODEL_DIR = "./model_cache"
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "llama-3.1-8b")

# Generation Settings
GENERATION_CONFIG = {
    "max_new_tokens": 250,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "return_full_text": False,
    "clean_up_tokenization_spaces": True
}

# Device Configuration
DEVICE_CONFIG = {
    "torch_dtype": "float16",
    "device_map": "auto",
    "low_cpu_mem_usage": True
}

# ========================================================================
# VODAFONE BRAND GUIDELINES
# ========================================================================

VODAFONE_BRAND_GUIDELINES = {
    "tone_attributes": {
        "friendly_approachable": "Use warm and conversational language to make customers feel valued. Avoid overly technical jargon.",
        "clear_concise": "Ensure messages are straightforward and easy to digest. Use short sentences and bullet points for clarity.",
        "positive_reassuring": "Highlight benefits and positive outcomes to reassure customers. Address concerns empathetically and provide solutions.", 
        "professional_trustworthy": "Maintain a respectful and courteous tone. Ensure information is accurate and reliable."
    },
    
    "email_structure": {
        "subject_line": "Friendly, enticing, and relevant to customer interests",
        "greeting": "Warm and personalized with customer name",
        "introduction": "Brief explanation of email purpose",
        "body": "Special offers/updates with bullet points for key information",
        "call_to_action": "Clear and compelling",
        "closing": "Warm and appreciative",
        "signature": "Friendly and professional from Vodafone Customer Care Team"
    },
    
    "personalization_rules": [
        "Always use customer's name",
        "Reference their tenure and services",
        "Tailor offers based on usage patterns",
        "Show appreciation for loyalty"
    ],
    
    "content_requirements": {
        "max_word_count": 150,
        "min_word_count": 50,
        "required_elements": ["subject_line", "greeting", "bullet_points", "cta", "signature"],
        "forbidden_words": ["hey", "sup", "yo", "awesome", "cool"],
        "mandatory_phrases": ["Thank you", "valued customer", "Vodafone"]
    }
}

# ========================================================================
# EMAIL TEMPLATES BY RISK LEVEL
# ========================================================================

EMAIL_TEMPLATES = {
    "ultra_high_risk": {
        "threshold": 0.9,
        "subject_template": "Important: Exclusive Offers for You, {name}!",
        "discount_percentage": 30,
        "offer_duration": "7 days",
        "urgency_level": "high",
        "benefits": [
            "30% discount on your monthly bill for 6 months",
            "Free premium streaming services (Netflix, Amazon Prime)",
            "Priority customer support with dedicated hotline",
            "Complimentary device upgrade worth up to £200"
        ]
    },
    
    "high_risk": {
        "threshold": 0.7,
        "subject_template": "Special Thank You Offer, {name}!",
        "discount_percentage": 25,
        "offer_duration": "14 days",
        "urgency_level": "medium",
        "benefits": [
            "25% discount on your next plan upgrade",
            "Free international calling for 3 months",
            "Priority technical support access",
            "Early access to new Vodafone services"
        ]
    },
    
    "medium_risk": {
        "threshold": 0.5,
        "subject_template": "Celebrating {tenure} Months Together, {name}!",
        "discount_percentage": 15,
        "offer_duration": "30 days",
        "urgency_level": "low",
        "benefits": [
            "Loyalty bonus: 50% more data at no extra cost",
            "Exclusive member pricing on latest devices",
            "Free Vodafone TV entertainment package trial",
            "VIP customer service priority status"
        ]
    }
}

# ========================================================================
# CUSTOMER SEGMENTS CONFIGURATION
# ========================================================================

CUSTOMER_SEGMENTS = {
    "senior_citizens": {
        "tone_adjustments": ["Simple language", "Security focus", "Clear instructions"],
        "preferred_benefits": ["Device protection", "Tech support", "Simple plans"]
    },
    
    "young_adults": {
        "tone_adjustments": ["Tech features", "Social benefits", "Modern language"],
        "preferred_benefits": ["Streaming services", "High-speed internet", "Mobile apps"]
    },
    
    "families": {
        "tone_adjustments": ["Family-focused", "Value emphasis", "Safety features"],
        "preferred_benefits": ["Bundle offers", "Parental controls", "Multi-device plans"]
    },
    
    "business_customers": {
        "tone_adjustments": ["ROI focus", "Enterprise features", "Professional language"],
        "preferred_benefits": ["Business solutions", "Reliability", "Support packages"]
    }
}

# ========================================================================
# COMPLIANCE CHECKING CONFIGURATION
# ========================================================================

COMPLIANCE_CHECKS = {
    "mandatory_elements": {
        "has_subject_line": {"pattern": "Subject:", "weight": 1.0},
        "personal_greeting": {"patterns": ["Hi ", "Dear "], "weight": 1.0},
        "uses_customer_name": {"dynamic": True, "weight": 1.0},
        "mentions_tenure": {"patterns": ["months", "tenure", "time with"], "weight": 0.8},
        "has_bullet_points": {"patterns": ["•", "- ", "* "], "weight": 0.8},
        "compelling_offers": {"patterns": ["discount", "free", "exclusive", "bonus"], "weight": 1.0},
        "clear_cta": {"patterns": ["Call", "Click", "Visit", "Contact"], "weight": 1.0},
        "professional_closing": {"patterns": ["Best regards", "Thank you", "Sincerely"], "weight": 1.0},
        "vodafone_signature": {"pattern": "Vodafone", "weight": 1.0}
    },
    
    "quality_checks": {
        "appropriate_length": {"min_words": 100, "max_words": 300, "weight": 0.8},
        "professional_tone": {"forbidden": ["hey", "awesome", "cool", "sup"], "weight": 1.0},
        "sentence_structure": {"max_sentence_length": 25, "weight": 0.6}
    },
    
    "scoring": {
        "minimum_compliance_score": 0.75,
        "excellent_threshold": 0.9,
        "good_threshold": 0.8
    }
}

# ========================================================================
# SAMPLE DATA CONFIGURATION
# ========================================================================

SAMPLE_HIGH_RISK_CUSTOMERS = [
    {
        "customerID": "7590-VHVEG",
        "name": "Sarah",
        "tenure": 3,
        "contract": "Month-to-month",
        "monthly_charges": 89.50,
        "internet_service": "Fiber optic", 
        "senior_citizen": False,
        "partner": True,
        "churn_probability": 0.85,
        "risk_level": "High",
        "key_risk_factors": ["Short tenure", "Month-to-month contract", "High monthly charges"],
        "segment": "young_adults"
    },
    {
        "customerID": "3668-QPYBK", 
        "name": "Michael",
        "tenure": 2,
        "contract": "Month-to-month",
        "monthly_charges": 75.30,
        "internet_service": "DSL",
        "senior_citizen": False,
        "partner": False,
        "churn_probability": 0.78,
        "risk_level": "High", 
        "key_risk_factors": ["Very short tenure", "No partner", "Month-to-month contract"],
        "segment": "young_adults"
    },
    {
        "customerID": "9237-HQITU",
        "name": "Emma",
        "tenure": 1,
        "contract": "Month-to-month", 
        "monthly_charges": 95.20,
        "internet_service": "Fiber optic",
        "senior_citizen": True,
        "partner": False,
        "churn_probability": 0.92,
        "risk_level": "High",
        "key_risk_factors": ["New customer", "Senior citizen", "High charges", "Living alone"],
        "segment": "senior_citizens"
    }
]

# ========================================================================
# FILE PATHS AND OUTPUT CONFIGURATION
# ========================================================================

OUTPUT_CONFIG = {
    "results_file": "generated_emails_results.json",
    "compliance_report": "compliance_report.json",
    "model_performance": "model_performance_log.txt",
    "timestamp_format": "%Y-%m-%d_%H-%M-%S"
}

# ========================================================================
# LOGGING CONFIGURATION - FIXED
# ========================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": "email_generation.log"  # Changed from "file" to "filename"
}