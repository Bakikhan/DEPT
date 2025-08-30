"""
Prompt Templates for DEPT Case Assignment - Email Generation
Contains all prompt engineering logic and template generation functions
"""

from config import VODAFONE_BRAND_GUIDELINES, EMAIL_TEMPLATES, CUSTOMER_SEGMENTS


class PromptTemplateManager:
    """Manages prompt templates for different customer segments and risk levels"""
    
    def __init__(self):
        self.brand_guidelines = VODAFONE_BRAND_GUIDELINES
        self.email_templates = EMAIL_TEMPLATES
        self.customer_segments = CUSTOMER_SEGMENTS
    
    def create_system_prompt(self):
        """Create the system prompt with brand guidelines"""
        
        system_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a professional customer retention specialist at Vodafone. Your task is to create personalized retention emails that STRICTLY follow Vodafone's brand guidelines.

CRITICAL BRAND GUIDELINES - MUST FOLLOW EXACTLY:

TONE REQUIREMENTS:
1. Friendly & Approachable: {self.brand_guidelines['tone_attributes']['friendly_approachable']}
2. Clear & Concise: {self.brand_guidelines['tone_attributes']['clear_concise']}
3. Positive & Reassuring: {self.brand_guidelines['tone_attributes']['positive_reassuring']}
4. Professional & Trustworthy: {self.brand_guidelines['tone_attributes']['professional_trustworthy']}

MANDATORY EMAIL STRUCTURE:
1. Subject Line: {self.brand_guidelines['email_structure']['subject_line']}
2. Greeting: {self.brand_guidelines['email_structure']['greeting']}
3. Introduction: {self.brand_guidelines['email_structure']['introduction']}  
4. Body: {self.brand_guidelines['email_structure']['body']}
5. Call to Action: {self.brand_guidelines['email_structure']['call_to_action']}
6. Closing: {self.brand_guidelines['email_structure']['closing']}
7. Signature: {self.brand_guidelines['email_structure']['signature']}

PERSONALIZATION REQUIREMENTS:
- Use customer's name throughout
- Reference their tenure with Vodafone
- Mention their current services
- Show appreciation for their loyalty
- Tailor offers based on risk level

CONTENT CONSTRAINTS:
- Maximum {self.brand_guidelines['content_requirements']['max_word_count']} words for email body
- Must include bullet points for offers
- Avoid words: {', '.join(self.brand_guidelines['content_requirements']['forbidden_words'])}
- Must include: {', '.join(self.brand_guidelines['content_requirements']['mandatory_phrases'])}

<|eot_id|>"""
        
        return system_prompt
    
    def create_user_prompt(self, customer_profile, campaign_type="retention"):
        """Create user prompt based on customer profile and campaign type"""
        
        # Determine risk template
        risk_template = self._get_risk_template(customer_profile['churn_probability'])
        
        # Get segment-specific adjustments
        segment_config = self.customer_segments.get(
            customer_profile.get('segment', 'young_adults'), 
            self.customer_segments['young_adults']
        )
        
        user_prompt = f"""<|start_header_id|>user<|end_header_id|>

Create a customer retention email for this HIGH-RISK customer:

CUSTOMER PROFILE:
- Name: {customer_profile['name']}
- Tenure: {customer_profile['tenure']} months with Vodafone
- Contract: {customer_profile['contract']}
- Monthly Charges: £{customer_profile['monthly_charges']:.2f}
- Internet Service: {customer_profile['internet_service']}
- Senior Citizen: {'Yes' if customer_profile.get('senior_citizen', False) else 'No'}
- Churn Risk: {customer_profile['churn_probability']:.0%} (VERY HIGH)
- Risk Factors: {', '.join(customer_profile.get('key_risk_factors', []))}
- Customer Segment: {customer_profile.get('segment', 'young_adults').replace('_', ' ').title()}

CAMPAIGN REQUIREMENTS:
- Risk Level: {risk_template['urgency_level'].upper()} PRIORITY
- Target Offers: {risk_template['discount_percentage']}% discount tier
- Offer Validity: {risk_template['offer_duration']}
- Tone Adjustments: {', '.join(segment_config['tone_adjustments'])}

SPECIFIC OFFERS TO INCLUDE:
{self._format_benefits_as_bullets(risk_template['benefits'])}

Generate the COMPLETE email following the exact structure above. Include subject line and full email body.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Subject: """
        
        return user_prompt
    
    def create_complete_prompt(self, customer_profile, campaign_type="retention"):
        """Create complete prompt by combining system and user prompts"""
        
        system_prompt = self.create_system_prompt()
        user_prompt = self.create_user_prompt(customer_profile, campaign_type)
        
        return system_prompt + user_prompt
    
    def _get_risk_template(self, churn_probability):
        """Get the appropriate risk template based on churn probability"""
        
        for template_name, template_config in self.email_templates.items():
            if churn_probability >= template_config['threshold']:
                return template_config
        
        # Default to medium risk if no match
        return self.email_templates['medium_risk']
    
    def _format_benefits_as_bullets(self, benefits):
        """Format benefits list as bullet points"""
        
        return '\n'.join([f"• {benefit}" for benefit in benefits])
    
    def generate_template_email(self, customer_profile):
        """Generate high-quality template email as fallback"""
        
        risk = customer_profile['churn_probability']
        name = customer_profile['name']
        tenure = customer_profile['tenure']
        risk_template = self._get_risk_template(risk)
        
        # Get customer segment for personalization
        segment = customer_profile.get('segment', 'young_adults')
        segment_config = self.customer_segments.get(segment, self.customer_segments['young_adults'])
        
        # Create subject line
        subject = risk_template['subject_template'].format(name=name, tenure=tenure)
        
        # Create email body based on risk level
        if risk >= 0.9:
            email_body = self._create_ultra_high_risk_email(name, tenure, risk_template, segment_config)
        elif risk >= 0.7:
            email_body = self._create_high_risk_email(name, tenure, risk_template, segment_config)
        else:
            email_body = self._create_medium_risk_email(name, tenure, risk_template, segment_config)
        
        # Combine subject and body
        complete_email = f"Subject: {subject}\n\n{email_body}"
        
        return complete_email
    
    def _create_ultra_high_risk_email(self, name, tenure, risk_template, segment_config):
        """Create ultra high-risk email template"""
        
        return f"""Hi {name},

After {tenure} months with Vodafone, you're one of our most valued customers. We want to ensure you're getting the best possible service.

We've prepared exclusive benefits just for you:
{self._format_benefits_as_bullets(risk_template['benefits'])}

These exclusive offers expire in {risk_template['offer_duration']} and are available only to select customers like you.

Call us now at 191 or click here to claim your benefits immediately.

Thank you for your continued trust in Vodafone. We're here to serve you better.

Best regards,
Sarah Mitchell, Senior Customer Care Specialist
Vodafone Customer Retention Team"""
    
    def _create_high_risk_email(self, name, tenure, risk_template, segment_config):
        """Create high-risk email template"""
        
        return f"""Hi {name},

Your {tenure} months with Vodafone mean so much to us, and we want to show our appreciation.

As a loyal customer, you've earned these exclusive benefits:
{self._format_benefits_as_bullets(risk_template['benefits'])}

We're committed to providing you with exceptional service and value.

Visit your account online or call 191 to explore these personalized offers.

Thank you for choosing Vodafone. Your loyalty drives everything we do.

Best regards,
Michael Chen, Customer Care Manager
Vodafone Customer Experience Team"""
    
    def _create_medium_risk_email(self, name, tenure, risk_template, segment_config):
        """Create medium-risk email template"""
        
        return f"""Hi {name},

It's been wonderful having you as part of the Vodafone family for {tenure} months!

To celebrate, we've arranged some special perks:
{self._format_benefits_as_bullets(risk_template['benefits'])}

We're always working to enhance your experience with us.

Check your personalized offers in the My Vodafone app or visit any store.

Thank you for being such a valued customer. Here's to many more months together!

Best regards,
Emma Thompson, Customer Success Manager  
Vodafone Customer Care Team"""


class AdvancedPromptTemplates:
    """Advanced prompt templates for different use cases"""
    
    @staticmethod
    def create_a_b_testing_prompts(customer_profile, variant="A"):
        """Create different prompt variants for A/B testing"""
        
        base_template = PromptTemplateManager()
        
        if variant == "A":
            # More formal approach
            return base_template.create_complete_prompt(customer_profile)
        
        elif variant == "B":
            # More casual, benefit-focused approach
            system_prompt = f"""You are a friendly Vodafone customer success specialist. Create warm, benefit-focused retention emails that feel personal and caring. Focus on showing genuine appreciation and providing immediate value to the customer."""
            
            user_prompt = f"""Create a warm, personal email for {customer_profile['name']} who has been with us for {customer_profile['tenure']} months and is at {customer_profile['churn_probability']:.0%} risk of leaving.

Make it feel like a personal message from a friend who works at Vodafone and truly cares about their experience.

Include specific offers but focus more on the relationship and value we provide."""
            
            return f"{system_prompt}\n\n{user_prompt}"
    
    @staticmethod
    def create_seasonal_campaign_prompt(customer_profile, season="default"):
        """Create seasonal campaign prompts"""
        
        seasonal_themes = {
            "christmas": "holiday spirit, family time, giving back",
            "summer": "vacation connectivity, outdoor adventures, travel plans",
            "back_to_school": "student discounts, family plans, educational content",
            "new_year": "fresh starts, new goals, improved service"
        }
        
        theme = seasonal_themes.get(season, "appreciation and loyalty")
        
        base_template = PromptTemplateManager()
        base_prompt = base_template.create_complete_prompt(customer_profile)
        
        # Add seasonal context
        seasonal_addition = f"\n\nSEASONAL CONTEXT: Incorporate themes of {theme} into the email while maintaining professional Vodafone tone."
        
        return base_prompt + seasonal_addition
    
    @staticmethod
    def create_journey_stage_prompt(customer_profile, journey_stage):
        """Create prompts based on customer journey stage"""
        
        journey_contexts = {
            "onboarding": "Welcome them warmly, help them get the most from their service",
            "activation": "Encourage them to explore features, show value of their plan", 
            "retention": "Show appreciation, provide exclusive benefits, prevent churn",
            "win_back": "Acknowledge their departure, offer compelling reasons to return"
        }
        
        context = journey_contexts.get(journey_stage, journey_contexts["retention"])
        
        base_template = PromptTemplateManager()
        base_prompt = base_template.create_complete_prompt(customer_profile)
        
        journey_addition = f"\n\nJOURNEY STAGE: {journey_stage.upper()} - {context}"
        
        return base_prompt + journey_addition


# Utility functions for prompt management
def get_prompt_for_customer(customer_profile, campaign_type="retention", variant="standard"):
    """Main function to get appropriate prompt for any customer"""
    
    prompt_manager = PromptTemplateManager()
    
    if variant == "standard":
        return prompt_manager.create_complete_prompt(customer_profile, campaign_type)
    elif variant == "template":
        return prompt_manager.generate_template_email(customer_profile)
    else:
        # Use advanced templates for other variants
        advanced = AdvancedPromptTemplates()
        return advanced.create_a_b_testing_prompts(customer_profile, variant)


def validate_prompt_requirements(prompt_text):
    """Validate that prompt contains all required elements"""
    
    required_elements = [
        "brand guidelines",
        "customer profile", 
        "personalization",
        "structure",
        "tone requirements"
    ]
    
    validation_results = {}
    
    for element in required_elements:
        validation_results[element] = element.lower() in prompt_text.lower()
    
    return validation_results