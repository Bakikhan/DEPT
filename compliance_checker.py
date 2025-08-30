"""
Compliance Checker for DEPT Case Assignment - Email Generation
Validates generated emails against Vodafone brand guidelines
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
from config import COMPLIANCE_CHECKS, VODAFONE_BRAND_GUIDELINES


class EmailComplianceChecker:
    """Comprehensive email compliance checker for Vodafone brand guidelines"""

    def __init__(self):
        self.compliance_config = COMPLIANCE_CHECKS
        self.brand_guidelines = VODAFONE_BRAND_GUIDELINES
        self.validation_results = {}

    def analyze_email_compliance(self, email_content: str, customer_name: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive compliance analysis on generated email

        Args:
            email_content (str): The generated email content
            customer_name (str): Customer name for dynamic validation

        Returns:
            Dict containing compliance score, detailed checks, and recommendations
        """

        # Perform all compliance checks
        mandatory_results = self._check_mandatory_elements(email_content, customer_name)
        quality_results = self._check_quality_elements(email_content)
        structure_results = self._check_email_structure(email_content)
        tone_results = self._check_tone_compliance(email_content)

        # Calculate overall score
        overall_score = self._calculate_compliance_score(
            mandatory_results, quality_results, structure_results, tone_results
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            mandatory_results, quality_results, structure_results, tone_results
        )

        # Determine compliance level
        compliance_level = self._determine_compliance_level(overall_score)

        return {
            "overall_score": overall_score,
            "compliance_level": compliance_level,
            "is_compliant": overall_score >= self.compliance_config["scoring"]["minimum_compliance_score"],
            "detailed_checks": {
                "mandatory_elements": mandatory_results,
                "quality_checks": quality_results,
                "structure_validation": structure_results,
                "tone_compliance": tone_results
            },
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "word_count": len(email_content.split()),
            "character_count": len(email_content)
        }

    def _check_mandatory_elements(self, email_content: str, customer_name: str = None) -> Dict[str, Any]:
        """Check all mandatory elements required by brand guidelines"""

        checks = {}

        for check_name, check_config in self.compliance_config["mandatory_elements"].items():
            if check_name == "uses_customer_name" and customer_name:
                # Dynamic check for customer name
                checks[check_name] = {
                    "passed": customer_name in email_content,
                    "weight": check_config["weight"],
                    "description": f"Email uses customer name '{customer_name}'"
                }
            elif "patterns" in check_config:
                # Multiple pattern check
                pattern_found = any(pattern in email_content for pattern in check_config["patterns"])
                checks[check_name] = {
                    "passed": pattern_found,
                    "weight": check_config["weight"],
                    "patterns_checked": check_config["patterns"],
                    "description": f"Email contains required patterns for {check_name.replace('_', ' ')}"
                }
            elif "pattern" in check_config:
                # Single pattern check
                checks[check_name] = {
                    "passed": check_config["pattern"] in email_content,
                    "weight": check_config["weight"],
                    "pattern_checked": check_config["pattern"],
                    "description": f"Email contains required element: {check_config['pattern']}"
                }

        return checks

    def _check_quality_elements(self, email_content: str) -> Dict[str, Any]:
        """Check quality elements like length, tone, sentence structure"""

        checks = {}
        word_count = len(email_content.split())

        # Word count check
        length_config = self.compliance_config["quality_checks"]["appropriate_length"]
        checks["appropriate_length"] = {
            "passed": length_config["min_words"] <= word_count <= length_config["max_words"],
            "weight": length_config["weight"],
            "actual_count": word_count,
            "expected_range": f"{length_config['min_words']}-{length_config['max_words']} words",
            "description": "Email has appropriate word count"
        }

        # Professional tone check
        tone_config = self.compliance_config["quality_checks"]["professional_tone"]
        forbidden_found = [word for word in tone_config["forbidden"] if word.lower() in email_content.lower()]
        checks["professional_tone"] = {
            "passed": len(forbidden_found) == 0,
            "weight": tone_config["weight"],
            "forbidden_words_found": forbidden_found,
            "description": "Email maintains professional tone"
        }

        # Sentence structure check
        sentence_config = self.compliance_config["quality_checks"]["sentence_structure"]
        sentences = re.split(r'[.!?]+', email_content)
        long_sentences = [s for s in sentences if len(s.split()) > sentence_config["max_sentence_length"]]
        checks["sentence_structure"] = {
            "passed": len(long_sentences) <= 2,  # Allow up to 2 longer sentences
            "weight": sentence_config["weight"],
            "long_sentences_count": len(long_sentences),
            "max_allowed_length": sentence_config["max_sentence_length"],
            "description": "Email has good sentence structure"
        }

        return checks

    def _check_email_structure(self, email_content: str) -> Dict[str, Any]:
        """Check if email follows proper structure"""

        structure_checks = {}

        # Check for subject line
        structure_checks["has_subject"] = {
            "passed": email_content.startswith("Subject:") or "Subject:" in email_content[:50],
            "description": "Email has proper subject line"
        }

        # Check for greeting
        greetings = ["Hi ", "Dear ", "Hello "]
        has_greeting = any(greeting in email_content for greeting in greetings)
        structure_checks["has_greeting"] = {
            "passed": has_greeting,
            "description": "Email has personal greeting"
        }

        # Check for call-to-action
        cta_phrases = ["click", "call", "visit", "contact", "explore", "discover"]
        has_cta = any(cta.lower() in email_content.lower() for cta in cta_phrases)
        structure_checks["has_call_to_action"] = {
            "passed": has_cta,
            "description": "Email has clear call-to-action"
        }

        # Check for professional closing
        closings = ["Best regards", "Kind regards", "Thank you", "Sincerely"]
        has_closing = any(closing in email_content for closing in closings)
        structure_checks["has_closing"] = {
            "passed": has_closing,
            "description": "Email has professional closing"
        }

        # Check for signature
        signature_indicators = ["Vodafone", "Customer Care", "Team", "Best regards"]
        has_signature = any(indicator in email_content for indicator in signature_indicators)
        structure_checks["has_signature"] = {
            "passed": has_signature,
            "description": "Email has professional signature"
        }

        return structure_checks

    def _check_tone_compliance(self, email_content: str) -> Dict[str, Any]:
        """Check if email tone matches Vodafone brand guidelines"""

        tone_checks = {}

        # Friendly and approachable check
        friendly_indicators = ["thank you", "appreciate", "valued", "welcome", "pleased"]
        friendly_score = sum(1 for indicator in friendly_indicators if indicator.lower() in email_content.lower())
        tone_checks["friendly_approachable"] = {
            "passed": friendly_score >= 2,
            "score": friendly_score,
            "description": "Email uses friendly and approachable language"
        }

        # Clear and concise check
        sentences = re.split(r'[.!?]+', email_content)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
        tone_checks["clear_concise"] = {
            "passed": avg_sentence_length <= 20,
            "average_sentence_length": avg_sentence_length,
            "description": "Email uses clear and concise language"
        }

        # Positive and reassuring check
        positive_indicators = ["benefit", "exclusive", "special", "improve", "better", "enhance"]
        positive_score = sum(1 for indicator in positive_indicators if indicator.lower() in email_content.lower())
        tone_checks["positive_reassuring"] = {
            "passed": positive_score >= 2,
            "score": positive_score,
            "description": "Email uses positive and reassuring language"
        }

        # Professional and trustworthy check
        professional_indicators = ["service", "commitment", "quality", "reliable", "support"]
        professional_score = sum(1 for indicator in professional_indicators if indicator.lower() in email_content.lower())
        tone_checks["professional_trustworthy"] = {
            "passed": professional_score >= 1,
            "score": professional_score,
            "description": "Email maintains professional and trustworthy tone"
        }

        return tone_checks

    def _calculate_compliance_score(self, mandatory_results: Dict, quality_results: Dict, 
                                   structure_results: Dict, tone_results: Dict) -> float:
        """Calculate overall compliance score based on weighted results"""

        total_score = 0
        total_weight = 0

        # Mandatory elements (highest weight)
        for check_name, check_result in mandatory_results.items():
            weight = check_result.get("weight", 1.0)
            score = 1.0 if check_result["passed"] else 0.0
            total_score += score * weight * 2  # Double weight for mandatory
            total_weight += weight * 2

        # Quality elements
        for check_name, check_result in quality_results.items():
            weight = check_result.get("weight", 1.0)
            score = 1.0 if check_result["passed"] else 0.0
            total_score += score * weight
            total_weight += weight

        # Structure elements
        for check_name, check_result in structure_results.items():
            score = 1.0 if check_result["passed"] else 0.0
            total_score += score
            total_weight += 1

        # Tone elements
        for check_name, check_result in tone_results.items():
            score = 1.0 if check_result["passed"] else 0.0
            total_score += score
            total_weight += 1

        return total_score / total_weight if total_weight > 0 else 0.0

    def _determine_compliance_level(self, score: float) -> str:
        """Determine compliance level based on score"""

        if score >= self.compliance_config["scoring"]["excellent_threshold"]:
            return "EXCELLENT"
        elif score >= self.compliance_config["scoring"]["good_threshold"]:
            return "GOOD"
        elif score >= self.compliance_config["scoring"]["minimum_compliance_score"]:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"

    def _generate_recommendations(self, mandatory_results: Dict, quality_results: Dict,
                                 structure_results: Dict, tone_results: Dict) -> List[str]:
        """Generate specific recommendations for improvement"""

        recommendations = []

        # Check mandatory elements
        for check_name, check_result in mandatory_results.items():
            if not check_result["passed"]:
                recommendations.append(f"Add {check_name.replace('_', ' ')}: {check_result['description']}")

        # Check quality elements
        for check_name, check_result in quality_results.items():
            if not check_result["passed"]:
                if check_name == "appropriate_length":
                    recommendations.append(f"Adjust email length to {check_result['expected_range']}")
                elif check_name == "professional_tone":
                    recommendations.append(f"Remove unprofessional words: {check_result['forbidden_words_found']}")
                elif check_name == "sentence_structure":
                    recommendations.append("Break down long sentences for better readability")

        # Check structure elements
        for check_name, check_result in structure_results.items():
            if not check_result["passed"]:
                recommendations.append(f"Add {check_name.replace('_', ' ')}: {check_result['description']}")

        # Check tone elements
        for check_name, check_result in tone_results.items():
            if not check_result["passed"]:
                recommendations.append(f"Improve {check_name.replace('_', ' ')}: {check_result['description']}")

        return recommendations


class ComplianceReporter:
    """Generate comprehensive compliance reports"""

    def __init__(self, compliance_checker: EmailComplianceChecker):
        self.checker = compliance_checker

    def generate_batch_report(self, email_results: List[Dict]) -> Dict[str, Any]:
        """Generate compliance report for multiple emails"""

        batch_results = []
        total_compliant = 0

        for email_data in email_results:
            email_content = email_data.get("generated_email", "")
            customer_name = email_data.get("customer_name", "")

            compliance_result = self.checker.analyze_email_compliance(email_content, customer_name)

            batch_results.append({
                "customer_id": email_data.get("customer_id", ""),
                "customer_name": customer_name,
                "compliance_result": compliance_result,
                "model_used": email_data.get("model_used", "Unknown")
            })

            if compliance_result["is_compliant"]:
                total_compliant += 1

        # Calculate batch statistics
        scores = [result["compliance_result"]["overall_score"] for result in batch_results]
        avg_score = sum(scores) / len(scores) if scores else 0

        compliance_rate = total_compliant / len(batch_results) if batch_results else 0

        return {
            "summary": {
                "total_emails": len(batch_results),
                "compliant_emails": total_compliant,
                "compliance_rate": compliance_rate,
                "average_score": avg_score,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": batch_results,
            "recommendations": self._generate_batch_recommendations(batch_results)
        }

    def _generate_batch_recommendations(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Generate recommendations for improving overall email quality"""

        # Collect all failed checks
        failed_checks = {}

        for result in batch_results:
            compliance = result["compliance_result"]

            for category, checks in compliance["detailed_checks"].items():
                for check_name, check_result in checks.items():
                    if not check_result["passed"]:
                        if check_name not in failed_checks:
                            failed_checks[check_name] = 0
                        failed_checks[check_name] += 1

        # Sort by frequency
        sorted_issues = sorted(failed_checks.items(), key=lambda x: x[1], reverse=True)

        return {
            "most_common_issues": sorted_issues[:5],
            "improvement_priorities": [
                f"Address {issue[0].replace('_', ' ')}: Failed in {issue[1]} emails"
                for issue in sorted_issues[:3]
            ]
        }

    def export_compliance_report(self, batch_report: Dict, filename: str = None) -> str:
        """Export compliance report to JSON file"""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"compliance_report_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(batch_report, f, indent=2, default=str)

        return filename


# Utility functions for easy access
def check_single_email_compliance(email_content: str, customer_name: str = None) -> Dict[str, Any]:
    """Utility function to check single email compliance"""

    checker = EmailComplianceChecker()
    return checker.analyze_email_compliance(email_content, customer_name)


def generate_compliance_report(email_results: List[Dict]) -> Dict[str, Any]:
    """Utility function to generate batch compliance report"""

    checker = EmailComplianceChecker()
    reporter = ComplianceReporter(checker)
    return reporter.generate_batch_report(email_results)


def validate_email_batch(email_results: List[Dict], export_report: bool = True) -> Tuple[Dict[str, Any], str]:
    """
    Complete validation pipeline for batch of emails

    Returns:
        Tuple of (compliance_report, report_filename)
    """

    checker = EmailComplianceChecker()
    reporter = ComplianceReporter(checker)

    batch_report = reporter.generate_batch_report(email_results)

    report_filename = None
    if export_report:
        report_filename = reporter.export_compliance_report(batch_report)

    return batch_report, report_filename


# Advanced compliance features
class AdvancedComplianceAnalyzer:
    """Advanced compliance analysis with ML-based insights"""

    def __init__(self):
        self.base_checker = EmailComplianceChecker()

    def analyze_email_sentiment(self, email_content: str) -> Dict[str, Any]:
        """Analyze email sentiment and emotional tone"""

        # Simple sentiment analysis based on word lists
        positive_words = ["thank", "appreciate", "value", "exclusive", "special", "benefit", "improve"]
        negative_words = ["unfortunately", "problem", "issue", "concern", "difficult", "sorry"]
        urgent_words = ["immediately", "urgent", "expires", "limited", "act now", "hurry"]

        words = email_content.lower().split()

        positive_count = sum(1 for word in positive_words if word in words)
        negative_count = sum(1 for word in negative_words if word in words)
        urgent_count = sum(1 for word in urgent_words if word in words)

        total_words = len(words)

        return {
            "positive_sentiment_score": positive_count / total_words if total_words > 0 else 0,
            "negative_sentiment_score": negative_count / total_words if total_words > 0 else 0,
            "urgency_score": urgent_count / total_words if total_words > 0 else 0,
            "overall_sentiment": "positive" if positive_count > negative_count else "neutral" if positive_count == negative_count else "negative",
            "sentiment_balance": "good" if positive_count > negative_count and urgent_count <= 2 else "needs_adjustment"
        }

    def check_accessibility_compliance(self, email_content: str) -> Dict[str, Any]:
        """Check email accessibility compliance"""

        accessibility_checks = {
            "clear_headings": any(indicator in email_content for indicator in ["Subject:", "Hi ", "Dear "]),
            "bullet_points_used": any(bullet in email_content for bullet in ["•", "- ", "* "]),
            "short_paragraphs": self._check_paragraph_length(email_content),
            "action_words_clear": any(action in email_content.lower() for action in ["click", "call", "visit", "contact"]),
            "no_all_caps": not re.search(r'\b[A-Z]{4,}\b', email_content)
        }

        accessibility_score = sum(accessibility_checks.values()) / len(accessibility_checks)

        return {
            "accessibility_score": accessibility_score,
            "accessibility_checks": accessibility_checks,
            "is_accessible": accessibility_score >= 0.8
        }

    def _check_paragraph_length(self, email_content: str) -> bool:
        """Check if paragraphs are appropriately sized"""

        paragraphs = [p.strip() for p in email_content.split('\n\n') if p.strip()]
        long_paragraphs = [p for p in paragraphs if len(p.split()) > 50]

        return len(long_paragraphs) <= 1  # Allow max 1 long paragraph