"""
Comprehensive Transformers Model Registry
Business-focused model configurations for real-world applications
"""
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class BusinessDomain(str, Enum):
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    LEGAL = "legal"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    CUSTOMER_SERVICE = "customer_service"
    MARKETING = "marketing"
    HR = "hr"
    CYBERSECURITY = "cybersecurity"
    EDUCATION = "education"

class ModelCategory(str, Enum):
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENTITY_RECOGNITION = "entity_recognition"
    TEXT_CLASSIFICATION = "text_classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    QUESTION_ANSWERING = "question_answering"
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    CONTENT_MODERATION = "content_moderation"

class TransformersRegistry:
    """
    Comprehensive registry of Transformers models organized by business use cases
    """
    
    def __init__(self):
        self.model_registry = self._initialize_registry()
        self.business_use_cases = self._initialize_business_use_cases()
        
        logger.info(f"TransformersRegistry initialized with {len(self.model_registry)} models")
    
    def _initialize_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive model registry"""
        return {
            # === SENTIMENT ANALYSIS ===
            "sentiment-distilbert": {
                "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
                "model_type": "classification",
                "task": "text-classification",
                "description": "Fast sentiment analysis for customer feedback",
                "domains": [BusinessDomain.CUSTOMER_SERVICE, BusinessDomain.MARKETING, BusinessDomain.RETAIL],
                "max_length": 512,
                "performance": {"speed": "fast", "accuracy": "high"},
                "business_impact": "High - Customer experience optimization"
            },
            
            "sentiment-roberta": {
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment",
                "model_type": "classification", 
                "task": "text-classification",
                "description": "Robust sentiment analysis for social media and reviews",
                "domains": [BusinessDomain.MARKETING, BusinessDomain.CUSTOMER_SERVICE],
                "max_length": 512,
                "performance": {"speed": "medium", "accuracy": "very_high"},
                "business_impact": "High - Brand monitoring and reputation management"
            },
            
            # === ENTITY RECOGNITION ===
            "ner-bert": {
                "model_name": "dslim/bert-base-NER",
                "model_type": "token_classification",
                "task": "token-classification",
                "description": "Named entity recognition for person, organization, location extraction",
                "domains": [BusinessDomain.LEGAL, BusinessDomain.FINANCE, BusinessDomain.HEALTHCARE],
                "max_length": 512,
                "performance": {"speed": "medium", "accuracy": "high"},
                "business_impact": "Critical - Document processing and compliance"
            },
            
            "ner-financial": {
                "model_name": "yiyanghkust/finbert-tone",
                "model_type": "classification",
                "task": "text-classification", 
                "description": "Financial sentiment and entity analysis",
                "domains": [BusinessDomain.FINANCE],
                "max_length": 512,
                "performance": {"speed": "medium", "accuracy": "high"},
                "business_impact": "High - Financial analysis and risk assessment"
            },
            
            # === TEXT GENERATION ===
            "chat-dialogpt": {
                "model_name": "microsoft/DialoGPT-medium",
                "model_type": "causal_lm",
                "task": "text-generation",
                "description": "Conversational AI for customer service and support",
                "domains": [BusinessDomain.CUSTOMER_SERVICE, BusinessDomain.EDUCATION],
                "max_length": 1024,
                "performance": {"speed": "fast", "accuracy": "medium"},
                "business_impact": "High - 24/7 customer support automation"
            },
            
            "code-codellama": {
                "model_name": "codellama/CodeLlama-7b-hf",
                "model_type": "causal_lm",
                "task": "text-generation",
                "description": "Code generation and programming assistance",
                "domains": [BusinessDomain.MANUFACTURING, BusinessDomain.FINANCE, BusinessDomain.CYBERSECURITY],
                "max_length": 2048,
                "performance": {"speed": "medium", "accuracy": "high"},
                "business_impact": "Critical - Developer productivity and automation"
            },
            
            # === SUMMARIZATION ===
            "summarization-bart": {
                "model_name": "facebook/bart-large-cnn",
                "model_type": "seq2seq",
                "task": "summarization",
                "description": "Text summarization for reports and documents",
                "domains": [BusinessDomain.LEGAL, BusinessDomain.FINANCE, BusinessDomain.HEALTHCARE],
                "max_length": 1024,
                "performance": {"speed": "medium", "accuracy": "high"},
                "business_impact": "High - Document processing efficiency"
            },
            
            "summarization-t5": {
                "model_name": "t5-small",
                "model_type": "seq2seq",
                "task": "summarization",
                "description": "Fast text summarization for real-time applications",
                "domains": [BusinessDomain.CUSTOMER_SERVICE, BusinessDomain.MARKETING],
                "max_length": 512,
                "performance": {"speed": "fast", "accuracy": "medium"},
                "business_impact": "Medium - Quick insights from text data"
            },
            
            # === TRANSLATION ===
            "translation-en-fr": {
                "model_name": "Helsinki-NLP/opus-mt-en-fr",
                "model_type": "seq2seq",
                "task": "translation",
                "description": "English to French translation for global business",
                "domains": [BusinessDomain.CUSTOMER_SERVICE, BusinessDomain.LEGAL, BusinessDomain.MARKETING],
                "max_length": 512,
                "performance": {"speed": "fast", "accuracy": "high"},
                "business_impact": "High - International business expansion"
            },
            
            "translation-multilingual": {
                "model_name": "facebook/m2m100_418M",
                "model_type": "seq2seq", 
                "task": "translation",
                "description": "Multilingual translation for 100+ languages",
                "domains": [BusinessDomain.CUSTOMER_SERVICE, BusinessDomain.EDUCATION],
                "max_length": 512,
                "performance": {"speed": "medium", "accuracy": "medium"},
                "business_impact": "Critical - Global customer support"
            },
            
            # === QUESTION ANSWERING ===
            "qa-bert": {
                "model_name": "deepset/roberta-base-squad2",
                "model_type": "question_answering",
                "task": "question-answering",
                "description": "Question answering for knowledge bases and FAQs",
                "domains": [BusinessDomain.CUSTOMER_SERVICE, BusinessDomain.EDUCATION, BusinessDomain.HEALTHCARE],
                "max_length": 512,
                "performance": {"speed": "medium", "accuracy": "high"},
                "business_impact": "High - Customer self-service and support"
            },
            
            # === CONTENT MODERATION ===
            "moderation-toxicity": {
                "model_name": "unitary/toxic-bert",
                "model_type": "classification",
                "task": "text-classification",
                "description": "Content moderation and toxicity detection",
                "domains": [BusinessDomain.CUSTOMER_SERVICE, BusinessDomain.EDUCATION],
                "max_length": 512,
                "performance": {"speed": "fast", "accuracy": "high"},
                "business_impact": "Critical - Platform safety and compliance"
            },
            
            # === LEGAL & COMPLIANCE ===
            "legal-bert": {
                "model_name": "nlpaueb/legal-bert-base-uncased",
                "model_type": "classification",
                "task": "text-classification",
                "description": "Legal document analysis and compliance checking",
                "domains": [BusinessDomain.LEGAL, BusinessDomain.FINANCE],
                "max_length": 512,
                "performance": {"speed": "medium", "accuracy": "high"},
                "business_impact": "Critical - Regulatory compliance and risk management"
            },
            
            # === HEALTHCARE ===
            "healthcare-ner": {
                "model_name": "samrawal/bert-base-uncased_clinical-ner",
                "model_type": "token_classification", 
                "task": "token-classification",
                "description": "Medical entity recognition for healthcare documents",
                "domains": [BusinessDomain.HEALTHCARE],
                "max_length": 512,
                "performance": {"speed": "medium", "accuracy": "high"},
                "business_impact": "Critical - Patient care and medical records"
            }
        }
    
    def _initialize_business_use_cases(self) -> Dict[BusinessDomain, List[Dict[str, Any]]]:
        """Initialize business use cases for each domain"""
        return {
            BusinessDomain.HEALTHCARE: [
                {
                    "name": "Medical Record Analysis",
                    "description": "Extract insights from patient records and medical documents",
                    "models": ["healthcare-ner", "summarization-bart", "qa-bert"],
                    "business_value": "Improve patient care and reduce administrative overhead",
                    "roi_impact": "High - Reduces manual review time by 70%"
                },
                {
                    "name": "Clinical Trial Matching", 
                    "description": "Match patients with appropriate clinical trials",
                    "models": ["ner-bert", "text-classification"],
                    "business_value": "Accelerate clinical research and patient enrollment",
                    "roi_impact": "Critical - Reduces trial recruitment time by 50%"
                }
            ],
            
            BusinessDomain.FINANCE: [
                {
                    "name": "Financial Sentiment Analysis",
                    "description": "Analyze market sentiment from news and social media",
                    "models": ["sentiment-roberta", "ner-financial"],
                    "business_value": "Real-time market intelligence and risk assessment",
                    "roi_impact": "High - Improves trading decisions by 25%"
                },
                {
                    "name": "Compliance Monitoring",
                    "description": "Monitor communications for regulatory compliance",
                    "models": ["legal-bert", "moderation-toxicity", "ner-bert"],
                    "business_value": "Automate compliance and reduce regulatory risks",
                    "roi_impact": "Critical - Prevents million-dollar fines"
                }
            ],
            
            BusinessDomain.LEGAL: [
                {
                    "name": "Contract Analysis",
                    "description": "Extract key clauses and obligations from legal documents",
                    "models": ["legal-bert", "ner-bert", "summarization-bart"],
                    "business_value": "Reduce legal review time and improve accuracy",
                    "roi_impact": "High - Cuts contract review time by 80%"
                },
                {
                    "name": "Legal Research Assistant",
                    "description": "Answer legal questions and find relevant case law",
                    "models": ["qa-bert", "text-generation"],
                    "business_value": "Accelerate legal research and case preparation",
                    "roi_impact": "Medium - Saves 10+ hours per week per lawyer"
                }
            ],
            
            BusinessDomain.CUSTOMER_SERVICE: [
                {
                    "name": "Intelligent Chatbots",
                    "description": "24/7 customer support with contextual understanding",
                    "models": ["chat-dialogpt", "sentiment-distilbert", "qa-bert"],
                    "business_value": "Reduce support costs and improve customer satisfaction",
                    "roi_impact": "High - Handles 60% of queries without human intervention"
                },
                {
                    "name": "Customer Feedback Analysis",
                    "description": "Analyze support tickets and feedback for insights",
                    "models": ["sentiment-roberta", "ner-bert", "summarization-bart"],
                    "business_value": "Identify product issues and customer needs",
                    "roi_impact": "Medium - Improves product development decisions"
                }
            ],
            
            BusinessDomain.MARKETING: [
                {
                    "name": "Brand Monitoring",
                    "description": "Track brand mentions and sentiment across channels",
                    "models": ["sentiment-roberta", "ner-bert"],
                    "business_value": "Real-time brand perception and crisis management",
                    "roi_impact": "High - Enables proactive reputation management"
                },
                {
                    "name": "Content Generation",
                    "description": "Create marketing copy and social media content",
                    "models": ["text-generation", "summarization-bart"],
                    "business_value": "Scale content creation and personalization",
                    "roi_impact": "Medium - Reduces content creation time by 50%"
                }
            ],
            
            BusinessDomain.RETAIL: [
                {
                    "name": "Customer Review Analysis",
                    "description": "Analyze product reviews for insights and trends",
                    "models": ["sentiment-distilbert", "ner-bert"],
                    "business_value": "Improve product development and customer satisfaction",
                    "roi_impact": "High - Identifies product issues 3x faster"
                }
            ],
            
            BusinessDomain.MANUFACTURING: [
                {
                    "name": "Quality Control Automation",
                    "description": "Automated quality inspection and defect detection",
                    "models": ["code-codellama", "text-classification"],
                    "business_value": "Reduce defect rates and improve efficiency",
                    "roi_impact": "Critical - Reduces quality issues by 60%"
                }
            ],
            
            BusinessDomain.HR: [
                {
                    "name": "Resume Screening",
                    "description": "Automated candidate screening and matching",
                    "models": ["ner-bert", "text-classification"],
                    "business_value": "Accelerate hiring and improve candidate quality",
                    "roi_impact": "High - Reduces screening time by 80%"
                }
            ],
            
            BusinessDomain.CYBERSECURITY: [
                {
                    "name": "Threat Detection",
                    "description": "Identify security threats from logs and communications",
                    "models": ["moderation-toxicity", "text-classification"],
                    "business_value": "Proactive threat detection and response",
                    "roi_impact": "Critical - Prevents security breaches"
                }
            ],
            
            BusinessDomain.EDUCATION: [
                {
                    "name": "Intelligent Tutoring",
                    "description": "Personalized learning assistance and Q&A",
                    "models": ["qa-bert", "chat-dialogpt"],
                    "business_value": "Improve learning outcomes and student engagement",
                    "roi_impact": "Medium - Increases student success by 30%"
                }
            ]
        }
    
    def get_models_by_domain(self, domain: BusinessDomain) -> List[Dict[str, Any]]:
        """Get all models relevant to a business domain"""
        return [model for model in self.model_registry.values() if domain in model.get("domains", [])]
    
    def get_models_by_category(self, category: ModelCategory) -> List[Dict[str, Any]]:
        """Get all models of a specific category"""
        return [model for model in self.model_registry.values() if model.get("task") == category.value]
    
    def get_business_use_cases(self, domain: BusinessDomain) -> List[Dict[str, Any]]:
        """Get business use cases for a domain"""
        return self.business_use_cases.get(domain, [])
    
    def get_model_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        return self.model_registry.get(model_id)
    
    def get_all_domains(self) -> List[BusinessDomain]:
        """Get all available business domains"""
        return list(self.business_use_cases.keys())
    
    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """Search models by name, description, or domain"""
        query = query.lower()
        results = []
        
        for model_id, model_info in self.model_registry.items():
            searchable_text = f"{model_id} {model_info.get('description', '')} {''.join([str(d) for d in model_info.get('domains', [])])}".lower()
            if query in searchable_text:
                results.append({"model_id": model_id, **model_info})
        
        return results

# Global registry instance
transformers_registry = TransformersRegistry()
