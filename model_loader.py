"""
Model Loader for DEPT Case Assignment - Email Generation
Handles LLM model loading, caching, and inference
"""

import os
import torch
import logging
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import warnings

from config import (
    HF_TOKEN, MODEL_NAME, MODEL_DIR, LOCAL_MODEL_PATH, 
    GENERATION_CONFIG, DEVICE_CONFIG, LOGGING_CONFIG
)

# Setup logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ModelManager:
    """Manages LLM model loading, caching, and inference"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.text_generator = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        logger.info(f"ModelManager initialized. Device: {self.device}")
    
    def authenticate_huggingface(self) -> bool:
        """Authenticate with Hugging Face"""
        
        try:
            login(token=HF_TOKEN)
            logger.info("Successfully authenticated with Hugging Face")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with Hugging Face: {e}")
            return False
    
    def load_or_cache_model(self, force_reload: bool = False) -> bool:
        """
        Load model from cache or download and cache it
        
        Args:
            force_reload: Force reload even if model is already loaded
            
        Returns:
            bool: Success status
        """
        
        if self.model_loaded and not force_reload:
            logger.info("Model already loaded")
            return True
        
        try:
            # Authenticate first
            if not self.authenticate_huggingface():
                return False
            
            # Check if model exists locally
            if self._model_exists_locally() and not force_reload:
                logger.info("Loading model from local cache...")
                success = self._load_from_cache()
            else:
                logger.info("Downloading and caching model (this may take several minutes)...")
                success = self._download_and_cache_model()
            
            if success:
                self._create_pipeline()
                self.model_loaded = True
                logger.info("Model loaded and pipeline created successfully")
                return True
            else:
                logger.error("Failed to load model")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _model_exists_locally(self) -> bool:
        """Check if model exists in local cache"""
        
        return (os.path.exists(LOCAL_MODEL_PATH) and 
                len(os.listdir(LOCAL_MODEL_PATH)) > 0 and
                os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")))
    
    def _load_from_cache(self) -> bool:
        """Load model from local cache"""
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
            
            # Configure torch dtype
            torch_dtype = getattr(torch, DEVICE_CONFIG["torch_dtype"])
            
            self.model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                torch_dtype=torch_dtype,
                device_map=DEVICE_CONFIG["device_map"] if self.device == "cuda" else None,
                low_cpu_mem_usage=DEVICE_CONFIG["low_cpu_mem_usage"]
            )
            
            logger.info("Model loaded from cache successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return False
    
    def _download_and_cache_model(self) -> bool:
        """Download model and save to cache"""
        
        try:
            # Download tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                token=HF_TOKEN,
                cache_dir=MODEL_DIR
            )
            
            # Configure torch dtype
            torch_dtype = getattr(torch, DEVICE_CONFIG["torch_dtype"])
            
            # Download model
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                token=HF_TOKEN,
                torch_dtype=torch_dtype,
                device_map=DEVICE_CONFIG["device_map"] if self.device == "cuda" else None,
                low_cpu_mem_usage=DEVICE_CONFIG["low_cpu_mem_usage"],
                cache_dir=MODEL_DIR
            )
            
            # Save to local directory for faster future loading
            logger.info("Saving model to local cache...")
            self.tokenizer.save_pretrained(LOCAL_MODEL_PATH)
            self.model.save_pretrained(LOCAL_MODEL_PATH)
            
            logger.info("Model downloaded and cached successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download and cache model: {e}")
            return False
    
    def _create_pipeline(self):
        """Create text generation pipeline"""
        
        try:
            self.text_generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=getattr(torch, DEVICE_CONFIG["torch_dtype"]),
                device_map=DEVICE_CONFIG["device_map"] if self.device == "cuda" else None,
                pad_token_id=self.tokenizer.eos_token_id,
                **GENERATION_CONFIG
            )
            
            logger.info("Text generation pipeline created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise e
    
    def generate_text(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate text using the loaded model
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text or None if failed
        """
        
        if not self.model_loaded:
            logger.error("Model not loaded. Call load_or_cache_model() first.")
            return None
        
        try:
            # Merge generation config with any additional parameters
            generation_params = {**GENERATION_CONFIG, **kwargs}
            
            # Generate text
            outputs = self.text_generator(
                prompt,
                **generation_params
            )
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text'].strip()
                logger.debug(f"Successfully generated {len(generated_text)} characters")
                return generated_text
            else:
                logger.warning("No text generated")
                return None
                
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        
        return {
            "model_name": MODEL_NAME,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "cache_path": LOCAL_MODEL_PATH,
            "cache_exists": self._model_exists_locally(),
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else None,
            "generation_config": GENERATION_CONFIG
        }
    
    def cleanup(self):
        """Clean up model resources"""
        
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            if self.text_generator is not None:
                del self.text_generator
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model_loaded = False
            logger.info("Model resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class ModelFallback:
    """Fallback system when LLM model fails"""
    
    def __init__(self):
        self.fallback_active = False
        logger.info("ModelFallback initialized")
    
    def is_api_available(self) -> bool:
        """Check if Hugging Face API is available as fallback"""
        
        try:
            import requests
            
            api_url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            # Test API availability
            response = requests.get(api_url, headers=headers, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"API availability check failed: {e}")
            return False
    
    def generate_via_api(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using Hugging Face API as fallback"""
        
        try:
            import requests
            
            api_url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get("max_new_tokens", 250),
                    "temperature": kwargs.get("temperature", 0.7),
                    "return_full_text": False
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').strip()
                    logger.info("Successfully generated text via API fallback")
                    self.fallback_active = True
                    return generated_text
            
            logger.error(f"API request failed: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            return None


class EmailGenerationEngine:
    """Main engine that combines model management with email generation"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.fallback = ModelFallback()
        self.is_initialized = False
        
        logger.info("EmailGenerationEngine initialized")
    
    def initialize(self, force_reload: bool = False) -> bool:
        """Initialize the email generation engine"""
        
        logger.info("Initializing Email Generation Engine...")
        
        # Try to load local model first
        if self.model_manager.load_or_cache_model(force_reload):
            self.is_initialized = True
            logger.info("Engine initialized with local model")
            return True
        
        # Check API fallback
        elif self.fallback.is_api_available():
            self.is_initialized = True
            logger.info("Engine initialized with API fallback")
            return True
        
        else:
            logger.error("Failed to initialize engine - no model or API available")
            return False
    
    def generate_email(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate email using available method (local model or API)"""
        
        if not self.is_initialized:
            logger.error("Engine not initialized")
            return None
        
        # Try local model first
        if self.model_manager.model_loaded:
            result = self.model_manager.generate_text(prompt, **kwargs)
            if result:
                return result
            
            logger.warning("Local model generation failed, trying API fallback...")
        
        # Try API fallback
        result = self.fallback.generate_via_api(prompt, **kwargs)
        if result:
            return result
        
        logger.error("All generation methods failed")
        return None
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of the generation engine"""
        
        return {
            "is_initialized": self.is_initialized,
            "model_info": self.model_manager.get_model_info(),
            "fallback_active": self.fallback.fallback_active,
            "api_available": self.fallback.is_api_available()
        }
    
    def cleanup(self):
        """Clean up all resources"""
        
        self.model_manager.cleanup()
        self.is_initialized = False
        logger.info("EmailGenerationEngine cleaned up")


# Utility functions for easy import
def create_email_generator() -> EmailGenerationEngine:
    """Create and initialize email generation engine"""
    
    engine = EmailGenerationEngine()
    engine.initialize()
    return engine


def get_model_status() -> Dict[str, Any]:
    """Get current model status without initializing"""
    
    manager = ModelManager()
    return manager.get_model_info()


# Testing utilities
class ModelTester:
    """Test model functionality"""
    
    def __init__(self, engine: EmailGenerationEngine):
        self.engine = engine
    
    def test_basic_generation(self) -> bool:
        """Test basic text generation capability"""
        
        test_prompt = "Hello, this is a test prompt for"
        result = self.engine.generate_email(test_prompt, max_new_tokens=20)
        
        if result and len(result) > len(test_prompt):
            logger.info("Basic generation test passed")
            return True
        else:
            logger.error("Basic generation test failed")
            return False
    
    def test_email_prompt(self) -> bool:
        """Test with a simple email prompt"""
        
        test_prompt = """Subject: Test Email

Hi John,

This is a test email to verify"""
        
        result = self.engine.generate_email(test_prompt, max_new_tokens=50)
        
        if result and "email" in result.lower():
            logger.info("Email prompt test passed")
            return True
        else:
            logger.error("Email prompt test failed")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all available tests"""
        
        return {
            "basic_generation": self.test_basic_generation(),
            "email_prompt": self.test_email_prompt(),
            "engine_status": self.engine.is_initialized
        }