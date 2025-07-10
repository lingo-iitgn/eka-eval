# eka_eval/core/api_loader.py

import os
import logging
import time
import json
from typing import Optional, Any, Dict, List, Union
from abc import ABC, abstractmethod
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standard response format for API calls"""
    generated_text: str
    model_name: str
    tokens_used: int = 0
    finish_reason: str = "completed"
    error: Optional[str] = None

class BaseAPIClient(ABC):
    """Base class for all API clients"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.request_count = 0
        self.total_tokens = 0
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> APIResponse:
        """Generate text from the API"""
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[APIResponse]:
        """Generate text for multiple prompts"""
        pass
    
    def _handle_rate_limiting(self, delay: float = 1.0):
        """Handle rate limiting with exponential backoff"""
        time.sleep(delay)
        self.request_count += 1

class OpenAIClient(BaseAPIClient):
    """OpenAI GPT API Client"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        super().__init__(api_key, model_name)
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized OpenAI client for model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> APIResponse:
        """Generate text using OpenAI API"""
        max_tokens = kwargs.get('max_new_tokens', 256)
        temperature = kwargs.get('temperature', 0.0)
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'choices' in data and len(data['choices']) > 0:
                generated_text = data['choices'][0]['message']['content']
                tokens_used = data.get('usage', {}).get('total_tokens', 0)
                finish_reason = data['choices'][0].get('finish_reason', 'completed')
                
                self.total_tokens += tokens_used
                self._handle_rate_limiting(0.1)  # Small delay for rate limiting
                
                return APIResponse(
                    generated_text=generated_text,
                    model_name=self.model_name,
                    tokens_used=tokens_used,
                    finish_reason=finish_reason
                )
            else:
                raise Exception("No choices in response")
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return APIResponse(
                generated_text="",
                model_name=self.model_name,
                error=str(e)
            )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[APIResponse]:
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
            # Add delay between requests for rate limiting
            if len(prompts) > 1:
                time.sleep(0.5)
        return results

class GeminiClient(BaseAPIClient):
    """Google Gemini API Client"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        super().__init__(api_key, model_name)
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        logger.info(f"Initialized Gemini client for model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> APIResponse:
        """Generate text using Gemini API"""
        max_tokens = kwargs.get('max_new_tokens', 256)
        temperature = kwargs.get('temperature', 0.0)
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        params = {"key": self.api_key}
        
        try:
            response = requests.post(
                self.base_url,
                params=params,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'candidates' in data and len(data['candidates']) > 0:
                content = data['candidates'][0]['content']
                if 'parts' in content and len(content['parts']) > 0:
                    generated_text = content['parts'][0]['text']
                    
                    # Gemini doesn't always provide token count
                    tokens_used = data.get('usageMetadata', {}).get('totalTokenCount', 0)
                    finish_reason = data['candidates'][0].get('finishReason', 'STOP')
                    
                    self.total_tokens += tokens_used
                    self._handle_rate_limiting(0.2)
                    
                    return APIResponse(
                        generated_text=generated_text,
                        model_name=self.model_name,
                        tokens_used=tokens_used,
                        finish_reason=finish_reason
                    )
            
            raise Exception("No valid response from Gemini")
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return APIResponse(
                generated_text="",
                model_name=self.model_name,
                error=str(e)
            )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[APIResponse]:
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
            if len(prompts) > 1:
                time.sleep(0.3)
        return results

class ClaudeClient(BaseAPIClient):
    """Anthropic Claude API Client"""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key, model_name)
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        logger.info(f"Initialized Claude client for model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> APIResponse:
        """Generate text using Claude API"""
        max_tokens = kwargs.get('max_new_tokens', 256)
        temperature = kwargs.get('temperature', 0.0)
        
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'content' in data and len(data['content']) > 0:
                generated_text = data['content'][0]['text']
                tokens_used = data.get('usage', {}).get('output_tokens', 0)
                finish_reason = data.get('stop_reason', 'end_turn')
                
                self.total_tokens += tokens_used
                self._handle_rate_limiting(0.2)
                
                return APIResponse(
                    generated_text=generated_text,
                    model_name=self.model_name,
                    tokens_used=tokens_used,
                    finish_reason=finish_reason
                )
            
            raise Exception("No content in Claude response")
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return APIResponse(
                generated_text="",
                model_name=self.model_name,
                error=str(e)
            )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[APIResponse]:
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
            if len(prompts) > 1:
                time.sleep(0.5)  # Claude has stricter rate limits
        return results

class APIModelPipeline:
    """Pipeline wrapper to make API clients compatible with transformers pipeline interface"""
    
    def __init__(self, api_client: BaseAPIClient):
        self.api_client = api_client
        self.model_name = api_client.model_name
        self.device = "api"  # Indicate this is an API model
        
        # Create a mock tokenizer object for compatibility
        self.tokenizer = self._create_mock_tokenizer()
    
    def _create_mock_tokenizer(self):
        """Create a mock tokenizer for API compatibility"""
        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1
        
        return MockTokenizer()
    
    def __call__(self, prompts: Union[str, List[str]], **kwargs) -> Union[List[Dict], List[List[Dict]]]:
        """Make the pipeline callable like transformers pipeline"""
        is_single_prompt = isinstance(prompts, str)
        
        if is_single_prompt:
            prompts = [prompts]
        
        try:
            api_responses = self.api_client.batch_generate(prompts, **kwargs)
            
            # Convert to transformers-like format
            results = []
            for api_response in api_responses:
                if api_response.error:
                    # Return error response in expected format
                    result = [{"generated_text": f"API Error: {api_response.error}"}]
                else:
                    result = [{"generated_text": api_response.generated_text}]
                results.append(result)
            
            # If single prompt, return single result
            if is_single_prompt:
                return results[0]
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline call error: {e}")
            error_result = [{"generated_text": f"Pipeline Error: {str(e)}"}]
            
            if is_single_prompt:
                return error_result
            
            return [error_result for _ in prompts]

def get_available_api_models() -> Dict[str, List[str]]:
    """Get available API models by provider"""
    return {
        "OpenAI": [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-mini"
        ],
        "Gemini": [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],
        "Claude": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620"
        ]
    }

def initialize_api_model_pipeline(
    provider: str,
    model_name: str,
    api_key: str
) -> Optional[APIModelPipeline]:
    """
    Initialize API model pipeline for proprietary models.
    
    Args:
        provider: API provider ('openai', 'gemini', 'claude')
        model_name: Model name/identifier
        api_key: API key for authentication
    
    Returns:
        APIModelPipeline instance or None if initialization fails
    """
    logger.info(f"Initializing API model: {provider}/{model_name}")
    
    try:
        provider_lower = provider.lower()
        
        if provider_lower == "openai":
            client = OpenAIClient(api_key, model_name)
        elif provider_lower == "gemini":
            client = GeminiClient(api_key, model_name)
        elif provider_lower == "claude":
            client = ClaudeClient(api_key, model_name)
        else:
            raise ValueError(f"Unsupported API provider: {provider}")
        
        # Test the API connection
        test_response = client.generate("Hello", max_new_tokens=5)
        if test_response.error:
            raise Exception(f"API test failed: {test_response.error}")
        
        pipeline = APIModelPipeline(client)
        logger.info(f"Successfully initialized {provider} API pipeline for {model_name}")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize API model {provider}/{model_name}: {e}")
        return None

def cleanup_api_model_resources(pipeline: Optional[APIModelPipeline]):
    """Clean up API model resources"""
    if pipeline and hasattr(pipeline, 'api_client'):
        logger.info(f"Cleaning up API model: {pipeline.model_name}")
        logger.info(f"Total tokens used: {pipeline.api_client.total_tokens}")
        logger.info(f"Total requests made: {pipeline.api_client.request_count}")
        
        # Clear the client
        pipeline.api_client = None
        pipeline = None
        
        logger.info("API model resources cleaned up")

# Environment variable helpers
def get_api_key_from_env(provider: str) -> Optional[str]:
    """Get API key from environment variables"""
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY", 
        "claude": "ANTHROPIC_API_KEY"
    }
    
    env_var = env_var_map.get(provider.lower())
    if env_var:
        return os.getenv(env_var)
    
    return None

def set_api_key_env(provider: str, api_key: str):
    """Set API key in environment variables"""
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY"
    }
    
    env_var = env_var_map.get(provider.lower())
    if env_var:
        os.environ[env_var] = api_key
        logger.info(f"Set {env_var} environment variable")