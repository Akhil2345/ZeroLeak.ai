import requests
import json
import time
from typing import Dict, Any, Optional
from config import LLM_PROVIDERS, DEFAULT_PROVIDER, DEFAULT_MODEL

class LLMClient:
    def __init__(self, provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL):
        self.provider = provider
        self.model = model
        self.config = LLM_PROVIDERS.get(provider, {})
        
    def call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Call LLM with retry logic and error handling"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if self.provider == "openrouter":
                    return self._call_openrouter(prompt, temperature, max_tokens)
                elif self.provider == "openai":
                    return self._call_openai(prompt, temperature, max_tokens)
                elif self.provider == "together":
                    return self._call_together(prompt, temperature, max_tokens)
                elif self.provider == "groq":
                    return self._call_groq(prompt, temperature, max_tokens)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error calling LLM: {str(e)}"
                time.sleep(retry_delay)
                retry_delay *= 2
                
    def _call_openrouter(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call OpenRouter API"""
        import os
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "OpenRouter API key not configured"
            
        url = self.config["base_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        model_name = self.config["models"].get(self.model, self.model)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    def _call_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call OpenAI API"""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OpenAI API key not configured"
            
        url = self.config["base_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        model_name = self.config["models"].get(self.model, self.model)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    def _call_together(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Together.ai API"""
        import os
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            return "Together.ai API key not configured"
            
        url = self.config["base_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        model_name = self.config["models"].get(self.model, self.model)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    def _call_groq(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Groq API"""
        import os
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Groq API key not configured"
            
        url = self.config["base_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        model_name = self.config["models"].get(self.model, self.model)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]

# Backward compatibility
def call_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Legacy function for backward compatibility"""
    client = LLMClient(DEFAULT_PROVIDER, model)
    return client.call_llm(prompt)