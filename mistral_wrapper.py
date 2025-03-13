import os
from typing import Dict, Any, Optional
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

class MistralWrapper:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('HUGGINGFACE_TOKEN')
        if not self.api_key:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
        self.client = InferenceClient(api_key=self.api_key)
        self.model = "mistralai/Mistral-Nemo-Instruct-2407"

    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = []
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response.append(chunk.choices[0].delta.content)
            
            return "".join(response)
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def generate_with_context(self, system_prompt: str, user_prompt: str, 
                            conversation_history: Optional[list] = None,
                            max_tokens: int = 1000, temperature: float = 0.7) -> str:
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": user_prompt})
            
            response = []
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response.append(chunk.choices[0].delta.content)
            
            return "".join(response)
        except Exception as e:
            raise Exception(f"Error generating response with context: {str(e)}")
