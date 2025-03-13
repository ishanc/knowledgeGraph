import os
from typing import List, Dict, Optional, Union
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class MistralWrapper:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("HUGGINGFACE_TOKEN")
        if not api_key:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
            
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )
        self.model = "mistralai/Mistral-Nemo-Instruct-2407"

    def _format_messages(self, content: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """Format content into proper chat message sequence"""
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        
        # Ensure alternating user/assistant pattern
        formatted_messages = []
        last_role = None
        
        for msg in content:
            if msg["role"] == "system":
                if not formatted_messages:
                    formatted_messages.append(msg)
                continue
                
            if last_role == "user" and msg["role"] == "user":
                formatted_messages.append({"role": "assistant", "content": "Understood."})
            elif last_role == "assistant" and msg["role"] == "assistant":
                continue
                
            formatted_messages.append(msg)
            last_role = msg["role"]
        
        # Ensure it ends with user message
        if formatted_messages[-1]["role"] == "assistant":
            formatted_messages.pop()
            
        return formatted_messages

    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Simple prompt-based generation"""
        try:
            messages = self._format_messages(prompt)
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

    def generate_with_context(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.1) -> Optional[str]:
        """Generate text using the Mistral model with streaming"""
        try:
            formatted_messages = self._format_messages(messages)
            response_chunks = []
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response_chunks.append(chunk.choices[0].delta.content)
            
            full_response = "".join(response_chunks)
            return full_response if full_response else None
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
