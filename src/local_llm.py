"""
Local LLM module using Transformers.
Uses Qwen2.5-0.5B-Instruct - a small, fast model (~500MB).
"""

import logging
from pathlib import Path
from typing import Generator
import threading

logger = logging.getLogger(__name__)

# Small model that works well for Q&A (~500MB)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODELS_DIR = Path(__file__).parent.parent / "models" / "llm"


class LocalLLM:
    """Local LLM using Transformers library."""
    
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_model()
    
    def _load_model(self):
        """Load the model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading model: {MODEL_NAME}")
            
            cache_dir = MODELS_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Force CPU for stability as MPS is causing segmentation faults with this model configuration
            # Qwen 0.5B is small enough to run fast on CPU
            self._device = "cpu"
            dtype = torch.float32
            
            logger.info(f"Using device: {self._device} with dtype={dtype}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                cache_dir=cache_dir
            )
            
            self._model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                cache_dir=cache_dir,
                torch_dtype=dtype,
            ).to(self._device)
            
            logger.info("Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from a prompt (blocking)."""
        output = ""
        for chunk in self.generate_stream(prompt, system_prompt, max_tokens, temperature):
            output += chunk
        return output
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Generate text stream."""
        from transformers import TextIteratorStreamer
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._device)
        
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
        
        thread = threading.Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
