from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import uvicorn
from typing import List, Optional

app = FastAPI(title="Llama3.1-Aloe-Beta-8B CPU API")

# Initialize the model and tokenizer
model_name = "HPAI-BSC/Llama3.1-Aloe-Beta-8B"

# Load config first
config = AutoConfig.from_pretrained(model_name)
# Keep the original RoPE scaling configuration
config.rope_scaling = {
    "type": "llama3",
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192
}

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Initialize model with modified config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float32,  # Use float32 for CPU
    low_cpu_mem_usage=True,
    device_map="cpu"
)
# Set model's pad token
model.config.pad_token_id = tokenizer.pad_token_id

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stop: Optional[List[str]] = None

class GenerationResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        # Tokenize input with attention mask
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        
        # Generate
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode and return
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        generated_text = generated_text[len(request.prompt):]
        
        return GenerationResponse(generated_text=generated_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 