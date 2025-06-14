from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
from typing import List, Optional

app = FastAPI(title="Llama3.1-Aloe-Beta-8B API")

# Initialize the model
model = LLM(
    model="HPAI-BSC/Llama3.1-Aloe-Beta-8B",
    tensor_parallel_size=1,  # Adjust based on your GPU setup
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

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
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop
        )
        
        outputs = model.generate(
            request.prompt,
            sampling_params
        )
        
        generated_text = outputs[0].outputs[0].text
        
        return GenerationResponse(generated_text=generated_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 