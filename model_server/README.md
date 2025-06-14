# Llama3.1-Aloe-Beta-8B API Server

This is a FastAPI server that serves the Llama3.1-Aloe-Beta-8B model. Two versions are available:
- GPU version using VLLM (server.py)
- CPU version using Transformers (server_cpu.py)

## Prerequisites

### For GPU Version
- Linux server with CUDA-compatible GPU
- Python 3.8 or higher
- CUDA 11.8 or higher
- At least 16GB of GPU memory (recommended)

### For CPU Version
- Linux server with Intel Xeon or similar CPU
- Python 3.8 or higher
- At least 32GB of RAM (recommended)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

For GPU version:
```bash
pip install -r requirements.txt
```

For CPU version:
```bash
pip install -r requirements_cpu.txt
```

3. Run the server:

For GPU version:
```bash
python server.py
```

For CPU version:
```bash
python server_cpu.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### Generate Text
- **URL**: `/generate`
- **Method**: `POST`
- **Request Body**:
```json
{
    "prompt": "Your prompt here",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": ["stop", "words"]
}
```
- **Response**:
```json
{
    "generated_text": "Generated text here"
}
```

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
```json
{
    "status": "healthy"
}
```

## Example Usage

Using curl:
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What are the symptoms of diabetes?", "max_tokens": 200}'
```

Using Python requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "What are the symptoms of diabetes?",
        "max_tokens": 200
    }
)
print(response.json())
```

## Notes

### GPU Version
- Adjust `tensor_parallel_size` in `server.py` based on your GPU setup
- The model requires significant GPU memory, ensure your system meets the requirements

### CPU Version
- The CPU version is significantly slower than the GPU version
- Consider using a machine with high RAM capacity
- Generation times will be longer, especially for longer sequences

### General Notes
- For production deployment, consider:
  - Using a process manager like systemd or supervisor
  - Setting up proper authentication
  - Configuring HTTPS
  - Implementing rate limiting 