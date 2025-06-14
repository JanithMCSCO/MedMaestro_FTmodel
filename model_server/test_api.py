import requests
import json

# Replace with your server's IP address
SERVER_URL = "http://<your-server-ip>:8000"

def test_health():
    response = requests.get(f"{SERVER_URL}/health")
    print("Health check response:", response.json())

def test_generation():
    payload = {
        "prompt": "What are the symptoms of diabetes?",
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    print("Sending request...")
    response = requests.post(
        f"{SERVER_URL}/generate",
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\nGenerated text:")
        print(result["generated_text"])
    else:
        print("Error:", response.status_code)
        print(response.text)

if __name__ == "__main__":
    print("Testing health endpoint...")
    test_health()
    
    print("\nTesting generation endpoint...")
    test_generation() 