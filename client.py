import requests
import json
import sys
import time
import base64
from pprint import pprint

# Base URL for the API
BASE_URL = "http://localhost:8020"

def list_models():
    """List all available models"""
    response = requests.get(f"{BASE_URL}/models")
    if response.status_code == 200:
        models = response.json()
        print(f"Found {len(models)} models")
        
        # Categorize models
        multimodal = [m for m in models if m.get('is_multimodal')]
        text_only = [m for m in models if not m.get('is_multimodal')]
        
        print(f"\n=== Multimodal Models ({len(multimodal)}) ===")
        for i, model in enumerate(multimodal[:10], 1):  # Show first 10
            print(f"{i}. {model['name']} ({model['parameters']}) - {model['type']}")
        
        print(f"\n=== Text-Only Models ({len(text_only)}) ===")
        for i, model in enumerate(text_only[:10], 1):  # Show first 10
            print(f"{i}. {model['name']} ({model['parameters']}) - {model['type']}")
        
        return models
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def deploy_model(model_id, gpu_id=0, port=8001):
    """Deploy a model"""
    data = {
        "model_id": model_id,
        "gpu_id": gpu_id,
        "port": port
    }
    
    # If model is multimodal, adjust parameters
    if "VL" in model_id or "vision" in model_id.lower() or "vl" in model_id.lower():
        data["max_model_len"] = 3000
        data["vision_batch_size"] = 2
        data["gpu_memory_utilization"] = 0.95
    
    print(f"Deploying model: {model_id}")
    response = requests.post(f"{BASE_URL}/deploy", json=data)
    if response.status_code == 200:
        result = response.json()
        print("Deployment status:", result["status"])
        print("Command:", result["deployment_command"])
        print("Log file:", result["log_file"])
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def check_deployments():
    """Check current deployments"""
    response = requests.get(f"{BASE_URL}/deployments")
    if response.status_code == 200:
        deployments = response.json()
        print(f"Active deployments: {len(deployments)}")
        for model_id, info in deployments.items():
            print(f"- {model_id}: {info['status']} (Port: {info['port']}, GPU: {info['gpu_id']})")
        return deployments
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def stop_deployment(model_id):
    """Stop a running deployment"""
    response = requests.delete(f"{BASE_URL}/deployments/{model_id}")
    if response.status_code == 200:
        result = response.json()
        print(f"Stopped deployment: {model_id}")
        print("Status:", result["status"])
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def test_deployed_model(port=8001, model_id="Qwen/Qwen2.5-7B-Instruct", multimodal=False, image_path=None):
    """Test a deployed model with a simple prompt"""
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    if multimodal and image_path:
        # For multimodal models with image
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        data = {
            "model": model_id.split("/")[-1],
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What can you see in this image?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
    else:
        # For text-only models
        data = {
            "model": model_id.split("/")[-1],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What are the main features of the SWIFT framework?"}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
    
    print(f"Testing model at port {port}...")
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print("\nModel response:")
            if 'choices' in result and len(result['choices']) > 0:
                print(result['choices'][0]['message']['content'])
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        print("The model might still be loading. Please wait and try again.")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <command> [args]")
        print("Commands: list, deploy, check, stop, test")
        print("Examples:")
        print("  python client.py list")
        print("  python client.py deploy Qwen/Qwen2.5-7B-Instruct")
        print("  python client.py check")
        print("  python client.py stop Qwen/Qwen2.5-7B-Instruct")
        print("  python client.py test 8001 Qwen/Qwen2.5-7B-Instruct")
        print("  python client.py test 8001 Qwen/Qwen2-VL-7B-Instruct True path/to/image.jpg")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_models()
    
    elif command == "deploy":
        if len(sys.argv) < 3:
            print("Error: Missing model_id")
            return
        model_id = sys.argv[2]
        gpu_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        port = int(sys.argv[4]) if len(sys.argv) > 4 else 8001
        deploy_model(model_id, gpu_id, port)
    
    elif command == "check":
        check_deployments()
    
    elif command == "stop":
        if len(sys.argv) < 3:
            print("Error: Missing model_id")
            return
        model_id = sys.argv[2]
        stop_deployment(model_id)
    
    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Missing port")
            return
        port = int(sys.argv[2])
        model_id = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-7B-Instruct"
        multimodal = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else False
        image_path = sys.argv[5] if len(sys.argv) > 5 else None
        test_deployed_model(port, model_id, multimodal, image_path)
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()