import requests
import json
import base64
import argparse
import os

def test_multimodal(image_path, model_name="Qwen2-VL-7B-Instruct", port=8001, prompt="What can you see in this image?"):
    """
    Test a multimodal model with an image
    
    Args:
        image_path: Path to the image file
        model_name: Name of the model (without organization prefix)
        port: Port where the model is deployed
        prompt: Text prompt to send with the image
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    print(f"Sending request to multimodal model at port {port}...")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            result = response.json()
            print("\nModel Response:")
            print("=" * 80)
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(content)
            
            print("\nFull Response:")
            print(json.dumps(result, indent=2))
            
            # Save the response to a file
            with open("mllm_response.json", "w") as f:
                json.dump(result, f, indent=2)
            print("\nResponse saved to mllm_response.json")
            
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        print("The model might still be loading or not properly deployed.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a multimodal model with an image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--model", default="Qwen2-VL-7B-Instruct", help="Model name (default: Qwen2-VL-7B-Instruct)")
    parser.add_argument("--port", type=int, default=8001, help="Port where the model is deployed (default: 8001)")
    parser.add_argument("--prompt", default="What can you see in this image?", help="Text prompt to send with the image")
    
    args = parser.parse_args()
    
    test_multimodal(args.image_path, args.model, args.port, args.prompt)