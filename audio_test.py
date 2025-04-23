import requests
import json
import base64
import argparse
import os

def test_audio_model(audio_path, port=8001, model_name="Qwen2-Audio-7B-Instruct", prompt="What is being said in this audio?"):
    """
    Test an audio model with an audio file
    
    Args:
        audio_path: Path to the audio file
        port: Port where the model is deployed
        model_name: Name of the model (without organization prefix)
        prompt: Text prompt to send with the audio
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    
    # Read and encode the audio file
    with open(audio_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Format for Qwen audio model
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{encoded_audio}"}}
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    print(f"Sending request to audio model at port {port}...")
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
            with open("audio_response.json", "w") as f:
                json.dump(result, f, indent=2)
            print("\nResponse saved to audio_response.json")
            
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        print("The model might still be loading or not properly deployed.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an audio model with an audio file")
    parser.add_argument("audio_path", help="Path to the audio file (WAV format recommended)")
    parser.add_argument("--model", default="Qwen2-Audio-7B-Instruct", help="Model name (default: Qwen2-Audio-7B-Instruct)")
    parser.add_argument("--port", type=int, default=8001, help="Port where the model is deployed (default: 8001)")
    parser.add_argument("--prompt", default="What is being said in this audio?", 
                        help="Text prompt to send with the audio")
    
    args = parser.parse_args()
    
    test_audio_model(args.audio_path, args.port, args.model, args.prompt)