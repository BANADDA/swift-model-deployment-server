import requests
import json
import sys
import base64

def test_audio_model(audio_path, prompt="What is being said in this audio?"):
    """Test audio model from inside the container"""
    # Read and encode the audio file
    with open(audio_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    
    url = "http://localhost:8001/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Format for Qwen audio model
    data = {
        "model": "Qwen2-Audio-7B-Instruct",
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
    
    print(f"Sending request to audio model at localhost:8001...")
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
            
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python container_test.py <audio_file_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    prompt = "Transcribe this audio."
    
    test_audio_model(audio_path, prompt)