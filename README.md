# Swift Model Deployment Server

A Docker-based service for deploying and managing AI models using the SWIFT framework. This server provides a REST API to deploy, manage, and interact with both text-only and multimodal language models.

## Features

- Deploy models with a simple API call
- Manage multiple model deployments
- Supports text, image, and audio models
- Automatically handles model requirements
- Provides OpenAI-compatible endpoints for deployed models

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

### Setup and Deployment

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/swift-model-deployment-server.git
   cd swift-model-deployment-server
   ```

2. Start the server:
   ```bash
   docker-compose up -d
   ```

3. Deploy a model:
   ```bash
   python client.py deploy Qwen/Qwen2.5-7B-Instruct
   ```

4. Test the deployed model:
   ```bash
   curl http://localhost:8001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen2.5-7B-Instruct",
       "messages": [
         {"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is the SWIFT framework?"}
       ],
       "temperature": 0.7,
       "max_tokens": 500
     }'
   ```

## Usage

### API Endpoints

- `GET /` - API information
- `GET /models` - List all available models
- `POST /deploy` - Deploy a model
- `GET /deployments` - List active deployments
- `DELETE /deployments/{model_id}` - Stop a deployment

### Client Commands

The `client.py` script provides a convenient command-line interface:

```bash
# List available models
python client.py list

# Deploy a text model
python client.py deploy Qwen/Qwen2.5-7B-Instruct

# Deploy a multimodal model
python client.py deploy Qwen/Qwen2-VL-7B-Instruct

# Check active deployments
python client.py check

# Stop a deployment
python client.py stop Qwen/Qwen2.5-7B-Instruct
```

### Testing Multimodal Models

#### Image Models

Use the `mllm_test.py` script to test vision models:

```bash
python mllm_test.py path/to/image.jpg --model Qwen2-VL-7B-Instruct --prompt "Describe this image in detail."
```

#### Audio Models

Use the `audio_test.py` script to test audio models:

```bash
python audio_test.py path/to/audio.wav --prompt "Transcribe this audio and tell me what language it's in."
```

## Supported Models

### Text Models

- Qwen2.5 models (7B, 72B)
- Gemma models (2-9B, 2-27B, 3-12B)  
- Llama 3 models (3B to 70B)
- Phi models (3.5, 4)
- Mistral models
- DeepSeek models
- Yi models
- GLM models

### Multimodal Models

- **Vision**: Qwen-VL, DeepSeek-VL, LLaVA, Phi-Vision
- **Audio**: Qwen2-Audio, Llama-Omni
- **Video**: InternVL, MiniCPM-V
- **Document/OCR**: Janus-Pro, GOT-OCR
- **Multi-capability**: MiniCPM-o, Phi-4-Multimodal

## Configuration

### Docker Compose

The `docker-compose.yml` file configures the environment:

```yaml
version: '3'
services:
  swift-deploy-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8020:8020"  # API server port
      - "8001:8001"  # Default model deployment port
    volumes:
      - ./:/app
      - ${HOME}/.cache/modelscope:/root/.cache/modelscope
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    command: bash -c "uvicorn api:app --host 0.0.0.0 --port 8020"
```

### Deployment Parameters

The deployment API accepts these parameters:

```json
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "gpu_id": 0,
  "max_model_len": 4096,
  "vision_batch_size": 2,
  "gpu_memory_utilization": 0.9,
  "port": 8001
}
```

## Troubleshooting

### Checking Model Status

To check if a model is properly deployed:

```bash
# Using the client
python client.py check

# Using curl directly
curl http://localhost:8001/v1/models
```

### Viewing Logs

To view deployment logs:

```bash
# API server logs
docker-compose logs -f

# Model deployment logs
tail -f deployment_Qwen_Qwen2.5-7B-Instruct_8001.log
```

### Common Issues

- **Connection refused** - The model is not yet fully loaded or the port is incorrect
- **CUDA out of memory** - Try reducing `max_model_len` or using a smaller model
- **Missing dependencies** - Check deployment logs for required packages

## License

MIT License