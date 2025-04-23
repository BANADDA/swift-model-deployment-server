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
   git clone https://github.com/BANADDA/swift-model-deployment-server.git
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

### Text-Only Models (LLMs)

| Family | Model | Parameters | Type | Model ID |
|--------|-------|------------|------|----------|
| **Qwen** | Qwen2.5-7B-Instruct | 7B | Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| | Qwen2.5-72B-Instruct | 72B | Instruct | `Qwen/Qwen2.5-72B-Instruct` |
| | Qwen2.5-Coder-7B-Instruct | 7B | Coding | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| | Qwen2.5-Math-7B-Instruct | 7B | Math | `Qwen/Qwen2.5-Math-7B-Instruct` |
| **Gemma** | Gemma-2-9B-Instruct | 9B | Instruct | `LLM-Research/gemma-2-9b-it` |
| | Gemma-2-27B-Instruct | 27B | Instruct | `LLM-Research/gemma-2-27b-it` |
| | Gemma-3-12B-Instruct | 12B | Instruct | `LLM-Research/gemma-3-12b-it` |
| **Llama** | Meta-Llama-3.1-8B-Instruct | 8B | Instruct | `LLM-Research/Meta-Llama-3.1-8B-Instruct` |
| | Meta-Llama-3.1-70B-Instruct | 70B | Instruct | `LLM-Research/Meta-Llama-3.1-70B-Instruct` |
| | Llama-3.2-3B-Instruct | 3B | Instruct | `LLM-Research/Llama-3.2-3B-Instruct` |
| | Llama-3.3-70B-Instruct | 70B | Instruct | `LLM-Research/Llama-3.3-70B-Instruct` |
| **Phi** | Phi-3-Medium-128k-Instruct | 14B | Instruct | `LLM-Research/Phi-3-medium-128k-instruct` |
| | Phi-3.5-Mini-Instruct | 3.8B | Instruct | `LLM-Research/Phi-3.5-mini-instruct` |
| | Phi-4-Mini-Instruct | 3.8B | Instruct | `LLM-Research/Phi-4-mini-instruct` |
| **Mistral** | Mistral-7B-Instruct-v0.3 | 7B | Instruct | `LLM-Research/Mistral-7B-Instruct-v0.3` |
| | Mistral-Large-Instruct-2407 | 36B | Instruct | `LLM-Research/Mistral-Large-Instruct-2407` |
| | Mixtral-8x7B-Instruct-v0.1 | 47B MoE | Instruct | `AI-ModelScope/Mixtral-8x7B-Instruct-v0.1` |
| **DeepSeek** | DeepSeek-V3 | 7B | Instruct | `deepseek-ai/DeepSeek-V3` |
| | DeepSeek-R1 | 7B | Instruct | `deepseek-ai/DeepSeek-R1` |
| | DeepSeek-Coder-V2-Instruct | 7B | Coding | `deepseek-ai/DeepSeek-Coder-V2-Instruct` |
| **Yi** | Yi-1.5-34B-Chat | 34B | Chat | `01ai/Yi-1.5-34B-Chat` |
| | Yi-Coder-9B-Chat | 9B | Coding | `01ai/Yi-Coder-9B-Chat` |
| **GLM** | GLM-4-9B-Chat | 9B | Chat | `ZhipuAI/glm-4-9b-chat` |
| | GLM-Z1-9B-0414 | 9B | Chat | `ZhipuAI/GLM-Z1-9B-0414` |

### Multimodal Models (MLLMs)

| Family | Model | Parameters | Modalities | Model ID |
|--------|-------|------------|------------|----------|
| **Qwen** | Qwen-VL-Chat | 7B | Vision | `Qwen/Qwen-VL-Chat` |
| | Qwen2-VL-7B-Instruct | 7B | Vision, Video | `Qwen/Qwen2-VL-7B-Instruct` |
| | Qwen2-Audio-7B-Instruct | 7B | Audio | `Qwen/Qwen2-Audio-7B-Instruct` |
| **DeepSeek** | DeepSeek-VL-7B-Chat | 7B | Vision | `deepseek-ai/deepseek-vl-7b-chat` |
| | DeepSeek-VL2 | 7B | Vision | `deepseek-ai/deepseek-vl2` |
| | Janus-Pro-7B | 7B | Vision, OCR | `deepseek-ai/Janus-Pro-7B` |
| **Llama** | LLaVA-1.5-7B-HF | 7B | Vision | `llava-hf/llava-1.5-7b-hf` |
| | Llama-3.2-11B-Vision-Instruct | 11B | Vision | `LLM-Research/Llama-3.2-11B-Vision-Instruct` |
| | Llama-3.1-8B-Omni | 8B | Audio | `ICTNLP/Llama-3.1-8B-Omni` |
| **InternVL** | InternVL2-8B | 8B | Vision, Video | `OpenGVLab/InternVL2-8B` |
| | InternVL3-8B | 8B | Vision, Video | `OpenGVLab/InternVL3-8B` |
| **Phi** | Phi-3-Vision-128k-Instruct | 4B | Vision | `LLM-Research/Phi-3-vision-128k-instruct` |
| | Phi-4-Multimodal-Instruct | 4B | Vision, Audio | `LLM-Research/Phi-4-multimodal-instruct` |
| **Other** | Yi-VL-6B | 6B | Vision | `01ai/Yi-VL-6B` |
| | MiniCPM-V-2_6 | 7B | Vision, Video | `OpenBMB/MiniCPM-V-2_6` |
| | MiniCPM-o-2_6 | 7B | Vision, Video, Audio | `OpenBMB/MiniCPM-o-2_6` |
| | GLM-4V-9B | 9B | Vision | `ZhipuAI/glm-4v-9b` |
| | GOT-OCR2_0 | 7B | Vision, OCR | `stepfun-ai/GOT-OCR2_0` |

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