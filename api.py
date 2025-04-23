import os
import json
import subprocess
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Initialize FastAPI app
app = FastAPI(title="Swift Model Deployment API")

# Load models config from JSON file
with open("models_config.json", "r") as f:
    models_config = json.load(f)

class DeployRequest(BaseModel):
    model_id: str
    gpu_id: int = 0
    max_model_len: Optional[int] = None
    vision_batch_size: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    port: int = 8001

class DeploymentStatus(BaseModel):
    status: str
    model_id: str
    deployment_command: str
    log_file: str
    port: int
    gpu_id: int

# Track deployments
active_deployments = {}

def find_model_config(model_id: str) -> Dict[str, Any]:
    """Find the model configuration by model_id"""
    # Search in multimodal models
    for category in models_config["multimodal_models"]:
        for model in models_config["multimodal_models"][category]:
            if model["model_id"] == model_id:
                model["is_multimodal"] = True
                return model
    
    # Search in text only models
    for category in models_config["text_only_models"]:
        for model in models_config["text_only_models"][category]:
            if model["model_id"] == model_id:
                model["is_multimodal"] = False
                return model
    
    raise ValueError(f"Model {model_id} not found in configuration")

def deploy_model_task(model_id: str, gpu_id: int, max_model_len: Optional[int], 
                     vision_batch_size: Optional[int], gpu_memory_utilization: float,
                     port: int) -> None:
    """Background task to deploy the model"""
    try:
        # Find model configuration
        model_config = find_model_config(model_id)
        
        # Determine if model is from HuggingFace
        use_hf = not model_id.startswith(("Qwen/", "modelscope/", "damo/", "iic/", "AI-ModelScope/"))
        
        # Set default max_model_len if not provided
        if max_model_len is None:
            max_model_len = 3000 if model_config["is_multimodal"] else 4096
        
        # Build command
        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "swift", "deploy",
            "--model", model_id,
            "--infer_backend", "vllm",
            "--max_model_len", str(max_model_len),
            "--gpu_memory_utilization", str(gpu_memory_utilization),
            "--port", str(port)
        ]
        
        # Add vision_batch_size for multimodal models
        if model_config["is_multimodal"]:
            if vision_batch_size is None:
                vision_batch_size = 2
            cmd.extend(["--vision_batch_size", str(vision_batch_size)])
        
        # Add use_hf flag if needed
        if use_hf:
            cmd.extend(["--use_hf", "true"])
        
        # Create log file
        log_file = f"deployment_{model_id.replace('/', '_')}_{port}.log"
        with open(log_file, "w") as f:
            # Execute deployment command
            deployment_process = subprocess.Popen(
                " ".join(cmd),
                shell=True,
                stdout=f,
                stderr=f
            )
        
        # Store deployment information
        active_deployments[model_id] = {
            "process": deployment_process,
            "command": " ".join(cmd),
            "log_file": log_file,
            "port": port,
            "gpu_id": gpu_id
        }
        
    except Exception as e:
        print(f"Error deploying model {model_id}: {str(e)}")
        if model_id in active_deployments:
            del active_deployments[model_id]

def install_requirements(model_id: str) -> None:
    """Install required packages for the model"""
    try:
        model_config = find_model_config(model_id)
        requires = model_config.get("requires", "-")
        
        if requires != "-" and requires:
            # Install requirements
            subprocess.run(
                f"pip install {requires} -U",
                shell=True,
                check=True
            )
            print(f"Installed requirements for {model_id}: {requires}")
        
    except Exception as e:
        print(f"Error installing requirements for {model_id}: {str(e)}")
        raise

@app.post("/deploy", response_model=DeploymentStatus)
async def deploy_model(deploy_request: DeployRequest, background_tasks: BackgroundTasks):
    """Deploy a model with the specified parameters"""
    try:
        model_id = deploy_request.model_id
        
        # Check if model is already deployed
        if model_id in active_deployments:
            return DeploymentStatus(
                status="already_deployed",
                model_id=model_id,
                deployment_command=active_deployments[model_id]["command"],
                log_file=active_deployments[model_id]["log_file"],
                port=active_deployments[model_id]["port"],
                gpu_id=active_deployments[model_id]["gpu_id"]
            )
        
        # Find model in config
        model_config = find_model_config(model_id)
        
        # Install requirements
        install_requirements(model_id)
        
        # Start deployment in background
        background_tasks.add_task(
            deploy_model_task,
            model_id=model_id,
            gpu_id=deploy_request.gpu_id,
            max_model_len=deploy_request.max_model_len,
            vision_batch_size=deploy_request.vision_batch_size,
            gpu_memory_utilization=deploy_request.gpu_memory_utilization,
            port=deploy_request.port
        )
        
        # Build the command for display
        cmd_parts = [
            f"CUDA_VISIBLE_DEVICES={deploy_request.gpu_id}",
            "swift", "deploy",
            "--model", model_id,
            "--infer_backend", "vllm",
            "--max_model_len", str(deploy_request.max_model_len or (3000 if model_config["is_multimodal"] else 4096)),
            "--gpu_memory_utilization", str(deploy_request.gpu_memory_utilization),
            "--port", str(deploy_request.port),
            "--host", "0.0.0.0"  # Ensure the model server accepts connections from all interfaces
        ]
        
        if model_config["is_multimodal"]:
            cmd_parts.extend(["--vision_batch_size", str(deploy_request.vision_batch_size or 2)])
            
        # Determine if model is from HuggingFace
        use_hf = not model_id.startswith(("Qwen/", "modelscope/", "damo/", "iic/", "AI-ModelScope/"))
        if use_hf:
            cmd_parts.extend(["--use_hf", "true"])
        
        command = " ".join(cmd_parts)
        log_file = f"deployment_{model_id.replace('/', '_')}_{deploy_request.port}.log"
        
        return DeploymentStatus(
            status="deploying",
            model_id=model_id,
            deployment_command=command,
            log_file=log_file,
            port=deploy_request.port,
            gpu_id=deploy_request.gpu_id
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deployment error: {str(e)}")

@app.get("/deployments", response_model=Dict[str, DeploymentStatus])
async def get_deployments():
    """Get all active deployments"""
    result = {}
    for model_id, deployment in active_deployments.items():
        # Check if process is still running
        if deployment["process"].poll() is None:
            status = "running"
        else:
            status = f"exited (code: {deployment['process'].returncode})"
            
        result[model_id] = DeploymentStatus(
            status=status,
            model_id=model_id,
            deployment_command=deployment["command"],
            log_file=deployment["log_file"],
            port=deployment["port"],
            gpu_id=deployment["gpu_id"]
        )
    
    return result

@app.delete("/deployments/{model_id}")
async def stop_deployment(model_id: str):
    """Stop a running deployment"""
    if model_id not in active_deployments:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found in active deployments")
    
    try:
        # Terminate the process
        deployment = active_deployments[model_id]
        if deployment["process"].poll() is None:
            deployment["process"].terminate()
            deployment["process"].wait(timeout=30)
        
        # Remove from active deployments
        del active_deployments[model_id]
        
        return {"status": "stopped", "model_id": model_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping deployment: {str(e)}")

@app.get("/models", response_model=List[Dict[str, Any]])
async def list_models():
    """List all available models"""
    all_models = []
    
    # Add multimodal models
    for category in models_config["multimodal_models"]:
        for model in models_config["multimodal_models"][category]:
            model_copy = model.copy()
            model_copy["category"] = category
            model_copy["is_multimodal"] = True
            all_models.append(model_copy)
    
    # Add text only models
    for category in models_config["text_only_models"]:
        for model in models_config["text_only_models"][category]:
            model_copy = model.copy()
            model_copy["category"] = category
            model_copy["is_multimodal"] = False
            all_models.append(model_copy)
    
    return all_models

@app.get("/")
async def root():
    return {
        "name": "Swift Model Deployment API",
        "version": "1.0.0",
        "endpoints": [
            "/deploy - Deploy a model",
            "/deployments - List active deployments",
            "/deployments/{model_id} - Stop a deployment",
            "/models - List all available models"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)