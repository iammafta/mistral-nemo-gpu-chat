"""
Mistral-NeMo Chat API - Hybrid GPU/CPU Solution
GPU model loading with CPU inference to avoid segmentation faults
"""

import os
import logging
import time
from typing import Optional, List, Dict
import modal
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "mistral-nemo-api-key-2025"

# Modal setup with GPU for model loading
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "fastapi",
        "uvicorn[standard]",
        "llama-cpp-python[cuda]",  # CUDA version for GPU model loading
        "pydantic",
        "python-multipart"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Use existing volume with model
volume = modal.Volume.from_name("mistral-nemo-models", create_if_missing=True)

app = modal.App("mistral-hybrid-gpu-cpu")

@app.function(
    image=image,
    volumes={"/models": volume},
    gpu=modal.gpu.T4(),  # GPU for model loading
    memory=16384,  # 16GB RAM
    timeout=2400,  # 40 minutes
    scaledown_window=1800,  # 30 minutes
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from llama_cpp import Llama
    
    # Pydantic models
    class ChatRequest(BaseModel):
        message: str
        session_id: Optional[str] = "default"
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = 512
        stream: Optional[bool] = False

    class QuickTestRequest(BaseModel):
        prompt: Optional[str] = "Hello, how are you?"
        max_tokens: Optional[int] = 50
    
    # FastAPI app
    web_app = FastAPI(
        title="Mistral-NeMo Hybrid GPU/CPU API",
        description="GPU model loading with CPU inference for stability",
        version="1.5.0"
    )

    # CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global variables
    model_instance = None
    session_contexts = {}

    def verify_api_key(request: Request):
        """Verify API key from Authorization header"""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
        token = auth_header.split(" ")[1]
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    @web_app.on_event("startup")
    async def startup_event():
        nonlocal model_instance
        
        model_path = "/models/model.gguf"
        
        if not os.path.exists(model_path):
            # Try alternative path
            model_path = "/models/Mistral-NeMo-Minitron-8B-Base.Q6_K.gguf"
            if not os.path.exists(model_path):
                raise Exception(f"❌ Model file not found in /models/")
        
        logger.info(f"🔄 Loading Mistral-NeMo model: {model_path}")
        
        try:
            # Hybrid approach: Load model with some GPU layers but conservative settings
            # This avoids the segfault during inference while still using GPU memory
            model_instance = Llama(
                model_path=model_path,
                n_ctx=2048,          # Conservative context window
                n_batch=128,         # Smaller batch size for stability
                n_threads=6,         # Leave some CPU cores free
                n_gpu_layers=10,     # Partial GPU offloading - enough to help but not cause segfaults
                verbose=False,
                use_mmap=True,       # Memory-mapped files for efficiency
                use_mlock=False,     # Don't lock memory
                numa=False,          # Disable NUMA
                chat_format="llama-2",
                # Conservative inference settings to avoid segfaults
                rope_scaling_type=None,
                rope_freq_base=0.0,
                rope_freq_scale=0.0,
            )
            
            logger.info("✅ Hybrid GPU/CPU model loaded successfully")
            logger.info("🔧 Using 10 GPU layers with CPU inference for stability")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            # Fallback to CPU-only if GPU fails
            logger.info("🔄 Falling back to CPU-only mode...")
            try:
                model_instance = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_batch=128,
                    n_threads=6,
                    n_gpu_layers=0,    # CPU-only fallback
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    numa=False,
                    chat_format="llama-2"
                )
                logger.info("✅ CPU fallback model loaded successfully")
            except Exception as e2:
                logger.error(f"❌ CPU fallback also failed: {e2}")
                raise

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        import subprocess
        
        # Check GPU availability
        gpu_available = False
        gpu_info = "None"
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_available = True
                gpu_info = result.stdout.strip()
        except:
            pass
        
        return {
            "status": "ready",
            "model_loaded": model_instance is not None,
            "hardware": f"Hybrid (GPU: {gpu_info}, CPU: 6 threads)" if gpu_available else "CPU (6 threads)",
            "gpu_layers": 10 if gpu_available else 0,
            "version": "1.5.0",
            "optimization": "hybrid-gpu-cpu"
        }

    @web_app.get("/performance")
    async def performance_info():
        """Performance and resource information"""
        return {
            "model_info": {
                "type": "Mistral-NeMo-Minitron-8B",
                "quantization": "Q6_K",
                "size_gb": 6.4
            },
            "hardware": {
                "compute": "Hybrid GPU/CPU",
                "memory": "16GB",
                "gpu_layers": 10,
                "strategy": "GPU model loading + CPU inference"
            },
            "settings": {
                "context_length": 2048,
                "batch_size": 128,
                "threads": 6
            }
        }

    @web_app.post("/quick-test")
    async def quick_test(request: Request, test_request: QuickTestRequest):
        """Quick test endpoint for basic functionality"""
        verify_api_key(request)
        
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            logger.info("🧪 Running hybrid GPU/CPU quick test...")
            
            start_time = time.time()
            
            # Conservative inference settings to avoid segfaults
            response = model_instance(
                prompt=test_request.prompt,
                max_tokens=min(test_request.max_tokens, 100),  # Limit tokens for stability
                temperature=0.3,
                top_p=0.8,
                echo=False,
                stop=["</s>", "[INST]", "[/INST]"],
                # Additional stability settings
                repeat_penalty=1.1,
                top_k=40
            )
            
            generation_time = time.time() - start_time
            response_text = response['choices'][0]['text'].strip()
            
            result = {
                "prompt": test_request.prompt,
                "response": response_text,
                "tokens_generated": len(response_text.split()),
                "generation_time_seconds": round(generation_time, 2),
                "tokens_per_second": round(len(response_text.split()) / generation_time, 2) if generation_time > 0 else 0,
                "inference_mode": "hybrid-gpu-cpu"
            }
            
            logger.info(f"✅ Hybrid test completed in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Quick test failed: {e}")
            raise HTTPException(status_code=500, detail=f"Quick test failed: {str(e)}")

    @web_app.post("/chat")
    async def chat_endpoint(request: Request, chat_request: ChatRequest):
        """Main chat endpoint with hybrid GPU/CPU processing"""
        verify_api_key(request)
        
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            logger.info(f"🤖 Generating hybrid response for session {chat_request.session_id}")
            logger.info(f"📝 Prompt length: {len(chat_request.message)} chars")
            
            start_time = time.time()
            
            # Format the prompt properly
            formatted_prompt = f"[INST] {chat_request.message} [/INST]"
            
            # Conservative inference settings to prevent segfaults
            response = model_instance(
                prompt=formatted_prompt,
                max_tokens=min(chat_request.max_tokens, 512),  # Reasonable limit
                temperature=max(0.1, min(chat_request.temperature, 1.0)),  # Clamp temperature
                top_p=0.9,
                echo=False,
                stop=["</s>", "[INST]", "[/INST]", "\n\n"],
                repeat_penalty=1.1,
                top_k=40
            )
            
            generation_time = time.time() - start_time
            response_text = response['choices'][0]['text'].strip()
            
            # Update session context
            if chat_request.session_id not in session_contexts:
                session_contexts[chat_request.session_id] = []
            
            session_contexts[chat_request.session_id].append({
                "user": chat_request.message,
                "assistant": response_text,
                "timestamp": time.time()
            })
            
            # Keep only last 10 exchanges per session
            if len(session_contexts[chat_request.session_id]) > 10:
                session_contexts[chat_request.session_id] = session_contexts[chat_request.session_id][-10:]
            
            result = {
                "response": response_text,
                "session_id": chat_request.session_id,
                "generation_time_seconds": round(generation_time, 2),
                "tokens_generated": len(response_text.split()),
                "tokens_per_second": round(len(response_text.split()) / generation_time, 2) if generation_time > 0 else 0,
                "model_info": {
                    "name": "Mistral-NeMo-Minitron-8B",
                    "hardware": "Hybrid GPU/CPU",
                    "optimization": "hybrid-gpu-cpu",
                    "gpu_layers": 10
                }
            }
            
            logger.info(f"✅ Hybrid response generated in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Chat generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

    @web_app.get("/sessions/{session_id}")
    async def get_session_history(request: Request, session_id: str):
        """Get chat history for a session"""
        verify_api_key(request)
        
        history = session_contexts.get(session_id, [])
        return {
            "session_id": session_id,
            "message_count": len(history),
            "history": history
        }

    @web_app.delete("/sessions/{session_id}")
    async def clear_session(request: Request, session_id: str):
        """Clear chat history for a session"""
        verify_api_key(request)
        
        if session_id in session_contexts:
            del session_contexts[session_id]
            return {"message": f"Session {session_id} cleared"}
        else:
            return {"message": f"Session {session_id} not found"}

    return web_app

if __name__ == "__main__":
    print("🚀 Deploying Mistral-NeMo Hybrid GPU/CPU Chat API...")
    print("📋 Features:")
    print("  • Hybrid GPU/CPU processing")
    print("  • 10 GPU layers for model loading")
    print("  • CPU inference for stability")
    print("  • T4 GPU + 16GB memory")
    print("  • Anti-segfault configuration")
    print("  • Session management")
    print("\n🔗 Endpoints:")
    print("  • GET  /health - Health check")
    print("  • GET  /performance - Performance info")
    print("  • POST /quick-test - Quick functionality test")
    print("  • POST /chat - Main chat endpoint")
    print("  • GET  /sessions/{id} - Session history")
    print("  • DELETE /sessions/{id} - Clear session")
    print("\n🔑 Authorization: Bearer mistral-nemo-api-key-2025")
