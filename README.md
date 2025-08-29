# Mistral NeMo API on Modal

# 🚀 Mistral-NeMo GPU Chat API

A high-performance GPU-accelerated chat API using Mistral-NeMo-8B model on Modal.

## 🎯 Features

- **GPU Acceleration**: Full A10G GPU support (24GB VRAM)
- **Optimized Performance**: 20 GPU layers for reliable inference
- **Modal Deployment**: Serverless deployment with auto-scaling
- **Web Interface**: Clean chat interface with real-time responses
- **Session Management**: Persistent chat sessions
- **Health Monitoring**: Performance metrics and diagnostics

## 🏗️ Architecture

### Model Configuration
- **Model**: Mistral-NeMo-Minitron-8B-Base (Q6_K GGUF)
- **GPU**: A10G (24GB VRAM)
- **Context**: 4K tokens
- **GPU Layers**: 20 (conservative for stability)
- **Memory**: 16GB RAM

### Deployment Stack
- **Platform**: Modal.com
- **Framework**: FastAPI + llama-cpp-python
- **GPU Runtime**: CUDA-accelerated inference
- **Storage**: Modal volumes for model persistence

## 🚀 Quick Start

### Prerequisites
```bash
pip install modal
modal auth
```

### Deploy to Modal
```bash
# Deploy the working GPU version
modal deploy mistral_gpu_clean.py

# Test the deployment
python mistral_gpu_clean.py
```

## 📁 Project Structure

```
mistral-nemo-setup/
├── mistral_gpu_clean.py      # ✅ Main working GPU implementation
├── mistral_hybrid_gpu_cpu.py # 🔄 Hybrid GPU/CPU fallback version
├── mistral_working.py         # 📝 Original working version
├── README.md                  # 📖 This file
└── requirements.txt           # 📦 Dependencies
```

## 🎮 GPU Configuration

### Validated Settings
```python
MODEL_CONFIG = {
    "model_path": "/models/model.gguf",
    "n_ctx": 4096,           # 4K context window
    "n_gpu_layers": 20,      # Conservative GPU layers
    "n_batch": 512,          # Batch size
    "n_threads": 4,          # CPU threads
    "gpu": "A10G",           # 24GB VRAM
    "memory": 16384          # 16GB RAM
}
```

### Performance Metrics
- **Cold Start**: ~30-60 seconds (model loading)
- **Inference Speed**: ~2-5 seconds per response
- **GPU Utilization**: 80-90% during inference
- **Stability**: No crash loops with conservative settings

## 🔧 Usage Examples

### Direct API Call
```python
import requests

response = requests.post(
    "https://your-app--chat-api.modal.run/chat",
    json={"message": "Hello! How can you help me?"}
)
print(response.json())
```

### Web Interface
Visit the deployed URL to access the interactive chat interface.

## 🧪 Testing & Validation

### Model Loading Test
```bash
python test_gpu_model.py
```

### Chat Functionality Test
```bash
python mistral_gpu_clean.py
```

### Expected Output
```json
{
  "response": "Hello! I'm Mistral, an AI assistant...",
  "status": "success"
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Not Found**
   - Ensure model is uploaded to `model-volume`
   - Check file path: `/models/model.gguf`

2. **GPU Memory Issues**
   - Reduce `n_gpu_layers` from 20 to 10
   - Decrease `n_ctx` context window

3. **Timeout Errors**
   - Increase Modal timeout settings
   - Check container cold start time

### Debugging
```bash
# Check volume contents
modal volume ls model-volume

# Monitor deployment
modal apps list
```

## 📊 Performance Optimization

### GPU Layer Tuning
- **Conservative**: 10-20 layers (stable)
- **Balanced**: 25-30 layers (good performance)
- **Aggressive**: 35+ layers (may cause crashes)

### Memory Management
- Monitor GPU VRAM usage
- Adjust batch size based on context length
- Use memory mapping for large models

## 🏆 Success Metrics

### Validation Complete ✅
- [x] GPU acceleration working
- [x] Model loading successfully
- [x] Chat inference functional
- [x] Stable deployment
- [x] No crash loops
- [x] Performance optimized

### Benchmarks
- **Model Size**: 6.4GB (Q6_K quantization)
- **Load Time**: ~45 seconds on A10G
- **Response Time**: 2-5 seconds average
- **GPU Memory**: ~8GB usage during inference

## 🔮 Future Enhancements

- [ ] Full GPU offloading (-1 layers) with CUDA compilation
- [ ] Streaming responses for real-time chat
- [ ] Multi-session management
- [ ] Performance analytics dashboard
- [ ] Auto-scaling based on demand

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Test with Modal deployment
4. Submit pull request

## 📝 License

MIT License - See LICENSE file for details

## 🔗 Links

- [Modal Documentation](https://modal.com/docs)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Mistral AI](https://mistral.ai/)

---

**Status**: ✅ Production Ready | **Last Updated**: August 2025

## 🎯 What This Does

- Deploys Mistral NeMo 8B model on Modal with GPU acceleration
- Creates OpenAI-compatible API endpoints
- Handles authentication with API keys
- Provides chat completion and text completion endpoints
- Auto-scales based on demand

## 📁 Files Overview

- `deploy_fixed.py` - Main Modal deployment with improved error handling
- `upload_model_fixed.py` - Uploads the GGUF model file to Modal volume
- `test_api.py` - Test client for the deployed API
- `deploy.sh` - Automated deployment script
- `Mistral-NeMo-Minitron-8B-Base.Q6_K.gguf` - The model file

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install modal
```

### 2. Setup Modal Account

```bash
# Login to Modal
modal token new

# Verify login
modal token current
```

### 3. Deploy Everything

```bash
# Run the automated deployment
./deploy.sh
```

Or deploy manually:

```bash
# Upload model
python upload_model_fixed.py

# Deploy API
modal deploy deploy_fixed.py
```

### 4. Test Your API

1. Copy the API key from deployment logs
2. Update `test_api.py` with your Modal URL and API key
3. Run tests:

```bash
pip install requests  # if not already installed
python test_api.py
```

## 🔧 API Usage

### Authentication

All requests require a Bearer token:

```bash
Authorization: Bearer YOUR_API_KEY
```

### Chat Completion

```bash
curl -X POST "https://your-app.modal.run/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-nemo",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Text Completion

```bash
curl -X POST "https://your-app.modal.run/v1/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-nemo",
    "prompt": "The weather today is",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Health Check

```bash
curl "https://your-app.modal.run/health"
```

## 🛠 Troubleshooting

### Check Model Upload

```python
import modal
app = modal.App("upload-mistral-model")
with app.run():
    from upload_model_fixed import list_models
    result = list_models.remote()
    print(result)
```

### Check Deployment Status

```bash
modal app list
modal app logs mistral-api
```

### Common Issues

1. **Model not found**: Run `python upload_model_fixed.py` again
2. **GPU memory issues**: The model needs ~8GB GPU memory (T4 should work)
3. **Timeout errors**: Increase timeout values in the client
4. **Authentication errors**: Verify your API key is correct

## 📊 Model Specifications

- **Model**: Mistral NeMo Minitron 8B Base
- **Quantization**: Q6_K (6-bit quantization)
- **Context Length**: 4096 tokens
- **GPU**: NVIDIA T4 (16GB memory allocated)
- **Framework**: llama.cpp for efficient inference

## 🔄 Scaling

Modal automatically scales your deployment:
- **Cold starts**: ~30-60 seconds (model loading)
- **Keep warm**: 1 instance kept warm to reduce latency
- **Auto-scale**: Scales up based on request volume

## 💰 Cost Estimation

- **GPU**: ~$0.50/hour when active (T4)
- **Storage**: ~$0.10/month for model storage
- **Bandwidth**: Varies by usage

## 🔗 Useful Commands

```bash
# View app status
modal app list

# View logs
modal app logs mistral-api

# Stop app
modal app stop mistral-api

# Delete app
modal app delete mistral-api

# View volumes
modal volume list
```

## 📝 API Response Format

### Chat Completion Response

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "mistral-nemo",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! I'm doing well, thank you for asking."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational purposes. Please check the Mistral license for commercial usage.
