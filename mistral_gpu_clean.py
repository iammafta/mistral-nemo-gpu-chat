import modal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple image with llama-cpp-python
image = modal.Image.debian_slim().pip_install(
    "llama-cpp-python[server]",
    "fastapi", 
    "uvicorn"
)

app = modal.App("mistral-gpu-working")

# Use existing model volume
volume = modal.Volume.from_name("model-volume")

@app.cls(
    gpu="A10G",
    timeout=600,
    memory=16384,
    volumes={"/models": volume},
    image=image,
)
class MistralChat:
    
    @modal.enter()
    def load_model(self):
        """Load model with GPU acceleration"""
        from llama_cpp import Llama
        import os
        
        model_path = "/models/model.gguf"
        logger.info(f"🚀 Loading model: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Conservative GPU settings that should work
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,        # 4K context
            n_gpu_layers=20,   # Conservative GPU layers
            n_batch=512,
            n_threads=4,
            verbose=True
        )
        
        logger.info("✅ Model loaded successfully!")
        
        # Test inference
        test = self.llm("Hello", max_tokens=5)
        logger.info(f"✅ Test inference: {test}")
    
    @modal.method()
    def chat(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, stop: list | None = None):
        """Generate response from a generic prompt (stateless).

        Accepts a raw prompt and optional generation parameters so this service is not chat-only.
        """
        try:
            if stop is None:
                stop = ["</s>", "[INST]", "[/INST]"]

            wrapped = f"[INST] {prompt} [/INST]"
            response = self.llm(
                wrapped,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )

            if isinstance(response, dict):
                # llama-cpp-python returns {'choices':[{'text': ...}], ...}
                text = response.get("choices", [{}])[0].get("text", "").strip()
            else:
                text = str(response).strip()

            return {
                "prompt": prompt,
                "response": text,
                "raw": response,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

@app.function(image=image)
@modal.web_endpoint(method="GET")
def web():
    """Simple web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Mistral GPU Chat</title></head>
    <body style="font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px;">
        <h1>🚀 Mistral GPU Chat</h1>
        <div style="background: #f0f0f0; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <h3>💡 Status: GPU-Accelerated</h3>
            <p>🎮 Hardware: A10G (24GB VRAM)</p>
            <p>🧠 Model: Mistral-NeMo-8B (Q6_K)</p>
            <p>⚡ GPU Layers: 20 (Conservative)</p>
        </div>
        
        <div>
            <input type="text" id="msg" placeholder="Type your message..." style="width: 70%; padding: 10px;">
            <button onclick="chat()" style="padding: 10px 20px; background: #007cba; color: white; border: none;">Send</button>
        </div>
        
        <div id="result" style="margin-top: 20px; padding: 15px; background: white; border: 1px solid #ddd; border-radius: 5px;"></div>
        
        <script>
            async function chat() {
                const msg = document.getElementById('msg').value;
                if (!msg) return;
                
                document.getElementById('result').innerHTML = '🚀 Generating...';
                
                try {
                    const resp = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: msg})
                    });
                    const data = await resp.json();
                    
                    if (data.status === 'success') {
                        document.getElementById('result').innerHTML = `
                            <strong>Response:</strong><br>${data.response}
                        `;
                    } else {
                        document.getElementById('result').innerHTML = `❌ Error: ${data.error}`;
                    }
                } catch (e) {
                    document.getElementById('result').innerHTML = `❌ Error: ${e.message}`;
                }
                
                document.getElementById('msg').value = '';
            }
            
            document.getElementById('msg').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') chat();
            });
        </script>
    </body>
    </html>
    """

@app.function(image=image)
@modal.web_endpoint(method="POST")
def chat_api(payload: dict = None, message: str = None):
    """Generic API endpoint: accept JSON body or legacy query 'message'.

    Prefer JSON body; fall back to query param for backward compatibility.
    """
    logger.info(f"chat_api called; payload present={payload is not None}, message={message}")

    body = payload if isinstance(payload, dict) else {}

    # If body absent, fall back to query param
    prompt = (body.get("prompt") or body.get("message") or body.get("input")) if body else None
    if not prompt and message:
        prompt = message

    max_tokens = (body.get("max_tokens") if body else None) or 100
    temperature = (body.get("temperature") if body else None) or 0.7
    stop = (body.get("stop") if body else None)

    if not prompt:
        return {"status": "failed", "error": "missing 'prompt' (or 'message'/'input')"}

    chat_instance = MistralChat()
    try:
        fut = chat_instance.chat.remote(prompt, max_tokens, temperature, stop)
        result = fut.get(timeout=600)
        return result
    except Exception as e:
        logger.error(f"chat_api error: {e}")
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    # Test locally
    with app.run():
        chat_instance = MistralChat()
        
        print("🧪 Testing chat...")
        result = chat_instance.chat.remote("Hello! Introduce yourself briefly.")
        print(f"Result: {result}")
