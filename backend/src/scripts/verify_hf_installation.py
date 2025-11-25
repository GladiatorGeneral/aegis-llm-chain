#!/usr/bin/env python3
"""
Verify Hugging Face installation and model loading
"""
import os
import sys

def verify_installation():
    print("🔍 Verifying Hugging Face Installation...")
    print("=" * 60)
    
    # Check Python version
    print(f"\n🐍 Python Version: {sys.version}")
    
    # Check core packages
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: True")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("💻 CUDA: Not available (using CPU)")
    except ImportError as e:
        print(f"❌ PyTorch: Not installed ({e})")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers: Not installed")
        return False
    
    try:
        import huggingface_hub
        print(f"✅ Hugging Face Hub: {huggingface_hub.__version__}")
    except ImportError:
        print("❌ Hugging Face Hub: Not installed")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError:
        print("⚠️  Datasets: Not installed (optional)")
    
    try:
        import accelerate
        print(f"✅ Accelerate: {accelerate.__version__}")
    except ImportError:
        print("⚠️  Accelerate: Not installed (optional)")
    
    # Check authentication
    print(f"\n🔐 Authentication:")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        print(f"✅ Hugging Face Token: Found ({len(token)} characters)")
    else:
        print("⚠️  Hugging Face Token: Not found")
        print("   Some models may require authentication")
        print("   Get token from: https://huggingface.co/settings/tokens")
    
    # Check cache directories
    print(f"\n📁 Cache Configuration:")
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"   HF_HOME: {hf_home}")
    
    transformers_cache = os.getenv("TRANSFORMERS_CACHE", hf_home)
    print(f"   TRANSFORMERS_CACHE: {transformers_cache}")
    
    # Test model loading
    print(f"\n🧪 Testing Model Loading...")
    try:
        from transformers import pipeline
        print("   Loading sentiment analysis pipeline...")
        
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        result = classifier("Hugging Face is amazing!")[0]
        print(f"✅ Model test successful!")
        print(f"   Test result: {result['label']} (confidence: {result['score']:.4f})")
        
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False
    
    print(f"\n{'=' * 60}")
    print("🎉 Hugging Face installation verified successfully!")
    print("\n✅ Next steps:")
    print("   1. Set HF_TOKEN in your .env file if not already set")
    print("   2. Run the server: python backend/src/main.py")
    print("   3. Test with: python backend/src/scripts/test_model_loading.py")
    
    return True

if __name__ == "__main__":
    try:
        success = verify_installation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
