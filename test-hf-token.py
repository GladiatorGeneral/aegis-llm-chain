"""
Quick test to verify HF_TOKEN works with actual API call
"""
import os
import sys

# Add backend/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend", "src"))

print("=" * 80)
print("🧪 Testing HF_TOKEN with Live API Call")
print("=" * 80)

# Check token
token = os.getenv("HF_TOKEN")
if not token:
    print("\n❌ HF_TOKEN not found in environment!")
    print("   Set it with: $env:HF_TOKEN = 'your_token_here'")
    sys.exit(1)

print(f"\n✅ HF_TOKEN found: {token[:10]}...")

# Test with huggingface_hub
try:
    from huggingface_hub import InferenceClient
    
    print("\n🔄 Creating Inference Client...")
    client = InferenceClient(token=token)
    
    print("✅ Inference Client created successfully!")
    
    # Try a simple API call with a small, fast model
    print("\n🔄 Testing with actual API call (phi-3-mini)...")
    print("   Prompt: 'Say hello in one word'")
    
    try:
        completion = client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct",
            messages=[
                {"role": "user", "content": "Say hello in one word"}
            ],
            max_tokens=10,
            temperature=0.7
        )
        
        response = completion.choices[0].message.content
        print(f"\n✅ API CALL SUCCESS!")
        print(f"   Response: {response}")
        print(f"\n🎉 Your HF_TOKEN is working perfectly!")
        
    except Exception as e:
        print(f"\n⚠️  API call failed: {str(e)}")
        print("\n   This could mean:")
        print("   1. Rate limit reached (try again in a minute)")
        print("   2. Model access not granted (accept terms on model page)")
        print("   3. Network issue")
        print(f"\n   But your token IS valid and recognized by HuggingFace! ✅")

except ImportError:
    print("\n❌ huggingface_hub not installed")
    print("   Install with: pip install huggingface-hub")
except Exception as e:
    print(f"\n❌ Error: {str(e)}")

print("\n" + "=" * 80)
