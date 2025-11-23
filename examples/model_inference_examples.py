"""
Model Inference Examples
Demonstrates how to use the organized model inference system
"""
import asyncio
import os
import sys

# Add backend/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "src"))

from models.inference_client import inference_client
from models.registry import model_registry

async def example_1_basic_chat():
    """Example 1: Basic Chat (Your Original Code Enhanced)"""
    print("=" * 80)
    print("🚀 Example 1: Basic Chat Completion")
    print("=" * 80)
    
    try:
        completion = await inference_client.chat_completion(
            model_key="cogito-671b",  # Using model key instead of raw ID
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"\n✅ Response from {completion['model_key']}:")
        print(f"📝 {completion['content']}")
        print(f"\n📊 Usage Statistics:")
        print(f"   - Prompt tokens: {completion['usage']['prompt_tokens']}")
        print(f"   - Completion tokens: {completion['usage']['completion_tokens']}")
        print(f"   - Total tokens: {completion['usage']['total_tokens']}")
        print(f"   - Finish reason: {completion['finish_reason']}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

async def example_2_complex_conversation():
    """Example 2: Multi-turn Conversation"""
    print("\n" + "=" * 80)
    print("🚀 Example 2: Multi-turn Conversation")
    print("=" * 80)
    
    try:
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful AI assistant that explains concepts simply."
            },
            {
                "role": "user",
                "content": "Can you explain quantum computing in one sentence?"
            },
            {
                "role": "assistant", 
                "content": "Quantum computing uses quantum bits or qubits, which can exist in multiple states simultaneously through superposition."
            },
            {
                "role": "user",
                "content": "How is that different from classical computing?"
            }
        ]
        
        completion = await inference_client.chat_completion(
            model_key="mistral-7b",  # Try different model
            messages=messages,
            temperature=0.8,
            max_tokens=200
        )
        
        print(f"\n✅ Response from {completion['model_key']}:")
        print(f"📝 {completion['content']}")
        print(f"\n📊 Total tokens used: {completion['usage']['total_tokens']}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

async def example_3_model_comparison():
    """Example 3: Compare Different Models"""
    print("\n" + "=" * 80)
    print("🚀 Example 3: Model Comparison")
    print("=" * 80)
    
    prompt = "Write a haiku about artificial intelligence"
    
    # Test multiple models
    models_to_test = ["cogito-671b", "mistral-7b", "phi-3-mini"]
    
    results = []
    
    for model_key in models_to_test:
        print(f"\n🎯 Testing {model_key}...")
        
        try:
            model_config = model_registry.get_model(model_key)
            if not model_config:
                print(f"   ⚠️  Model not found in registry")
                continue
            
            completion = await inference_client.text_completion(
                model_key=model_key,
                prompt=prompt,
                max_tokens=100,
                temperature=0.9
            )
            
            print(f"   ✅ {model_config.name}")
            print(f"   📝 {completion['content']}")
            print(f"   📊 Tokens: {completion['usage']['total_tokens']}")
            
            results.append({
                "model": model_key,
                "response": completion['content'],
                "tokens": completion['usage']['total_tokens']
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Summary
    print("\n📊 Comparison Summary:")
    print(f"   Models tested: {len(results)}")
    if results:
        avg_tokens = sum(r['tokens'] for r in results) / len(results)
        print(f"   Average tokens: {avg_tokens:.0f}")

async def example_4_embeddings():
    """Example 4: Generate Embeddings"""
    print("\n" + "=" * 80)
    print("🚀 Example 4: Text Embeddings")
    print("=" * 80)
    
    texts = [
        "Artificial intelligence is transforming the world",
        "Machine learning helps computers learn from data",
        "Deep learning uses neural networks"
    ]
    
    try:
        result = await inference_client.embedding(
            model_key="bge-large",
            texts=texts
        )
        
        print(f"\n✅ Generated embeddings for {result['count']} texts")
        print(f"📊 Embedding dimension: {result['dimension']}")
        print(f"📝 Model: {result['model']}")
        
        print("\n📐 Embedding vectors (first 5 dimensions):")
        for i, (text, embedding) in enumerate(zip(texts, result['embeddings'])):
            print(f"   {i+1}. \"{text[:50]}...\"")
            print(f"      [{', '.join(f'{x:.4f}' for x in embedding[:5])}, ...]")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

async def example_5_list_models():
    """Example 5: Browse Available Models"""
    print("\n" + "=" * 80)
    print("🚀 Example 5: Browse Available Models")
    print("=" * 80)
    
    try:
        # Get all models
        all_models = inference_client.get_available_models()
        
        print(f"\n📊 Total models available: {len(all_models)}")
        
        # Group by type
        by_type = {}
        for model in all_models:
            model_type = model['type']
            if model_type not in by_type:
                by_type[model_type] = []
            by_type[model_type].append(model)
        
        print("\n📋 Models by Type:")
        for model_type, models in by_type.items():
            print(f"\n   {model_type.upper()} ({len(models)} models):")
            for model in models:
                local_tag = " [LOCAL]" if model['is_local'] else ""
                quant_tag = f" [{model['quantization']}]" if model['quantization'] else ""
                print(f"      • {model['key']}: {model['name']}{local_tag}{quant_tag}")
                print(f"        Tasks: {', '.join(model['supported_tasks'])}")
                print(f"        Context: {model['context_length']} | Max tokens: {model['max_tokens']}")
        
        # Search for coding models
        print("\n\n🔍 Searching for 'coding' models:")
        coding_models = model_registry.search_models("coding")
        for config in coding_models:
            # Find key
            model_key = None
            for key, cfg in model_registry._registry.items():
                if cfg == config:
                    model_key = key
                    break
            if model_key:
                print(f"   • {model_key}: {config.name}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

async def example_6_model_health():
    """Example 6: Check Model Health"""
    print("\n" + "=" * 80)
    print("🚀 Example 6: Model Health Checks")
    print("=" * 80)
    
    models_to_check = ["cogito-671b", "mistral-7b", "phi-3-mini"]
    
    print("\n🏥 Checking model health...")
    
    for model_key in models_to_check:
        print(f"\n   Checking {model_key}...")
        
        try:
            # Test with simple prompt
            result = await inference_client.chat_completion(
                model_key=model_key,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            
            print(f"   ✅ Healthy - Response received")
            
        except Exception as e:
            print(f"   ❌ Unhealthy - {str(e)}")

async def example_7_streaming_simulation():
    """Example 7: Batch Processing Simulation"""
    print("\n" + "=" * 80)
    print("🚀 Example 7: Batch Processing Multiple Prompts")
    print("=" * 80)
    
    prompts = [
        "What is AI?",
        "Explain machine learning",
        "What is deep learning?"
    ]
    
    print(f"\n📦 Processing {len(prompts)} prompts...")
    
    # Process sequentially (could be made parallel)
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   {i}. Prompt: \"{prompt}\"")
        
        try:
            result = await inference_client.text_completion(
                model_key="phi-3-mini",  # Fast model
                prompt=prompt,
                max_tokens=50,
                temperature=0.7
            )
            
            print(f"      Response: {result['content'][:100]}...")
            print(f"      Tokens: {result['usage']['total_tokens']}")
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")

async def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "🎪 AEGIS Model Inference Examples" + " " * 24 + "║")
    print("║" + " " * 18 + "Organized Model Management System" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Check if HF_TOKEN is set
    if not os.getenv("HF_TOKEN"):
        print("\n⚠️  WARNING: HF_TOKEN environment variable not set!")
        print("   Set it with: export HF_TOKEN=your_token_here")
        print("   Get token from: https://huggingface.co/settings/tokens")
        return
    
    print("\n✅ HF_TOKEN detected - proceeding with examples")
    
    try:
        # Run examples
        await example_1_basic_chat()
        await example_2_complex_conversation()
        await example_3_model_comparison()
        await example_4_embeddings()
        await example_5_list_models()
        await example_6_model_health()
        await example_7_streaming_simulation()
        
        print("\n" + "=" * 80)
        print("🎉 All Examples Completed!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
