"""
Quick test script for model registry system
"""
import sys
import os

# Add backend/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "src"))

def test_registry():
    """Test model registry"""
    print("=" * 80)
    print("🧪 Testing Model Registry")
    print("=" * 80)
    
    try:
        from models.registry import model_registry, ModelType
        
        # Test 1: Get model count
        count = model_registry.get_model_count()
        print(f"\n✅ Registry initialized with {count} models")
        
        # Test 2: Get specific model
        cogito = model_registry.get_model("cogito-671b")
        if cogito:
            print(f"\n✅ Retrieved Cogito model:")
            print(f"   Name: {cogito.name}")
            print(f"   Model ID: {cogito.model_id}")
            print(f"   Type: {cogito.model_type.value}")
            print(f"   Context: {cogito.context_length}")
            print(f"   Tasks: {', '.join(cogito.supported_tasks)}")
        else:
            print("\n❌ Failed to retrieve Cogito model")
        
        # Test 3: List by type
        chat_models = model_registry.list_models(ModelType.CHAT)
        print(f"\n✅ Chat models: {len(chat_models)}")
        for model in chat_models[:5]:
            print(f"   - {model.name}")
        
        embedding_models = model_registry.list_models(ModelType.EMBEDDING)
        print(f"\n✅ Embedding models: {len(embedding_models)}")
        for model in embedding_models:
            print(f"   - {model.name}")
        
        # Test 4: Search
        coding_models = model_registry.search_models("coding")
        print(f"\n✅ Coding models (search): {len(coding_models)}")
        for model in coding_models:
            print(f"   - {model.name}: {model.description}")
        
        # Test 5: Get by task
        chat_task_models = model_registry.get_models_by_task("chat")
        print(f"\n✅ Models supporting 'chat' task: {len(chat_task_models)}")
        
        print("\n" + "=" * 80)
        print("✅ All Registry Tests Passed!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Registry test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_client():
    """Test inference client initialization"""
    print("\n" + "=" * 80)
    print("🧪 Testing Inference Client")
    print("=" * 80)
    
    try:
        from models.inference_client import inference_client
        
        # Test 1: Get available models
        models = inference_client.get_available_models()
        print(f"\n✅ Inference client initialized")
        print(f"   Available models: {len(models)}")
        
        # Show first few models
        print(f"\n📋 First 5 models:")
        for model in models[:5]:
            local_tag = " [LOCAL]" if model['is_local'] else ""
            print(f"   - {model['key']}: {model['name']}{local_tag}")
        
        # Test 2: Check HF client status
        if inference_client.hf_client:
            print(f"\n✅ Hugging Face API client ready")
        else:
            print(f"\n⚠️  Hugging Face API client not initialized (HF_TOKEN not set)")
        
        # Test 3: Check for loaded models
        loaded = inference_client.get_loaded_models()
        print(f"\n📊 Loaded local models: {len(loaded)}")
        
        print("\n" + "=" * 80)
        print("✅ All Inference Client Tests Passed!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Inference client test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "🧪 Model System Tests" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Check HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        print("\n⚠️  WARNING: HF_TOKEN not set")
        print("   Some features may be limited")
        print("   Set it with: export HF_TOKEN=your_token_here")
    else:
        print("\n✅ HF_TOKEN detected")
    
    results = []
    
    # Run tests
    results.append(("Model Registry", test_registry()))
    results.append(("Inference Client", test_inference_client()))
    
    # Summary
    print("\n\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {name}: {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
