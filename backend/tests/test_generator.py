import asyncio
import sys
import os

# Add src directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(test_dir)
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, src_dir)

from engines.base import GenerationRequest, GenerationTask
from engines.lightweight_generator import LightweightGenerator

async def test_lightweight_generator():
    """Test the lightweight generator"""
    print("🧪 TESTING LIGHTWEIGHT GENERATOR...")
    
    generator = LightweightGenerator()
    
    test_cases = [
        (GenerationTask.CHAT, "Hello, how are you?"),
        (GenerationTask.CODE_GENERATION, "Write a Python function to calculate factorial"),
        (GenerationTask.TEXT_SUMMARIZATION, "This is a long text that needs to be summarized."),
    ]
    
    for task, prompt in test_cases:
        print(f"\n--- Testing {task.value} ---")
        print(f"Prompt: {prompt}")
        
        request = GenerationRequest(
            task=task,
            prompt=prompt,
            parameters={"max_tokens": 50}
        )
        
        response = await generator.generate(request)
        
        print(f"Response: {response.content}")
        print(f"Model: {response.model_used}")
        print(f"Latency: {response.latency:.2f}s")
        print(f"Safety Flagged: {response.safety_flagged}")

async def test_security_features():
    """Test security features"""
    print("\n🔒 TESTING SECURITY FEATURES...")
    
    generator = LightweightGenerator()
    
    # Test PII detection
    pii_prompt = "My email is test@example.com and my phone is 555-123-4567"
    request = GenerationRequest(
        task=GenerationTask.CHAT,
        prompt=pii_prompt
    )
    
    response = await generator.generate(request)
    
    print(f"Original prompt had PII: {pii_prompt}")
    print(f"Sanitized response: {response.content}")
    print(f"Safety flagged: {response.safety_flagged}")

if __name__ == "__main__":
    print("🚀 STARTING GENERATOR TESTS...")
    asyncio.run(test_lightweight_generator())
    asyncio.run(test_security_features())
    print("\n✅ ALL TESTS COMPLETED!")
