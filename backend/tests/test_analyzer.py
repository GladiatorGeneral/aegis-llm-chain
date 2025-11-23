import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from engines.lightweight_analyzer import LightweightAnalyzer
from engines.analyzer import AnalysisRequest, AnalysisTask
from engines.cognitive import UnifiedCognitiveEngine, CognitiveRequest, CognitiveObjective

async def test_lightweight_analyzer():
    """Test the lightweight analyzer"""
    print("🧪 TESTING LIGHTWEIGHT ANALYZER...")
    
    analyzer = LightweightAnalyzer()
    
    test_cases = [
        (AnalysisTask.SENTIMENT_ANALYSIS, "I absolutely love this product! It's amazing."),
        (AnalysisTask.ENTITY_EXTRACTION, "John Smith works at Google in London."),
        (AnalysisTask.TEXT_CLASSIFICATION, "The new AI model achieves state-of-the-art performance."),
        (AnalysisTask.QUESTION_ANSWERING, "What is the capital of France?"),
    ]
    
    for task, input_text in test_cases:
        print(f"\n--- Testing {task.value} ---")
        print(f"Input: {input_text}")
        
        request = AnalysisRequest(
            task=task,
            input_data=input_text,
            require_reasoning_chain=True
        )
        
        response = await analyzer.analyze(request)
        
        print(f"Result: {response.result}")
        print(f"Confidence: {response.confidence:.2f} ({response.confidence_level.value})")
        print(f"Processing Time: {response.processing_time:.2f}s")
        
        if response.reasoning_chain:
            print("Reasoning Chain:")
            for step in response.reasoning_chain:
                print(f"  - {step}")

async def test_cognitive_engine():
    """Test the unified cognitive engine"""
    print("\n🧠 TESTING UNIFIED COGNITIVE ENGINE...")
    
    engine = UnifiedCognitiveEngine()
    
    # Test multi-objective processing
    test_requests = [
        {
            "input": "Analyze the sentiment of this text and then generate a response.",
            "objectives": [CognitiveObjective.ANALYZE, CognitiveObjective.GENERATE],
            "description": "Analysis + Generation"
        },
        {
            "input": "Extract entities from this text and evaluate the content.",
            "objectives": [CognitiveObjective.ANALYZE, CognitiveObjective.EVALUATE],
            "description": "Analysis + Evaluation"
        }
    ]
    
    for test_case in test_requests:
        print(f"\n--- Testing {test_case['description']} ---")
        print(f"Input: {test_case['input']}")
        print(f"Objectives: {[obj.value for obj in test_case['objectives']]}")
        
        request = CognitiveRequest(
            input=test_case["input"],
            objectives=test_case["objectives"],
            use_lightweight=True
        )
        
        response = await engine.process(request)
        
        print(f"Final Output: {response.final_output}")
        print(f"Processing Sequence: {response.processing_sequence}")
        print("Reasoning Trace:")
        for trace in response.reasoning_trace:
            print(f"  - {trace['step']}: {trace['result']} (confidence: {trace.get('confidence', 'N/A')})")
        
        print(f"Results Keys: {list(response.results.keys())}")

async def test_intelligent_routing():
    """Test the engine's intelligent task routing"""
    print("\n🎯 TESTING INTELLIGENT TASK ROUTING...")
    
    engine = UnifiedCognitiveEngine()
    
    test_inputs = [
        "What is the sentiment of this sentence?",
        "Generate Python code for a fibonacci function",
        "Extract all person names from this paragraph",
        "Classify this text into categories"
    ]
    
    for input_text in test_inputs:
        print(f"\nInput: {input_text}")
        
        # Let the engine determine the best approach
        request = CognitiveRequest(
            input=input_text,
            objectives=[CognitiveObjective.ANALYZE, CognitiveObjective.GENERATE],
            use_lightweight=True
        )
        
        response = await engine.process(request)
        
        # Show which engines were used
        analysis_used = "analysis" in response.results
        generation_used = "generation" in response.results
        
        print(f"Analysis Used: {analysis_used}")
        print(f"Generation Used: {generation_used}")
        print(f"Primary Output Type: {type(response.final_output).__name__}")

if __name__ == "__main__":
    print("🚀 STARTING ANALYSIS ENGINE TESTS...")
    print("Universal Analysis Engine + Cognitive Orchestration\n")
    
    asyncio.run(test_lightweight_analyzer())
    asyncio.run(test_cognitive_engine())
    asyncio.run(test_intelligent_routing())
    
    print("\n✅ ANALYSIS ENGINE TESTS COMPLETED!")
