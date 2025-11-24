"""
Converter Engine Advanced Examples
Demonstrating true cross-modal understanding
"""
import asyncio
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.multimodal_engine import multimodal_engine
from models.registry import model_registry

async def example_1_cross_modal_reasoning():
    """True cross-modal reasoning with converter engine"""
    print("🚀 Example 1: Advanced Cross-Modal Reasoning")
    print("=" * 60)
    
    # Create sample business chart
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw chart background
    draw.rectangle([50, 50, 550, 350], outline='black', width=2)
    
    # Draw bars for quarterly sales
    quarters = [(100, 250), (200, 200), (300, 150), (400, 100)]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    for i, ((x, y), color) in enumerate(zip(quarters, colors)):
        draw.rectangle([x, y, x+70, 330], fill=color, outline='black')
        draw.text((x+10, 340), f"Q{i+1}", fill='black')
    
    # Add title
    draw.text((200, 20), "Quarterly Sales Performance", fill='black')
    draw.text((50, 360), "$0", fill='black')
    draw.text((45, 250), "$500K", fill='black')
    draw.text((45, 100), "$1.2M", fill='black')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    modalities = {
        "text": "Analyze this quarterly sales chart and provide three strategic recommendations for sustaining growth",
        "vision": img_bytes
    }
    
    result = await multimodal_engine.advanced_cross_modal_reasoning(
        model_key="llava-13b-converter",
        modalities=modalities,
        task="business_analysis",
        fusion_strategy="cross_attention"
    )
    
    print(f"\n📊 Cross-Modal Analysis:")
    print(f"   Model: {result['model']}")
    print(f"   Converter: {result.get('converter_used', 'unknown')}")
    print(f"   Confidence: {result.get('confidence', 0):.1%}")
    print(f"\n📝 Analysis Result:")
    print(f"   {result['content']}")
    print("\n")

async def example_2_contrastive_search():
    """CLIP-style cross-modal retrieval"""
    print("🚀 Example 2: Contrastive Cross-Modal Search")
    print("=" * 60)
    
    # Text-to-image search
    result = await multimodal_engine.contrastive_cross_modal_search(
        query_modality="text",
        query_data="a modern glass office building with blue sky background",
        target_modality="vision",
        top_k=3
    )
    
    print(f"\n🔍 Search Configuration:")
    print(f"   Query Modality: {result['query_modality']}")
    print(f"   Target Modality: {result['target_modality']}")
    print(f"   Search Strategy: {result['search_strategy']}")
    print(f"   Model: {result['model']}")
    
    print(f"\n📋 Top Matches:")
    for i, match in enumerate(result["top_matches"], 1):
        print(f"\n   Match {i}:")
        print(f"      Similarity Score: {match['score']:.1%}")
        print(f"      Description: {match['description']}")
        print(f"      Source: {match['metadata']['source']}")
    print("\n")

async def example_3_complex_multimodal_qa():
    """Complex QA requiring true cross-modal understanding"""
    print("🚀 Example 3: Complex Multi-Modal QA")
    print("=" * 60)
    
    # Create a complex comparative chart
    img = Image.new('RGB', (700, 450), color='white')
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((150, 20), "Revenue: Our Company vs Competitor", fill='black')
    
    # Draw grid
    for i in range(5):
        y = 100 + i * 60
        draw.line([(50, y), (650, y)], fill='lightgray')
    
    # Our company (blue bars)
    our_data = [(100, 280), (220, 240), (340, 200), (460, 160)]
    for x, y in our_data:
        draw.rectangle([x, y, x+60, 340], fill='#3498db', outline='black')
    
    # Competitor (red bars - smaller)
    comp_data = [(110, 300), (230, 280), (350, 260), (470, 240)]
    for x, y in comp_data:
        draw.rectangle([x, y, x+40, 340], fill='#e74c3c', outline='black', width=2)
    
    # Labels
    for i, x in enumerate([115, 235, 355, 475]):
        draw.text((x, 355), f"Q{i+1}", fill='black')
    
    # Legend
    draw.rectangle([550, 100, 570, 120], fill='#3498db', outline='black')
    draw.text((575, 105), "Our Company", fill='black')
    draw.rectangle([550, 130, 570, 150], fill='#e74c3c', outline='black')
    draw.text((575, 135), "Competitor", fill='black')
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    modalities = {
        "text": """Based on this comparative revenue chart, answer:
        1. What is the growth trend for our company versus the competitor?
        2. Which quarter showed the most significant competitive advantage?
        3. What strategic insights can we derive from Q3 performance?
        4. What might explain the competitor's consistent underperformance?""",
        "vision": img_bytes
    }
    
    result = await multimodal_engine.advanced_cross_modal_reasoning(
        model_key="llava-13b-converter", 
        modalities=modalities,
        task="complex_business_analysis",
        fusion_strategy="q_former"
    )
    
    print(f"\n📈 Complex Analysis:")
    print(f"   Model: {result['model']}")
    print(f"   Converter Type: {result.get('converter_used')}")
    print(f"   Modalities Used: {', '.join(result.get('modalities_used', []))}")
    
    print(f"\n🎯 Detailed Analysis:")
    print(f"   {result['content']}")
    
    print(f"\n🔍 Reasoning Process:")
    for i, step in enumerate(result.get('reasoning_steps', []), 1):
        print(f"   {i}. {step}")
    print("\n")

async def example_4_fusion_strategy_comparison():
    """Compare different fusion strategies"""
    print("🚀 Example 4: Fusion Strategy Comparison")
    print("=" * 60)
    
    # Simple visual
    img = Image.new('RGB', (400, 300), color='#FFF8DC')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple icon
    draw.ellipse([150, 100, 250, 200], fill='#FFD700', outline='black', width=2)
    draw.text((175, 220), "Success!", fill='black')
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    modalities = {
        "text": "Create a creative social media caption for this motivational image",
        "vision": img_bytes
    }
    
    strategies = {
        "linear_projection": "llava-13b-converter",
        "q_former": "blip2-converter",
        "cross_attention": "llava-13b-converter"
    }
    
    print("\n🔧 Testing Different Fusion Strategies:\n")
    
    for strategy, model_key in strategies.items():
        print(f"   Strategy: {strategy.upper()}")
        print(f"   Model: {model_key}")
        
        try:
            result = await multimodal_engine.advanced_cross_modal_reasoning(
                model_key=model_key,
                modalities=modalities,
                task="creative_captioning", 
                fusion_strategy=strategy
            )
            
            print(f"   ✓ Response Preview: {result['content'][:80]}...")
            print(f"   ✓ Confidence: {result.get('confidence', 0):.1%}")
            print(f"   ✓ Reasoning Steps: {len(result.get('reasoning_steps', []))}")
            print()
            
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            print()

async def example_5_model_discovery():
    """Discover converter-enabled models"""
    print("🚀 Example 5: Converter Model Discovery")
    print("=" * 60)
    
    # Get all converter-enabled models
    converter_models = model_registry.get_converter_enabled_models()
    
    print(f"\n📚 Total Converter-Enabled Models: {len(converter_models)}\n")
    
    for model in converter_models:
        # Find the key
        model_key = [k for k, v in model_registry._registry.items() if v == model][0]
        
        print(f"   Model: {model.name}")
        print(f"   Key: {model_key}")
        print(f"   Converter Type: {model.converter_type}")
        print(f"   Alignment Strategy: {model.alignment_strategy}")
        print(f"   Cross-Modal Tasks: {', '.join(model.supported_cross_modal_tasks[:3])}...")
        print(f"   Capabilities: {len(model.multimodal_capabilities)} modalities")
        print()
    
    # Group by converter type
    print("\n📊 Models by Converter Type:")
    for conv_type in ["linear_projection", "q_former", "cross_attention", "contrastive"]:
        models = model_registry.get_models_by_converter_type(conv_type)
        print(f"   {conv_type}: {len(models)} model(s)")
    print()

async def example_6_chart_deep_analysis():
    """Deep chart analysis with visual reasoning"""
    print("🚀 Example 6: Deep Chart Analysis")
    print("=" * 60)
    
    # Create detailed financial chart
    img = Image.new('RGB', (800, 500), color='white')
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((250, 20), "Quarterly Financial Metrics", fill='black')
    
    # Draw three different metrics
    metrics = [
        ("Revenue", [(100, 200), (200, 150), (300, 120), (400, 100)], '#2ecc71'),
        ("Costs", [(100, 250), (200, 240), (300, 235), (400, 230)], '#e74c3c'),
        ("Profit", [(100, 270), (200, 240), (300, 200), (400, 170)], '#3498db')
    ]
    
    legend_y = 100
    for name, data, color in metrics:
        # Draw line
        for i in range(len(data) - 1):
            draw.line([data[i], data[i+1]], fill=color, width=3)
        
        # Draw points
        for x, y in data:
            draw.ellipse([x-5, y-5, x+5, y+5], fill=color, outline='black')
        
        # Legend
        draw.rectangle([650, legend_y, 670, legend_y+10], fill=color)
        draw.text((675, legend_y), name, fill='black')
        legend_y += 25
    
    # Axis labels
    for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        x = 100 + i * 100
        draw.text((x-10, 290), q, fill='black')
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    modalities = {
        "text": """Perform comprehensive financial analysis:
        1. Identify all trends across the three metrics
        2. Calculate approximate growth rates
        3. Assess profit margin trajectory
        4. Identify potential risk factors
        5. Provide actionable recommendations""",
        "vision": img_bytes
    }
    
    result = await multimodal_engine.advanced_cross_modal_reasoning(
        model_key="llava-13b-converter",
        modalities=modalities,
        task="chart_analysis",
        fusion_strategy="cross_attention"
    )
    
    print(f"\n📊 Deep Chart Analysis:")
    print(f"   Converter: {result.get('converter_used')}")
    print(f"   Confidence Level: {result.get('confidence', 0):.1%}")
    
    print(f"\n💡 Analysis:")
    print(result['content'])
    print("\n")

async def main():
    """Run all converter engine examples"""
    print("\n" + "="*70)
    print(" 🎪 AEGIS LLM CHAIN - CONVERTER ENGINE ADVANCED EXAMPLES")
    print(" 🔬 Demonstrating True Cross-Modal Understanding")
    print("="*70 + "\n")
    
    examples = [
        ("Cross-Modal Business Analysis", example_1_cross_modal_reasoning),
        ("Contrastive Cross-Modal Search", example_2_contrastive_search),
        ("Complex Multi-Modal QA", example_3_complex_multimodal_qa),
        ("Fusion Strategy Comparison", example_4_fusion_strategy_comparison),
        ("Converter Model Discovery", example_5_model_discovery),
        ("Deep Chart Analysis", example_6_chart_deep_analysis)
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
            print("✅ Example completed successfully\n")
        except Exception as e:
            print(f"❌ Example failed: {str(e)}\n")
        
        print("-" * 70 + "\n")
    
    print("="*70)
    print(" 🎉 ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\n💡 Key Takeaways:")
    print("   • Converter engine enables true cross-modal understanding")
    print("   • Multiple fusion strategies for different use cases")
    print("   • Shared semantic space across modalities")
    print("   • Production-ready for business intelligence")
    print()

if __name__ == "__main__":
    asyncio.run(main())
