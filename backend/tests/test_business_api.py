"""
Test Business Use Cases API
"""
import requests
import json

BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "test-token"

def test_business_domains():
    """Test getting business domains"""
    print("\n=== Testing Business Domains ===")
    response = requests.get(f"{BASE_URL}/api/v1/business/domains")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    print(f"Domains: {json.dumps(data.get('data', []), indent=2)}")
    return data.get('data', [])

def test_domain_use_cases(domain_id):
    """Test getting use cases for a domain"""
    print(f"\n=== Testing Use Cases for {domain_id} ===")
    response = requests.get(f"{BASE_URL}/api/v1/business/domains/{domain_id}/use-cases")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    print(f"Use Cases: {json.dumps(data.get('data', []), indent=2)[:500]}...")
    return data.get('data', [])

def test_domain_models(domain_id):
    """Test getting models for a domain"""
    print(f"\n=== Testing Models for {domain_id} ===")
    response = requests.get(f"{BASE_URL}/api/v1/business/domains/{domain_id}/models")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    models = data.get('data', [])
    print(f"Found {len(models)} models")
    if models:
        print(f"First model: {json.dumps(models[0], indent=2)}")
    return models

def test_business_generation():
    """Test business-focused generation"""
    print("\n=== Testing Business Generation ===")
    
    request_data = {
        "business_domain": "healthcare",
        "use_case": "Medical Record Analysis",
        "prompt": "Summarize patient symptoms: fever, cough, fatigue for 3 days",
        "business_context": {
            "industry": "healthcare",
            "use_case": "medical_documentation",
            "tone": "professional",
            "audience": "medical_staff"
        },
        "parameters": {
            "max_tokens": 200
        }
    }
    
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    response = requests.post(
        f"{BASE_URL}/api/v1/business/generate",
        json=request_data,
        headers=headers
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    if data.get('success'):
        print(f"Generated Content: {json.dumps(data.get('data', {}), indent=2)[:500]}...")
    else:
        print(f"Error: {data.get('error')}")

def test_business_analysis():
    """Test business-focused analysis"""
    print("\n=== Testing Business Analysis ===")
    
    request_data = {
        "business_domain": "finance",
        "use_case": "Financial Sentiment Analysis",
        "input_data": "The stock market showed strong gains today with tech stocks leading the rally.",
        "business_context": {
            "industry": "finance",
            "use_case": "sentiment_analysis"
        },
        "require_reasoning_chain": True
    }
    
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    response = requests.post(
        f"{BASE_URL}/api/v1/business/analyze",
        json=request_data,
        headers=headers
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    if data.get('success'):
        print(f"Analysis Result: {json.dumps(data.get('data', {}), indent=2)[:500]}...")
    else:
        print(f"Error: {data.get('error')}")

def test_model_search():
    """Test model search"""
    print("\n=== Testing Model Search ===")
    query = "sentiment"
    response = requests.get(f"{BASE_URL}/api/v1/business/models/search?query={query}")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    results = data.get('data', [])
    print(f"Found {len(results)} models matching '{query}'")
    if results:
        print(f"First result: {json.dumps(results[0], indent=2)}")

if __name__ == "__main__":
    print("=" * 70)
    print("BUSINESS USE CASES API TEST")
    print("=" * 70)
    
    # Test all endpoints
    try:
        # 1. Get domains
        domains = test_business_domains()
        
        # 2. Test first domain if available
        if domains:
            first_domain = domains[0]['id']
            test_domain_use_cases(first_domain)
            test_domain_models(first_domain)
        
        # 3. Test generation
        test_business_generation()
        
        # 4. Test analysis
        test_business_analysis()
        
        # 5. Test search
        test_model_search()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
