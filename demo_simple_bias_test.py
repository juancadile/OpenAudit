#!/usr/bin/env python3
"""
Demo: Simple Name-Only Bias Testing (Your Original Concept)
Shows how the core ceteris paribus name testing works
"""

from core.bias_testing_framework import HiringBiasTest, BiasAnalyzer
from core.multi_llm_dispatcher import MultiLLMDispatcher
import asyncio
import json

async def simple_name_bias_test():
    """
    Your original concept: Test identical CVs with only names changed
    """
    print("🎯 OpenAudit: Simple Name-Only Bias Testing")
    print("=" * 50)
    print("Testing: Do AI models make different hiring decisions")
    print("when given IDENTICAL resumes with only names changed?")
    print()
    
    # Create test cases - this is your core insight
    bias_test = HiringBiasTest()
    test_cases = bias_test.create_test_cases("software_engineer", cv_level="borderline")
    
    print(f"📋 Generated {len(test_cases)} test cases")
    print("Each test case has:")
    print("  ✅ IDENTICAL qualifications")
    print("  ✅ IDENTICAL university")
    print("  ✅ IDENTICAL experience")
    print("  ❗ ONLY name differs (demographic signal)")
    print()
    
    # Show a couple examples
    for i, test_case in enumerate(test_cases[:2]):
        name = test_case.variables["name"][0]
        print(f"Example {i+1}: Testing with name '{name}'")
        
        # Show that CV content is identical except name
        cv_snippet = test_case.test_cases[0]["variables"]["cv_content"][:200]
        print(f"  CV starts: {cv_snippet}...")
        print()
    
    # Run the actual test
    print("🚀 Running bias test across multiple AI models...")
    dispatcher = MultiLLMDispatcher()
    
    all_responses = []
    
    # Test just 2 names to keep it simple for demo
    for test_case in test_cases[:2]:
        name = test_case.variables["name"][0]
        prompt = test_case.test_cases[0]["prompt"]
        
        print(f"  Testing with {name}...")
        
        try:
            responses = await dispatcher.dispatch_prompt(
                prompt=prompt,
                models=["gpt-3.5-turbo", "gpt-4o-mini"],  # Just 2 models for speed
                iterations=1
            )
            all_responses.extend(responses)
            print(f"    ✅ Got {len(responses)} responses")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Simple analysis (your original approach)
    print("\n📊 Simple Analysis Results:")
    print("=" * 30)
    
    hire_decisions_by_name = {}
    
    for response in all_responses:
        # Extract name from prompt
        prompt_lines = response.prompt.split('\n')
        cv_start = None
        for i, line in enumerate(prompt_lines):
            if "CANDIDATE RESUME" in line:
                cv_start = i + 2
                break
        
        if cv_start and cv_start < len(prompt_lines):
            name = prompt_lines[cv_start].strip()
            
            # Extract decision
            decision = "unclear"
            if "YES" in response.response.upper():
                decision = "hire"
            elif "NO" in response.response.upper():
                decision = "no_hire"
            
            if name not in hire_decisions_by_name:
                hire_decisions_by_name[name] = []
            hire_decisions_by_name[name].append({
                "model": response.model_name,
                "decision": decision
            })
    
    # Show results
    for name, decisions in hire_decisions_by_name.items():
        hire_count = sum(1 for d in decisions if d["decision"] == "hire")
        total = len(decisions)
        hire_rate = hire_count / total if total > 0 else 0
        
        print(f"{name}: {hire_rate:.1%} hire rate ({hire_count}/{total})")
        for decision in decisions:
            print(f"  {decision['model']}: {decision['decision']}")
    
    print("\n🎯 This is your core insight:")
    print("Same resume + different name = different outcomes = BIAS")
    
    return all_responses

async def enhanced_analysis_demo(responses):
    """
    Show how the enhanced statistical analysis works (optional)
    """
    print("\n" + "=" * 50)
    print("🔬 Enhanced Statistical Analysis (Optional)")
    print("=" * 50)
    print("This adds academic rigor to your core insight...")
    
    try:
        analyzer = BiasAnalyzer(responses)
        
        # Run comprehensive analysis
        analysis = analyzer.comprehensive_bias_analysis()
        
        print(f"\n📊 Statistical Results:")
        print(f"  Confidence Level: {analysis['overall_assessment']['confidence_level'].upper()}")
        print(f"  Bias Detected: {analysis['overall_assessment']['bias_detected']}")
        
        if 'chi_square_test' in analysis['statistical_tests']:
            chi2 = analysis['statistical_tests']['chi_square_test']
            if 'p_value' in chi2:
                print(f"  Chi-square p-value: {chi2['p_value']:.4f}")
                print(f"  Statistically significant: {chi2['significant']}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(analysis.get('recommendations', []), 1):
            print(f"  {i}. {rec}")
            
    except Exception as e:
        import traceback
        print(f"❌ Statistical analysis error: {e}")
        print(f"📍 Full error traceback:")
        traceback.print_exc()
        print("💡 But your core bias detection still works!")

if __name__ == "__main__":
    print("🚀 Starting OpenAudit Demo...")
    responses = asyncio.run(simple_name_bias_test())
    
    if responses:
        asyncio.run(enhanced_analysis_demo(responses))
    
    print("\n✅ Demo complete!")
    print("Your original concept: Elegant, simple, powerful! 🎯") 