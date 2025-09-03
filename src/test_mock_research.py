#!/usr/bin/env python3
"""
Test script for Mock Research Service
Tests the research retry logic without full agent overhead
"""

import sys
import os
sys.path.append('src')

from mock_research_service import MockResearchService

def test_mock_research():
    """Test the mock research service functionality"""
    
    print("🧪 Testing Mock Research Service")
    print("=" * 50)
    
    # Create mock research service
    research_service = MockResearchService()
    
    # Test control options (similar to what agent would provide)
    test_control_options = [
        {"power": 250, "dwell_0": 60, "dwell_1": 80},
        {"power": 275, "dwell_0": 70, "dwell_1": 90},
        {"power": 225, "dwell_0": 50, "dwell_1": 70}
    ]
    
    test_planned_controls = [
        {"power": 200, "dwell_0": 40, "dwell_1": 60},
        {"power": 300, "dwell_0": 80, "dwell_1": 100}
    ]
    
    print("\n🔬 Test 1: First request (should perform research)")
    result1 = research_service.get_research_background_strict_first_only(
        is_first_request=True,
        control_options=test_control_options,
        planned_controls=test_planned_controls,
        force=False
    )
    
    print(f"📊 Result 1:")
    print(f"   - Skipped: {result1.get('skipped', 'N/A')}")
    print(f"   - Citations: {len(result1.get('citations', []))}")
    print(f"   - From cache: {result1.get('from_cache', 'N/A')}")
    print(f"   - Mock data: {result1.get('mock_data', 'N/A')}")
    print(f"   - Findings length: {len(result1.get('research_findings', ''))}")
    
    print("\n🔬 Test 2: Second request (should skip research)")
    result2 = research_service.get_research_background_strict_first_only(
        is_first_request=False,
        control_options=test_control_options,
        planned_controls=test_planned_controls,
        force=False
    )
    
    print(f"📊 Result 2:")
    print(f"   - Skipped: {result2.get('skipped', 'N/A')}")
    print(f"   - Citations: {len(result2.get('citations', []))}")
    print(f"   - From cache: {result2.get('from_cache', 'N/A')}")
    print(f"   - Mock data: {result2.get('mock_data', 'N/A')}")
    
    print("\n🔬 Test 3: Forced research (should perform research)")
    result3 = research_service.get_research_background_strict_first_only(
        is_first_request=False,
        control_options=test_control_options,
        planned_controls=test_planned_controls,
        force=True
    )
    
    print(f"📊 Result 3:")
    print(f"   - Skipped: {result3.get('skipped', 'N/A')}")
    print(f"   - Citations: {len(result3.get('citations', []))}")
    print(f"   - From cache: {result3.get('from_cache', 'N/A')}")
    print(f"   - Mock data: {result3.get('mock_data', 'N/A')}")
    
    print("\n📋 Sample Citation from Test 1:")
    if result1.get('citations'):
        citation = result1['citations'][0]
        print(f"   Title: {citation.get('title', 'N/A')}")
        print(f"   Authors: {citation.get('authors', 'N/A')}")
        print(f"   Year: {citation.get('year', 'N/A')}")
        print(f"   Abstract: {citation.get('abstract', 'N/A')[:100]}...")
    
    print("\n✅ Mock Research Service Test Complete!")
    
    # Test the research retry logic
    print("\n🔄 Testing Research Retry Logic")
    print("=" * 50)
    
    # Simulate the agent's logic
    class MockLifespanContext:
        def __init__(self):
            self.research_started = False
    
    lifespan_ctx = MockLifespanContext()
    
    def test_research_flow(layer_num, lifespan_ctx, research_service):
        """Simulate the agent's research flow"""
        print(f"\n🏗️ Layer {layer_num} Processing:")
        
        # Determine if this is first request
        if hasattr(lifespan_ctx, 'research_started'):
            research_already_started = getattr(lifespan_ctx, 'research_started')
            is_first_request = not research_already_started
            print(f"🔍 Research status check: research_started={research_already_started}, is_first_request={is_first_request}")
        else:
            is_first_request = True
            print(f"🔍 No research status found - treating as first request: is_first_request={is_first_request}")
            setattr(lifespan_ctx, 'research_started', False)
        
        # Get research
        research_context = research_service.get_research_background_strict_first_only(
            is_first_request=is_first_request,
            control_options=test_control_options,
            planned_controls=test_planned_controls,
            force=False
        )
        
        # Update research status
        if is_first_request and not research_context.get("skipped", False):
            citations = research_context.get("citations", []) or research_context.get("web_sources", [])
            if citations:
                print(f"✅ Research succeeded with {len(citations)} citations - marking as started")
                setattr(lifespan_ctx, 'research_started', True)
            else:
                print(f"⚠️ Research attempt failed/empty - will retry on next layer")
        elif not is_first_request:
            print(f"📚 Skipping research (not first request) - using cached/empty results")
        
        return research_context
    
    # Test the flow across multiple layers
    for layer in [1, 2, 3, 4]:
        result = test_research_flow(layer, lifespan_ctx, research_service)
        citations_count = len(result.get('citations', []))
        print(f"   Result: {citations_count} citations, skipped={result.get('skipped', False)}")

if __name__ == "__main__":
    test_mock_research()