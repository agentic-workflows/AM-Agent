"""
Mock Research Service for Testing
Provides fake literature review results without making real API calls
"""

import time
import json
import hashlib
from typing import Dict, Any, List

class MockResearchService:
    """Mock research service that returns realistic fake data for testing"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = "mock-research-model"
        self.call_count = 0
        
    def get_research_background_strict_first_only(self, *, is_first_request: bool, control_options, planned_controls, force: bool = False) -> Dict[str, Any]:
        """Mock version that returns fake research data"""
        
        self.call_count += 1
        
        # Simulate the same caching logic as real service
        messages = self._build_mock_messages(control_options, planned_controls)
        prompt_hash, prompt_json = self._prompt_signature(messages)
        
        print(f"\n🧪 MOCK RESEARCH SERVICE (call #{self.call_count})")
        print(f"   📋 is_first_request: {is_first_request}")
        print(f"   🔒 force: {force}")
        print(f"   🔑 prompt_hash: {prompt_hash[:8]}...")
        
        # If not first request and not forced, skip research
        if not is_first_request and not force:
            print(f"   ⏭️  Skipping research (not first request)")
            return {
                "research_findings": "",
                "citations": [],
                "web_sources": [],
                "timestamp": 0.0,
                "parameter_context": {},
                "from_cache": False,
                "skipped": True,
                "prompt_hash": prompt_hash,
            }
        
        # Simulate some processing time
        print(f"   ⏳ Simulating research processing...")
        time.sleep(1)  # Shorter than real API call
        
        # Generate mock research results
        mock_citations = self._generate_mock_citations(control_options)
        mock_findings = self._generate_mock_findings(control_options)
        
        print(f"   ✅ Generated {len(mock_citations)} mock citations")
        print(f"   📄 Generated mock findings ({len(mock_findings)} chars)")
        
        return {
            "research_findings": mock_findings,
            "citations": mock_citations,
            "web_sources": mock_citations,  # Same as citations for mock
            "timestamp": time.time(),
            "parameter_context": self._extract_parameter_context(control_options),
            "from_cache": False,
            "skipped": False,
            "prompt_hash": prompt_hash,
            "mock_data": True  # Flag to indicate this is mock data
        }
    
    def _build_mock_messages(self, control_options, planned_controls):
        """Build messages similar to real service"""
        return [
            {"role": "system", "content": "Mock researcher"},
            {"role": "user", "content": f"Research for {len(control_options)} options"}
        ]
    
    def _prompt_signature(self, messages: List[Dict[str, str]]) -> tuple[str, str]:
        """Generate prompt hash like real service"""
        prompt_json = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        prompt_hash = hashlib.sha256(prompt_json.encode("utf-8")).hexdigest()
        return prompt_hash, prompt_json
    
    def _extract_parameter_context(self, control_options):
        """Extract parameter ranges from options"""
        if not control_options:
            return {}
        
        powers = [opt.get("power", 0) for opt in control_options]
        dwell_0s = [opt.get("dwell_0", 0) for opt in control_options]
        dwell_1s = [opt.get("dwell_1", 0) for opt in control_options]
        
        return {
            "power_range": f"{min(powers)}-{max(powers)}W",
            "dwell_0_range": f"{min(dwell_0s)}-{max(dwell_0s)}ms",
            "dwell_1_range": f"{min(dwell_1s)}-{max(dwell_1s)}ms",
            "option_count": len(control_options)
        }
    
    def _generate_mock_citations(self, control_options) -> List[Dict[str, Any]]:
        """Generate realistic mock citations"""
        
        # Get parameter context for realistic citations
        if control_options:
            avg_power = sum(opt.get("power", 0) for opt in control_options) / len(control_options)
            avg_dwell = sum(opt.get("dwell_0", 0) for opt in control_options) / len(control_options)
        else:
            avg_power = 250
            avg_dwell = 60
        
        mock_citations = [
            {
                "title": f"Optimizing Laser Power Parameters for Powder Bed Fusion: A Study on {int(avg_power)}W Range",
                "url": "https://www.sciencedirect.com/science/article/pii/S2214860423004567",
                "doi": "10.1016/j.addma.2023.103891",
                "authors": "Smith, J.; Johnson, A.; Williams, R.",
                "year": "2024",
                "abstract": f"This study investigates optimal laser power settings around {int(avg_power)}W for selective laser melting processes. Results show significant improvement in surface quality and mechanical properties when using controlled dwell times. The research demonstrates that power levels in the {int(avg_power-50)}-{int(avg_power+50)}W range provide optimal thermal management while maintaining manufacturing precision.",
                "source_note": "mock_generated"
            },
            {
                "title": f"Thermal Management in Additive Manufacturing: Dwell Time Optimization for {int(avg_dwell)}ms Cycles",
                "url": "https://www.nature.com/articles/s41598-024-58492-1",
                "doi": "10.1038/s41598-024-58492-1", 
                "authors": "Chen, L.; Rodriguez, M.; Thompson, K.",
                "year": "2024",
                "abstract": f"Investigation of dwell time parameters in the {int(avg_dwell-20)}-{int(avg_dwell+20)}ms range reveals critical thermal management insights. Optimal dwell times reduce thermal stress while maintaining build quality. The study provides guidelines for parameter selection in industrial powder bed fusion applications.",
                "source_note": "mock_generated"
            },
            {
                "title": "Recent Advances in Process Parameter Optimization for Metal 3D Printing",
                "url": "https://www.sciencedirect.com/science/article/pii/S1359646224001234",
                "doi": "10.1016/j.actamat.2024.119567",
                "authors": "Anderson, P.; Lee, S.; Brown, D.; Wilson, T.",
                "year": "2024",
                "abstract": "Comprehensive review of parameter optimization strategies in metal additive manufacturing. Analysis of 200+ studies reveals best practices for laser power, scan speed, and dwell time selection. Provides decision framework for industrial applications with emphasis on quality-cost optimization.",
                "source_note": "mock_generated"
            },
            {
                "title": "Machine Learning Approaches to Powder Bed Fusion Parameter Selection",
                "url": "https://www.mdpi.com/2075-4701/14/3/289",
                "doi": "10.3390/met14030289",
                "authors": "Kumar, R.; Zhang, Y.; Patel, N.",
                "year": "2024", 
                "abstract": "Novel machine learning algorithms for automatic parameter optimization in selective laser melting. The proposed approach reduces defects by 35% compared to traditional methods. Validation across multiple materials demonstrates robust performance for industrial deployment.",
                "source_note": "mock_generated"
            },
            {
                "title": "Quality Control and Monitoring in Metal Additive Manufacturing Processes",
                "url": "https://www.sciencedirect.com/science/article/pii/S2214860423007891",
                "doi": "10.1016/j.addma.2023.103945",
                "authors": "Garcia, M.; Liu, X.; Miller, J.",
                "year": "2024",
                "abstract": "Real-time monitoring and control strategies for maintaining consistent quality in powder bed fusion processes. Integration of thermal sensors and adaptive parameter control shows promising results for industrial applications. Recommendations for implementation provided.",
                "source_note": "mock_generated"
            }
        ]
        
        return mock_citations
    
    def _generate_mock_findings(self, control_options) -> str:
        """Generate realistic mock research findings"""
        
        if control_options:
            avg_power = sum(opt.get("power", 0) for opt in control_options) / len(control_options)
            avg_dwell_0 = sum(opt.get("dwell_0", 0) for opt in control_options) / len(control_options)
            avg_dwell_1 = sum(opt.get("dwell_1", 0) for opt in control_options) / len(control_options)
        else:
            avg_power = 250
            avg_dwell_0 = 60
            avg_dwell_1 = 80
        
        findings = f"""
EXECUTIVE SUMMARY:
Recent research in additive manufacturing process optimization shows that laser power settings around {int(avg_power)}W with dwell times of {int(avg_dwell_0)}-{int(avg_dwell_1)}ms provide optimal balance between quality and efficiency. Critical thermal management considerations must be applied to prevent defects.

KEY RECOMMENDATIONS:
• Maintain laser power within {int(avg_power-30)}-{int(avg_power+30)}W for optimal surface quality
• Use dwell time ratios between {avg_dwell_0/avg_dwell_1:.1f}:1 to {avg_dwell_1/avg_dwell_0:.1f}:1 for thermal stability
• Implement real-time monitoring to detect thermal anomalies
• Consider material-specific thermal properties when setting parameters
• Apply adaptive control strategies for complex geometries

PARAMETER OPTIMIZATION INSIGHTS:
Power Range ({int(avg_power-50)}-{int(avg_power+50)}W): Research indicates optimal thermal management occurs at moderate power levels with precise control. Higher powers risk overheating while lower powers may cause incomplete fusion.

Dwell_0 Range ({int(avg_dwell_0-20)}-{int(avg_dwell_0+20)}ms): Primary dwell time directly affects heat input and melt pool stability. Optimal settings depend on material thermal conductivity and part geometry.

Dwell_1 Range ({int(avg_dwell_1-20)}-{int(avg_dwell_1+20)}ms): Secondary dwell time critical for thermal relaxation and preventing residual stress buildup.

THERMAL MANAGEMENT CONSIDERATIONS:
Thermal management is crucial for maintaining consistent part quality. Key strategies include controlled cooling rates, optimized scan patterns, and adaptive power control based on real-time thermal feedback.

QUALITY CONTROL APPROACHES:
Latest monitoring methods include thermal imaging, acoustic emission monitoring, and machine learning-based defect prediction. Integration of multiple sensors provides comprehensive process control.

DETAILED RESEARCH FINDINGS:
Recent studies demonstrate that parameter optimization using data-driven approaches can reduce defects by 30-40% compared to traditional trial-and-error methods. Machine learning algorithms show particular promise for real-time parameter adjustment.

INDUSTRY APPLICATIONS:
Aerospace and automotive industries report significant quality improvements when implementing research-based parameter optimization. Cost savings of 15-25% achieved through reduced post-processing and improved first-time-right rates.

CITATIONS WITH ABSTRACTS (JSON):
{json.dumps(self._generate_mock_citations(control_options), indent=2)}
"""
        
        return findings.strip()