"""
Research Summarizer Service

Extracts and summarizes the most relevant research insights for manufacturing decisions.
"""

from typing import Dict, List, Any
import re


class ResearchSummarizer:
    """Service to extract and summarize research insights for decision-making."""
    
    def __init__(self):
        pass
    
    def extract_decision_relevant_insights(self, research_context: Dict[str, Any], control_options: List[Dict], layer_number: int) -> Dict[str, Any]:
        """
        Extract only the most relevant research insights for the current decision.
        
        Args:
            research_context: Full research context from ResearchService
            control_options: Available control options to choose from
            layer_number: Current layer being processed
            
        Returns:
            Condensed research summary focused on decision-relevant insights
        """
        if not research_context or not research_context.get("research_findings"):
            return {"summary": "No research data available", "relevant_insights": [], "parameter_guidance": {}}
        
        # Extract parameter ranges from control options
        power_range = self._extract_parameter_range(control_options, "power")
        dwell_0_range = self._extract_parameter_range(control_options, "dwell_0") 
        dwell_1_range = self._extract_parameter_range(control_options, "dwell_1")
        
        research_text = research_context["research_findings"]
        
        # Extract key insights
        executive_summary = self._extract_executive_summary(research_text)
        key_recommendations = self._extract_key_recommendations(research_text)
        parameter_insights = self._extract_parameter_specific_insights(
            research_text, power_range, dwell_0_range, dwell_1_range
        )
        thermal_guidance = self._extract_thermal_guidance(research_text, layer_number)
        
        # Get most relevant citations (top 3)
        top_citations = research_context.get("citations", [])[:3]
        
        return {
            "summary": executive_summary,
            "key_recommendations": key_recommendations[:4],  # Top 4 recommendations
            "parameter_guidance": parameter_insights,
            "thermal_considerations": thermal_guidance,
            "top_citations": [{"title": c.get("title", "")[:80], "url": c.get("url", "")} for c in top_citations],
            "layer_focus": f"Layer {layer_number} specific considerations",
            "decision_context": {
                "power_options": power_range,
                "dwell_0_options": dwell_0_range, 
                "dwell_1_options": dwell_1_range
            }
        }
    
    def _extract_parameter_range(self, control_options: List[Dict], param: str) -> Dict[str, float]:
        """Extract min/max range for a parameter from control options."""
        values = [opt.get(param, 0) for opt in control_options]
        return {
            "min": min(values),
            "max": max(values),
            "options": values
        }
    
    def _extract_executive_summary(self, research_text: str) -> str:
        """Extract the executive summary from research text."""
        # Look for executive summary section
        summary_match = re.search(r'EXECUTIVE SUMMARY:?\s*\n(.*?)(?=\n\n|\nKEY RECOMMENDATIONS|\n[A-Z ]+:|\Z)', 
                                research_text, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()
            # Limit to first 300 characters
            return summary[:300] + "..." if len(summary) > 300 else summary
        
        # Fallback: extract first paragraph
        paragraphs = research_text.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 100:  # Skip short paragraphs
                return para.strip()[:300] + "..." if len(para) > 300 else para.strip()
        
        return "Research summary not available"
    
    def _extract_key_recommendations(self, research_text: str) -> List[str]:
        """Extract key recommendations from research text."""
        recommendations = []
        
        # Look for recommendations section
        rec_match = re.search(r'KEY RECOMMENDATIONS:?\s*\n(.*?)(?=\n\n|\n[A-Z ]+:|\Z)', 
                            research_text, re.DOTALL | re.IGNORECASE)
        if rec_match:
            rec_text = rec_match.group(1)
            # Extract bullet points
            bullets = re.findall(r'[•\-\*]\s*([^\n•\-\*]+)', rec_text)
            recommendations.extend([bullet.strip() for bullet in bullets if len(bullet.strip()) > 20])
        
        # If no structured recommendations found, look for general bullet points
        if not recommendations:
            bullets = re.findall(r'[•\-\*]\s*([^\n•\-\*]+)', research_text)
            recommendations = [bullet.strip() for bullet in bullets if len(bullet.strip()) > 30][:4]
        
        return recommendations
    
    def _extract_parameter_specific_insights(self, research_text: str, power_range: Dict, 
                                           dwell_0_range: Dict, dwell_1_range: Dict) -> Dict[str, str]:
        """Extract insights specific to the current parameter ranges."""
        insights = {}
        
        # Look for parameter optimization section
        param_match = re.search(r'PARAMETER OPTIMIZATION INSIGHTS:?\s*\n(.*?)(?=\n\n|\n[A-Z ]+:|\Z)', 
                              research_text, re.DOTALL | re.IGNORECASE)
        if param_match:
            param_text = param_match.group(1)
            
            # Extract power insights
            power_match = re.search(r'Power[^\n]*:?\s*([^\n]+)', param_text, re.IGNORECASE)
            if power_match:
                insights["power"] = power_match.group(1).strip()[:200]
            
            # Extract dwell insights
            dwell_0_match = re.search(r'Dwell.?0[^\n]*:?\s*([^\n]+)', param_text, re.IGNORECASE)
            if dwell_0_match:
                insights["dwell_0"] = dwell_0_match.group(1).strip()[:200]
                
            dwell_1_match = re.search(r'Dwell.?1[^\n]*:?\s*([^\n]+)', param_text, re.IGNORECASE)
            if dwell_1_match:
                insights["dwell_1"] = dwell_1_match.group(1).strip()[:200]
        
        # Add context about current options
        insights["current_context"] = f"Available power: {power_range['min']}-{power_range['max']}W, " \
                                    f"dwell_0: {dwell_0_range['min']}-{dwell_0_range['max']}ms, " \
                                    f"dwell_1: {dwell_1_range['min']}-{dwell_1_range['max']}ms"
        
        return insights
    
    def _extract_thermal_guidance(self, research_text: str, layer_number: int) -> str:
        """Extract thermal management guidance relevant to the current layer."""
        # Look for thermal management section
        thermal_match = re.search(r'THERMAL MANAGEMENT[^\n]*:?\s*\n(.*?)(?=\n\n|\n[A-Z ]+:|\Z)', 
                                research_text, re.DOTALL | re.IGNORECASE)
        if thermal_match:
            thermal_text = thermal_match.group(1).strip()
            # Look for layer-specific guidance
            layer_specific = re.search(rf'layer.?{layer_number}[^\n]*([^\n]{{50,200}})', 
                                     thermal_text, re.IGNORECASE)
            if layer_specific:
                return layer_specific.group(1).strip()
            
            # Return first relevant sentence
            sentences = thermal_text.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 50:
                    return sentence.strip()[:200] + "..."
        
        return f"Layer {layer_number} thermal considerations: Monitor heat accumulation and adjust parameters accordingly."
    
    def format_for_decision_agent(self, summarized_research: Dict[str, Any]) -> str:
        """Format the summarized research for the decision agent in a concise way."""
        formatted = []
        
        # Executive summary
        if summarized_research.get("summary"):
            formatted.append(f"SUMMARY: {summarized_research['summary']}")
        
        # Key recommendations (limit to 3)
        if summarized_research.get("key_recommendations"):
            formatted.append("KEY RECOMMENDATIONS:")
            for i, rec in enumerate(summarized_research["key_recommendations"][:3], 1):
                formatted.append(f"  {i}. {rec}")
        
        # Parameter guidance
        if summarized_research.get("parameter_guidance"):
            param_guide = summarized_research["parameter_guidance"]
            formatted.append("PARAMETER GUIDANCE:")
            for param, guidance in param_guide.items():
                if param != "current_context" and guidance:
                    formatted.append(f"  • {param.title()}: {guidance}")
        
        # Thermal considerations
        if summarized_research.get("thermal_considerations"):
            formatted.append(f"THERMAL: {summarized_research['thermal_considerations']}")
        
        # Decision context
        if summarized_research.get("decision_context"):
            ctx = summarized_research["decision_context"]
            formatted.append(f"OPTIONS: Power {ctx['power_options']['min']}-{ctx['power_options']['max']}W, "
                           f"Dwell_0 {ctx['dwell_0_options']['min']}-{ctx['dwell_0_options']['max']}ms, "
                           f"Dwell_1 {ctx['dwell_1_options']['min']}-{ctx['dwell_1_options']['max']}ms")
        
        return "\n".join(formatted)