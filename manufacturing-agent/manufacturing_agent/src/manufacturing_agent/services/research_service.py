"""
Research Service Module

Contains the ResearchService class for Perplexity deep research functionality.
"""

from typing import Any, Dict
import time
import json
import os
import httpx


class ResearchService:
    """Perplexity 'sonar-deep-research' service for manufacturing insights."""

    DEFAULT_ENDPOINT = "https://api.perplexity.ai/chat/completions"

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialise the Perplexity Deep Research client.

        Parameters
        ----------
        config : dict | None
            Optional dictionary that may contain::

              api_key   – Perplexity API key (will fall back to env var PERPLEXITY_API_KEY)
              endpoint  – Override API URL
              model     – Model name (defaults to sonar-deep-research)
              temperature – Generation temperature (float)
        """

        cfg = config or {}

        # Retrieve API key with graceful fall-backs for backward-compat names
        self.api_key: str | None = (
            cfg.get("api_key")
            or os.getenv("PERPLEXITY_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Perplexity API key missing. Set PERPLEXITY_API_KEY or pass api_key in config."
            )

        self.endpoint: str = cfg.get("endpoint", self.DEFAULT_ENDPOINT)
        self.model: str = cfg.get("model", "sonar-deep-research")
        self.temperature: float = float(cfg.get("temperature", 0.2))

        # Prepare persistent HTTP client – Deep-Research calls can be slow (>30 s)
        default_timeout = float(cfg.get("timeout", 800))  # seconds (Perplexity deep research can exceed 2 min)
        self._client = httpx.Client(
            timeout=httpx.Timeout(default_timeout, connect=10),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        print(
            f"Perplexity Deep Research initialised (model={self.model}, endpoint={self.endpoint})"
        )

    # ---------------------------
    # Internal helper
    # ---------------------------
    def _chat_complete(self, messages: list[Dict[str, str]]) -> Dict[str, Any]:
        """Call Perplexity chat completions endpoint and return the JSON."""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": 1,
            # keep schema compatible
            "stream": False,
        }

        response = self._client.post(self.endpoint, json=payload)
        if response.status_code != 200:
            raise Exception(
                f"Perplexity API error {response.status_code}: {response.text}"
            )
        return response.json()

    # ---------------------------
    # Public API
    # ---------------------------
    def _parse_research_structure(self, research_text: str) -> dict:
        """Parse the structured research text into organized sections."""
        sections = {
            'executive_summary': '',
            'key_recommendations': [],
            'parameter_insights': {},
            'thermal_management': '',
            'quality_control': '',
            'detailed_findings': '',
            'industry_applications': '',
            'unparsed_content': ''
        }
        
        # Split text into lines for processing
        lines = research_text.split('\n')
        current_section = 'unparsed_content'
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            # Identify section headers
            if 'EXECUTIVE SUMMARY:' in line.upper():
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'executive_summary'
                current_content = []
            elif 'KEY RECOMMENDATIONS:' in line.upper():
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'key_recommendations'
                current_content = []
            elif 'PARAMETER OPTIMIZATION INSIGHTS:' in line.upper():
                if current_content:
                    if current_section == 'key_recommendations':
                        sections[current_section] = self._parse_recommendations(current_content)
                    else:
                        sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'parameter_insights'
                current_content = []
            elif 'THERMAL MANAGEMENT CONSIDERATIONS:' in line.upper():
                if current_content:
                    if current_section == 'parameter_insights':
                        sections[current_section] = self._parse_parameter_insights(current_content)
                    else:
                        sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'thermal_management'
                current_content = []
            elif 'QUALITY CONTROL APPROACHES:' in line.upper():
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'quality_control'
                current_content = []
            elif 'DETAILED RESEARCH FINDINGS:' in line.upper():
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'detailed_findings'
                current_content = []
            elif 'INDUSTRY APPLICATIONS:' in line.upper():
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'industry_applications'
                current_content = []
            elif line and not line.startswith('Research Focus Areas:'):
                current_content.append(line)
        
        # Process the last section
        if current_content:
            if current_section == 'key_recommendations':
                sections[current_section] = self._parse_recommendations(current_content)
            elif current_section == 'parameter_insights':
                sections[current_section] = self._parse_parameter_insights(current_content)
            else:
                sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _parse_recommendations(self, content_lines: list) -> list:
        """Extract bullet-point recommendations from content."""
        recommendations = []
        for line in content_lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                recommendations.append(line[1:].strip())
            elif line and not any(line.startswith(prefix) for prefix in ['EXECUTIVE', 'KEY', 'PARAMETER']):
                recommendations.append(line)
        return recommendations
    
    def _parse_parameter_insights(self, content_lines: list) -> dict:
        """Parse parameter-specific insights."""
        insights = {'power': '', 'dwell_0': '', 'dwell_1': '', 'general': ''}
        current_param = 'general'
        
        for line in content_lines:
            line = line.strip()
            if 'Power Range' in line or 'power' in line.lower():
                current_param = 'power'
                # Extract content after the colon
                if ':' in line:
                    insights[current_param] = line.split(':', 1)[1].strip()
                else:
                    insights[current_param] = line
            elif 'Dwell_0 Range' in line or 'dwell_0' in line.lower() or 'primary dwell' in line.lower():
                current_param = 'dwell_0'
                if ':' in line:
                    insights[current_param] = line.split(':', 1)[1].strip()
                else:
                    insights[current_param] = line
            elif 'Dwell_1 Range' in line or 'dwell_1' in line.lower() or 'secondary dwell' in line.lower():
                current_param = 'dwell_1'
                if ':' in line:
                    insights[current_param] = line.split(':', 1)[1].strip()
                else:
                    insights[current_param] = line
            elif line and current_param != 'general':
                insights[current_param] += ' ' + line
            elif line:
                insights['general'] += line + ' '
        
        return insights

    # ---------------------------
    # LLM abstracts extraction/merge
    # ---------------------------
    def _extract_llm_citation_abstracts(self, answer_text: str) -> list[dict] | None:
        """Extract the trailing 'CITATIONS WITH ABSTRACTS (JSON):' block and parse JSON array.

        Returns a list of dicts or None if not found/parseable.
        """
        marker = "CITATIONS WITH ABSTRACTS (JSON):"
        idx = answer_text.rfind(marker)
        if idx == -1:
            return None

        json_part = answer_text[idx + len(marker):].strip()
        # Some models may add surrounding fences or extra newlines; try to locate the first '[' and matching ']'
        start = json_part.find('[')
        if start == -1:
            return None
        # take substring starting from first '['; try json.loads directly
        candidate = json_part[start:]
        # Heuristic: cut off any trailing prose after the JSON array by finding the last ']'
        end = candidate.rfind(']')
        if end != -1:
            candidate = candidate[: end + 1]
        # Attempt to load JSON
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                # Normalize keys and trim abstract length
                normalized = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    abstract = (item.get('abstract') or '').strip()
                    if abstract and len(abstract) > 600:
                        abstract = abstract[:600].rstrip() + "…"
                    normalized.append({
                        'title': (item.get('title') or '').strip(),
                        'url': (item.get('url') or '').strip(),
                        'doi': (item.get('doi') or '').strip(),
                        'authors': (item.get('authors') or '').strip(),
                        'year': (item.get('year') or '').strip(),
                        'abstract': abstract or 'abstract_unavailable',
                    })
                return normalized
        except Exception:
            return None

        return None

    def _merge_llm_abstracts_into_citations(self, citations: list[dict], supplied: list[dict]) -> int:
        """Merge abstracts supplied by the LLM into the citations list in-place.

        Matching priority: DOI -> URL -> Title (case-insensitive). Returns number attached.
        """
        # Build quick indices for faster match
        def norm(s: str) -> str:
            return (s or '').strip().lower()

        attached = 0
        for sup in supplied:
            sup_doi = norm(sup.get('doi', ''))
            sup_url = norm(sup.get('url', ''))
            sup_title = norm(sup.get('title', ''))
            sup_abs = sup.get('abstract')
            if not sup_abs or sup_abs == 'abstract_unavailable':
                continue

            matched = None
            # Pass 1: DOI
            if sup_doi:
                for c in citations:
                    c_doi = norm(c.get('doi', ''))
                    if c_doi and c_doi == sup_doi:
                        matched = c
                        break
            # Pass 2: URL
            if matched is None and sup_url:
                for c in citations:
                    c_url = norm(c.get('url', ''))
                    if c_url and c_url == sup_url:
                        matched = c
                        break
            # Pass 3: Title
            if matched is None and sup_title:
                for c in citations:
                    c_title = norm(c.get('title', ''))
                    if c_title and c_title == sup_title:
                        matched = c
                        break

            if matched is not None:
                if not matched.get('abstract'):
                    matched['abstract'] = sup_abs
                    attached += 1
        return attached
    
    def _format_enhanced_research_output(self, layer_number: int, answer: str, citations: list, parameter_context: dict):
        """Format research output with enhanced structure and natural parsing."""
        
        # Parse the structured research content
        sections = self._parse_research_structure(answer)
        
        # Header with layer information
        print("\n" + "🔬" + "=" * 78 + "🔬")
        print(f"📊 DEEP RESEARCH ANALYSIS - LAYER {layer_number}")
        print("🔬" + "=" * 78 + "🔬")
        
        # Parameter context section
        print("\n📋 PARAMETER CONTEXT:")
        print(f"   ├─ Layer Number: {layer_number}")
        print(f"   └─ Parameter Ranges: {parameter_context.get('power_range', 'N/A')} | {parameter_context.get('dwell_0_range', 'N/A')} | {parameter_context.get('dwell_1_range', 'N/A')}")
        
        # Executive Summary
        if sections['executive_summary']:
            print(f"\n🎯 EXECUTIVE SUMMARY:")
            summary_lines = sections['executive_summary'].split('\n')[:3]  # Max 3 lines
            for line in summary_lines:
                if line.strip():
                    print(f"   {line.strip()}")
        
        # Key Recommendations
        if sections['key_recommendations']:
            print(f"\n💡 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(sections['key_recommendations'][:4], 1):  # Max 4 recommendations
                if rec.strip():
                    print(f"   ✓ {rec.strip()}")
        
        # Parameter Insights
        if sections['parameter_insights']:
            print(f"\n🔧 PARAMETER INSIGHTS:")
            insights = sections['parameter_insights']
            if insights.get('power'):
                print(f"   • Power: {insights['power'][:100]}{'...' if len(insights['power']) > 100 else ''}")
            if insights.get('dwell_0'):
                print(f"   • Dwell_0: {insights['dwell_0'][:100]}{'...' if len(insights['dwell_0']) > 100 else ''}")
            if insights.get('dwell_1'):
                print(f"   • Dwell_1: {insights['dwell_1'][:100]}{'...' if len(insights['dwell_1']) > 100 else ''}")
        
        # Top Citations (limit to 5)
        print(f"\n📚 TOP CITATIONS ({min(5, len(citations))} most relevant):")
        if citations:
            for i, citation in enumerate(citations[:5], 1):
                title = citation.get('title', 'Unknown Title')[:50]
                if len(citation.get('title', '')) > 50:
                    title += "..."
                url = citation.get('url', 'No URL')
                print(f"   {i}. {title}")
                print(f"      🔗 {url[:60]}{'...' if len(url) > 60 else ''}")
        else:
            print("   └─ No citations available")
        
        # Research Sections (collapsed view)
        print(f"\n📖 RESEARCH SECTIONS:")
        section_map = {
            'thermal_management': '🌡️  Thermal Management',
            'quality_control': '🔍 Quality Control', 
            'detailed_findings': '📊 Detailed Findings',
            'industry_applications': '🏭 Industry Applications'
        }
        
        for key, display_name in section_map.items():
            content = sections.get(key, '')
            if content:
                # Show first 100 characters of each section
                preview = content.replace('\n', ' ').strip()[:100]
                if len(content) > 100:
                    preview += "..."
                print(f"   📑 {display_name}: {preview}")
        
        # Summary statistics
        total_sections = sum(1 for v in sections.values() if v)
        print(f"\n📊 ANALYSIS COMPLETE: ✅ {len(citations)} citations • {total_sections} sections • {len(answer):,} chars")
        print("🔬" + "=" * 78 + "🔬\n")
    
    def _format_research_output(self, layer_number: int, answer: str, citations: list, parameter_context: dict):
        """Format research output - now uses enhanced formatting."""
        self._format_enhanced_research_output(layer_number, answer, citations, parameter_context)

    def get_research_background(self, layer_number: int, control_options, planned_controls):
        """Perform deep research with web search & citations via Perplexity."""

        # Initial progress indicator
        print(f"\n🔍 Initiating Deep Research for Layer {layer_number}...")
        print(f"   🤖 Model: {self.model}")
        print(f"   🌐 Live Web Search: Enabled")
        print(f"   ⏱️  Status: Preparing research query...")

        # Extract parameter ranges for context
        if control_options:
            powers = [opt.get("power", 0) for opt in control_options]
            dwell_0s = [opt.get("dwell_0", 0) for opt in control_options]
            dwell_1s = [opt.get("dwell_1", 0) for opt in control_options]
            power_range = f"{min(powers)}-{max(powers)}W"
            dwell0_range = f"{min(dwell_0s)}-{max(dwell_0s)}ms"
            dwell1_range = f"{min(dwell_1s)}-{max(dwell_1s)}ms"
        else:
            power_range = dwell0_range = dwell1_range = "N/A"

        print(f"   📊 Parameter Analysis: {power_range} | {dwell0_range} | {dwell1_range}")

        base_query = f"""
        Research latest developments in additive manufacturing control parameters for layer {layer_number} powder bed fusion processes.
        
        Please structure your response in the following format for easy parsing:

        EXECUTIVE SUMMARY:
        [Provide a 2-3 sentence overview of the most critical findings for layer {layer_number} optimization]

        KEY RECOMMENDATIONS:
        • [Specific actionable recommendation 1]
        • [Specific actionable recommendation 2]  
        • [Specific actionable recommendation 3]
        • [Additional recommendations as needed]

        PARAMETER OPTIMIZATION INSIGHTS:
        Power Range ({power_range}): [Specific guidance on laser power optimization]
        Dwell_0 Range ({dwell0_range}): [Specific guidance on primary dwell time]
        Dwell_1 Range ({dwell1_range}): [Specific guidance on secondary dwell time]

        THERMAL MANAGEMENT CONSIDERATIONS:
        [Key thermal management insights and strategies for layer {layer_number}]

        QUALITY CONTROL APPROACHES:
        [Latest monitoring and quality control methods relevant to these parameters]

        DETAILED RESEARCH FINDINGS:
        [Comprehensive technical details and research evidence supporting the above recommendations]

        INDUSTRY APPLICATIONS:
        [Practical implementation examples and case studies]

        Research Focus Areas:
        1. Recent (2023-2025) research papers on laser power optimisation for powder bed fusion
        2. Latest dwell time studies and thermal management advances  
        3. Parameter interaction findings and optimisation strategies
        4. Current industry best practices and case studies
        5. Quality control methods and monitoring approaches

        Provide specific recommendations with citations. Include publication dates and author information where available.
        Focus on actionable insights for manufacturing engineers. Limit to 10 most relevant sources.
        """

        abstracts_block = """
        IMPORTANT: At the very end of your response, append the following section EXACTLY:

        CITATIONS WITH ABSTRACTS (JSON):
        [
          {
            "title": "...",
            "url": "https://...",
            "doi": "10.XXXX/xxxxx" ,
            "authors": "Last, F.; Last2, F.",
            "year": "2024",
            "abstract": "Up to 600 characters summarizing the paper's abstract, or the exact string 'abstract_unavailable' if not accessible.",
            "source_note": "reason why available or not; e.g., 'from page meta', 'from search snippet', 'abstract_unavailable'"
          }
          // repeat for each of the most relevant sources, total up to 10
        ]
        
        STRICT JSON REQUIREMENTS:
        - The line 'CITATIONS WITH ABSTRACTS (JSON):' must be followed by a JSON array only (no markdown, no code fences, no extra text)
        - Each object must include all keys listed above
        - Use "abstract_unavailable" if the abstract cannot be determined from accessible content
        """

        research_query = base_query + abstracts_block

        messages = [
            {
                "role": "system",
                "content": "You are an expert additive-manufacturing researcher. Be precise and include citations.",
            },
            {"role": "user", "content": research_query},
        ]

        print(f"   🚀 Executing research query...")

        try:
            resp_json = self._chat_complete(messages)

            answer = resp_json["choices"][0]["message"]["content"]
            citations = resp_json.get("search_results", [])

            # Optionally enrich citations with abstracts supplied by the model in the answer tail
            if os.getenv("ABSTRACT_FROM_LLM", "true").lower() in ("1", "true", "yes"): 
                try:
                    supplied = self._extract_llm_citation_abstracts(answer)
                    if supplied:
                        attached = self._merge_llm_abstracts_into_citations(citations, supplied)
                        print(f"   🧩 Attached {attached}/{len(citations)} abstracts from LLM synthesis")
                except Exception as _e:
                    print(f"   ⚠️  Unable to parse LLM-provided abstracts: {_e}")

            research_result = {
                "research_findings": answer,
                "citations": citations,
                "web_sources": citations,  # maintain previous key for downstream code
                "layer_number": layer_number,
                "timestamp": time.time(),
                "parameter_context": {
                    "power_range": power_range,
                    "dwell_0_range": dwell0_range,
                    "dwell_1_range": dwell1_range,
                },
            }

            print(f"   ✅ Research completed successfully!")
            
            # Display formatted research output
            self._format_research_output(layer_number, answer, citations, research_result["parameter_context"])

            return research_result

        except Exception as e:
            print(f"   ❌ Research failed for layer {layer_number}: {e}")
            raise Exception(
                f"Mandatory Perplexity Deep Research failed for layer {layer_number}: {str(e)}"
            )