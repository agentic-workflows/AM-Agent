"""
Research Service Module

Contains the ResearchService class for Perplexity deep research functionality.
"""

from typing import Any, Dict
import time
import json
import os
import httpx
import hashlib
from pathlib import Path

from ..utils.research_cache import ResearchCache


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

    def _db_path(self) -> str:
        """Return robust database path that works across deployment scenarios."""
        from ..utils.db_utils import get_robust_db_path
        return get_robust_db_path("manufacturing_runs.db")

    def _build_messages_for_prompt(self, control_options, planned_controls) -> list[Dict[str, str]]:
        """Build canonical messages for strict prompt-based caching (layer-agnostic)."""
        # Extract parameter ranges for context (same logic as get_research_background)
        if control_options:
            powers = [opt.get("power", 0) for opt in control_options]
            dwell_0s = [opt.get("dwell_0", 0) for opt in control_options]
            dwell_1s = [opt.get("dwell_1", 0) for opt in control_options]
            power_range = f"{min(powers)}-{max(powers)}W"
            dwell0_range = f"{min(dwell_0s)}-{max(dwell_0s)}ms"
            dwell1_range = f"{min(dwell_1s)}-{max(dwell_1s)}ms"
        else:
            power_range = dwell0_range = dwell1_range = "N/A"

        base_query = f"""
        RESEARCH REQUIREMENTS: 
        - Conduct LIGHTWEIGHT deep research on additive manufacturing control parameters for powder bed fusion processes
        - TIME LIMIT: Maximum 2 minutes for research and response generation
        - FOCUS: Latest developments and most critical findings only
        - EFFICIENCY: Prioritize high-impact, recent sources over exhaustive coverage
        
        Please structure your response in the following format for easy parsing:

        EXECUTIVE SUMMARY:
        [Provide a 2-3 sentence overview of the most critical findings]

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
        [Key thermal management insights and strategies]

        QUALITY CONTROL APPROACHES:
        [Latest monitoring and quality control methods relevant to these parameters]

        DETAILED RESEARCH FINDINGS:
        [Comprehensive technical details and research evidence supporting the above recommendations]

        INDUSTRY APPLICATIONS:
        [Practical implementation examples and case studies]

        Research Focus Areas (PRIORITIZED FOR SPEED):
        1. Most recent (2024-2025) high-impact research papers on laser power optimization for powder bed fusion
        2. Latest breakthrough dwell time studies and thermal management advances  
        3. Critical parameter interaction findings and optimization strategies
        4. Current industry best practices and proven case studies
        5. Essential quality control methods and monitoring approaches

        RESEARCH EFFICIENCY GUIDELINES:
        - Target 5-7 most relevant and recent sources (instead of exhaustive coverage)
        - Prioritize peer-reviewed papers from top-tier journals
        - Focus on actionable insights for manufacturing engineers
        - Include publication dates and author information where available
        - Emphasize findings directly applicable to the given parameter ranges
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
          // repeat for each of the most relevant sources, total 5-7 sources max
        ]
        
        STRICT JSON REQUIREMENTS:
        - The line 'CITATIONS WITH ABSTRACTS (JSON):' must be followed by a JSON array only (no markdown, no code fences, no extra text)
        - Each object must include all keys listed above
        - Use "abstract_unavailable" if the abstract cannot be determined from accessible content
        """

        research_query = base_query + abstracts_block

        return [
            {
                "role": "system",
                "content": "You are an expert additive-manufacturing researcher. Be precise and include citations.",
            },
            {"role": "user", "content": research_query},
        ]

    def _prompt_signature(self, messages: list[Dict[str, str]]) -> tuple[str, str]:
        prompt_json = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        prompt_hash = hashlib.sha256(prompt_json.encode("utf-8")).hexdigest()
        return prompt_hash, prompt_json

    def get_research_background_strict_first_only(self, *, is_first_request: bool, control_options, planned_controls, force: bool = False) -> Dict[str, Any]:
        """Strict prompt-based reuse; only first request may perform research API call.

        - If prompt exists in cache: return cached.
        - Else if not first request: skip API call and return {skipped=True}.
        - Else perform research, store, and return.
        """

        messages = self._build_messages_for_prompt(control_options, planned_controls)
        prompt_hash, prompt_json = self._prompt_signature(messages)

        # Initialize cache with error handling
        try:
            cache = ResearchCache(self._db_path())
            # Try cache first
            hit = cache.get_by_prompt(prompt_hash)
        except Exception as e:
            print(f"Warning: Research cache initialization failed: {e}")
            print("   Proceeding without caching...")
            cache = None
            hit = None
        if hit and not force:
            result = hit["result_json"]
            citations = hit["citations_json"]
            result.setdefault("citations", citations)
            result.setdefault("web_sources", citations)
            # Log cache usage
            try:
                age_min = (time.time() - float(hit.get("created_at", time.time()))) / 60.0
                print(f"\n Using cached literature review (age {age_min:.1f} min, {len(citations)} citations) — prompt_hash={prompt_hash[:8]}")
            except Exception:
                print(f"\nUsing cached literature review — prompt_hash={prompt_hash[:8]}")
            result["from_cache"] = True
            result["skipped"] = False
            result["prompt_hash"] = prompt_hash
            return result

        if not is_first_request and not force:
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

        # Perform research (first request or forced)
        print(f"\n🔍 Initiating Deep Research (first request)...")
        print(f"   🤖 Model: {self.model}")
        print(f"   🌐 Live Web Search: Enabled")
        print(f"   ⏱️  Status: Preparing research query...")

        # Compute parameter ranges for pretty output and context
        if control_options:
            powers = [opt.get("power", 0) for opt in control_options]
            dwell_0s = [opt.get("dwell_0", 0) for opt in control_options]
            dwell_1s = [opt.get("dwell_1", 0) for opt in control_options]
            power_range = f"{min(powers)}-{max(powers)}W"
            dwell0_range = f"{min(dwell_0s)}-{max(dwell_0s)}ms"
            dwell1_range = f"{min(dwell_1s)}-{max(dwell_1s)}ms"
        else:
            power_range = dwell0_range = dwell1_range = "N/A"

        try:
            print(f"   🚀 Executing research query...")
            resp_json = self._chat_complete(messages)
            answer = resp_json["choices"][0]["message"]["content"]
            
            # Temporary debug: Log API response structure
            print(f"🔍 API Response Keys: {list(resp_json.keys())}")
            for key in resp_json.keys():
                if key != "choices":
                    value = resp_json[key]
                    if isinstance(value, list):
                        print(f"🔍 {key}: list with {len(value)} items")
                    else:
                        print(f"🔍 {key}: {type(value).__name__}")
            
            # Extract citations from Perplexity API response
            # Perplexity API returns citations in different possible fields
            citations = []
            
            # Try the standard search_results field first
            if "search_results" in resp_json and resp_json["search_results"]:
                citations = resp_json["search_results"]
                print(f"📚 Found {len(citations)} citations in 'search_results' field")
            
            # Try citations field as fallback
            elif "citations" in resp_json and resp_json["citations"]:
                citations = resp_json["citations"]
                print(f"📚 Found {len(citations)} citations in 'citations' field")
            
            # If no citations in API response, try to extract them from the content
            else:
                print(f"⚠️ No citations found in API response fields. Checking content...")
                # Some models embed citations directly in the response text
                citations = self._extract_citations_from_content(answer)
                if citations:
                    print(f"📚 Extracted {len(citations)} citations from response content")
                else:
                    print(f"❌ No citations found. Research will proceed without external sources.")

            if os.getenv("ABSTRACT_FROM_LLM", "true").lower() in ("1", "true", "yes"):
                try:
                    supplied = self._extract_llm_citation_abstracts(answer)
                    if supplied:
                        attached = self._merge_llm_abstracts_into_citations(citations, supplied)
                        print(f"Attached {attached}/{len(citations)} abstracts from LLM synthesis")
                except Exception as _e:
                    print(f"nable to parse LLM-provided abstracts: {_e}")

            research_result: Dict[str, Any] = {
                "research_findings": answer,
                "citations": citations,
                "web_sources": citations,
                "timestamp": time.time(),
                "parameter_context": {
                    "power_range": power_range,
                    "dwell_0_range": dwell0_range,
                    "dwell_1_range": dwell1_range,
                },
            }

            print(f"Research completed successfully!")
            self._format_research_output(1, answer, citations, research_result["parameter_context"])  # layer-agnostic display

            # Persist in cache (if cache is available)
            if cache is not None:
                try:
                    cache.put_by_prompt(
                        prompt_hash=prompt_hash,
                        prompt_json=prompt_json,
                        result_json=research_result,
                        citations_json=citations,
                        model=self.model,
                    )
                    print(f"Research results cached for future use")
                except Exception as e:
                    print(f"Warning: Failed to cache research results: {e}")
            else:
                print(f"Research results not cached (cache unavailable)")

            research_result["from_cache"] = False
            research_result["skipped"] = False
            research_result["prompt_hash"] = prompt_hash
            return research_result

        except Exception as e:
            print(f"Research failed: {e}")
            raise Exception(
                f"Mandatory Perplexity Deep Research failed: {str(e)}"
            )
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
    
    def _extract_citations_from_content(self, content: str) -> list:
        """Extract citations from response content when they're embedded in the text."""
        citations = []
        
        # Look for common citation patterns in the content
        import re
        
        # Pattern 1: [1] Title - URL format
        pattern1 = r'\[(\d+)\]\s*([^-\n]+)\s*-\s*(https?://[^\s\)]+)'
        matches1 = re.findall(pattern1, content)
        for match in matches1:
            citations.append({
                "title": match[1].strip(),
                "url": match[2].strip(),
                "index": int(match[0])
            })
        
        # Pattern 2: (Source: URL) format
        pattern2 = r'\(Source:\s*(https?://[^\s\)]+)\)'
        matches2 = re.findall(pattern2, content)
        for i, url in enumerate(matches2):
            citations.append({
                "title": f"Source {i+1}",
                "url": url.strip(),
                "index": i+1
            })
        
        # Pattern 3: Direct URLs in text
        if not citations:
            pattern3 = r'(https?://[^\s\)\]]+)'
            matches3 = re.findall(pattern3, content)
            for i, url in enumerate(matches3[:10]):  # Limit to 10 URLs
                citations.append({
                    "title": f"Reference {i+1}",
                    "url": url.strip(),
                    "index": i+1
                })
        
        return citations

    def _format_research_output(self, layer_number: int, answer: str, citations: list, parameter_context: dict):
        """Format research output - now uses enhanced formatting."""
        self._format_enhanced_research_output(layer_number, answer, citations, parameter_context)

    def get_research_background(self, layer_number: int, control_options, planned_controls):
        """Backward-compatible method; delegates to strict first-only API.

        Callers should set is_first_request=True only for the first processed request.
        Since we can't infer that here, we conservatively set is_first_request=False
        so this method will only use cache (or skip) and never trigger an API call.
        """
        return self.get_research_background_strict_first_only(
            is_first_request=False,
            control_options=control_options,
            planned_controls=planned_controls,
            force=False,
        )