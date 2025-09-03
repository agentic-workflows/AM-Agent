"""
Citation Classification Crew Module

Contains the CitationClassificationCrew class for analyzing research citations.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_core.language_models import LLM
from typing import Any, Dict
import json


@CrewBase
class CitationClassificationCrew:
    """Crew that analyzes research citations to determine literature support for manufacturing decisions."""

    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

    def __init__(self, llm: LLM | None = None):
        self.llm = llm

    @agent
    def citation_classifier(self) -> Agent:
        return Agent(
            config=self.agents_config['citation_classifier'],
            verbose=True,
            llm=self.llm,
        )

    @task
    def citation_classification_task(self) -> Task:
        return Task(
            config=self.tasks_config['citation_classification_task'],
            agent=self.citation_classifier(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,  # Reduce verbose output to hide large inputs
        )
    
    # Public helper
    def classify(self, layer_number: int, decision_result: Dict[str, Any], research_context: Dict[str, Any], control_options) -> Dict[str, Any]:
        """Aggregate all papers into a single classification with reasoning and cited evidence."""

        # Extract chosen option details
        best_option_index = decision_result.get("best_option", 0)
        num_options = len(control_options) if control_options else 0
        if num_options and best_option_index < num_options:
            chosen_option = control_options[best_option_index]
            chosen_power = chosen_option.get("power", 0)
            chosen_dwell_0 = chosen_option.get("dwell_0", 0)
            chosen_dwell_1 = chosen_option.get("dwell_1", 0)
        else:
            chosen_power = chosen_dwell_0 = chosen_dwell_1 = "Unknown"

        # Extract individual papers from research context
        citations = research_context.get('citations', [])

        # Only keep citations that have a usable abstract/snippet/summary
        def _extract_abstract(c: Dict[str, Any]) -> str:
            return (
                c.get('abstract')
                or c.get('snippet')
                or c.get('description')
                or c.get('summary')
                or c.get('content')
                or c.get('text', '')
            ) or ''

        eligible_citations = [c for c in citations if _extract_abstract(c).strip()]

        print(f"\n📊 Literature Evaluation (Aggregate): Layer {layer_number} | {chosen_power}W, {chosen_dwell_0}ms, {chosen_dwell_1}ms")
        print(f"   📚 Aggregating {len(eligible_citations)} abstracts (from {len(citations)} citations)...")

        # Prepare compact papers list for a single aggregated analysis
        papers: list[dict[str, Any]] = []
        for i, citation in enumerate(eligible_citations, start=1):
            papers.append({
                "paper_index": i,
                "title": citation.get('title', f'Paper {i}') or f'Paper {i}',
                "authors": citation.get('authors', 'Unknown') or 'Unknown',
                "year": citation.get('year', 'Unknown') or 'Unknown',
                "abstract": _extract_abstract(citation)[:1200],  # keep reasonably short
                "url": citation.get('url', ''),
            })

        inputs = {
            "layer_number": layer_number,
            "decision_reasoning": decision_result.get("reasoning", ""),
            "chosen_power": chosen_power,
            "chosen_dwell_0": chosen_dwell_0,
            "chosen_dwell_1": chosen_dwell_1,
            "num_papers": len(papers),
            # Embed as JSON string to avoid prompt formatting issues
            "papers_json": json.dumps(papers, ensure_ascii=False),
        }

        # Single aggregated analysis
        try:
            output = self.crew().kickoff(inputs=inputs)
            raw_text = str(output.raw) if hasattr(output, "raw") else str(output)

            result = json.loads(raw_text)

            classification = (result.get("classification") or "neutral").strip().lower()
            # Allow "mixed" for aggregate; normalize others
            if classification not in ("positive", "neutral", "negative", "mixed"):
                classification = "neutral"
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")
            evidence = result.get("evidence", [])

            # Backward-compat mapping
            attitude_map = {
                "positive": "POSITIVE",
                "neutral": "NEUTRAL",
                "negative": "NEGATIVE",
                "mixed": "MIXED",
            }
            overall_attitude = attitude_map.get(classification, "NEUTRAL") if len(papers) > 0 else "NO_DATA"

            print(f"\n   📈 Literature Attitude (Aggregate): {overall_attitude}")
            print(f"   🧠 Confidence: {confidence:.2f}")
            if evidence:
                print("   🔎 Evidence:")
                for idx, ev in enumerate(evidence[:5], start=1):
                    title = (ev.get("title") or "Untitled")[:60]
                    quote = (ev.get("quote") or "").replace("\n", " ")[:120]
                    print(f"     {idx}. {title} — \"{quote}\"")

            return {
                # New aggregate fields
                "classification": classification,
                "confidence": confidence,
                "reasoning": reasoning,
                "evidence": evidence,
                # Backward-compatible fields expected by downstream code
                "overall_attitude": overall_attitude,
                "distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "total_papers": len(papers),
                "paper_analyses": [],
                "decision_summary": {
                    "layer_number": layer_number,
                    "chosen_option": best_option_index,
                    "chosen_power": chosen_power,
                    "chosen_dwell_0": chosen_dwell_0,
                    "chosen_dwell_1": chosen_dwell_1,
                    "decision_reasoning": decision_result.get("reasoning", ""),
                },
            }

        except Exception as exc:
            print(f"   ❌ Aggregate classification failed: {exc}")
            return {
                "classification": "neutral",
                "confidence": 0.3,
                "reasoning": f"Aggregate analysis failed: {exc}",
                "evidence": [],
                "overall_attitude": "NO_DATA" if not eligible_citations else "NEUTRAL",
                "distribution": {"positive": 0, "neutral": len(eligible_citations), "negative": 0},
                "total_papers": len(eligible_citations),
                "paper_analyses": [],
                "decision_summary": {
                    "layer_number": layer_number,
                    "chosen_option": best_option_index,
                    "chosen_power": chosen_power,
                    "chosen_dwell_0": chosen_dwell_0,
                    "chosen_dwell_1": chosen_dwell_1,
                    "decision_reasoning": decision_result.get("reasoning", ""),
                },
            }