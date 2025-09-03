# Manufacturing Agent Crew

Welcome to the Manufacturing Agent Crew project, powered by [crewAI](https://crewai.com). This is a specialized multi-agent AI system designed for additive manufacturing process control and optimization. The system consists of four specialized crews that collaborate to generate, evaluate, validate, and research-support manufacturing control decisions for layer-by-layer additive manufacturing processes.

## Architecture Overview

The Manufacturing Agent system consists of specialized crews and services that work together:

```
┌─────────────────────┐    ┌─────────────────────┐
│ Option Generation   │    │ Decision Making     │
│ Crew               │───▶│ Crew               │
└─────────────────────┘    └─────────────────────┘
           │                         │
           ▼                         ▼
┌─────────────────────┐    ┌─────────────────────┐
│ Safety Validation   │    │ Citation            │
│ Crew               │    │ Classification Crew │
└─────────────────────┘    └─────────────────────┘
                                    ▲
                                    │
                           ┌─────────────────────┐
                           │ Deep Research       │
                           │ Service             │
                           │ (Perplexity API)    │
                           └─────────────────────┘
```

1. **Option Generation Crew**: Generates candidate control parameters (power, dwell_0, dwell_1) for each manufacturing layer
2. **Decision Making Crew**: Evaluates simulation scores and selects the optimal control option
3. **Safety Validation Crew**: Validates decisions for safety compliance and logical consistency
4. **Deep Research Service**: Performs web-based research using Perplexity's sonar-deep-research model to gather relevant scientific literature
5. **Citation Classification Crew**: Analyzes research abstracts and papers to determine literature support for manufacturing decisions

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling.

First, install uv if you haven't already:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

```bash
# Install from lock file (recommended)
uv pip install -r uv.lock

# Or install from pyproject.toml and resolve fresh
uv pip install -e .
```

Alternative using crewAI CLI:
```bash
crewai install
```
### Configuration

**Environment Setup:**
- Add your `OPENAI_API_KEY` or Azure OpenAI credentials to the `.env` file
- Configure `PERPLEXITY_API_KEY` for research functionality

**Configuration Files:**
- `src/manufacturing_agent/config/agents.yaml` - Defines the four specialized agents (option_designer, decision_maker, safety_validator, citation_classifier)
- `src/manufacturing_agent/config/tasks.yaml` - Defines tasks for generation, decision-making, safety validation, and citation analysis
- `src/manufacturing_agent/crew.py` - Main crew orchestration with backward compatibility
- `src/manufacturing_agent/main.py` - CLI entry point and execution logic

The system will:
1. Generate control options for the specified manufacturing layer
2. Evaluate options using simulation scores 
3. Select the optimal control parameters
4. Validate the decision for safety compliance
5. Perform deep research on relevant manufacturing literature (using Perplexity API)
6. Classify research literature support for the decision with evidence citations
7. Output results and generate reports in the `src/output/` directory

## Project Structure

```
src/manufacturing_agent/
├── __init__.py                 # Package initialization
├── config/
│   ├── agents.yaml            # Agent definitions and configurations
│   └── tasks.yaml             # Task definitions and workflows
├── crew.py                    # Main crew orchestration module
├── main.py                    # CLI entry point
├── crews/                     # Specialized crew implementations
│   ├── __init__.py
│   ├── citation_classification_crew.py
│   ├── decision_crew.py
│   ├── option_generation_crew.py
│   └── safety_validation_crew.py
├── services/                  # External service integrations
│   ├── __init__.py
│   ├── research_service.py    # Perplexity API integration
│   └── research_summarizer.py # Research synthesis
├── tools/                     # Custom tools (currently minimal)
│   └── __init__.py
└── utils/                     # Utility functions and helpers
    ├── __init__.py
    ├── crew_cache.py          # Caching functionality
    └── json_fixer_crew.py     # JSON validation utilities

src/output/                    # Runtime artifacts
└── manufacturing_runs.db     # SQLite database for run tracking
```

## Key Components

### Agents (defined in `config/agents.yaml`)
- **option_designer**: Generates candidate control parameters within domain constraints
- **decision_maker**: Selects optimal options based on simulation scores  
- **safety_validator**: Validates decisions for safety and logical consistency
- **citation_classifier**: Analyzes research literature for decision support

### Services
- **ResearchService**: Integrates with Perplexity's sonar-deep-research model for comprehensive web-based research on manufacturing parameters and processes. Supports advanced research queries with citation extraction and structured analysis.
- **ResearchSummarizer**: Processes and condenses research findings into decision-relevant insights, extracting parameter-specific guidance and thermal management recommendations.

### Crews
Each crew is specialized for a specific phase of the manufacturing decision process:
- **OptionGenerationCrew**: Parameter generation and constraint validation
- **DecisionCrew**: Score-based optimization and selection
- **SafetyValidationCrew**: Safety compliance and decision validation  
- **CitationClassificationCrew**: Literature analysis and evidence synthesis

## Usage Examples

### Basic Manufacturing Control Decision
```python
from manufacturing_agent import ManufacturingAgentCrew

# Initialize the crew
crew = ManufacturingAgentCrew()

# Run decision-making for a specific layer
result = crew.kickoff(inputs={
    "layer_number": 1,
    "number_of_options": 5,
    "planned_controls": {"power": 250, "dwell_0": 50, "dwell_1": 60},
    "scores": [2.1, 3.4, 1.8, 4.2, 2.9]
})
```

### Integration with Flowcept
This project is designed to integrate with Flowcept for workflow orchestration and data provenance tracking. The main entry point through `main.py` includes placeholder code for Flowcept integration.

## Development

### Adding New Tools
1. Create tool files in `src/manufacturing_agent/tools/`
2. Register tools in `tools/__init__.py`
3. Reference tools in agent configurations in `config/agents.yaml`

### Extending Crews
1. Create new crew classes in `src/manufacturing_agent/crews/`
2. Follow the pattern of existing crews (inherit from CrewBase)
3. Add utility functions in `utils/` for crew instantiation

### Deep Research Integration
The system includes sophisticated research capabilities:

**Research Service Features:**
- **Perplexity Integration**: Uses sonar-deep-research model for comprehensive web search and analysis
- **Citation Extraction**: Automatically extracts and structures relevant papers with abstracts, authors, and URLs
- **Parameter-Specific Research**: Queries focused on specific manufacturing parameters (power, dwell times, thermal management)
- **Structured Analysis**: Organizes findings into executive summaries, key recommendations, and parameter insights

**Research Workflow:**
1. **Query Generation**: Automatically generates research queries based on manufacturing parameters and layer context
2. **Deep Search**: Executes comprehensive web search using Perplexity's research model (timeout: 800s for thorough analysis)
3. **Content Processing**: Extracts abstracts, citations, and key insights from research papers
4. **Synthesis**: ResearchSummarizer condenses findings into decision-relevant recommendations
5. **Literature Classification**: Citation Classification Crew analyzes aggregate literature support with confidence scoring and evidence citations

### Caching
The system includes intelligent caching via `utils/crew_cache.py` to optimize repeated operations and reduce API costs.

## Support

For support, questions, or feedback regarding the Manufacturing Agent Crew:
- Visit the [crewAI documentation](https://docs.crewai.com)
- Check the [crewAI GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join the crewAI Discord](https://discord.com/invite/X4JWnZnxPb)

This project leverages the power and simplicity of crewAI for manufacturing process optimization.
