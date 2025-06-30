import os

from flowcept.configs import AGENT
from flowcept.flowceptor.consumers.agent.base_agent_context_manager import BaseAgentContextManager
from flowcept.instrumentation.flowcept_agent_task import FlowceptLLM
from langchain_community.llms.sambanova import SambaStudio

os.environ["SAMBASTUDIO_URL"] = AGENT.get("llm_server_url")
os.environ["SAMBASTUDIO_API_KEY"] = AGENT.get("api_key")


def build_llm(agent_id=None):
    model_kwargs = AGENT.get("model_kwargs", {}).copy()
    model_kwargs["model"] = AGENT.get("model")

    llm = FlowceptLLM(SambaStudio(model_kwargs=model_kwargs))
    if agent_id is None:
        agent_id = BaseAgentContextManager.agent_id
    llm.agent_id = agent_id
    return llm
