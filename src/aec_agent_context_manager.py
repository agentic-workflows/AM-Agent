from dataclasses import dataclass
from typing import Dict, List
import json
import re

from flowcept.flowceptor.consumers.agent.base_agent_context_manager import BaseAgentContextManager
from flowcept.agents.agent_client import run_tool


@dataclass
class AeCContext:
    """
    Container for storing agent context data during the lifespan of an application session.

    Attributes
    ----------
    tasks : list of dict
        A list of task messages received from the message queue. Each task message is stored as a dictionary.
    user_messages : dict of int -> str
        A dictionary mapping layer numbers to user messages. Empty string means no user message for that layer.
    """
    history: List[Dict]
    user_messages: Dict[int, str]


class AdamantineAeCContextManager(BaseAgentContextManager):

    def __init__(self):
        super().__init__()

    def message_handler(self, msg_obj: Dict) -> bool:
        if msg_obj.get('type', '') == 'task':
            subtype = msg_obj.get("subtype", '')
            activity_id = msg_obj.get("activity_id", '')
            
            # Handle data_message subtype to capture user_messages
            # TODO:
            if subtype == 'call_agent_task':
                print(msg_obj)
                tool_name = msg_obj["custom_metadata"]["tool_name"]
                campaign_id = msg_obj.get("campaign_id", None)
                tool_args = msg_obj.get("used", {})
                tool_args["campaign_id"] = campaign_id
                self.logger.debug(f"Going to run {tool_name}, {tool_args}")
                tool_result = run_tool(tool_name, kwargs=tool_args)
                if len(tool_result):
                    if tool_name == 'choose_option':
                        this_history = dict()
                        this_history["layer"] = tool_args.get("layer")
                        # Support both list and non-list return formats
                        raw_result = tool_result[0] if isinstance(tool_result, list) else tool_result
                        this_history["scores"] = tool_args.get("scores", [])
                        this_history["control_options"] = tool_args.get("control_options", [])
                        # Parse result robustly: dict → use as-is; JSON-like str → json.loads; otherwise fallback
                        if isinstance(raw_result, dict):
                            parsed = raw_result
                        elif isinstance(raw_result, str):
                            raw_trim = raw_result.strip()
                            try:
                                if raw_trim and raw_trim[0] in "{[":
                                    parsed = json.loads(raw_trim)
                                else:
                                    parsed = {"option": None, "explanation": raw_trim}
                            except Exception:
                                self.logger.warning(f"Unexpected tool result string; cannot parse JSON: {raw_trim!r}")
                                parsed = {"option": None, "explanation": ""}
                        else:
                            self.logger.warning(f"Unexpected tool result type: {type(raw_result)}")
                            parsed = {"option": None, "explanation": ""}
                        this_history["chosen_option"] = parsed.get("option")
                        this_history["explanation"] = parsed.get("explanation", "")
                        self.context.history.append(this_history)
                else:
                    self.logger.error(f"Something wrong happened when running tool {tool_name}.")
            elif subtype == 'agent_task':
                if activity_id == "hmi_message":
                    used_data = msg_obj.get("used", {})
                    content = used_data.get("content", None)
                    if content is not None:
                        print("Received user messages from HMI mock")
                        normalized: Dict[int, str] = {}
                        try:
                            if isinstance(content, dict):
                                for k, v in content.items():
                                    try:
                                        normalized[int(k)] = v
                                    except Exception:
                                        self.logger.warning(f"Skipping non-integer layer key: {k!r}")
                            elif isinstance(content, str):
                                trimmed = content.strip()
                                parsed = None
                                if trimmed and trimmed[0] in "{[":
                                    try:
                                        parsed = json.loads(trimmed)
                                    except Exception:
                                        parsed = None
                                if isinstance(parsed, dict):
                                    for k, v in parsed.items():
                                        try:
                                            normalized[int(k)] = v
                                        except Exception:
                                            self.logger.warning(f"Skipping non-integer layer key: {k!r}")
                                else:
                                    match = re.search(r'layer\s+(\d+)', content, re.IGNORECASE)
                                    if match:
                                        layer_index = int(match.group(1))
                                        normalized[layer_index] = content
                                    else:
                                        self.logger.warning("HMI content is a plain string without a layer number; ignoring")
                            else:
                                self.logger.warning(f"Unexpected HMI content type: {type(content)}")
                        except Exception as e:
                            self.logger.error(f"Failed to normalize HMI content: {e}")
                            normalized = {}
                        if normalized:
                            self.context.user_messages.update(normalized)
                            print(f"Stored user messages for {len(self.context.user_messages)} layers")
                print('Tool result', msg_obj["activity_id"])
            if msg_obj.get("subtype", '') == "llm_query":
                print("Msg from agent.")
        else:
            print(f"We got a msg with different type: {msg_obj.get("type", None)}")
        return True

    def reset_context(self):
        self.context = AeCContext(history=[], user_messages={})
