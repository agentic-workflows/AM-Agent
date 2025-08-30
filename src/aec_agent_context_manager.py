from dataclasses import dataclass
from typing import Dict, List
import json

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
                        tool_result = tool_result[0]
                        this_history["scores"] = tool_args["scores"]
                        tool_result = json.loads(tool_result)
                        this_history["chosen_option"] = tool_result["option"]
                        this_history["explanation"] = tool_result["explanation"]
                        self.context.history.append(this_history)
                else:
                    self.logger.error(f"Something wrong happened when running tool {tool_name}.")
            elif subtype == 'agent_task':
                if activity_id == "publish_experiment_setup":
                    # Handle setup message from HMI mock
                    print("Start the setup")
                    used_data = msg_obj.get("used", {})
                    if "user_messages" in used_data:
                        print("Received user messages from HMI mock")
                        self.context.user_messages.update(used_data["user_messages"])
                        print(f"Stored user messages for {len(self.context.user_messages)} layers")
                elif activity_id == "hmi_message":
                    used_data = msg_obj.get("used", {})
                    if "user_messages" in used_data:
                        print("Received user messages from HMI mock")
                        self.context.user_messages.update(used_data["user_messages"])
                        print(f"Stored user messages for {len(self.context.user_messages)} layers")
                print('Tool result', msg_obj["activity_id"])
            if msg_obj.get("subtype", '') == "llm_query":
                print("Msg from agent.")
        else:
            print(f"We got a msg with different type: {msg_obj.get("type", None)}")
        return True

    def reset_context(self):
        self.context = AeCContext(history=[], user_messages={})
