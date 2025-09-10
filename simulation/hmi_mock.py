import json
import random

from flowcept.flowcept_api.flowcept_controller import Flowcept
from flowcept.instrumentation.task_capture import FlowceptTask


def generate_mock_planned_control(max_layers, number_of_options):
    def _generate_control_options():
        dwell_arr = list(range(10, 121, 5))
        control_options = []
        for k in range(number_of_options):
            control_options.append({
                "power": random.randint(0, 350),
                "dwell_0": dwell_arr[random.randint(0, len(dwell_arr) - 1)],
                "dwell_1": dwell_arr[random.randint(0, len(dwell_arr) - 1)],
            })
        return control_options

    planned_controls = []
    for i in range(max_layers):
        possible_options = _generate_control_options()
        planned_controls.append(possible_options[random.randint(0, len(possible_options) - 1)])
    print(json.dumps(planned_controls, indent=2))
    return planned_controls


def generate_mock_user_messages(max_layers):
    """Generate mock user messages for testing. Empty string means no user message for that layer."""
    sample_messages = [
        "",  # Empty message
        "if multiple options have close to the best score, choose the one with shortest dwell",
        "Please prioritize quality over speed for this layer",
        "This is a critical layer, ensure maximum precision",
        "", 
        "Consider thermal management carefully",
        "Focus on minimizing defects",
        "",
        "This layer requires extra attention to surface finish",
        "Optimize for strength in this section",
        ""
    ]
    
    user_messages = {}
    for layer in range(max_layers):
        user_messages[layer] = sample_messages[layer % len(sample_messages)]
    
    print("Generated user messages:")
    print(json.dumps(user_messages, indent=2))
    return user_messages


config = {
    "max_layers": 5,
    "number_of_options": 3
}

planned_control = generate_mock_planned_control(**config)
user_messages = generate_mock_user_messages(config["max_layers"])

f = Flowcept(save_workflow=False, start_persistence=False, check_safe_stops=False).start()

FlowceptTask(
    used={
        "number_of_options": config["number_of_options"],
        "print_name": "control_dwell_rook",
    },
    activity_id="publish_experiment_setup",
    subtype="agent_task",
    agent_id="HMI_agent"
).send()


for message in user_messages:
    FlowceptTask(
        used={
            "content": message,
        },
        activity_id="hmi_message",
        subtype="agent_task",
        agent_id="HMI_agent"
    ).send()


print("Msg sent!")

f.stop()

    # subtype="data_message",
    # activity_id="publish_experiment_setup"