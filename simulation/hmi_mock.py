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


config = {
    "max_layers": 5,
    "number_of_options": 3
}

planned_control = generate_mock_planned_control(**config)

f = Flowcept(save_workflow=False, start_persistence=False, check_safe_stops=False).start()

FlowceptTask(
    used={
        "number_of_options": config["number_of_options"],
        "planned_control": planned_control,
    },
    subtype="data_message",
    activity_id="publish_experiment_setup"
).send()

print("Msg sent!")

f.stop()
