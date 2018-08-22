import sys
import time

import yaml

from actions.gui import (
    ClickAction,
    FindTextAction,
    KeySequenceAction,
    MATCH_STRATEGY,
    ScrollForAppear,
    TypeWriteAction,
    WaitForAppear,
    WaitUntilVisible,
)

actions = [
    ClickAction,
    FindTextAction,
    KeySequenceAction,
    ScrollForAppear,
    TypeWriteAction,
    WaitForAppear,
    WaitUntilVisible,
]
name_to_action = {action.instruction_type: action for action in actions}


def parse_step(step, defaults_provider=None):
    action = name_to_action[step["type"]]

    required = {arg: step[arg] for arg in action.required}
    optional = {arg: step[arg] for arg in action.optional if arg in step}

    if defaults_provider is not None:
        optional = {
            **{
                key: value
                for key, value in defaults_provider.items()
                if key in action.optional
            },
            **optional,
        }

    return action(**required, **optional)


def run_yaml(path):
    with open(path) as fp:
        instruction = yaml.load(fp)

    defaults_provider = {}
    default_match_strategy = MATCH_STRATEGY.DEFAULT
    default_sleep = 1
    if instruction[0]["type"] == "config":
        default_sleep = instruction[0].get("between_sleep", 1)
        default_match_strategy = instruction[0].get(
            "default_match_strategy", default_match_strategy
        )
        instruction[:] = instruction[1:]

    defaults_provider["strategy"] = default_match_strategy

    actions = [parse_step(step, defaults_provider) for step in instruction]
    all_actions_len = len(actions)

    for i, (action, step) in enumerate(zip(actions, instruction), start=1):
        print(f"RUNNING ({i}/{all_actions_len}) {step['type']}:", action.name)
        time.sleep(step.get("before_sleep", 0))
        action.run()
        time.sleep(step.get("after_sleep", default_sleep))


if __name__ == "__main__":
    run_yaml(sys.argv[1])
