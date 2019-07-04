import subprocess

from .base import Action


class CommandAction(Action):
    instruction_type = "command"
    required = ["name", "command"]
    optional = ["timeout"]

    def __init__(self, name, command, timeout=30):
        self.command = command.split()
        self.timeout = timeout
        super().__init__(name)

    def run(self):
        try:
            proc = subprocess.run(
                self.command, capture_output=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command '{self.command}' reached the timeout")

        print(proc.stdout)
