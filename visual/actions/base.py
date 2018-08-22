class Action:
    instruction_type = None
    required = []
    optional = []

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "<{}:{}:{}>".format(
            self.instruction_type,
            ",".join(f"{arg}={getattr(self, arg)}" for arg in self.required),
            ",".join(f"{arg}={getattr(self, arg)}" for arg in self.optional),
        )

    __repr__ = __str__
