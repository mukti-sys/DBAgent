"""Command registry — routes slash commands to handlers."""


class CommandRegistry:
    def __init__(self):
        self._commands: dict[str, dict] = {}

    def register(self, name, handler, description, usage="", category="General"):
        self._commands[name.lower()] = {
            "handler": handler, "description": description,
            "usage": usage or f"/{name}", "category": category,
        }

    def get(self, name):
        return self._commands.get(name.lower())

    def execute(self, name, args, context):
        cmd = self.get(name)
        if cmd is None:
            return False
        cmd["handler"](args, context)
        return True

    def get_help(self):
        groups = {}
        for name, info in sorted(self._commands.items()):
            cat = info["category"]
            if cat not in groups:
                groups[cat] = []
            groups[cat].append((info["usage"], info["description"]))
        return groups

    def get_command_names(self):
        return [f"/{name}" for name in self._commands.keys()]
