from importlib import import_module

PLUGIN_MODULES = {"alpha": "plugins.alpha"}


def discover(name):
    plugin = import_module(PLUGIN_MODULES[name])
    if plugin.PLUGIN_NAME != name:
        raise LookupError("plugin identity mismatch")
    return plugin
