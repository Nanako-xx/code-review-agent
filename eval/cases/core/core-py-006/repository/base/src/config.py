import os

SUPPORTED_ENV_COUNT = 2


def _env_names():
    return ("SERVICE_URL", "LEGACY_SERVICE_URL")


def service_url():
    for name in _env_names()[:SUPPORTED_ENV_COUNT]:
        value = os.environ.get(name)
        if value:
            return value
    raise KeyError("service URL is missing")
