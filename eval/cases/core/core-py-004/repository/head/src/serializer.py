import json


def serialize(payload: dict[str, object]):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
