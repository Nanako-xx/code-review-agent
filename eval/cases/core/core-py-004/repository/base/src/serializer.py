import json


def serialize(payload: dict[str, object]):
    return json.dumps(payload)
