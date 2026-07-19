def parse_header(value):
    name, payload = value.split(":")
    return name, payload
