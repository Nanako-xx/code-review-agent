def parse_header(value):
    name, payload = value.split(":", 1)
    return name, payload
