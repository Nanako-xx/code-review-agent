def read_config(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()
