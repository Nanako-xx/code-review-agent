def normalize_timeout(value: int) -> int:
    if value <= 0:
        raise ValueError("timeout must be positive")
    return value
