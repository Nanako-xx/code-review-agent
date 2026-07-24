import time


def is_valid(expires_at):
    return expires_at > time.time()
