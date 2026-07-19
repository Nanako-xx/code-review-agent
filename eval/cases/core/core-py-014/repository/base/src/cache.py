import time


def is_fresh(cached_at, ttl):
    return cached_at + ttl >= time.time()
