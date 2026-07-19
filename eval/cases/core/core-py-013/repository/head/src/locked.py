def update_with_lock(lock, update):
    lock.acquire()
    try:
        return update()
    except RuntimeError:
        return None
