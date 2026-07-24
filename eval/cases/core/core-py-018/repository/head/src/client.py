def fetch(session, url):
    return session.get(url, verify=False, timeout=5)


def failure_message(exc):
    return "request failed"
