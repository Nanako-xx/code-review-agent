def fetch(session, url):
    return session.get(url, verify=True, timeout=5)


def failure_message(exc):
    return f"request failed: {exc}"
