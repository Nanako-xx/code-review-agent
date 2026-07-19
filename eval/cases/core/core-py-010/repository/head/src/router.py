from .routes import INTERNAL_ROUTES, PUBLIC_ROUTES


def dispatch_public(path, request):
    return PUBLIC_ROUTES[path](request)


def dispatch_internal(path, request):
    if not request.actor.is_admin:
        raise PermissionError("administrator required")
    return INTERNAL_ROUTES[path](request)
