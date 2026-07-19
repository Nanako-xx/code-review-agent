from .database import export_all_customers


def export_customers(request):
    if request.token is None:
        raise PermissionError("token required")
    return export_all_customers()
