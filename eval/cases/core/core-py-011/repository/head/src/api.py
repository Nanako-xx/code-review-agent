from .database import export_all_customers


def export_customers(request):
    return export_all_customers()
