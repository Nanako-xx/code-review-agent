from .auth import is_admin


def delete_user(actor, target_id):
    if not is_admin(actor):
        raise PermissionError("administrator required")
    return {"deleted": target_id}
