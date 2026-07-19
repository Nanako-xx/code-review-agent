def is_admin(user):
    return user is not None and "admin" in user.roles
