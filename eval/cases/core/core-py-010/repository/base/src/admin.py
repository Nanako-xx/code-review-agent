def delete_user(request):
    request.user_store.delete(request.target_id)
    return {"deleted": request.target_id}
