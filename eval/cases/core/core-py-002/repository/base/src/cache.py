def cache_key(tenant_id: str, resource_id: str) -> str:
    return f"{tenant_id}:{resource_id}"
