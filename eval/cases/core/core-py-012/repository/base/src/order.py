from datetime import datetime, timezone


def update_order(order, total):
    if total < 0:
        raise ValueError("negative total")
    order.total = total
    order.updated_at = datetime.now(timezone.utc)
    return order
