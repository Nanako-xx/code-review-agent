from datetime import datetime, timezone


def update_order(order, total):
    order.total = total
    return order
