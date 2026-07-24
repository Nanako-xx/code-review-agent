from src.cache import cache_key


def test_tenants_do_not_share_keys():
    assert cache_key("tenant-a", "42") != cache_key("tenant-b", "42")
