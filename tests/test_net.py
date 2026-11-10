import httpx
import pytest

from mlarchive import net

URL = "https://x.test/a"


def test_retries_then_succeeds(server):
    server.add(URL, b"ok")
    server.fail[URL] = [503, httpx.ConnectError("dns"), 429]
    assert net.request(server.client(), "GET", URL).content == b"ok"
    assert server.count(URL) == 4


def test_gives_up_after_tries(server):
    server.add(URL, b"ok")
    server.fail[URL] = [502] * 10
    with pytest.raises(net.TransientHTTPError):
        net.request(server.client(), "GET", URL, tries=3)
    assert server.count(URL) == 3


def test_permanent_error_not_retried(server):
    with pytest.raises(net.PermanentHTTPError) as e:
        net.request(server.client(), "GET", "https://x.test/missing")
    assert e.value.status == 404
    assert server.count("missing") == 1


def test_short_range_read_retried(server):
    server.add(URL, b"0123456789")
    assert net.get_range(server.client(), URL, 2, 3) == b"234"
