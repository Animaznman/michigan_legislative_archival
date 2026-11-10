import json

import httpx

from mlarchive import sources

HOUSE_HTML = """
<html><body>
<li>
  <div data-toggle="collapse" data-target="#set_3" class="text-clickable">
    <strong>Appropriations - Agriculture | <span>2 Videos</span></strong>
  </div>
  <div class="collapse page-search-collapsible" id="set_3"><div class="p-1">
    <div class="page-search-object">
      <a href="/VideoArchivePlayer?video=AGRR-022526.mp4">Wednesday, February 25, 2026  </a>
    </div>
    <div class="page-search-object">
      <a href="/VideoArchivePlayer?video=AGRR-031126-2.mp4">Wednesday, March 11, 2026</a>
    </div>
  </div></div>
</li>
<li>
  <div data-target="#set_9"><strong>House Session | <span>1 Videos</span></strong></div>
  <div class="collapse" id="set_9">
    <a href="/VideoArchivePlayer?video=Session-011426.mp4">garbled text</a>
    <a href="/VideoArchivePlayer?video=../etc/passwd">evil</a>
  </div>
</li>
</body></html>
"""


def test_house_listing_parse():
    items = {i.source_id: i for i in sources.parse_house_listing(HOUSE_HTML)}
    assert set(items) == {"AGRR-022526.mp4", "AGRR-031126-2.mp4", "Session-011426.mp4"}
    a = items["AGRR-022526.mp4"]
    assert a.committee == "Appropriations - Agriculture"
    assert a.recorded_on == "2026-02-25"
    assert a.media_url == "https://www.house.mi.gov/ArchiveVideoFiles/AGRR-022526.mp4"
    assert items["AGRR-031126-2.mp4"].meta["part"] == "2"
    assert "(part 2)" in items["AGRR-031126-2.mp4"].title
    # Unparseable link text falls back to the date in the filename.
    assert items["Session-011426.mp4"].recorded_on == "2026-01-14"


def _doc(i, name="Senate Session 26-11-04", **kw):
    d = {"_id": f"id{i:04d}", "access": "open", "transcoded": True, "transcoding": False,
         "metadata": {"filename": name, "duration": 22}, "date": "2026-11-04T14:59:08Z",
         "size": 100}
    d.update(kw)
    return d


def test_senate_item_mapping():
    it = sources.senate_item(_doc(1, "Finance, Insurance, and Consumer Protection 26-10-14"))
    assert it.recorded_on == "2026-10-14"
    assert it.committee == "Finance, Insurance, and Consumer Protection"
    assert it.page_url.endswith("/video/id0001")
    assert sources.senate_item(_doc(2, transcoded=False)) is None
    assert sources.senate_item(_doc(3, access="paid")) is None
    # No date in title -> upload date.
    assert sources.senate_item(_doc(4, "Special Event")).recorded_on == "2026-11-04"


def senate_api(docs, too_big_pages=()):
    """Fake Lambda API; pages listed in too_big_pages 502 like the real one."""
    def handler(req: httpx.Request):
        url = str(req.url)
        if url == sources.SENATE_CONFIG_URL:
            return httpx.Response(200, json={"user": "U"})
        if url == sources.SENATE_LIST_URL:
            body = json.loads(req.content)
            page, n = body["page"], body["results"]
            if (page, n) in too_big_pages:
                return httpx.Response(502, json={"message": "Internal server error"})
            chunk = docs[(page - 1) * n: page * n]
            return httpx.Response(200, json={"record": len(docs), "allFiles": chunk})
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_senate_full_walk():
    docs = [_doc(i) for i in range(70)]
    got = list(sources.discover_senate(senate_api(docs), full=True, page_size=32))
    assert len(got) == 70


def test_senate_incremental_stops_at_known_page():
    docs = [_doc(i) for i in range(100)]
    known = {f"id{i:04d}" for i in range(3, 100)}    # only 3 newest are new
    client = senate_api(docs)
    got = list(sources.discover_senate(client, known=known, page_size=4))
    # page 1 (ids 0-3) has new items -> read page 2 (all known) -> stop.
    assert len(got) == 8


def test_senate_oversized_page_is_split():
    docs = [_doc(i) for i in range(64)]
    # Page 2 of size 32 is "too big"; so is page 3 of size 16 (= items 32..47).
    client = senate_api(docs, too_big_pages={(2, 32), (3, 16)})
    got = list(sources.discover_senate(client, full=True, page_size=32))
    assert [g.source_id for g in got] == [d["_id"] for d in docs]
