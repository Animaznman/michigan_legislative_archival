"""Discovery: turn each portal's listing into a stream of `Discovered` items.

House  (house.mi.gov/VideoArchive): one server-rendered HTML page grouping
       links like /VideoArchivePlayer?video=AGRR-022526.mp4 under committee
       headings. The player page plays
       https://www.house.mi.gov/ArchiveVideoFiles/<video> directly.

Senate (cloud.castus.tv/vod/misenate): a React app over a JSON API.
       POST .../default/api/all {_id, page, results} lists videos newest
       first; POST .../upload/get {file, type} returns the media URL.
       We read the API, not the rendered page.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterator
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from . import net

log = logging.getLogger(__name__)


@dataclass
class Discovered:
    source: str
    source_id: str
    title: str
    committee: str | None
    recorded_on: str            # ISO date
    page_url: str | None
    media_url: str | None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------- House
HOUSE_LIST_URL = "https://house.mi.gov/VideoArchive"
HOUSE_PLAYER_URL = "https://house.mi.gov/VideoArchivePlayer?video={}"
HOUSE_MEDIA_URL = "https://www.house.mi.gov/ArchiveVideoFiles/{}"
_HOUSE_FILE = re.compile(r"^(?P<code>[A-Za-z]+)-(?P<mmddyy>\d{6})(?:-(?P<part>\d+))?\.mp4$")


def _house_date(filename: str, link_text: str) -> str | None:
    text = " ".join(link_text.split())
    try:
        return datetime.strptime(text, "%A, %B %d, %Y").date().isoformat()
    except ValueError:
        pass
    m = _HOUSE_FILE.match(filename)
    if m:
        d = m["mmddyy"]
        try:
            return date(2000 + int(d[4:]), int(d[:2]), int(d[2:4])).isoformat()
        except ValueError:
            return None
    return None


def parse_house_listing(html: str, base_url: str = HOUSE_LIST_URL) -> list[Discovered]:
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, Discovered] = {}
    for a in soup.select('a[href*="VideoArchivePlayer"]'):
        qs = parse_qs(urlparse(urljoin(base_url, a["href"])).query)
        filename = (qs.get("video") or [""])[0].strip()
        if not filename or "/" in filename or "\\" in filename:
            continue
        # Committee heading lives in the collapsible group's toggle element.
        committee = None
        group = a.find_parent(class_="collapse")
        if group is not None and group.get("id"):
            toggle = soup.find(attrs={"data-target": f"#{group['id']}"})
            if toggle is not None:
                strong = toggle.find("strong") or toggle
                committee = strong.get_text(" ", strip=True).split("|")[0].strip() or None
        link_text = a.get_text(" ", strip=True)
        recorded_on = _house_date(filename, link_text)
        if recorded_on is None:
            log.warning("house: cannot date %r (%r), skipping", filename, link_text)
            continue
        m = _HOUSE_FILE.match(filename)
        part = m["part"] if m else None
        title = f"{committee or (m['code'] if m else filename)} {recorded_on}"
        if part:
            title += f" (part {part})"
        out[filename] = Discovered(
            source="house", source_id=filename, title=title, committee=committee,
            recorded_on=recorded_on, page_url=HOUSE_PLAYER_URL.format(filename),
            media_url=HOUSE_MEDIA_URL.format(filename),
            meta={"code": m["code"] if m else None, "part": part, "link_text": link_text})
    return list(out.values())


def discover_house(client: httpx.Client, **_) -> Iterator[Discovered]:
    resp = net.request(client, "GET", HOUSE_LIST_URL)
    items = parse_house_listing(resp.text)
    if not items:
        # An empty listing is far more likely a markup change than reality.
        raise RuntimeError("house: listing parsed to zero videos; page layout changed?")
    yield from items


# -------------------------------------------------------------------- Senate
SENATE_CONFIG_URL = "https://837sc3bew0.execute-api.us-west-2.amazonaws.com/misenate"
SENATE_LIST_URL = "https://tf4pr3wftk.execute-api.us-west-2.amazonaws.com/default/api/all"
SENATE_MEDIA_API = "https://imd0mxanj2.execute-api.us-west-2.amazonaws.com/upload/get"
SENATE_PAGE_URL = "https://cloud.castus.tv/vod/misenate/video/{}"
_SENATE_DATE = re.compile(r"(\d{2})-(\d{2})-(\d{2})\s*$")


def senate_item(doc: dict) -> Discovered | None:
    """Map one API record to a Discovered, or None if not processable yet."""
    if doc.get("access") != "open" or not doc.get("transcoded") or doc.get("transcoding"):
        return None
    vid = doc["_id"]
    md = doc.get("metadata") or {}
    title = (md.get("filename") or vid).strip()
    m = _SENATE_DATE.search(title)
    recorded_on = None
    if m:
        try:
            recorded_on = date(2000 + int(m[1]), int(m[2]), int(m[3])).isoformat()
        except ValueError:
            pass
    if recorded_on is None:
        recorded_on = (doc.get("original_date") or doc.get("date") or "")[:10]
    if not recorded_on:
        return None
    committee = _SENATE_DATE.sub("", title).strip() or None
    try:
        duration = float(md.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    return Discovered(
        source="senate", source_id=vid, title=title, committee=committee,
        recorded_on=recorded_on, page_url=SENATE_PAGE_URL.format(vid), media_url=None,
        meta={"duration": duration, "size": doc.get("size"), "uploaded": doc.get("date")})


def senate_user_id(client: httpx.Client) -> str:
    return net.request(client, "GET", SENATE_CONFIG_URL).json()["user"]


def _senate_slice(client: httpx.Client, user: str, offset: int, count: int) -> tuple[list[dict], int]:
    """Fetch records [offset, offset+count) newest-first.

    The API runs on AWS Lambda, whose responses are capped at 6 MB. Each
    record embeds its full view history, so a page of popular videos can
    exceed the cap and come back as HTTP 502. When that happens we fetch the
    same range as two half-size pages (offset is always a multiple of count,
    and count a power of two, so the halves line up with real page numbers).
    """
    try:
        body = net.request(client, "POST", SENATE_LIST_URL, tries=2 if count > 1 else 6,
                           json={"_id": user, "page": offset // count + 1, "results": count}).json()
        return body.get("allFiles") or [], int(body.get("record") or 0)
    except net.TransientHTTPError:
        if count == 1:
            log.error("senate: record at offset %d cannot be listed; skipping it", offset)
            return [], -1
        half = count // 2
        log.info("senate: page offset=%d size=%d failed; splitting into size %d", offset, count, half)
        a, ta = _senate_slice(client, user, offset, half)
        b, tb = _senate_slice(client, user, offset + half, half)
        return a + b, max(ta, tb)


def discover_senate(client: httpx.Client, *, known: set[str] = frozenset(),
                    full: bool = False, page_size: int = 32,
                    max_pages: int = 1000) -> Iterator[Discovered]:
    """Page newest-first. Incremental mode stops at the first page that has
    nothing new; `full` walks everything (done periodically to catch late
    transcodes or re-dated items)."""
    assert page_size & (page_size - 1) == 0, "page_size must be a power of two"
    user = senate_user_id(client)
    total_known = 0
    for page in range(max_pages):
        docs, total = _senate_slice(client, user, page * page_size, page_size)
        if not docs and total != -1:
            break
        new_on_page = 0
        for doc in docs:
            item = senate_item(doc)
            if item is None:
                if doc.get("_id") not in known:
                    new_on_page += 1   # not ready yet; keep looking deeper this round
                continue
            if item.source_id not in known:
                new_on_page += 1
            yield item
        total_known = max(total_known, total)
        if (page + 1) * page_size >= total_known or (not full and new_on_page == 0):
            break


def resolve_senate_media(client: httpx.Client, video_id: str, kind: str = "HLS") -> str:
    body = net.request(client, "POST", SENATE_MEDIA_API,
                       json={"file": video_id, "type": kind, "user": ""}).json()
    url = (((body.get("response") or {}).get("payload") or {}).get("data") or "")
    if not url.startswith("http"):
        raise RuntimeError(f"senate: no media URL for {video_id}: {body!r:.200}")
    return url.split("?")[0]


SOURCES = {"house": discover_house, "senate": discover_senate}
