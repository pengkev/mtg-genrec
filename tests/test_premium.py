import json
from types import SimpleNamespace

import pytest

from scrape import premium
from scrape.sources.moxfield import moxfield_seed_searches
from scrape.state import CorpusOutputs


def tcg_raw(player="Andrea Mengucci"):
    return {"id": 42, "deck": {"name": "Test", "format": "modern", "game": "magic", "playerName": player,
            "subDecks": {"maindeck": [{"cardID": 1, "quantity": 4}],
                         "sideboard": [{"cardID": 2, "quantity": 2}]}},
            "cards": {"1": {"name": "Lightning Bolt"}, "2": {"name": "Pyroblast"}}}


def test_creator_searches_never_seed_global_discovery():
    jobs = moxfield_seed_searches(["BoshNRoll", "ThrabenU"])
    assert [j["params"]["authorUserNames"] for j in jobs] == ["BoshNRoll", "ThrabenU"]
    assert all(j["key"].startswith("author:") for j in jobs)


def test_tcg_author_and_quantities():
    record = premium.tcgplayer_record(tcg_raw(), "Andrea Mengucci", "author-id")
    assert record["mainboard"] == [{"name": "Lightning Bolt", "quantity": 4}]
    assert record["sideboard"] == [{"name": "Pyroblast", "quantity": 2}]
    assert record["metadata"]["author_id"] == "author-id"
    with pytest.raises(ValueError, match="creator"):
        premium.tcgplayer_record(tcg_raw("Somebody Else"), "Andrea Mengucci", "author-id")


def test_archidekt_categories_exclude_maybeboard_and_do_not_duplicate():
    def card(name, cats, quantity=1):
        return {"card": {"oracleCard": {"name": name}}, "categories": cats, "quantity": quantity}
    raw = {"id": 1, "name": "Deck", "deckFormat": 3, "owner": {"username": "Creator"},
           "categories": [{"name": "Ramp", "includedInDeck": True},
                          {"name": "Artifact", "includedInDeck": True},
                          {"name": "Maybeboard", "includedInDeck": False}],
           "cards": [card("Sol Ring", ["Ramp", "Artifact"]), card("Island", ["Maybeboard"], 10),
                     card("Urza, Lord High Artificer", ["Commander", "Ramp"]), card("Negate", ["Sideboard"], 2)]}
    record = premium.archidekt_record(raw, 614906, ["Home", "Brewing"])
    assert record["mainboard"] == [{"name": "Sol Ring", "quantity": 1}]
    assert record["commanders"][0]["name"] == "Urza, Lord High Artificer"
    assert record["sideboard"][0]["quantity"] == 2
    assert record["metadata"]["folder_path"] == ["Home", "Brewing"]


def test_folder_requires_correct_embedded_identity():
    with pytest.raises(premium.PaginationError):
        premium.folder_payload("<html>Access denied</html>", 1)
    data = {"props": {"pageProps": {"redux": {"folders": {"rootFolder": {"id": 2, "decks": []}}}}}}
    with pytest.raises(premium.PaginationError):
        premium.folder_payload('<script id="__NEXT_DATA__">' + json.dumps(data) + '</script>', 1)


def test_tcg_partial_page_resume_and_author_query(tmp_path, monkeypatch):
    args = SimpleNamespace(output_dir=tmp_path, refresh=False, formats=["modern"], limit=1, max_pages=2, delay=0)
    outputs = CorpusOutputs(tmp_path / "cards.jsonl", tmp_path / "formats", flush_every=1,
                            seen_path=tmp_path / "seen.sqlite3")
    outputs.load(["modern"])
    collector = premium.Collector(args, outputs, set())
    fetched = []
    def get(url, **kwargs):
        if "/c/author/" in url:
            return SimpleNamespace(json=lambda: {"result": {"name": "Andrea Mengucci", "uuid": "verified-id"}})
        if "/content/decks/" in url:
            assert kwargs["params"]["authorID"] == "verified-id"
            return SimpleNamespace(json=lambda: {"offset": 0, "count": 2, "total": 2,
                    "result": [{"deckID": str(i), "deckData": {"format": "modern"}} for i in (42, 43)]})
        raw = tcg_raw()
        raw["id"] = int(url.rstrip("/").split("/")[-1])
        fetched.append(raw["id"])
        return SimpleNamespace(json=lambda: {"result": raw})
    monkeypatch.setattr(collector, "get", get)
    def append(record):
        outputs.mark_source("modern", "tcgplayer", record["source_id"])
        collector.added += 1
    monkeypatch.setattr(collector, "append", append)
    try:
        collector.tcgplayer("Andrea Mengucci")
        assert fetched == [42]
        assert next(iter(collector.state.values()))["offset"] == 0
        collector.added = 0
        collector.tcgplayer("Andrea Mengucci")
        assert fetched == [42, 43]
    finally:
        outputs.close()
        collector.session.close()


def test_unknown_cards_reject_whole_deck(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, refresh=False, formats=["modern"])
    collector = premium.Collector(args, None, {"lightning bolt"})
    try:
        with pytest.raises(ValueError, match="Unrecognized"):
            collector.append(premium.tcgplayer_record(tcg_raw(), "Andrea Mengucci", "id"))
    finally:
        collector.session.close()
