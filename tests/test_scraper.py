import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests


from scrape import scraper, state, http, sources
from scrape.sources import moxfield, mtgtop8, deckbox
from mtgdeck import data, metadata


def test_default_sources_are_constructed_only():
    args = scraper.parse_args([])
    assert args.sources == ("moxfield", "mtgtop8", "deckbox")
    assert args.deckbox_discovery == "balanced"
    assert not hasattr(args, "limited_output")


def test_normalize_deck_cards_collapses_quantities_faces_and_invalid_text():
    valid = {"fire", "ice", "sol ring", "island"}
    cards = data.normalize_deck_cards(
        ["2 Sol Ring", "Sol Ring", "Fire // Ice", "Island", "not a card"],
        valid,
        min_cards=3,
    )
    assert cards == ["fire", "island", "sol ring"]
    assert data.normalize_deck_cards(["Sol Ring"], valid, min_cards=2) is None


def test_oracle_loader_streams_scryfall_compressed_jsonl(tmp_path):
    path = tmp_path / "oracle_cards.jsonl.gz"
    cards = [
        {"object": "card", "name": "Fire // Ice", "card_faces": [{"name": "Fire"}, {"name": "Ice"}]},
        {"object": "card", "name": "Sol Ring"},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card) + "\n")
    assert metadata.load_oracle_names(path) == {"fire", "ice", "sol ring"}


def test_oracle_profiles_assign_color_and_multiple_strategic_roles(tmp_path):
    path = tmp_path / "oracle_cards.jsonl.gz"
    card = {
        "name": "Baleful Strix",
        "color_identity": ["U", "B"],
        "type_line": "Artifact Creature — Bird",
        "oracle_text": "Flying, deathtouch\nWhen this enters, draw a card.",
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(card) + "\n")
    assert metadata.load_oracle_profiles(path)["baleful strix"] == {
        "color": "multicolor",
        "roles": ("card-advantage", "creature", "engine"),
        "is_land": False,
    }


def test_mtgtop8_parsers_preserve_quantities_dates_and_export_sections():
    html = """
    <table><tr class="hover_tr"><td>02/03/26</td><td>
      <a href="event?e=42&d=123&f=MO">Deck</a>
    </td></tr></table>
    """
    assert [row["url"] for row in mtgtop8.mtgtop8_search_records(html)] == [
        "https://www.mtgtop8.com/event?e=42&d=123&f=MO"
    ]
    event_html = """
    <a href="mtgo?d=123&f=Modern_Example">MTGO</a>
    <a href="/dec?d=123&f=Modern_Example_by_Player">.dec</a>
    """
    assert mtgtop8.mtgtop8_export_url(event_html) == (
        "https://www.mtgtop8.com/dec?d=123&f=Modern_Example_by_Player"
    )
    boards = mtgtop8.parse_mtgo_decklist(
        "Deck\n4 Lightning Bolt\n1x Island\n\nSideboard\n2 Pyroblast\n", "modern"
    )
    assert boards["mainboard"] == [
        {"name": "Lightning Bolt", "quantity": 4},
        {"name": "Island", "quantity": 1},
    ]
    assert boards["sideboard"] == [{"name": "Pyroblast", "quantity": 2}]
    assert mtgtop8.mtgtop8_search_records(html)[0]["date"] == "02/03/26"

    commander_boards = mtgtop8.parse_mtgo_decklist(
        "1 Sol Ring\nSideboard\n1 Tymna the Weaver\n", "cedh"
    )
    assert commander_boards["commanders"] == [
        {"name": "Tymna the Weaver", "quantity": 1}
    ]

    dec_boards = mtgtop8.parse_mtgo_decklist(
        "// FORMAT : Legacy\n4 [ZEN] Arid Mesa\n2 [] Static Prison\n"
        "SB:  3 [IA] Pyroblast\n",
        "legacy",
    )
    assert dec_boards["mainboard"] == [
        {"name": "Arid Mesa", "quantity": 4},
        {"name": "Static Prison", "quantity": 2},
    ]
    assert dec_boards["sideboard"] == [{"name": "Pyroblast", "quantity": 3}]


def test_moxfield_boards_exclude_maybeboard():
    payload = {
        "boards": {
            "mainboard": {"cards": {"a": {"card": {"name": "Island"}}}},
            "commanders": {"cards": {"b": {"card": {"name": "Niv-Mizzet"}}}},
            "sideboard": {"cards": {"c": {"card": {"name": "Pyroblast"}}}},
            "maybeboard": {"cards": {"d": {"card": {"name": "Mountain"}}}},
        }
    }
    boards = moxfield.moxfield_boards(
        payload, {"island", "niv-mizzet", "pyroblast", "mountain"}
    )
    assert boards["mainboard"] == [{"name": "Island", "quantity": 1}]
    assert "maybeboard" not in boards


def test_deckbox_parsers_keep_constructed_rows_and_deck_zones():
    listing = """
    <div id="users_list_container"><div class="pagination_controls">
      <span>Page 1</span><a href="/decks/mtg?f=337&amp;p=2">Next</a>
    </div><table class="simple_table">
      <tr><th>Name</th><th>User</th><th>Color</th><th>Format</th></tr>
      <tr>
        <td><a href="/sets/42"><img class="sprite s_brick"><span>Meren</span></a></td>
        <td><a href="/users/tester">Deck Builder</a></td><td></td><td>Commander</td>
        <td>0</td><td>2</td><td>1</td><td><span>02-Sep-2026 20:50</span></td>
      </tr>
      <tr>
        <td><a href="/sets/43"><span>Draft Pool</span></a></td>
        <td>User</td><td></td><td>Draft</td><td><span>02-Sep-2026 20:40</span></td>
      </tr>
    </table></div>
    """
    assert deckbox.deckbox_search_records(listing, "commander") == [
        {
            "source_id": "42",
            "url": "https://deckbox.org/sets/42",
            "name": "Meren",
            "format": "commander",
            "date": "02-Sep-2026 20:50",
            "user": "Deck Builder",
            "built": True,
        }
    ]
    assert deckbox.deckbox_next_page_url(listing) is not None
    assert data.normal_date("02-Sep-2026 20:50") == "2026-09-02"

    detail = """
    <table class="set_cards main">
      <tr data-is-commander="1"><td class="card_count">1</td>
        <td class="card_name"><a>Meren of Clan Nel Toth</a></td></tr>
      <tr><td class="card_count">2</td><td class="card_name"><a>Forest</a></td></tr>
      <tr><td class="card_count">1</td><td class="card_name"><a>Not a Card</a></td></tr>
    </table>
    <table class="set_cards sideboard"><tr><td class="card_count">1</td>
      <td class="card_name"><a>Haywire Mite</a></td></tr></table>
    """
    boards = deckbox.deckbox_boards(
        detail, {"meren of clan nel toth", "forest", "haywire mite"}
    )
    assert boards["commanders"] == [
        {"name": "Meren of Clan Nel Toth", "quantity": 1}
    ]
    assert boards["mainboard"] == [{"name": "Forest", "quantity": 2}]
    assert boards["sideboard"] == [{"name": "Haywire Mite", "quantity": 1}]


def test_deckbox_collection_writes_dedicated_corpus_not_claimed_format(tmp_path):
    listing = """
    <div id="users_list_container"><div class="pagination_controls">Page 1</div>
    <table class="simple_table"><tr>
      <td><a href="/sets/42"><span>Janky Modern</span></a></td>
      <td><a href="/users/tester">Tester</a></td><td></td><td>Modern</td>
      <td>0</td><td>2</td><td>0</td><td><span>03-Sep-2026 10:00</span></td>
    </tr></table></div>
    """
    detail = """
    <table class="set_cards main">
      <tr><td class="card_count">4</td><td class="card_name"><a>Island</a></td></tr>
      <tr><td class="card_count">4</td><td class="card_name"><a>Mountain</a></td></tr>
    </table>
    """

    class Response:
        status_code = 200
        headers = {}

        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    class Session:
        def __init__(self):
            self.responses = [Response(listing), Response(detail)]

        def request(self, *_args, **_kwargs):
            return self.responses.pop(0)

    outputs = state.CorpusOutputs(
        tmp_path / "embedding.jsonl", tmp_path / "format_corpora"
    )
    outputs.load(["modern", "deckbox"])
    checkpoint = state.Checkpoint(tmp_path / "checkpoint.json")
    args = SimpleNamespace(
        max_pages_per_format=1,
        limit_per_format=None,
        timeout=1,
        retries=1,
        min_cards=2,
        deckbox_delay=0,
    )
    assert deckbox.collect_deckbox_format(
        "modern",
        outputs,
        checkpoint,
        {"island", "mountain"},
        args,
        Session(),
    ) == (1, 1, 1)
    outputs.close()

    assert not (tmp_path / "format_corpora" / "modern.jsonl").exists()
    record = json.loads(
        (tmp_path / "format_corpora" / "deckbox.jsonl").read_text()
    )
    assert record["source"] == "deckbox"
    assert record["format"] == "modern"


def test_migrate_deckbox_records_preserves_other_rows_and_deduplicates(tmp_path):
    directory = tmp_path / "format_corpora"
    directory.mkdir()
    deckbox = data.make_decklist_record(
        source="deckbox",
        source_id="42",
        format_name="legacy",
        url="https://deckbox.org/sets/42",
        boards={"mainboard": [{"name": "Island", "quantity": 60}]},
    )
    competitive = data.make_decklist_record(
        source="mtgtop8",
        source_id="99",
        format_name="legacy",
        url="https://www.mtgtop8.com/event?d=99",
        boards={"mainboard": [{"name": "Mountain", "quantity": 60}]},
    )
    competitive_line = json.dumps(competitive) + "\n"
    (directory / "legacy.jsonl").write_text(
        competitive_line + json.dumps(deckbox) + "\n", encoding="utf-8"
    )
    duplicate = dict(deckbox)
    duplicate["source_id"] = "43"
    duplicate["deck_id"] = "deckbox:43"
    (directory / "modern.jsonl").write_text(
        json.dumps(duplicate) + "\n", encoding="utf-8"
    )

    assert state.migrate_deckbox_records(directory) == (2, 1, 2)
    assert (directory / "legacy.jsonl").read_text() == competitive_line
    assert (directory / "modern.jsonl").read_text() == ""
    migrated = [
        json.loads(line)
        for line in (directory / "deckbox.jsonl").read_text().splitlines()
    ]
    assert migrated == [deckbox]
    assert state.migrate_deckbox_records(directory) == (0, 0, 0)


def test_corpus_writer_indexes_and_appends_without_duplicate_rows(tmp_path):
    corpus = tmp_path / "embedding.jsonl"
    corpus.write_text(json.dumps({"cards": ["island", "sol ring"]}) + "\n", encoding="utf-8")
    writer = state.CorpusWriter(corpus)
    assert writer.load() == 1
    assert not writer.append(["island", "sol ring"])
    assert writer.append(["mountain", "sol ring"])
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"cards": ["island", "sol ring"]},
        {"cards": ["mountain", "sol ring"]},
    ]


def test_corpus_outputs_writes_format_even_when_deck_already_exists_globally(tmp_path):
    combined = tmp_path / "embedding.jsonl"
    cards = ["island", "sol ring"]
    combined.write_text(json.dumps({"cards": cards}) + "\n", encoding="utf-8")
    format_directory = tmp_path / "format_corpora"
    outputs = state.CorpusOutputs(combined, format_directory)
    combined_count, format_counts = outputs.load(["modern", "legacy"])
    assert combined_count == 1
    assert format_counts == {"modern": 0, "legacy": 0}

    decklist = data.make_decklist_record(
        source="mtgtop8",
        source_id="123",
        format_name="modern",
        url="https://www.mtgtop8.com/event?d=123",
        deck_date="02/03/26",
        boards={
            "mainboard": [
                {"name": "Island", "quantity": 4},
                {"name": "Sol Ring", "quantity": 1},
            ],
            "sideboard": [{"name": "Pyroblast", "quantity": 2}],
        },
    )
    assert outputs.append("modern", cards, decklist) == (False, True)
    assert outputs.append("modern", cards, decklist) == (False, False)
    rows = [
        json.loads(line)
        for line in (format_directory / "modern.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["format"] == "modern"
    assert rows[0]["date"] == "2026-03-02"
    assert rows[0]["mainboard"][0] == {"name": "Island", "quantity": 4}
    assert rows[0]["sideboard"] == [{"name": "Pyroblast", "quantity": 2}]
    assert "cards" not in rows[0]
    assert not (format_directory / "legacy.jsonl").exists()


def test_checkpoint_is_atomic_and_tracks_each_bucket(tmp_path):
    path = tmp_path / "scrape.checkpoint.json"
    checkpoint = state.Checkpoint(path)
    checkpoint.save("mtgtop8", "modern", 7)
    checkpoint.save("moxfield", "commander", 3, complete=True)
    resumed = state.Checkpoint(path)
    assert resumed.next_page("mtgtop8", "modern", 0) == 7
    assert resumed.is_complete("moxfield", "commander")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == state.Checkpoint.VERSION
    assert not path.with_name(path.name + ".tmp").exists()


def test_scrape_buckets_support_round_robin_page_scheduling():
    buckets = scraper.build_scrape_buckets(
        ["moxfield", "mtgtop8", "deckbox"], ["legacy", "premodern", "cedh"]
    )
    assert buckets == [
        ("moxfield", "legacy"),
        ("mtgtop8", "legacy"),
        ("deckbox", "legacy"),
        ("moxfield", "premodern"),
        ("mtgtop8", "premodern"),
        ("deckbox", "premodern"),
        ("mtgtop8", "cedh"),
    ]

    # The scheduler processes this complete bucket list once per cycle, so
    # page depth advances evenly rather than exhausting Legacy first.
    visits = [bucket for _cycle in range(2) for bucket in buckets]
    assert visits == buckets + buckets


@pytest.mark.parametrize("page_limit", [None, 2])
def test_main_scrapes_until_exhausted_unless_page_limit_is_explicit(tmp_path, monkeypatch, page_limit):
    visits = []
    final_pages = {"modern": 102, "cedh": 104}

    def collect(format_name, outputs, checkpoint, valid_names, args, session):
        page = checkpoint.next_page("mtgtop8", format_name, 0)
        visits.append((format_name, page))
        checkpoint.save(
            "mtgtop8", format_name, page + 1,
            complete=page + 1 == final_pages[format_name],
        )
        return 1, 0, 0

    monkeypatch.setattr(scraper, "refresh_oracle_cards", lambda *_args: None)
    monkeypatch.setattr(scraper, "load_oracle_names", lambda *_args: {"island"})
    monkeypatch.setattr(scraper, "make_mtgtop8_session", object)
    monkeypatch.setattr(scraper, "collect_mtgtop8_format", collect)
    checkpoint_path = tmp_path / "checkpoint.json"
    argv = [
        "--sources", "mtgtop8", "--formats", "modern", "cedh",
        "--output", str(tmp_path / "embedding.jsonl"),
        "--format-output-dir", str(tmp_path / "formats"),
        "--checkpoint", str(checkpoint_path),
    ]
    if page_limit is not None:
        argv.extend(["--max-pages-per-format", str(page_limit)])

    assert scraper.main(argv) == 0
    expected = [
        (format_name, page)
        for page in range(page_limit or max(final_pages.values()))
        for format_name in final_pages
        if page < final_pages[format_name]
    ]
    assert visits == expected
    checkpoint = state.Checkpoint(checkpoint_path)
    for format_name, final_page in final_pages.items():
        assert checkpoint.next_page("mtgtop8", format_name, 0) == (page_limit or final_page)
        assert checkpoint.is_complete("mtgtop8", format_name) == (page_limit is None)


class ScrapeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def scrape_response(payload=None, *, text=None, status=200, url=None):
    import requests
    response = requests.Response()
    response.status_code = status
    response.url = url
    response._content = (text if text is not None else json.dumps(payload)).encode()
    return response


def scrape_outputs(tmp_path):
    outputs = state.CorpusOutputs(tmp_path / 'embedding.jsonl', tmp_path / 'formats',
                                   flush_every=500, seen_path=tmp_path / 'seen.sqlite3')
    outputs.load(['modern', 'commander', 'deckbox'])
    return outputs


def scrape_args(**values):
    args = scraper.parse_args(['--oracle-refresh', 'never', '--max-pages-per-format', '1',
                              '--moxfield-delay', '0', '--mtgtop8-delay', '0', '--deckbox-delay', '0',
                              '--min-cards', '1', '--retries', '1'])
    for key, value in values.items():
        setattr(args, key, value)
    return args


def mox_listing(ids, *, page=1, pages=1, total=None):
    return scrape_response({'pageNumber': page, 'pageSize': 100, 'totalPages': pages,
                            'totalResults': len(ids) if total is None else total,
                            'data': [{'publicId': i, 'format': 'modern'} for i in ids]})


def mox_deck(name='Island', card_id=None):
    return scrape_response({'format': 'modern', 'createdAtUtc': '2026-09-01T00:00:00Z',
                            'boards': {'mainboard': {'cards': {'x': {'quantity': 4, 'card': {
                                'name': name, 'id': card_id, 'type_line': 'Instant',
                            }}}}}})


@pytest.mark.parametrize('source', ['moxfield', 'mtgtop8'])
def test_full_corpus_restart_indexes_all_sources_and_rejects_duplicates(tmp_path, source):
    path = tmp_path / 'modern.jsonl'
    record = data.make_decklist_record(source=source, source_id='a', format_name='modern',
                                         url='https://example.test/a',
                                         boards={'mainboard': [{'name': 'Island', 'quantity': 4}]})
    writer = state.DecklistWriter(path)
    assert writer.append(record)
    writer.close()
    resumed = state.DecklistWriter(path)
    assert resumed.load() == 1
    assert not resumed.append(record)
    assert (source, 'a') in resumed.source_keys


def test_identity_cache_remembers_content_duplicates_across_restarts(tmp_path):
    outputs = scrape_outputs(tmp_path)
    record = data.make_decklist_record(source='moxfield', source_id='a', format_name='modern',
                                         url='https://example.test/a',
                                         boards={'mainboard': [{'name': 'Island', 'quantity': 4}]})
    assert outputs.append('modern', ['island'], record) == (True, True)
    assert outputs.append('modern', ['island'], {**record, 'source_id': 'b'}) == (False, False)
    outputs.close()
    resumed = scrape_outputs(tmp_path)
    assert resumed.has_source('modern', 'moxfield', 'a')
    assert resumed.has_source('modern', 'moxfield', 'b')
    assert not resumed.has_source('commander', 'moxfield', 'b')
    resumed.close()


def test_v3_completed_moxfield_search_opens_new_queries_without_losing_resume(tmp_path):
    path = tmp_path / 'checkpoint.json'
    path.write_text(json.dumps({'schema_version': 3, 'buckets': {
        'moxfield:modern': {'next_page': 101, 'complete': True},
        'moxfield:commander': {'next_page': 97, 'complete': False},
        'deckbox:modern': {'next_page': 101, 'complete': False},
    }}))
    checkpoint = state.Checkpoint(path)
    for fmt in ['modern', 'commander']:
        checkpoint.add_searches('moxfield', fmt, moxfield.moxfield_seed_searches())
    assert not checkpoint.is_complete('moxfield', 'modern')
    assert checkpoint.current_search('moxfield', 'modern')['params']['sortType'] == 'created'
    assert checkpoint.current_search('moxfield', 'commander')['next_page'] == 97
    checkpoint.add_searches('deckbox', 'modern', deckbox.deckbox_seed_searches('modern'))
    assert checkpoint.current_search('deckbox', 'modern')['next_page'] == 1
    checkpoint.write()
    resumed = state.Checkpoint(path)
    assert resumed.current_search('moxfield', 'commander')['next_page'] == 97


def test_missing_moxfield_deck_does_not_stop_later_decks_or_lose_date(tmp_path):
    outputs = scrape_outputs(tmp_path)
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    session = ScrapeSession([mox_listing(['gone', 'good']), scrape_response(status=404), mox_deck()])
    assert moxfield.collect_moxfield_format('modern', outputs, checkpoint, {'island'}, scrape_args(), session) == (1, 1, 1)
    saved = json.loads((tmp_path / 'formats/modern.jsonl').read_text())
    assert saved['source_id'] == 'good'
    assert saved['date'] == '2026-09-01'
    assert checkpoint.state['moxfield:modern']['searches']['views:descending']['complete']
    assert not checkpoint.is_complete('moxfield', 'modern')
    outputs.close()


@pytest.mark.parametrize(
    ('status', 'error_type'),
    [(403, http.ScrapeHTTPError),
     (429, http.TransientRequestError),
     (503, http.TransientRequestError)],
)
def test_site_errors_preserve_moxfield_page_and_are_not_treated_as_missing(
        tmp_path, status, error_type):
    outputs = scrape_outputs(tmp_path)
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    session = ScrapeSession([mox_listing(['first', 'failed']), mox_deck(), scrape_response(status=status)])
    with pytest.raises(error_type) as error:
        moxfield.collect_moxfield_format('modern', outputs, checkpoint, {'island'}, scrape_args(), session)
    assert error.value.status == status
    assert checkpoint.current_search('moxfield', 'modern')['next_page'] == 1
    outputs.close()
    assert json.loads((tmp_path / 'formats/modern.jsonl').read_text())['source_id'] == 'first'


def test_moxfield_skips_known_details_and_discovers_nonland_queries(tmp_path):
    outputs = scrape_outputs(tmp_path)
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    first = ScrapeSession([mox_listing(['known']), mox_deck('Lightning Bolt', 'bolt-id')])
    moxfield.collect_moxfield_format('modern', outputs, checkpoint, {'lightning bolt'}, scrape_args(), first)
    second = ScrapeSession([mox_listing(['known'])])
    assert moxfield.collect_moxfield_format('modern', outputs, checkpoint, {'lightning bolt'}, scrape_args(), second) == (0, 0, 0)
    assert len(second.calls) == 1
    jobs = checkpoint.state['moxfield:modern']['searches']
    assert jobs['cardId:lightning bolt']['params']['cardId'] == 'bolt-id'
    assert 'q' not in jobs['cardId:lightning bolt']['params']
    outputs.close()


def test_moxfield_discovery_excludes_lands_and_maybeboard_and_deduplicates_printings():
    def entry(name, card_id, type_line):
        return {'card': {'name': name, 'id': card_id, 'type_line': type_line}}
    payload = {'boards': {
        'mainboard': {'cards': {'1': entry('Forest', 'f', 'Basic Land'),
                                '2': entry('Bolt', 'b1', 'Instant'),
                                '3': entry('Bolt', 'b2', 'Instant')}},
        'commanders': {'cards': {'4': entry('Winota', 'w', 'Legendary Creature')}},
        'maybeboard': {'cards': {'5': entry('Never', 'n', 'Instant')}},
    }}
    searches = list(moxfield.moxfield_card_searches(payload, 'cards'))
    assert {q['key'] for q in searches} == {'cardId:bolt', 'commanderCardId:winota'}
    assert len(list(moxfield.moxfield_card_searches(payload, 'commanders'))) == 1
    assert not list(moxfield.moxfield_card_searches(payload, 'off'))


def test_moxfield_capped_query_expands_and_remains_resumable(tmp_path):
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    checkpoint.add_searches('moxfield', 'modern', [{'key': 'commander:test',
        'next_page': 100, 'params': {'commanderCardId': 'test', 'sortType': 'created', 'sortDirection': 'descending'}}])
    outputs = scrape_outputs(tmp_path)
    session = ScrapeSession([mox_listing(['one'], page=100, pages=100, total=10000), mox_deck()])
    moxfield.collect_moxfield_format('modern', outputs, checkpoint, {'island'}, scrape_args(), session)
    jobs = checkpoint.state['moxfield:modern']['searches']
    assert jobs['commander:test']['stop_reason'] == 'result window cap'
    assert jobs['commander:test:bracket:1']['params']['minBracket'] == 1
    assert jobs['commander:test:ascending']['params']['sortDirection'] == 'ascending'
    assert not checkpoint.is_complete('moxfield', 'modern')
    outputs.close()


@pytest.mark.parametrize('payload', [{}, {'data': []}, {'pageNumber': 2, 'data': []},
                                    {'pageNumber': 1, 'data': [{'publicId': 'a', 'format': 'legacy'}]}])
def test_moxfield_invalid_search_cannot_complete_bucket(tmp_path, payload):
    outputs = scrape_outputs(tmp_path)
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    with pytest.raises(http.PaginationError):
        moxfield.collect_moxfield_format('modern', outputs, checkpoint, {'island'}, scrape_args(),
                                        ScrapeSession([scrape_response(payload)]))
    assert checkpoint.current_search('moxfield', 'modern')['next_page'] == 1
    outputs.close()


def test_repeated_page_detection_survives_checkpoint_reload(tmp_path):
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    checkpoint.add_searches('moxfield', 'modern', [{'key': 'test', 'params': {}}])
    job = checkpoint.current_search('moxfield', 'modern')
    job['last_fingerprint'] = http.page_fingerprint(['b', 'a'])
    checkpoint.write()
    resumed = state.Checkpoint(checkpoint.path)
    with pytest.raises(http.PaginationError):
        http.validate_page_fingerprint(resumed.current_search('moxfield', 'modern'), ['a', 'b'])


def test_deckbox_searches_partition_colors_and_all_public_sort_orders():
    jobs = list(deckbox.deckbox_seed_searches('modern'))
    assert len(jobs) == 264
    assert [job['key'].split(':')[1] for job in jobs[:8]] == [
        'all', '1', '2', '3', '4', '5', '1.2', '6',
    ]
    assert any(j['params'] == {'f': '335!2a1.2', 's': 'c', 'o': 'a'} for j in jobs)
    assert any(j['params']['f'] == '335!2a6' for j in jobs)
    assert {(j['params']['s'], j['params']['o']) for j in jobs} == {
        ('c', 'a'), ('c', 'd'), ('b', 'a'), ('b', 'd'),
        ('a', 'a'), ('a', 'd'), ('n', 'a'), ('n', 'd'),
    }


def test_capped_deckbox_card_search_expands_to_alternate_sort_windows():
    job = {
        'key': 'card:659:updated:d',
        'label': 'Dralnu, Lich Lord',
        'params': {'f': '337!51659', 's': 'c', 'o': 'd'},
    }
    searches = list(deckbox.expand_capped_deckbox_search(job))
    assert {search['key'] for search in searches} == {
        'card:659:views:a', 'card:659:views:d',
        'card:659:stars:a', 'card:659:stars:d',
        'card:659:name:a', 'card:659:name:d',
    }
    assert all(search['params']['f'] == '337!51659' for search in searches)
    assert list(deckbox.expand_capped_deckbox_search(job)) == []


@pytest.mark.parametrize('response', [scrape_response(status=404), scrape_response(
    text='<div id="users_list_container"><div class="pagination_controls">Page 1</div></div>')])
def test_deckbox_dead_page_finishes_only_that_search(tmp_path, response):
    outputs = scrape_outputs(tmp_path)
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    checkpoint.add_searches('deckbox', 'modern', [{'key': 'old', 'next_page': 101, 'params': {'f': '335'}}])
    assert deckbox.collect_deckbox_format('modern', outputs, checkpoint, {'island'}, scrape_args(),
                                         ScrapeSession([response])) == (0, 0, 0)
    assert checkpoint.state['deckbox:modern']['searches']['old']['stop_reason'] == 'pagination unavailable'
    assert not checkpoint.is_complete('deckbox', 'modern')
    assert checkpoint.current_search('deckbox', 'modern')['next_page'] == 1
    outputs.close()


def test_deckbox_empty_or_redirected_response_is_not_exhaustion(tmp_path):
    outputs = scrape_outputs(tmp_path)
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    response = scrape_response(text='<html>Please log in</html>')
    with pytest.raises(http.PaginationError):
        deckbox.collect_deckbox_format('modern', outputs, checkpoint, {'island'}, scrape_args(), ScrapeSession([response]))
    assert checkpoint.current_search('deckbox', 'modern')['next_page'] == 1
    outputs.close()


def test_deckbox_discovery_extracts_canonical_card_id_and_skips_lands():
    html = '''<script>Tcg.set = new Tcg.MtgDeck({"id":42},
        {"row-1":{"id":659,"name":"Dralnu, Lich Lord"},"row-2":{"id":99,"name":"Forest"}});</script>
        <table class="set_cards main"><tr data-id="row-1"><td>1</td><td data-tt="2066590">Dralnu</td>
        <td></td><td></td><td>Legendary Creature</td></tr>
        <tr data-id="row-2"><td>1</td><td>Forest</td><td></td><td></td><td>Basic Land</td></tr></table>'''
    candidates = list(deckbox.deckbox_card_candidates(html, {
        'dralnu, lich lord': {
            'color': 'multicolor', 'roles': ('graveyard', 'creature'), 'is_land': False,
        },
        'forest': {'color': 'G', 'roles': ('mana',), 'is_land': True},
    }))
    assert candidates == [{
        'card_id': '659', 'name': 'Dralnu, Lich Lord',
        'color': 'multicolor', 'roles': ('graveyard', 'creature'),
    }]
    jobs = list(deckbox.deckbox_card_searches(candidates, 'commander'))
    assert len(jobs) == 2
    assert {j['params']['f'] for j in jobs} == {'337!51659'}


def test_deckbox_card_selection_uses_popularity_with_color_and_role_balance(tmp_path):
    outputs = scrape_outputs(tmp_path)
    popular = [
        {'card_id': '1', 'name': 'White Answer', 'color': 'W', 'roles': ('interaction',)},
        {'card_id': '2', 'name': 'Blue Draw', 'color': 'U', 'roles': ('card-advantage',)},
        {'card_id': '3', 'name': 'Black Graveyard', 'color': 'B', 'roles': ('graveyard',)},
        {'card_id': '4', 'name': 'Red Answer', 'color': 'R', 'roles': ('interaction',)},
        {'card_id': '5', 'name': 'Green Ramp', 'color': 'G', 'roles': ('mana',)},
        {'card_id': '6', 'name': 'Gold Creature', 'color': 'multicolor', 'roles': ('creature',)},
        {'card_id': '7', 'name': 'Colorless Engine', 'color': 'colorless', 'roles': ('engine',)},
    ]
    less_popular = {
        'card_id': '8', 'name': 'Less Popular White Answer',
        'color': 'W', 'roles': ('interaction',),
    }
    for index, color in enumerate(sources.DECKBOX_CARD_COLORS):
        candidates = [*popular, *([less_popular] if index < 2 else [])]
        assert outputs.record_deckbox_card_sample('modern', str(index), color, candidates)
    selected = outputs.select_balanced_deckbox_cards(
        'modern', limit=7, min_samples=7, min_color_buckets=7, min_decks=1,
    )
    assert {card['card_id'] for card in selected} == {str(index) for index in range(1, 8)}
    assert next(card for card in selected if card['color'] == 'W')['name'] == 'White Answer'
    outputs.close()


def test_mtgtop8_current_codes_and_full_history_are_separate_queries():
    assert sources.MTGTOP8_FORMATS['explorer'] == 'EXP'
    assert sources.MTGTOP8_FORMATS['extended'] == 'EX'
    assert sources.MTGTOP8_FORMATS['duel-commander'] == 'EDH'
    assert sources.MTGTOP8_FORMATS['mtgo-commander'] == 'EDHM'
    assert 'commander' not in sources.MTGTOP8_FORMATS
    jobs = mtgtop8.mtgtop8_seed_searches('15/03/1993')
    assert jobs[-1]['params']['date_start'] == '15/03/1993'
    assert jobs[-1]['params']['date_end'] == '31/12/1993'
    assert jobs[0]['params']['date_end'].endswith(str(mtgtop8.date.today().year))


def test_mtgtop8_fetches_direct_export_once_and_saves_page_after_output(tmp_path):
    outputs = scrape_outputs(tmp_path)
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    listing = '''<select name="format"><option value="MO" selected>Modern</option></select>
        <input name="current_page" value="1"><div>1 decks matching</div>
        <tr class="hover_tr"><td><a href="event?e=1&d=123&f=MO">Deck</a></td><td>01/09/26</td></tr>'''
    session = ScrapeSession([scrape_response(text=listing), scrape_response(text='4 [SET] Island\nSB: 2 [] Mountain')])
    assert mtgtop8.collect_mtgtop8_format('modern', outputs, checkpoint, {'island', 'mountain'},
                                         scrape_args(), session) == (1, 1, 1)
    assert session.calls[0][2]['data']['current_page'] == '1'
    assert session.calls[1][1] == 'https://www.mtgtop8.com/dec?d=123'
    record = json.loads((tmp_path / 'formats/modern.jsonl').read_text())
    assert record['mainboard'] == [{'name': 'Island', 'quantity': 4}]
    assert record['sideboard'] == [{'name': 'Mountain', 'quantity': 2}]
    assert checkpoint.path.exists()
    outputs.close()


def test_refresh_searches_reopens_completed_jobs_but_keeps_active_progress(tmp_path):
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    checkpoint.add_searches('moxfield', 'modern', [
        {'key': 'done', 'params': {}, 'complete': True, 'next_page': 101, 'last_fingerprint': 'old'},
        {'key': 'active', 'params': {}, 'next_page': 17}])
    checkpoint.refresh_completed()
    assert checkpoint.current_search('moxfield', 'modern')['next_page'] == 17
    reopened = checkpoint.state['moxfield:modern']['searches']['done']
    assert reopened['next_page'] == 1
    assert not reopened['complete']
    assert 'last_fingerprint' not in reopened


def test_moxfield_format_names_match_public_client():
    assert sources.MOXFIELD_FORMATS['duel-commander'] == 'duelCommander'
    assert sources.MOXFIELD_FORMATS['pauper-commander'] == 'pauperEdh'
    assert sources.MOXFIELD_FORMATS['canadian-highlander'] == 'highlanderCanadian'
    assert sources.MOXFIELD_FORMATS['penny-dreadful'] == 'pennyDreadful'


def test_migration_repairs_proven_mtgtop8_labels_and_preserves_other_sources(tmp_path):
    record = data.make_decklist_record(
        source='mtgtop8', source_id='42', format_name='commander',
        url='https://www.mtgtop8.com/event?e=1&d=42&f=EDH',
        boards={'mainboard': [{'name': 'Island', 'quantity': 4}]})
    other = {**record, 'source': 'moxfield', 'source_id': 'keep', 'url': 'https://moxfield.com/decks/keep'}
    text = json.dumps(other) + '\n'
    (tmp_path / 'commander.jsonl').write_text(json.dumps(record) + '\n' + text)
    assert state.migrate_mtgtop8_formats(tmp_path) == 1
    assert (tmp_path / 'commander.jsonl').read_text() == text
    assert json.loads((tmp_path / 'duel-commander.jsonl').read_text()) == {**record, 'format': 'duel-commander'}
    assert state.migrate_mtgtop8_formats(tmp_path) == 0


def test_mtgtop8_wrong_format_or_page_cannot_advance():
    html = '<select name="format"><option value="EX" selected>Extended</option></select><input name="current_page" value="1">0 decks matching'
    with pytest.raises(http.PaginationError):
        mtgtop8.validate_mtgtop8_search(html, 'EXP', 1)
    with pytest.raises(http.PaginationError):
        mtgtop8.validate_mtgtop8_search(html, 'EX', 2)


def test_main_flushes_partial_progress_on_interrupt(tmp_path, monkeypatch):
    def collect(format_name, outputs, checkpoint, *_args):
        record = data.make_decklist_record(
            source='moxfield', source_id='42', format_name=format_name, url='https://moxfield.com/decks/42',
            boards={'mainboard': [{'name': 'Island', 'quantity': 4}]})
        outputs.append(format_name, ['island'], record)
        raise KeyboardInterrupt
    monkeypatch.setattr(scraper, 'refresh_oracle_cards', lambda *_args: None)
    monkeypatch.setattr(scraper, 'load_oracle_names', lambda *_args: {'island'})
    monkeypatch.setattr(scraper, 'make_moxfield_session', object)
    monkeypatch.setattr(scraper, 'collect_moxfield_format', collect)
    argv = ['--sources', 'moxfield', '--formats', 'modern', '--output', str(tmp_path / 'embedding.jsonl'),
            '--format-output-dir', str(tmp_path / 'formats'), '--checkpoint', str(tmp_path / 'checkpoint.json')]
    with pytest.raises(KeyboardInterrupt):
        scraper.main(argv)
    assert json.loads((tmp_path / 'embedding.jsonl').read_text()) == {'cards': ['island']}
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    assert checkpoint.current_search('moxfield', 'modern')['next_page'] == 1
    outputs = state.CorpusOutputs(tmp_path / 'embedding.jsonl', tmp_path / 'formats',
                                   seen_path=tmp_path / 'formats/.diverse_scraper.seen.sqlite3')
    outputs.load(['modern'])
    assert outputs.has_source('modern', 'moxfield', '42')
    outputs.close()


def test_rate_limit_honors_retry_after_and_does_not_sleep_after_final_attempt(monkeypatch):
    sleeps = []
    monkeypatch.setattr(http.time, 'sleep', sleeps.append)
    limited = scrape_response(status=429)
    limited.headers['Retry-After'] = '17'
    session = ScrapeSession([limited, limited])
    with pytest.raises(http.TransientRequestError) as error:
        http.request_with_retries(session, 'GET', 'https://example.test', timeout=1, retries=2)
    assert error.value.status == 429
    assert sleeps == [17]


def test_select_searches_preserves_suspended_progress_when_scope_changes(tmp_path):
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    checkpoint.add_searches('mtgtop8', 'modern', [
        {'key': 'old', 'params': {}, 'next_page': 70},
        {'key': 'new', 'params': {}, 'next_page': 5}])
    checkpoint.select_searches('mtgtop8', 'modern', ['new'])
    assert checkpoint.state['mtgtop8:modern']['queue'] == ['new']
    checkpoint.select_searches('mtgtop8', 'modern', ['old', 'new'])
    assert checkpoint.state['mtgtop8:modern']['queue'] == ['new', 'old']
    assert checkpoint.state['mtgtop8:modern']['searches']['old']['next_page'] == 70


def test_remove_searches_discards_only_obsolete_generated_jobs(tmp_path):
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    checkpoint.add_searches('deckbox', 'modern', [
        {'key': 'colors:all:updated:d', 'params': {}},
        {'key': 'card:1:updated:d', 'params': {}, 'next_page': 70},
        {'key': 'card:2:updated:d', 'params': {}, 'next_page': 5},
    ])
    assert checkpoint.remove_searches('deckbox', 'modern', ['card:1:updated:d']) == 1
    jobs = checkpoint.state['deckbox:modern']['searches']
    assert set(jobs) == {'colors:all:updated:d', 'card:2:updated:d'}
    assert 'card:1:updated:d' not in checkpoint.state['deckbox:modern']['queue']


def test_transport_failure_is_structured_and_stops_after_local_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(http.time, 'sleep', sleeps.append)
    session = ScrapeSession([requests.ConnectionError('reset'), requests.Timeout('slow')])
    with pytest.raises(http.TransientRequestError) as error:
        http.request_with_retries(
            session, 'GET', 'https://example.test', timeout=1, retries=2)
    assert error.value.status is None
    assert isinstance(error.value.cause, requests.Timeout)
    assert len(session.calls) == 2
    assert sleeps == [2]


def test_main_opens_source_circuit_after_one_exhausted_transport_failure(tmp_path, monkeypatch):
    visits = []

    def collect(format_name, *_args):
        visits.append(format_name)
        raise http.TransientRequestError(
            'https://www.mtgtop8.com/search', requests.Timeout('site unavailable'))

    monkeypatch.setattr(scraper, 'refresh_oracle_cards', lambda *_args: None)
    monkeypatch.setattr(scraper, 'load_oracle_names', lambda *_args: {'island'})
    monkeypatch.setattr(scraper, 'make_mtgtop8_session', object)
    monkeypatch.setattr(scraper, 'collect_mtgtop8_format', collect)
    result = scraper.main([
        '--sources', 'mtgtop8', '--formats', 'modern', 'legacy', 'vintage',
        '--oracle-refresh', 'never', '--output', str(tmp_path / 'embedding.jsonl'),
        '--format-output-dir', str(tmp_path / 'formats'),
        '--checkpoint', str(tmp_path / 'checkpoint.json'),
    ])
    assert result == 1
    assert visits == ['modern']
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    assert checkpoint.current_search('mtgtop8', 'modern')['next_page'] == 1
    assert checkpoint.current_search('mtgtop8', 'legacy')['next_page'] == 1


def test_permanently_missing_id_is_cached_without_a_deck_record(tmp_path):
    outputs = scrape_outputs(tmp_path)
    outputs.mark_source('modern', 'moxfield', 'gone')
    outputs.flush()
    outputs.close()
    resumed = scrape_outputs(tmp_path)
    assert resumed.has_source('modern', 'moxfield', 'gone')
    assert not (tmp_path / 'formats/modern.jsonl').exists()
    resumed.close()


def test_mtgtop8_malformed_deck_is_skipped_without_losing_later_decks(tmp_path):
    listing = '''<select name="format"><option value="MO" selected>Modern</option></select>
        <input name="current_page" value="1"><div>2 decks matching</div>
        <tr class="hover_tr"><td><a href="event?e=1&d=bad&f=MO">Bad</a></td><td>01/09/26</td></tr>
        <tr class="hover_tr"><td><a href="event?e=1&d=good&f=MO">Good</a></td><td>01/09/26</td></tr>'''
    session = ScrapeSession([
        scrape_response(text=listing),
        scrape_response(text='<html>not a deck</html>'),
        scrape_response(text='<html>no export link</html>'),
        scrape_response(text='4 [SET] Island'),
    ])
    outputs = scrape_outputs(tmp_path)
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    assert mtgtop8.collect_mtgtop8_format(
        'modern', outputs, checkpoint, {'island'}, scrape_args(), session) == (1, 1, 1)
    assert outputs.has_source('modern', 'mtgtop8', 'bad')
    saved = json.loads((tmp_path / 'formats/modern.jsonl').read_text())
    assert saved['source_id'] == 'good'
    outputs.close()


@pytest.mark.parametrize('kind', ['embedding', 'decklist'])
def test_corpus_index_memory_stays_bounded_as_corpus_grows(tmp_path, kind):
    import tracemalloc

    path = tmp_path / 'corpus.jsonl'
    count = 12_000

    def record(number):
        if kind == 'embedding':
            return {'cards': [f'card {number}', 'island']}
        return data.make_decklist_record(
            source='moxfield', source_id=str(number), format_name='modern',
            url=f'https://example.test/{number}',
            boards={'mainboard': [{'name': f'Card {number}', 'quantity': 4}]},
        )

    with path.open('w', encoding='utf-8') as handle:
        for number in range(count):
            handle.write(json.dumps(record(number)) + '\n')
    writer_type = state.CorpusWriter if kind == 'embedding' else state.DecklistWriter
    writer = writer_type(path, flush_every=500)
    index_directory = Path(writer._index.directory.name)
    try:
        tracemalloc.start()
        assert writer.load() == count
        for number in range(count, count + 1_000):
            row = record(number)
            assert writer.append(row['cards'] if kind == 'embedding' else row)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # The former Python sets exceed this budget at this size. This checks
        # both startup indexing and subsequent growth without retaining rows.
        assert peak < 1024 * 1024
        first = record(0)
        assert not writer.append(first['cards'] if kind == 'embedding' else first)
        assert len(writer.fingerprints) == count + 1_000
    finally:
        tracemalloc.stop()
        writer.close()
    assert not index_directory.exists()


@pytest.mark.parametrize('kind', ['embedding', 'decklist'])
def test_restart_rebuilds_index_after_corpus_replacement(tmp_path, kind):
    path = tmp_path / 'corpus.jsonl'
    if kind == 'embedding':
        writer_type = state.CorpusWriter
        original = ['island']
    else:
        writer_type = state.DecklistWriter
        original = data.make_decklist_record(
            source='moxfield', source_id='1', format_name='modern',
            url='https://example.test/1',
            boards={'mainboard': [{'name': 'Island', 'quantity': 4}]},
        )
    writer = writer_type(path)
    assert writer.append(original)
    writer.close()
    path.write_text('', encoding='utf-8')
    resumed = writer_type(path)
    try:
        assert resumed.load() == 0
        assert resumed.append(original)
    finally:
        resumed.close()


def test_deckbox_migration_streams_large_legacy_corpus(tmp_path):
    import tracemalloc

    path = tmp_path / 'modern.jsonl'
    count = 1_000
    with path.open('w', encoding='utf-8') as handle:
        for number in range(count):
            record = data.make_decklist_record(
                source='deckbox', source_id=str(number), format_name='modern',
                url=f'https://deckbox.org/sets/{number}',
                boards={'mainboard': [
                    {'name': f'Card {number}-{card}', 'quantity': 1} for card in range(30)
                ]},
            )
            handle.write(json.dumps(record) + '\n')
    try:
        tracemalloc.start()
        assert state.migrate_deckbox_records(tmp_path) == (count, count, 1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024
    assert path.read_text() == ''
    with (tmp_path / 'deckbox.jsonl').open() as handle:
        assert sum(1 for _ in handle) == count


@pytest.mark.parametrize('source', ['deckbox', 'mtgtop8'])
def test_migration_can_resume_after_interrupted_source_replacement(tmp_path, monkeypatch, source):
    source_path = tmp_path / 'commander.jsonl'
    target_path = tmp_path / ('deckbox.jsonl' if source == 'deckbox' else 'duel-commander.jsonl')
    migrate = state.migrate_deckbox_records if source == 'deckbox' else state.migrate_mtgtop8_formats
    record = data.make_decklist_record(
        source=source, source_id='123', format_name='commander',
        url=('https://deckbox.org/sets/123' if source == 'deckbox'
             else 'https://www.mtgtop8.com/event?d=123&f=EDH'),
        boards={'mainboard': [{'name': 'Island', 'quantity': 60}]},
    )
    original = json.dumps(record) + '\n'
    source_path.write_text(original, encoding='utf-8')
    replace = state.os.replace

    def interrupt_replacement(temporary, destination):
        if destination == source_path:
            # The target is already durable before source removal starts.
            assert len(target_path.read_text().splitlines()) == 1
            raise OSError('simulated interruption')
        return replace(temporary, destination)

    with monkeypatch.context() as patch:
        patch.setattr(state.os, 'replace', interrupt_replacement)
        with pytest.raises(OSError, match='simulated interruption'):
            migrate(tmp_path)
    assert source_path.read_text() == original
    result = migrate(tmp_path)
    assert result == ((1, 0, 1) if source == 'deckbox' else 1)
    assert source_path.read_text() == ''
    assert len(target_path.read_text().splitlines()) == 1


def test_large_checkpoint_save_avoids_a_second_full_serialized_copy(tmp_path):
    import tracemalloc

    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    checkpoint.add_searches('moxfield', 'modern', (
        {'key': f'card:{number}', 'label': f'Card {number}',
         'params': {'cardId': str(number), 'sortType': 'created', 'sortDirection': 'descending'}}
        for number in range(6_000)
    ))
    checkpoint.current_search('moxfield', 'modern')['next_page'] = 17
    try:
        tracemalloc.start()
        checkpoint.write()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < checkpoint.path.stat().st_size
    resumed = state.Checkpoint(checkpoint.path)
    assert resumed.state == checkpoint.state
    assert resumed.current_search('moxfield', 'modern')['next_page'] == 17


def test_interrupted_checkpoint_stream_preserves_previous_progress(tmp_path, monkeypatch):
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    checkpoint.save('mtgtop8', 'modern', 7)
    previous = checkpoint.path.read_bytes()

    def interrupted_dump(payload, handle, **kwargs):
        handle.write('{"schema_version":')
        raise OSError('simulated interrupted write')

    with monkeypatch.context() as patch:
        patch.setattr(state.json, 'dump', interrupted_dump)
        with pytest.raises(OSError, match='simulated interrupted write'):
            checkpoint.save('mtgtop8', 'modern', 8)
    assert checkpoint.path.read_bytes() == previous
    resumed = state.Checkpoint(checkpoint.path)
    assert resumed.next_page('mtgtop8', 'modern', 0) == 7
    resumed.save('mtgtop8', 'modern', 8)
    assert state.Checkpoint(checkpoint.path).next_page('mtgtop8', 'modern', 0) == 8


@pytest.mark.parametrize('failures', [1, 3])
def test_checkpoint_handles_brief_and_persistent_file_locks(tmp_path, monkeypatch, failures):
    checkpoint = state.Checkpoint(tmp_path / 'checkpoint.json')
    checkpoint.save('mtgtop8', 'modern', 7)
    original = checkpoint.path.read_bytes()
    replace = state.os.replace
    calls = []
    sleeps = []

    def locked_replace(source, destination):
        calls.append(destination)
        if len(calls) <= failures:
            raise PermissionError('simulated file lock')
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(state.os, 'replace', locked_replace)
        patch.setattr(http.time, 'sleep', sleeps.append)
        if failures == 3:
            with pytest.raises(PermissionError, match='simulated file lock'):
                checkpoint.save('mtgtop8', 'modern', 8)
            assert checkpoint.path.read_bytes() == original
        else:
            checkpoint.save('mtgtop8', 'modern', 8)
    resumed = state.Checkpoint(checkpoint.path)
    assert resumed.next_page('mtgtop8', 'modern', 0) == (7 if failures == 3 else 8)
    assert len(calls) == min(failures + 1, 3)
    assert sleeps == ([0.1, 0.2] if failures == 3 else [0.1])


def test_mtgtop8_event_metadata_parser_retains_tournament_fields():
    html = '<p>128 players</p><tr class="chosen_tr"><td class="S14">3-4</td><td><a href="event?e=2&d=123">Deck</a></td></tr>'
    assert mtgtop8.parse_event_metadata(html, '123') == (3, 128)
    assert mtgtop8.parse_event_metadata(html, 'other') == (None, 128)
    assert mtgtop8.parse_event_metadata('', '123') == (None, None)
    assert scraper.parse_args(['--mtgtop8-event-metadata']).mtgtop8_event_metadata
