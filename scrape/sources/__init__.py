"""Supported constructed formats and source identifiers."""

MOXFIELD_API = "https://api2.moxfield.com"


MTGTOP8 = "https://www.mtgtop8.com"


DECKBOX = "https://deckbox.org"


DECKBOX_CORPUS = "deckbox"


MOXFIELD_FORMATS: dict[str, str] = {
    "standard": "standard",
    "pioneer": "pioneer",
    "explorer": "explorer",
    "modern": "modern",
    "legacy": "legacy",
    "vintage": "vintage",
    "pauper": "pauper",
    "commander": "commander",
    "duel-commander": "duelCommander",
    "pauper-commander": "pauperEdh",
    "oathbreaker": "oathbreaker",
    "brawl": "brawl",
    "historic-brawl": "historicBrawl",
    "historic": "historic",
    "alchemy": "alchemy",
    "timeless": "timeless",
    "premodern": "premodern",
    "old-school": "oldSchool",
    "canadian-highlander": "highlanderCanadian",
    "gladiator": "gladiator",
    "penny-dreadful": "pennyDreadful",
}


MTGTOP8_FORMATS: dict[str, str] = {
    "standard": "ST",
    "pioneer": "PI",
    "explorer": "EXP",
    "modern": "MO",
    "legacy": "LE",
    "vintage": "VI",
    "pauper": "PAU",
    "cedh": "cEDH",
    "duel-commander": "EDH",
    "mtgo-commander": "EDHM",
    "extended": "EX",
    "canadian-highlander": "CHL",
    "highlander": "HIGH",
    "historic": "HI",
    "alchemy": "ALCH",
    "premodern": "PREM",
    "peasant": "PEA",
    "block": "BL",
}


DECKBOX_FORMATS: dict[str, str] = {
    "standard": "1",
    "vintage": "3",
    "legacy": "4",
    "modern": "5",
    "pioneer": "6",
    "commander": "7",
    "pauper": "11",
    "oathbreaker": "18",
    "historic": "19",
    "premodern": "20",
    "penny-dreadful": "21",
    "pauper-commander": "22",
    "brawl": "24",
    "alchemy": "25",
    "duel-commander": "26",
}


DECKBOX_FORMAT_NAMES = {
    "standard": "Standard",
    "vintage": "Vintage",
    "legacy": "Legacy",
    "modern": "Modern",
    "pioneer": "Pioneer",
    "commander": "Commander",
    "pauper": "Pauper",
    "oathbreaker": "Oathbreaker",
    "historic": "Historic",
    "premodern": "Premodern",
    "penny-dreadful": "Penny Dreadful",
    "pauper-commander": "Pauper Commander",
    "brawl": "Brawl",
    "alchemy": "Alchemy",
    "duel-commander": "Duel Commander",
}


DEFAULT_FORMATS = tuple(
    dict.fromkeys((*MOXFIELD_FORMATS, *MTGTOP8_FORMATS, *DECKBOX_FORMATS))
)


DECKBOX_CARD_COLORS = ("W", "U", "B", "R", "G", "multicolor", "colorless")


DECKBOX_CARD_ROLES = (
    "interaction",
    "card-advantage",
    "mana",
    "graveyard",
    "creature",
    "engine",
    "other",
)


EXPORT_SECTION_HEADERS = {
    "deck",
    "mainboard",
    "sideboard",
    "commander",
    "commanders",
    "companion",
    "companions",
}
