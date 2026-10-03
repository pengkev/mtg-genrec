"""Copy limits and companion construction checks against the Oracle snapshot.

Companion constraints describe the starting mainboard (and command zone), not
sideboard cards. A partial deck cannot establish minimum final deck size.
Rules reference: Wizards' Ikoria release notes and Comprehensive Rules 702.139.
"""
from __future__ import annotations

import re
from typing import Mapping, Any

from .legality import allowed_copy_count

COMPANIONS = (
    'Gyruda, Doom of Depths', 'Jegantha, the Wellspring', 'Kaheera, the Orphanguard',
    'Keruga, the Macrosage', 'Lurrus of the Dream-Den', 'Lutri, the Spellchaser',
    'Obosh, the Preypiercer', 'Umori, the Collector', 'Yorion, Sky Nomad', 'Zirda, the Dawnwaker',
)
CARD_TYPES = {'artifact', 'battle', 'creature', 'enchantment', 'instant', 'kindred', 'planeswalker', 'sorcery'}
PERMANENT_TYPES = {'artifact', 'battle', 'creature', 'enchantment', 'land', 'planeswalker'}
# Activated keywords can omit their reminder text in Oracle records.
ACTIVATED_KEYWORDS = {
    'cycling', 'equip', 'fortify', 'crew', 'reconfigure', 'transfigure', 'transmute',
    'unearth', 'scavenge', 'embalm', 'eternalize', 'outlast', 'adapt', 'monstrosity',
    'level up', 'ninjutsu', 'commander ninjutsu', 'channel', 'reinforce', 'boast',
    'forecast', 'craft', 'saddle',
}


def copy_limit(card: Mapping[str, Any], format_name: str, companion=None):
    special = allowed_copy_count(card)
    if special is None or special != 1:
        limit = special
    else:
        text = card.get('oracle_text', '') or (card.get('card_faces') or [{}])[0].get('oracle_text', '')
        limit = 1 if format_name == 'commander' or 'a deck can have up to one card named' in text.casefold() else 4
    if companion and companion['name'] == 'Lutri, the Spellchaser' and 'land' not in card_types(card):
        limit = 1
    return limit


def characteristics(card):
    # Only split cards combine both halves outside the stack. DFC/adventure
    # cards use their front/normal characteristics for construction restrictions.
    if card.get('layout') == 'split':
        return card
    faces = card.get('card_faces') or []
    return {**card, **faces[0]} if faces else card


def card_types(card):
    return set(re.findall(r'\w+', characteristics(card).get('type_line', '').split('—')[0].casefold()))


def has_activated_ability(card):
    front = characteristics(card)
    types = card_types(card)
    type_line = front.get('type_line', '').casefold()
    if 'land' in types and set(re.findall(r'\w+', type_line)) & {'plains', 'island', 'swamp', 'mountain', 'forest'}:
        return True
    text = front.get('oracle_text', '')
    # Abilities granted to other objects do not give this card an ability.
    unquoted = re.sub(r'["“][^"”]*["”]', '', text)
    if ':' in unquoted:
        return True
    keywords = {str(word).casefold() for word in card.get('keywords', [])}
    if any(word in ACTIVATED_KEYWORDS or word.endswith('cycling') for word in keywords):
        return True
    return any(re.match(r'(?:' + '|'.join(re.escape(k) for k in ACTIVATED_KEYWORDS) + r')\b', line.casefold())
               for line in text.splitlines())


def companion_card_allowed(companion, card, shared_types=None):
    name = companion['name']
    front = characteristics(card)
    types = card_types(card)
    value = float(front.get('cmc', card.get('cmc', 0)))
    land = 'land' in types
    if name == 'Gyruda, Doom of Depths':
        return value % 2 == 0
    if name == 'Jegantha, the Wellspring':
        symbols = re.findall(r'\{([^}]+)\}', front.get('mana_cost', '').casefold())
        return len(symbols) == len(set(symbols))
    if name == 'Kaheera, the Orphanguard':
        subtypes = set(re.findall(r'\w+', front.get('type_line', '').partition('—')[2].casefold()))
        return ('creature' not in types or bool(subtypes & {'cat', 'elemental', 'nightmare', 'dinosaur', 'beast'})
                or 'changeling' in {str(k).casefold() for k in card.get('keywords', [])}
                or 'changeling' in front.get('oracle_text', '').casefold())
    if name == 'Keruga, the Macrosage':
        return land or value >= 3
    if name == 'Lurrus of the Dream-Den':
        return not (types & PERMANENT_TYPES) or value <= 2
    if name == 'Obosh, the Preypiercer':
        return land or value % 2 == 1
    if name == 'Umori, the Collector':
        return land or bool((types & CARD_TYPES) & (CARD_TYPES if shared_types is None else shared_types))
    if name == 'Zirda, the Dawnwaker':
        return not (types & PERMANENT_TYPES) or has_activated_ability(card)
    return True  # Lutri uses counts; Yorion uses final deck size.


def shared_card_types(cards):
    types = set(CARD_TYPES)
    for card, _ in cards:
        if 'land' not in card_types(card):
            types &= card_types(card) & CARD_TYPES
    return types


def companion_errors(companion, cards):
    shared = shared_card_types(cards)
    errors = []
    for card, quantity in cards:
        if not companion_card_allowed(companion, card, shared):
            errors.append(card['name'])
        elif companion['name'] == 'Lutri, the Spellchaser' and 'land' not in card_types(card) and quantity > 1:
            errors.append(card['name'])
    return sorted(set(errors))



def companion_is_legal(card, format_name):
    # Wizards' February 9, 2026 Commander update separates companion legality
    # from ordinary card legality. Lutri may be in the 99 or command zone.
    return (card['name'] in COMPANIONS
            and card.get('legalities', {}).get(format_name) == 'legal'
            and not (format_name == 'commander' and card['name'] in {'Lutri, the Spellchaser', 'Yorion, Sky Nomad'}))


def companion_choices(catalog, format_name):
    return [name for name in COMPANIONS if (card := catalog.resolve(name))
            and companion_is_legal(card, format_name)]
