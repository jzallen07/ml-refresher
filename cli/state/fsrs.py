from __future__ import annotations

from fsrs import Card, Rating, Scheduler


_scheduler = Scheduler()


def new_card() -> str:
    return Card().to_json()


def review_card(card_json: str, score: float) -> str:
    card = Card.from_json(card_json)

    if score >= 0.9:
        rating = Rating.Easy
    elif score >= 0.7:
        rating = Rating.Good
    elif score >= 0.4:
        rating = Rating.Hard
    else:
        rating = Rating.Again

    updated_card, _log = _scheduler.review_card(card, rating)
    return updated_card.to_json()
