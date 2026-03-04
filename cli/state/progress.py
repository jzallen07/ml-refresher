from __future__ import annotations

import json
from datetime import datetime, timezone

from cli.state.db import StateDB
from cli.state.fsrs import new_card, review_card


_db: StateDB | None = None


def _get_db() -> StateDB:
    global _db
    if _db is None:
        _db = StateDB()
    return _db


def _ensure_topics_loaded():
    db = _get_db()
    if db.get_all_topics():
        return
    from cli.api import MLRefresherAPI

    api = MLRefresherAPI()
    for t in api.list_all_topics():
        topic_id = t.get("slug") or t["id"]
        display_name = t["title"]
        category = t["category"]
        db.ensure_topic(topic_id, display_name, category)


def get_progress(topic: str | None = None) -> dict:
    _ensure_topics_loaded()
    db = _get_db()

    if topic:
        t = db.get_topic(topic)
        if not t:
            return {"error": f"Topic '{topic}' not found"}
        events = db.get_events(topic, limit=20)
        scored = [e for e in events if e["score"] is not None]
        recent_scores = [e["score"] for e in scored[:5]]
        avg = sum(recent_scores) / len(recent_scores) if recent_scores else None

        due_cards = db.get_due_cards(topic_id=topic)
        weak = []
        for card in due_cards:
            card_data = json.loads(card["card_json"])
            if card_data.get("stability") and card_data["stability"] < 2.0:
                weak.append(card["concept"])

        last_event = events[0] if events else None
        return {
            "topic": topic,
            "display_name": t["display_name"],
            "category": t["category"],
            "level": t["level"],
            "total_interactions": t["total_interactions"],
            "recent_scores": recent_scores,
            "average_score": avg,
            "weak_concepts": weak,
            "due_cards": len(due_cards),
            "last_session": last_event["timestamp"] if last_event else None,
        }

    # Overall progress
    topics = db.get_all_topics()
    started = [t for t in topics if t["total_interactions"] > 0]
    level_counts = {"novice": 0, "intermediate": 0, "advanced": 0}
    for t in topics:
        level_counts[t["level"]] = level_counts.get(t["level"], 0) + 1

    all_scores = []
    weakest = None
    strongest = None
    weakest_avg = float("inf")
    strongest_avg = 0.0

    for t in started:
        events = _get_db().get_events(t["id"], limit=5)
        scored = [e["score"] for e in events if e["score"] is not None]
        if scored:
            avg = sum(scored) / len(scored)
            all_scores.extend(scored)
            if avg < weakest_avg:
                weakest_avg = avg
                weakest = t["id"]
            if avg > strongest_avg:
                strongest_avg = avg
                strongest = t["id"]

    return {
        "total_topics": len(topics),
        "topics_started": len(started),
        "level_counts": level_counts,
        "overall_average": sum(all_scores) / len(all_scores) if all_scores else None,
        "weakest_topic": weakest,
        "strongest_topic": strongest,
    }


def update_progress(
    topic: str,
    event_type: str,
    score: float | None = None,
    concepts_tested: list[str] | None = None,
) -> dict:
    _ensure_topics_loaded()
    db = _get_db()

    t = db.get_topic(topic)
    if not t:
        return {"error": f"Topic '{topic}' not found"}

    metadata = {}
    if concepts_tested:
        metadata["concepts_tested"] = concepts_tested

    db.insert_event(topic, event_type, score, metadata or None)
    db.increment_interactions(topic)

    # Check level promotion
    old_level = t["level"]
    new_level = old_level
    if score is not None:
        events = db.get_events(topic, limit=5)
        recent_scores = [e["score"] for e in events if e["score"] is not None]
        if len(recent_scores) >= 5:
            avg = sum(recent_scores[:5]) / 5
            if old_level == "novice" and avg >= 0.7:
                new_level = "intermediate"
            elif old_level == "intermediate" and avg >= 0.85:
                new_level = "advanced"

    if new_level != old_level:
        db.update_topic_level(topic, new_level)

    # Update FSRS cards
    if concepts_tested and score is not None:
        for concept in concepts_tested:
            card = db.get_card(topic, concept)
            if card:
                updated_json = review_card(card["card_json"], score)
            else:
                updated_json = review_card(new_card(), score)
            card_data = json.loads(updated_json)
            due_date = card_data.get("due", datetime.now(timezone.utc).isoformat())
            state_val = str(card_data.get("state", "new"))
            db.upsert_card(topic, concept, updated_json, due_date, state_val)

    return {
        "topic": topic,
        "event_type": event_type,
        "score": score,
        "old_level": old_level,
        "new_level": new_level,
        "level_changed": new_level != old_level,
        "total_interactions": t["total_interactions"] + 1,
    }


def get_review_schedule() -> dict:
    _ensure_topics_loaded()
    db = _get_db()

    due_cards = db.get_due_cards(limit=50)
    cards = []
    for c in due_cards:
        cards.append({
            "topic_id": c["topic_id"],
            "concept": c["concept"],
            "due_date": c["due_date"],
            "state": c["state"],
        })

    return {
        "total_due": len(cards),
        "cards": cards,
    }
