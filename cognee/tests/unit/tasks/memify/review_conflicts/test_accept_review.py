"""What of the model's answer a review keeps: labels, statuses, drops and partial answers."""

from copy import deepcopy

import pytest

from cognee.tasks.memify.review_conflicts import schema
from cognee.tasks.memify.review_conflicts.accept_review import accept_review_response
from cognee.tests.unit.tasks.memify.review_conflicts.review_builders import (
    build_prompt,
    existing_conflict,
    fact,
    response,
)


def accept(scope, output):
    prompt = build_prompt(scope, list(scope.entities))
    return accept_review_response(output, scope, prompt)


def test_dated_conflict_uses_relationship_identity_and_preserves_facts(scope):
    before = deepcopy(scope.facts)
    batch = accept(scope, response())
    accepted = batch.accepted_conflicts[0]
    assert accepted.conflict.attribute == "has_ceo"
    assert accepted.conflict.status == "resolved"
    assert accepted.conflict.values == ["alice", "bob"]
    assert accepted.conflict.sources == ["chunk-e1", "chunk-e2"]
    assert accepted.fact_statuses == {"e2": "current", "e1": "superseded"}
    assert batch.scope is scope
    assert scope.facts == before


@pytest.mark.parametrize(
    "case", ["undated", "time_words", "fixed", "restored", "reversed", "mixed"]
)
def test_status_rules(scope, case):
    output = response()
    result = output.conflicts[0]
    expected = "resolved"
    if case in {"undated", "time_words"}:
        for value in scope.facts.values():
            value["effective_date"] = None
            value["sources"][0]["effective_date"] = None
        expected = "unresolved"
        if case == "undated":
            result.current = result.superseded = []
            result.conflicting = ["r1", "r2"]
        else:
            # Time words alone no longer order values: only dates of different documents do.
            scope.facts["e1"]["properties"]["edge_text"] = "Alice was the former CEO."
    elif case == "fixed":
        result.kind = "fixed"
        expected = "unresolved"
    elif case == "restored":
        scope.facts["e1"]["effective_date"] = "2027-03-01"
        result.current, result.superseded = ["r1"], ["r2"]
    else:
        scope.facts["e1"]["effective_date"] = "2027-03-01"
        expected = "unresolved"
        if case == "mixed":
            scope.facts["e1"]["relationship"] = "led_by"
            result.attribute = "CEO's Office"
    accepted = accept(scope, output).accepted_conflicts[0]
    assert accepted.conflict.status == expected
    if expected == "unresolved":
        assert set(accepted.fact_statuses.values()) == {"conflicting"}
    if case in {"reversed", "mixed"}:
        assert "dates disagree" in accepted.conflict.text
    if case == "time_words":
        assert "do not order these values" in accepted.conflict.text
    if case == "mixed":
        assert accepted.conflict.attribute == "ceos office"


@pytest.mark.parametrize(
    "case",
    ["unknown", "overlap", "unrelated", "duplicate", "none_reviewed", "unknown_subject", "empty"],
)
def test_invalid_output_rejects_call(scope, case):
    output = response()
    if case == "unknown":
        output.conflicts[0].current = ["r99"]
    elif case == "overlap":
        output.conflicts[0].current = ["r1"]
    elif case == "unrelated":
        output.conflicts[0].about = "n2"
    elif case == "duplicate":
        output.conflicts.append(output.conflicts[0])
    elif case == "none_reviewed":
        output.reviewed.clear()
    elif case == "unknown_subject":
        output.reviewed[0] = "n99"
    else:
        output.conflicts[0].current = output.conflicts[0].superseded = []
    with pytest.raises(ValueError):
        accept(scope, output)


def test_conflict_about_an_unreviewed_subject_is_not_accepted(scope):
    output = response()
    output.reviewed.remove("n1")
    batch = accept(scope, output)
    assert batch.accepted_conflicts == []
    assert batch.unreviewed_entity_ids == ["acme"]


def test_drop_checks_all_surviving_support_and_replacement_collision(scope):
    conflict_id = existing_conflict(scope)
    output = response()
    output.drop = ["f1"]
    with pytest.raises(ValueError, match="replaced drop"):
        accept(scope, output)
    output.conflicts = []
    assert accept(scope, output).dropped_conflict_ids == [conflict_id]
    prompt = build_prompt(scope, list(scope.entities), leaving_out=["e1"])
    assert accept_review_response(output, scope, prompt).dropped_conflict_ids == []


@pytest.mark.parametrize(
    "forgotten, reviewed, retired",
    [
        (True, ["n1", "n2", "n3"], True),
        (False, ["n1", "n2", "n3"], False),
        (True, ["n2", "n3"], False),
    ],
)
def test_a_user_decision_whose_selected_fact_is_gone_retires_with_its_subjects_review(
    scope, forgotten, reviewed, retired
):
    """A response that leaves the decision out still retires it once its fact is forgotten."""
    conflict_id = existing_conflict(scope)
    scope.conflicts[conflict_id].update(
        resolution_source="user", selected_fact_id="e1", rejected_fact_ids=["e2"]
    )
    if forgotten:
        del scope.facts["e1"]
    batch = accept(scope, schema.ReviewResponse(reviewed=reviewed))
    assert batch.dropped_conflict_ids == ([conflict_id] if retired else [])


def test_partial_answer_writes_nothing_for_an_unreviewed_subject(scope):
    existing_conflict(scope)
    output = response()
    output.conflicts = []
    output.drop = ["f1"]
    output.reviewed.remove("n1")
    batch = accept(scope, output)
    assert batch.unreviewed_entity_ids == ["acme"]
    assert batch.dropped_conflict_ids == []


def test_pending_subject_header_requests_reconciliation_without_auto_acceptance(scope):
    conflict_id = existing_conflict(scope)
    scope.conflicts[conflict_id]["write_pending"] = True
    assert "Existing [f1] has_ceo (subject; pending write)" in build_prompt(scope, ["acme"]).text
    assert "pending write" not in build_prompt(scope, ["alice"]).text

    # Incomplete evidence still permits omission, and cannot authorize a drop.
    prompt = build_prompt(scope, ["acme"], leaving_out=["e1"])
    output = schema.ReviewResponse(reviewed=["n1"], drop=["f1"])
    batch = accept_review_response(output, scope, prompt)
    assert batch.accepted_conflicts == [] and batch.dropped_conflict_ids == []
    assert scope.conflicts[conflict_id]["write_pending"] is True


def test_acceptance_uses_the_snapshot_sent_to_the_llm(scope):
    sent = build_prompt(scope, list(scope.entities))
    later = build_prompt(scope, list(scope.entities), leaving_out=["e1"])
    assert later.fact_labels != sent.fact_labels
    batch = accept_review_response(response(), scope, sent)
    assert batch.shown_fact_ids == {"e1", "e2"}
    assert batch.accepted_conflicts[0].fact_statuses == {"e2": "current", "e1": "superseded"}


@pytest.mark.parametrize(
    "other, kept",
    [("2015-05-12", True), ("Bristol", False)],
    ids=["two dates compete", "a date beside a place does not"],
)
def test_a_date_competes_only_with_another_date(scope, other, kept):
    # A year alone is no ISO date; the Timestamp type says it is one.
    for name, node_type in (("2014", "Timestamp"), (other, "Timestamp" if kept else "Entity")):
        scope.nodes[name] = {"id": name, "name": name, "type": node_type}
    scope.facts = {
        "e1": fact("e1", "2014", None, relationship="incorporated_at"),
        "e2": fact("e2", other, "2026-01-10", relationship="incorporated_at"),
    }
    # A stored conflict's marks keep both facts in the section whether or not they group.
    existing_conflict(scope)
    output = schema.ReviewResponse(
        reviewed=["n1", "n2", "n3"],
        conflicts=[
            schema.ReportedConflict(
                about="n1",
                attribute="incorporated_at",
                kind="fixed",
                text="Two incorporation dates are recorded.",
                conflicting=["r1", "r2"],
            )
        ],
    )
    batch = accept(scope, output)
    assert [item.conflict.values for item in batch.accepted_conflicts] == (
        [sorted(["2014", other])] if kept else []
    )


def _ordering(scope, superseded, current, conflicting=()):
    """Accept one time_varying conflict ordering these facts as the LLM said."""
    # A stored conflict's marks keep every fact in the section, grouped or not: one
    # document's two facts form no comparison of their own.
    existing_conflict(scope)
    prompt = build_prompt(scope, list(scope.entities))
    labels = {fact_id: label for label, fact_id in prompt.fact_labels.items()}
    output = response()
    result = output.conflicts[0]
    result.superseded = [labels[fact_id] for fact_id in superseded]
    result.current = [labels[fact_id] for fact_id in current]
    result.conflicting = [labels[fact_id] for fact_id in conflicting]
    return accept_review_response(output, scope, prompt).accepted_conflicts[0]


def _set_source(scope, fact_id, *sources):
    """Restate a fact's citing documents as (data_id, date) pairs; its date is their latest."""
    scope.facts[fact_id]["sources"] = [
        {
            "chunk_id": f"chunk-{fact_id}-{data_id}",
            "data_id": data_id,
            "document": data_id,
            "effective_date": date,
        }
        for data_id, date in sources
    ]
    dates = [date for _, date in sources if date]
    scope.facts[fact_id]["effective_date"] = max(dates) if dates else None


@pytest.mark.parametrize(
    "old_sources, new_sources",
    [
        # S04: one document states both, under one date.
        ([("doc-s04", "2026-11-05")], [("doc-s04", "2026-11-05")]),
        # One document states both, even when another, later one also states the current fact.
        ([("doc-a", "2026-01-01")], [("doc-a", "2026-01-01"), ("doc-b", "2026-03-01")]),
        # Different documents, one date.
        ([("doc-a", "2026-11-05")], [("doc-b", "2026-11-05")]),
        # No dates, or a date on one side only.
        ([("doc-a", None)], [("doc-b", None)]),
        ([("doc-a", None)], [("doc-b", "2026-03-01")]),
        ([("doc-a", "2026-01-01")], [("doc-b", None)]),
    ],
    ids=[
        "same document",
        "shared document",
        "equal dates",
        "undated",
        "old undated",
        "new undated",
    ],
)
def test_supersession_without_an_order_of_documents_and_dates_is_conflicting(
    scope, old_sources, new_sources
):
    _set_source(scope, "e1", *old_sources)
    _set_source(scope, "e2", *new_sources)
    accepted = _ordering(scope, superseded=["e1"], current=["e2"])
    assert accepted.fact_statuses == {"e1": "conflicting", "e2": "conflicting"}
    assert accepted.conflict.status == "unresolved"
    assert accepted.conflict.text.endswith(
        "Supplied documents and dates do not order these values."
    )


def test_supersession_needs_a_current_fact(scope):
    """Superseded with nothing current names no direction."""
    accepted = _ordering(scope, superseded=["e1"], current=[], conflicting=["e2"])
    assert set(accepted.fact_statuses.values()) == {"conflicting"}


@pytest.mark.parametrize(
    "dates",
    [
        {"e1": "2020-05-01", "e2": "2026-01-10"},
        # The superseded fact sits between two current ones: the later one orders it.
        {"e1": "2026-02-01", "e2": "2026-01-01", "e3": "2026-03-01"},
    ],
    ids=["two documents", "one later current of two"],
)
def test_different_documents_with_ordered_dates_keep_superseded_and_current(scope, dates):
    if "e3" in dates:
        scope.facts["e3"] = fact("e3", "carol", None)
        scope.entities["carol"] = scope.nodes["carol"] = {
            "id": "carol",
            "name": "Carol",
            "types": ["Person"],
        }
    for fact_id, date in dates.items():
        _set_source(scope, fact_id, (f"doc-{fact_id}", date))
    current = sorted(set(dates) - {"e1"})
    accepted = _ordering(scope, superseded=["e1"], current=current)
    assert accepted.fact_statuses == {"e1": "superseded", **dict.fromkeys(current, "current")}
    assert accepted.conflict.status == "resolved"
    assert "order" not in accepted.conflict.text
