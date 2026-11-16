from leadfinder import outreach

PRACTICE = {"name": "Maple Grove Pediatrics", "specialty": "Pediatrics", "providers": 4, "city": "Philadelphia",
            "ehr": "athenahealth", "contact_name": "Priya Patel"}
REASONS = [{"label": "Runs athenahealth, which the product integrates with", "points": 15, "source": "NPI record"},
           {"label": "Visited /demo", "points": 20, "source": "site visits"}]


def test_the_draft_passes_its_own_guardrail():
    subject, body = outreach.draft(PRACTICE, REASONS)
    assert outreach.check(subject, body) == []
    assert "Hi Priya" in body and "athenahealth" in body and "demo page" in body
    assert "unsubscribe" in body


def test_patient_information_is_stopped():
    subject, body = outreach.draft(PRACTICE, REASONS)
    problems = outreach.check(subject, body + "\nWe saw your patient's diagnosis.")
    assert any("patient information" in p for p in problems)


def test_unbackable_claims_are_stopped():
    subject, body = outreach.draft(PRACTICE, REASONS)
    assert any("claim" in p for p in outreach.check(subject, body.replace("Best,", "We guarantee results. Best,")))
    assert any("claim" in p for p in outreach.check(subject, body + "\nWe are HIPAA certified."))


def test_words_inside_other_words_do_not_trip_it():
    subject, body = outreach.draft(PRACTICE, REASONS)        # "Philadelphia" holds "phi"
    assert outreach.check(subject, body + "\nYour data stays secure.") == []   # "secure" holds "cure"


def test_the_footer_must_stay():
    subject, body = outreach.draft(PRACTICE, REASONS)
    problems = outreach.check(subject, body.split("--")[0])
    assert any("CAN-SPAM" in p for p in problems)


def test_size_line_only_for_the_size_it_fits():
    _, solo = outreach.draft(PRACTICE | {"providers": 1}, REASONS)
    _, team = outreach.draft(PRACTICE, REASONS)
    assert "size it was built for" not in solo and "A team of 4 providers" in team
