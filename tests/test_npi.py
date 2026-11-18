import json

import httpx
import pytest

from leadfinder import npi
from leadfinder.research import (
    AgentToolCall,
    ModelTurn,
    PracticeLookupRequest,
    ResearchAgent,
    ResearchRequest,
    WebsiteEvidence,
    normalize_name,
    rank_candidates,
)
from leadfinder.seed import valid_npi
from leadfinder.synth import SPECIALTIES


def org(number="1063576312", name="MADISON IRVING PEDIATRICS, PC", code="2080A0000X", status="A", subpart="NO",
        location=("6711 TOWPATH RD STE 235", "SYRACUSE", "NY", "130571234")):
    street, city, state, postal = location
    return {
        "number": number,
        "enumeration_type": "NPI-2",
        "basic": {"organization_name": name, "status": status, "organizational_subpart": subpart,
                  "authorized_official_first_name": "JANE", "authorized_official_last_name": "DOE",
                  "authorized_official_title_or_position": "President", "enumeration_date": "2006-01-01",
                  "last_updated": "2020-01-01"},
        "addresses": [
            {"address_purpose": "MAILING", "address_1": "PO BOX 1", "city": "ATLANTA", "state": "GA",
             "postal_code": "30384", "country_code": "US"},
            {"address_purpose": "LOCATION", "address_1": street, "city": city, "state": state,
             "postal_code": postal, "country_code": "US", "telephone_number": "315-555-0100"},
        ],
        "taxonomies": [{"code": "193400000X", "desc": "Single Specialty", "primary": False},
                       {"code": code, "desc": "Pediatrics, Adolescent Medicine", "primary": True}],
    }


def clinician(number, street, postal, code="208000000X"):
    return {"number": number, "enumeration_type": "NPI-1",
            "addresses": [{"address_purpose": "LOCATION", "address_1": street, "postal_code": postal}],
            "taxonomies": [{"code": code, "primary": True}]}


def record(npi_number="1063576312", name="Madison Irving Pediatrics, PC", city="Syracuse", state="NY", **kw):
    fields = dict(npi=npi_number, name=name, specialty="Pediatrics", taxonomy_code="208000000X",
                  taxonomy_desc="Pediatrics", street="6711 Towpath Rd", city=city, state=state, zip5="13057",
                  phone=None, providers=3, providers_source="npi:clinicians-at-address", contact_name="Jane Doe",
                  contact_title="President", enumerated="2006-01-01", last_updated="2020-01-01")
    return npi.PracticeRecord(**(fields | kw))


@pytest.mark.parametrize("code, specialty", [
    ("207RC0000X", "Cardiology"),
    ("207R00000X", "Internal Medicine"),
    ("207Q00000X", "Family Medicine"),
    ("2080A0000X", "Pediatrics"),
    ("207X00000X", "Orthopedic Surgery"),
    ("225100000X", "Physical Therapy"),
    ("261QP2000X", "Physical Therapy"),
    ("207W00000X", "Ophthalmology"),
    ("207N00000X", "Dermatology"),
    ("122300000X", None),            # dentist
    (None, None),
])
def test_taxonomy_codes_map_to_the_scorers_specialties(code, specialty):
    assert npi.specialty_for(code) == specialty
    assert specialty is None or specialty in SPECIALTIES


def test_normalize_name_drops_legal_forms_and_joins_initials_only():
    assert normalize_name("Madison Irving Pediatrics, P.C.") == normalize_name("Madison Irving Pediatrics")
    assert normalize_name("Smith & Jones Eye Care, LLC") == normalize_name("Smith and Jones Eye Care")
    assert normalize_name("The Heart Center, P.L.L.C.") == "heart center"
    assert normalize_name("Dr J Smith Family Medicine") == "dr j smith family medicine"
    assert normalize_name("PC") == "pc"                                   # never normalizes a name to nothing


def test_parse_organization_keeps_the_practice_location_not_the_mailing_address():
    fields, why = npi.parse_organization(org())
    assert why == ""
    assert fields["npi"] == "1063576312"
    assert fields["name"] == "Madison Irving Pediatrics, PC"
    assert (fields["city"], fields["state"], fields["zip5"]) == ("Syracuse", "NY", "13057")
    assert fields["specialty"] == "Pediatrics"
    assert fields["contact_name"] == "Jane Doe"
    assert "phone" in fields and "authorized_official_telephone_number" not in json.dumps(fields)


@pytest.mark.parametrize("change, reason", [
    ({"status": "D"}, "inactive"),
    ({"subpart": "YES"}, "subpart of a larger organization"),
    ({"name": "CROUSE HEALTH HOSPITAL INC"}, "institution name"),
    ({"name": "CNY HOSPITALIST GROUP PC"}, "institution name"),
    ({"code": "122300000X"}, "primary specialty not one of the eight"),
    ({"number": "1063576313"}, "invalid NPI"),
])
def test_parse_organization_leaves_out_what_is_not_an_independent_practice(change, reason):
    fields, why = npi.parse_organization(org(**change))
    assert fields is None and why == reason


def test_provider_estimate_counts_clinicians_at_the_same_address_and_specialty():
    fields, _ = npi.parse_organization(org())
    clinicians = [
        clinician("1", "6711 Towpath Road Suite 100", "130570000"),   # same building, other suite and spelling
        clinician("2", "6711 TOWPATH RD", "13057"),
        clinician("2", "6711 TOWPATH RD", "13057"),                   # the same person twice counts once
        clinician("3", "6711 TOWPATH RD", "13057", code="207Q00000X"),  # another specialty
        clinician("4", "90 PRESIDENTIAL PLZ", "13202"),               # elsewhere
    ]
    assert npi.estimate_providers(fields, clinicians) == (2, "npi:clinicians-at-address")
    assert npi.estimate_providers(fields, []) == (1, "npi:floor")


def test_resolve_entities_merges_one_business_with_several_npis():
    a = record("1376551861", "Maine Medical Partners", "Portland", "ME", street="22 Bramhall St", zip5="04102",
               specialty="Family Medicine", providers=4)
    b = record("1790976074", "MAINE MEDICAL PARTNERS", "Portland", "ME", street="22 Bramhall Street", zip5="04102",
               specialty="Family Medicine", providers=4)          # same address and specialty: not added twice
    c = record("1992719272", "Maine Medical Partners", "Portland", "ME", street="887 Congress St", zip5="04102",
               specialty="Pediatrics", providers=2)
    other_city = record("1124335591", "Maine Medical Partners", "Bangor", "ME")
    merged, count = npi.resolve_entities([c, other_city, b, a])
    assert count == 2
    portland = next(r for r in merged if r.city == "Portland")
    assert portland.practice_id == "npi:1376551861"                 # lowest NPI is the stable ID
    assert portland.npis == ("1376551861", "1790976074", "1992719272")
    assert portland.specialty == "Family Medicine"
    assert portland.providers == 6
    assert next(r for r in merged if r.city == "Bangor").npis == ("1124335591",)


def test_registry_search_pages_until_a_short_page():
    pages = []

    def handler(request):
        skip = int(request.url.params["skip"])
        pages.append(skip)
        count = npi.PAGE_SIZE if skip == 0 else 3
        return httpx.Response(200, json={"result_count": count, "results": [{"number": str(i)} for i in range(count)]})

    registry = npi.NpiRegistry(httpx.Client(transport=httpx.MockTransport(handler)), pause=0)
    results = registry.search(enumeration_type="NPI-2", city="ITHACA", state="NY")
    assert len(results) == npi.PAGE_SIZE + 3
    assert pages == [0, npi.PAGE_SIZE]
    assert registry.requests == 2


@pytest.mark.parametrize("response", [
    httpx.Response(500),
    httpx.Response(200, json={"Errors": [{"description": "No valid search criteria"}]}),
    httpx.Response(200, text="not json"),
])
def test_registry_errors_are_reported_not_swallowed(response):
    registry = npi.NpiRegistry(httpx.Client(transport=httpx.MockTransport(lambda request: response)), pause=0)
    with pytest.raises(npi.NpiRegistryError):
        registry.search(city="ITHACA")


def test_build_table_drops_out_of_state_locations_and_reports_why():
    out_of_state = org("1285799163", "24 ON PHYSICIAN NEW YORK, PC", "207R00000X",
                       location=("1 MAIN ST", "AMARILLO", "TX", "79101"))

    def handler(request):
        if request.url.params["enumeration_type"] == "NPI-2":
            return httpx.Response(200, json={"results": [org(), out_of_state, org(subpart="YES")]})
        return httpx.Response(200, json={"results": [clinician("1", "6711 Towpath Rd", "13057")]})

    registry = npi.NpiRegistry(httpx.Client(transport=httpx.MockTransport(handler)), pause=0)
    records, report = npi.build_table(registry, cities=[("Syracuse", "NY")], specialties=["Pediatrics"])
    assert [r.practice_id for r in records] == ["npi:1063576312"]
    assert records[0].providers == 1 and records[0].providers_source == "npi:clinicians-at-address"
    assert report["dropped"] == {"practice location outside the searched state": 1,
                                 "subpart of a larger organization": 1}


def test_table_round_trips(tmp_path):
    path = tmp_path / "table.json"
    rows = [record(), record("1376551861", "Maine Medical Partners", "Portland", "ME", npis=("1376551861", "1790976074"))]
    npi.save_table(rows, {"kept": 2}, path)
    assert npi.load_table(path) == rows


def test_directory_finds_by_name_narrows_by_location_and_has_no_domains():
    directory = npi.TablePracticeDirectory([record(), record("1366813313", "Madison Pediatric Group", "Albany")])
    found = directory.lookup(PracticeLookupRequest("Madison Irving Pediatrics", None, "Syracuse, NY",
                                                   "name_domain_location"))
    assert [c.practice_id for c in found] == ["npi:1063576312"]
    assert len(directory.lookup(PracticeLookupRequest("Madison Irving Pediatrics", None, None, "name_only"))) == 2
    assert directory.lookup(PracticeLookupRequest("Madison Irving Pediatrics", "madison.example", None,
                                                  "domain_only")) == []
    assert all(c.domain is None and c.ehr is None and c.source == "npi" for c in found)


def test_missing_table_is_a_lookup_error(tmp_path):
    from leadfinder.research import PracticeLookupError

    with pytest.raises(PracticeLookupError):
        npi.TablePracticeDirectory.from_file(tmp_path / "missing.json")


# ---------------------------------------------------------------- the committed snapshot of real registry data

@pytest.fixture(scope="module")
def table():
    return npi.load_table()


def test_committed_table_is_valid_registry_data(table):
    assert len(table) >= 500
    ids = [r.practice_id for r in table]
    assert len(ids) == len(set(ids))
    every_npi = [n for r in table for n in r.npis]
    assert len(every_npi) == len(set(every_npi))
    for r in table:
        assert all(valid_npi(n) for n in r.npis) and r.npi == min(r.npis)
        assert r.specialty in SPECIALTIES
        assert r.providers >= 1 and r.providers_source in ("npi:clinicians-at-address", "npi:floor")
        assert r.state in {"NY", "VT", "CT", "PA", "MA", "ME", "RI"}
        assert not npi.INSTITUTION_WORDS.search(r.name)


def test_a_real_practice_is_found_and_matched_by_its_registry_name(table):
    target = next(r for r in table if r.city == "Ithaca")
    directory = npi.TablePracticeDirectory(table)
    request = ResearchRequest(target.name, None, target.location)
    found = directory.lookup(PracticeLookupRequest(target.name, None, target.location, "name_domain_location"))
    ranked = rank_candidates(request, found)
    assert ranked[0].practice_id == target.practice_id and ranked[0].confidence >= 0.85


class ScriptedModel:
    def __init__(self, *turns):
        self.turns = list(turns)

    def next_turn(self, messages):
        return self.turns.pop(0)


def turn(call_id, name, **arguments):
    call = AgentToolCall(call_id, name, arguments)
    return ModelTurn((call,), ({"type": "tool_use", "id": call_id, "name": name, "input": arguments},), "tool_use")


class NoWebsite:
    def inspect(self, domain):
        raise AssertionError("registry candidates have no domain, so no website is fetched")


def test_research_agent_runs_on_the_real_table_and_leaves_ehr_unknown(table):
    target = next(r for r in table if r.city == "Ithaca")
    model = ScriptedModel(
        turn("1", "practice_lookup", strategy="name_domain_location"),
        turn("2", "complete_research", candidate_id=target.practice_id, rationale="Registry name and city match."),
    )
    result = ResearchAgent(npi.TablePracticeDirectory(table), NoWebsite(), model).research(
        ResearchRequest(target.name, None, target.location))
    out = result.as_dict()
    assert out["status"] == "matched" and out["practice_id"] == target.practice_id
    assert out["facts"]["specialty"] == {"value": target.specialty, "source": "npi"}
    assert out["facts"]["ehr"] == {"value": None, "source": "unknown"}


def test_seeding_from_the_table_uses_real_practices(tmp_path):
    from leadfinder.db import connect
    from leadfinder import ml
    from leadfinder.seed import seed

    conn = connect(tmp_path / "npi.db")
    ml.ensure(conn)
    counts = seed(conn, n=12, practices="npi")
    rows = conn.execute("SELECT * FROM practices").fetchall()
    assert counts["practices"] == 12 == len(rows)
    assert {r["source"] for r in rows} == {"npi"}
    assert all(valid_npi(r["npi"]) and r["ehr"] is None and r["contact_email"] is None for r in rows)
    assert conn.execute("SELECT COUNT(*) n FROM leads").fetchone()["n"] == 12
    with pytest.raises(ValueError):
        seed(connect(tmp_path / "other.db"), practices="made-up")


def test_app_started_on_real_practices_shows_no_invented_email(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from leadfinder.service.app import create_app
    from leadfinder.writer import Offline
    from tests.conftest import login

    monkeypatch.setenv("LEADFINDER_PRACTICES", "npi")
    app = create_app(tmp_path / "npi-app.db", demo=True, writer=Offline())
    client = login(TestClient(app))
    lead_id = app.state.conn.execute("SELECT id FROM leads ORDER BY id LIMIT 1").fetchone()["id"]
    page = client.get(f"/lead/{lead_id}")
    assert page.status_code == 200
    assert "no email on record" in page.text and "&lt;None&gt;" not in page.text


def test_match_evaluation_counts_each_query_once(table):
    from leadfinder.evaluation import npi_match

    sample = sorted(table, key=lambda r: r.npi)[:60]
    rows = npi_match.run(sample, seed=1)
    result = npi_match.score(rows, 0.85, 0.10)
    held_out = len(sample) - int(0.7 * len(sample))
    assert result["negatives"] == held_out * len(npi_match.VARIANTS) * 2
    assert result["positives"] + result["negatives"] == len(rows)
    assert result["false_match_different_rate"] <= result["false_match_rate"]
