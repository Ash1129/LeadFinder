"""The data agent: real independent practices from the NPI registry (NPPES), kept as a local practice table that the
research loop searches through `TablePracticeDirectory`.

This is option A from the brainstorm: a bounded lookup over a table built ahead of time, not a live LLM data agent.
`build_table` asks the public NPPES API (https://npiregistry.cms.hhs.gov/api/, version 2.1, no key) for the
organizations in each territory city and specialty, keeps the ones that look like independent practices, and
estimates each one's size. `python -m leadfinder.npi build` writes the result to `data/npi_practices.json`, which is
committed so the app, the tests and the evaluation run offline on the same snapshot.

What the registry gives, and what it does not:

- **Stable ID.** An NPI is issued once and never reused, so `practice_id` is `npi:<10 digits>`.
- **Name, specialty, location.** The organization's legal name, its primary taxonomy code (mapped to the eight
  specialties the scorer knows; anything else is left out), and its LOCATION address, not its mailing address.
- **No provider count.** The registry lists organizations and clinicians separately and does not link them. The
  size here is an estimate: the number of individual clinicians whose practice address and specialty match the
  organization's, floored at 1 (`providers_source` says which). A clinician registered at home or under another
  address is missed, so the estimate is a lower bound.
- **No website, no EHR.** Neither is in the registry. `domain` is always None (so the `domain_only` strategy finds
  nothing here) and EHR stays unknown until website research finds one; neither is ever guessed.
- **"Independent" is a heuristic.** Records that are subparts of a larger organization, or whose names say hospital,
  health system, university and the like, are dropped. A practice owned by a system under its own name gets through.

Everything kept is public business information the registry publishes. The authorized official's name and title are
kept as the practice's listed contact; their phone number is not.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

import httpx

from .research import (
    MAX_DIRECTORY_RESULTS,
    PracticeCandidate,
    PracticeLookupError,
    PracticeLookupRequest,
    normalize_name,
)
from .seed import valid_npi
from .synth import CITIES

REGISTRY_URL = "https://npiregistry.cms.hhs.gov/api/"
API_VERSION = "2.1"
PAGE_SIZE = 200                     # the API's maximum `limit`
MAX_SKIP = 1000                     # the API's maximum `skip`; results past 1,200 cannot be reached
TABLE_PATH = Path(__file__).resolve().parents[2] / "data" / "npi_practices.json"
TABLE_VERSION = 1

# The search term the API matches against taxonomy descriptions, per specialty the scorer knows.
SEARCH_TERMS = {
    "Family Medicine": "Family Medicine",
    "Pediatrics": "Pediatrics",
    "Dermatology": "Dermatology",
    "Internal Medicine": "Internal Medicine",
    "Orthopedic Surgery": "Orthopaedic Surgery",
    "Physical Therapy": "Physical Therapy",
    "Cardiology": "Cardiovascular Disease",
    "Ophthalmology": "Ophthalmology",
}
# Taxonomy code prefixes, most specific first: 207RC is cardiology inside the internal medicine 207R family.
TAXONOMY_PREFIXES = (
    ("207RC", "Cardiology"),
    ("207RI0011", "Cardiology"),
    ("207Q", "Family Medicine"),
    ("2080", "Pediatrics"),
    ("207N", "Dermatology"),
    ("207R", "Internal Medicine"),
    ("207X", "Orthopedic Surgery"),
    ("2251", "Physical Therapy"),
    ("261QP2000", "Physical Therapy"),
    ("207W", "Ophthalmology"),
)
INSTITUTION_WORDS = re.compile(
    r"\b(?:hospitals?|health system|healthcare system|medical center|university|college|county|state of|"
    r"department|veterans|va |school|nursing|rehabilitation center|home health|hospice|hospitalists?)\b",
    re.IGNORECASE,
)


class NpiRegistryError(RuntimeError):
    """The registry could not be reached or answered with an error."""


def specialty_for(taxonomy_code: Optional[str]) -> Optional[str]:
    """The scorer's specialty for an NPPES taxonomy code, or None when it is not one of the eight."""
    code = (taxonomy_code or "").upper()
    for prefix, specialty in TAXONOMY_PREFIXES:
        if code.startswith(prefix):
            return specialty
    return None


def _street_key(street: str, zip5: str) -> str:
    """A practice address as a comparable key: suite and punctuation removed, common abbreviations unified."""
    s = re.sub(r"[^a-z0-9 ]+", " ", street.casefold())
    s = re.sub(r"\b(?:ste|suite|unit|rm|room|fl|floor|bldg|building|apt)\b.*$", "", s)
    words = {"street": "st", "avenue": "ave", "road": "rd", "drive": "dr", "boulevard": "blvd", "lane": "ln",
             "place": "pl", "plaza": "plz", "court": "ct", "parkway": "pkwy", "highway": "hwy", "east": "e",
             "west": "w", "north": "n", "south": "s"}
    s = " ".join(words.get(w, w) for w in s.split())
    return f"{s}|{zip5}"


@dataclass(frozen=True)
class PracticeRecord:
    """One independent practice as the registry describes it, with where each fact came from."""

    npi: str
    name: str
    specialty: str
    taxonomy_code: str
    taxonomy_desc: str
    street: str
    city: str
    state: str
    zip5: str
    phone: Optional[str]
    providers: int
    providers_source: str
    contact_name: Optional[str]
    contact_title: Optional[str]
    enumerated: Optional[str]
    last_updated: Optional[str]
    source: str = "npi"
    npis: tuple[str, ...] = ()      # every NPI merged into this practice (see resolve_entities); empty means just `npi`

    def __post_init__(self) -> None:
        object.__setattr__(self, "npis", tuple(self.npis) or (self.npi,))

    @property
    def practice_id(self) -> str:
        return f"npi:{self.npi}"

    @property
    def location(self) -> str:
        return f"{self.city}, {self.state}"

    def candidate(self) -> PracticeCandidate:
        """The record in the shape research.py consumes. No domain and no EHR: the registry has neither."""
        return PracticeCandidate(
            practice_id=self.practice_id,
            name=self.name,
            domain=None,
            location=self.location,
            specialty=self.specialty,
            provider_count=self.providers,
            source=self.source,
        )


def _title(value: str) -> str:
    """Registry names are upper case; keep short all-caps tokens (PC, MD, LLC) and title-case the rest."""
    keep = {"PC", "PLLC", "LLC", "LLP", "PA", "MD", "DO", "PT", "INC", "II", "III", "USA", "ENT", "OB", "GYN"}
    return " ".join(w.upper() if w.upper().strip(",") in keep or "." in w else w.capitalize() for w in value.split())


def parse_organization(result: dict) -> tuple[Optional[dict], str]:
    """Pick the fields of one NPI-2 result that the table keeps. Returns (fields, "") or (None, why it was left out).

    The provider count is filled in later by `estimate_providers`; it needs the clinicians at the same address."""
    basic = result.get("basic") or {}
    npi = str(result.get("number") or "")
    if result.get("enumeration_type") != "NPI-2":
        return None, "not an organization"
    if not valid_npi(npi):
        return None, "invalid NPI"
    if basic.get("status") != "A":
        return None, "inactive"
    if basic.get("organizational_subpart") == "YES":
        return None, "subpart of a larger organization"
    name = (basic.get("organization_name") or "").strip()
    if not name:
        return None, "no organization name"
    if INSTITUTION_WORDS.search(name):
        return None, "institution name"
    primary = next((t for t in result.get("taxonomies") or [] if t.get("primary")), None)
    specialty = specialty_for(primary.get("code") if primary else None)
    if specialty is None:
        return None, "primary specialty not one of the eight"
    location = next((a for a in result.get("addresses") or [] if a.get("address_purpose") == "LOCATION"), None)
    if location is None or location.get("country_code") != "US":
        return None, "no US practice location"
    contact = " ".join(p for p in (basic.get("authorized_official_first_name"),
                                   basic.get("authorized_official_last_name")) if p and p != "--")
    return {
        "npi": npi,
        "name": _title(name),
        "specialty": specialty,
        "taxonomy_code": primary["code"],
        "taxonomy_desc": primary.get("desc") or "",
        "street": _title(location.get("address_1") or ""),
        "city": _title(location.get("city") or ""),
        "state": (location.get("state") or "").upper(),
        "zip5": (location.get("postal_code") or "")[:5],
        "phone": location.get("telephone_number") or None,
        "contact_name": _title(contact) if contact else None,
        "contact_title": basic.get("authorized_official_title_or_position") or None,
        "enumerated": basic.get("enumeration_date"),
        "last_updated": basic.get("last_updated"),
    }, ""


def estimate_providers(org: dict, clinicians: Iterable[dict]) -> tuple[int, str]:
    """Count individual (NPI-1) clinicians whose practice address matches the organization's and whose primary
    specialty is the same. Floored at 1: an organization has at least one provider even when none is registered at
    its address."""
    key = _street_key(org["street"], org["zip5"])
    seen = set()
    for c in clinicians:
        location = next((a for a in c.get("addresses") or [] if a.get("address_purpose") == "LOCATION"), None)
        primary = next((t for t in c.get("taxonomies") or [] if t.get("primary")), None)
        if location is None or primary is None or specialty_for(primary.get("code")) != org["specialty"]:
            continue
        if _street_key(location.get("address_1") or "", (location.get("postal_code") or "")[:5]) == key:
            seen.add(c.get("number"))
    if seen:
        return len(seen), "npi:clinicians-at-address"
    return 1, "npi:floor"


class NpiRegistry:
    """A small client for the NPPES API. Inject `client` (an httpx.Client) in tests; nothing is fetched at import."""

    def __init__(self, client: Optional[httpx.Client] = None, pause: float = 0.25) -> None:
        self.client = client
        self.pause = pause          # seconds between requests, to be polite to a free public service
        self.requests = 0

    def search(self, **params) -> list[dict]:
        """Every result for one query, paging until the registry runs out or the API's skip limit is reached."""
        owned = self.client is None
        client = self.client or httpx.Client(timeout=20.0, headers={"User-Agent": "LeadFinder-DataAgent/1.0"})
        results: list[dict] = []
        try:
            skip = 0
            while skip <= MAX_SKIP:
                query = {"version": API_VERSION, "limit": PAGE_SIZE, "skip": skip} | params
                try:
                    response = client.get(REGISTRY_URL, params=query)
                    response.raise_for_status()
                    body = response.json()
                except (httpx.HTTPError, ValueError) as error:
                    raise NpiRegistryError(f"the NPI registry could not answer {params}") from error
                self.requests += 1
                if body.get("Errors"):
                    raise NpiRegistryError(f"the NPI registry refused {params}: {body['Errors']}")
                page = body.get("results") or []
                results.extend(page)
                if len(page) < PAGE_SIZE:
                    break
                skip += PAGE_SIZE
                time.sleep(self.pause)
        finally:
            if owned:
                client.close()
        time.sleep(self.pause)
        return results


def build_table(
    registry: NpiRegistry,
    cities: Sequence[tuple[str, str]] = CITIES,
    specialties: Sequence[str] = tuple(SEARCH_TERMS),
    log=lambda message: None,
) -> tuple[list[PracticeRecord], dict]:
    """Query each (city, specialty), keep independent practices, estimate their size. Returns (records, report)."""
    orgs: dict[str, dict] = {}
    dropped: Counter = Counter()
    for city, state in cities:
        for specialty in specialties:
            found = registry.search(enumeration_type="NPI-2", city=city, state=state,
                                    taxonomy_description=SEARCH_TERMS[specialty])
            kept = 0
            for result in found:
                fields, why = parse_organization(result)
                if fields is None:
                    dropped[why] += 1
                elif fields["state"] != state:
                    dropped["practice location outside the searched state"] += 1
                elif fields["npi"] not in orgs:
                    orgs[fields["npi"]] = fields
                    kept += 1
            log(f"{city}, {state} / {specialty}: {len(found)} found, {kept} new practices")

    # One clinician query per (ZIP, specialty) covers every organization that shares them.
    clinicians: dict[tuple[str, str], list[dict]] = {}
    for zip5, specialty in sorted({(o["zip5"], o["specialty"]) for o in orgs.values()}):
        clinicians[(zip5, specialty)] = registry.search(enumeration_type="NPI-1", postal_code=zip5,
                                                        taxonomy_description=SEARCH_TERMS[specialty])
    records = []
    for o in orgs.values():
        providers, source = estimate_providers(o, clinicians.get((o["zip5"], o["specialty"]), ()))
        records.append(PracticeRecord(**o, providers=providers, providers_source=source))
    records, merged = resolve_entities(records)
    report = {
        "kept": len(records),
        "npis_kept": len(orgs),
        "npis_merged_into_another_practice": merged,
        "dropped": dict(dropped.most_common()),
        "providers_estimated": sum(r.providers_source == "npi:clinicians-at-address" for r in records),
        "requests": registry.requests,
    }
    return records, report


def resolve_entities(records: Sequence[PracticeRecord]) -> tuple[list[PracticeRecord], int]:
    """Merge records that are one business: the same normalized name in the same city and state.

    The registry often gives one practice several NPIs (one per site or billing entity), and to a seller they are one
    account; left apart, two identical names tie in matching and the research agent can never pick one. The merged
    practice keeps the lowest NPI as its ID (stable while that NPI stays active), every NPI in `npis`, the most
    common specialty, the lowest NPI's address and contact, and a provider estimate that counts each
    (address, specialty) pair once so clinicians at a shared address are not added twice. Returns (records, merged)."""
    groups: dict[tuple[str, str, str], list[PracticeRecord]] = {}
    for r in records:
        groups.setdefault((normalize_name(r.name), r.city.casefold(), r.state), []).append(r)
    out, merged = [], 0
    for members in groups.values():
        members.sort(key=lambda r: r.npi)
        first = members[0]
        if len(members) == 1:
            out.append(first)
            continue
        merged += len(members) - 1
        specialties = Counter(r.specialty for r in members)
        # Most common; a tie goes to the lowest NPI's specialty, then alphabetically, so a rebuild picks the same one.
        specialty = min(specialties, key=lambda sp: (-specialties[sp], sp != first.specialty, sp))
        sites: dict[tuple[str, str], PracticeRecord] = {}
        for r in members:
            key = (_street_key(r.street, r.zip5), r.specialty)
            if key not in sites or r.providers > sites[key].providers:
                sites[key] = r
        estimated = [r for r in sites.values() if r.providers_source == "npi:clinicians-at-address"]
        providers = sum(r.providers for r in estimated) or 1
        out.append(PracticeRecord(**(asdict(first) | {
            "specialty": specialty,
            "providers": providers,
            "providers_source": "npi:clinicians-at-address" if estimated else "npi:floor",
            "npis": tuple(r.npi for r in members),
        })))
    out.sort(key=lambda r: (r.state, r.city, r.name, r.npi))
    return out, merged


def save_table(records: Sequence[PracticeRecord], report: dict, path: Path = TABLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": TABLE_VERSION,
        "source": f"{REGISTRY_URL} (NPPES API v{API_VERSION})",
        "retrieved": date.today().isoformat(),
        "territory": [f"{c}, {s}" for c, s in CITIES],
        "report": report,
        "practices": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def load_table(path: Path = TABLE_PATH) -> list[PracticeRecord]:
    doc = json.loads(Path(path).read_text())
    if doc.get("version") != TABLE_VERSION:
        raise ValueError(f"{path} is practice table version {doc.get('version')}, expected {TABLE_VERSION}")
    return [PracticeRecord(**(row | {"npis": tuple(row.get("npis") or ())})) for row in doc["practices"]]


def _location_parts(location: Optional[str]) -> tuple[str, str]:
    """('ithaca', 'ny') from 'Ithaca, NY'; a bare state ('NY') or city ('Ithaca') fills only its part."""
    if not location:
        return "", ""
    parts = [p.strip().casefold() for p in location.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[-1][:2]
    return ("", parts[0]) if len(parts[0]) == 2 else (parts[0], "")


class TablePracticeDirectory:
    """research.PracticeDirectory over the practice table.

    Retrieval is deliberately generous (any candidate sharing a distinctive name word, or close overall) so that
    research.rank_candidates, which applies the match threshold and margin, makes the precision decision. A location
    narrows only when it is supplied and the strategy uses it."""

    def __init__(self, records: Sequence[PracticeRecord], min_similarity: float = 0.5) -> None:
        self.records = list(records)
        self.min_similarity = min_similarity
        self._names = [normalize_name(r.name) for r in self.records]

    @classmethod
    def from_file(cls, path: Path = TABLE_PATH) -> "TablePracticeDirectory":
        try:
            return cls(load_table(path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise PracticeLookupError(f"the practice table at {path} could not be read") from error

    def lookup(self, request: PracticeLookupRequest) -> Sequence[PracticeCandidate]:
        if request.strategy == "domain_only":
            return []           # the registry has no websites, so no table row has a domain to match
        wanted = normalize_name(request.practice_name)
        wanted_words = set(wanted.split())
        city, state = _location_parts(request.location) if request.strategy == "name_domain_location" else ("", "")
        scored = []
        for record, name in zip(self.records, self._names):
            if state and record.state.casefold() != state:
                continue
            if city and record.city.casefold() != city:
                continue
            similarity = difflib.SequenceMatcher(None, wanted, name).ratio()
            if similarity >= self.min_similarity or wanted_words & set(name.split()) - GENERIC_WORDS:
                scored.append((similarity, record.npi, record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record.candidate() for _, _, record in scored[:MAX_DIRECTORY_RESULTS]]


# Words too common in practice names to make two names look alike on their own.
GENERIC_WORDS = {"medical", "medicine", "family", "pediatrics", "pediatric", "associates", "group", "practice",
                 "care", "health", "center", "clinic", "physicians", "physician", "of", "the", "and", "internal",
                 "orthopaedic", "orthopedic", "orthopedics", "physical", "therapy", "eye", "dermatology", "cardiology",
                 "heart", "services", "partners", "primary", "specialists", "surgery", "sports", "new", "york"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m leadfinder.npi",
                                     description="Build the practice table from the public NPI registry.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="query NPPES for the territory and write the practice table")
    build.add_argument("--out", type=Path, default=TABLE_PATH)
    build.add_argument("--pause", type=float, default=0.25, help="seconds between requests (default 0.25)")
    args = parser.parse_args(argv)

    registry = NpiRegistry(pause=args.pause)
    records, report = build_table(registry, log=lambda m: print(m, file=sys.stderr))
    save_table(records, report, args.out)
    print(json.dumps(report, indent=1))
    print(f"wrote {len(records)} practices to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
