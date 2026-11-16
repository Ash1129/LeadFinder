"""Bounded practice research over an injected directory and a public practice website.

The module owns the agent loop and output contract, but deliberately does not choose how the team's practice table
is delivered. Daniel's directory adapter is the integration point. Website HTML is reduced to exact EHR-name
mentions locally; raw page content is never sent to the model or retained in the result.
"""
from __future__ import annotations

import difflib
import ipaddress
import json
import re
import socket
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable, Literal, Optional, Protocol, Sequence
from urllib.parse import urlsplit

import httpx

from .outreach import KNOWN_EHRS
from .writer import MODEL

MAX_TOOL_CALLS = 10
MATCH_THRESHOLD = 0.85
AMBIGUOUS_THRESHOLD = 0.45
MATCH_MARGIN = 0.10
MAX_DIRECTORY_RESULTS = 50
MAX_HTML_BYTES = 1_000_000
LOOKUP_STRATEGIES = ("name_domain_location", "name_only", "domain_only")
EHR_NAMES = tuple(sorted(KNOWN_EHRS, key=len, reverse=True))

RESEARCH_SYSTEM = """You research one medical practice for a sales lead. You may use only the supplied
practice name, domain and location hints, the practice-directory results, and the structured website evidence.
Use the tools to look up candidate practices and, only for a returned candidate, inspect its public website.
The website tool returns exact EHR-name mentions only; it does not provide page text.
Call complete_research when you have enough evidence. Select only a candidate ID returned by practice_lookup,
or use an empty candidate_id when no candidate can be supported. Do not invent or infer an EHR, specialty, size,
location, practice ID, or fact. A tool call is not permission to override an ambiguous-match warning."""

RESEARCH_TOOLS = [
    {
        "name": "practice_lookup",
        "description": "Search the practice directory using the original business hints and one unused search strategy.",
        "input_schema": {
            "type": "object",
            "properties": {"strategy": {"type": "string", "enum": list(LOOKUP_STRATEGIES)}},
            "required": ["strategy"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch_practice_website",
        "description": "Fetch public EHR-name evidence for a candidate returned by practice_lookup.",
        "input_schema": {
            "type": "object",
            "properties": {"practice_id": {"type": "string"}},
            "required": ["practice_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "complete_research",
        "description": "Finish the bounded research run by selecting a returned candidate, or none.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["candidate_id", "rationale"],
            "additionalProperties": False,
        },
    },
]


class ResearchError(RuntimeError):
    """Base error for a research run that cannot produce a trustworthy result."""


class PracticeLookupError(ResearchError):
    """The practice-directory adapter could not complete its lookup."""


class WebsiteFetchError(ResearchError):
    """A public website could not be fetched safely or parsed."""


class ResearchModelError(ResearchError):
    """The tool-using model could not complete a research turn."""


@dataclass(frozen=True)
class ResearchRequest:
    practice_name: str
    domain_hint: Optional[str] = None
    location_hint: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.practice_name.strip():
            raise ValueError("practice_name is required")
        if len(self.practice_name) > 200:
            raise ValueError("practice_name must be at most 200 characters")
        if self.domain_hint is not None and len(self.domain_hint) > 253:
            raise ValueError("domain_hint must be at most 253 characters")
        if self.location_hint is not None and len(self.location_hint) > 200:
            raise ValueError("location_hint must be at most 200 characters")


@dataclass(frozen=True)
class PracticeLookupRequest:
    practice_name: str
    domain: Optional[str]
    location: Optional[str]
    strategy: Literal["name_domain_location", "name_only", "domain_only"]


@dataclass(frozen=True)
class PracticeCandidate:
    """One result from Daniel's practice-table adapter; IDs must be stable across runs."""

    practice_id: str
    name: str
    domain: Optional[str] = None
    location: Optional[str] = None
    specialty: Optional[str] = None
    provider_count: Optional[int] = None
    source: str = "npi"
    ehr: Optional[str] = None
    ehr_source: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.practice_id.strip() or not self.name.strip():
            raise ValueError("practice_id and name are required")
        if self.provider_count is not None and self.provider_count < 1:
            raise ValueError("provider_count must be positive when known")
        if not self.source.strip():
            raise ValueError("source is required")
        if self.ehr is not None and not self.ehr_source:
            raise ValueError("ehr_source is required when the candidate includes an EHR")


class PracticeDirectory(Protocol):
    """Bounded adapter over the team's table; wrap source failures as PracticeLookupError."""

    def lookup(self, request: PracticeLookupRequest) -> Sequence[PracticeCandidate]: ...


@dataclass(frozen=True)
class WebsiteEvidence:
    url: str
    ehr_mentions: tuple[str, ...]

    def __post_init__(self) -> None:
        canonical = {name.casefold(): name for name in EHR_NAMES}
        mentions = []
        for mention in self.ehr_mentions:
            vendor = canonical.get(mention.casefold())
            if vendor is None:
                raise ValueError("website evidence may contain only recognized EHR names")
            if vendor not in mentions:
                mentions.append(vendor)
        object.__setattr__(self, "ehr_mentions", tuple(mentions))


class PracticeWebsite(Protocol):
    def inspect(self, domain: str) -> WebsiteEvidence: ...


@dataclass(frozen=True)
class CandidateMatch:
    practice_id: str
    name: str
    domain: Optional[str]
    location: Optional[str]
    specialty: Optional[str]
    provider_count: Optional[int]
    source: str
    confidence: float
    match_reason: str


@dataclass(frozen=True)
class SourcedFact:
    value: Optional[str | int]
    source: str


@dataclass(frozen=True)
class ResearchResult:
    status: Literal["matched", "ambiguous", "not_found", "partial", "failed"]
    practice: Optional[PracticeCandidate]
    match_confidence: Optional[float]
    top_candidates: tuple[CandidateMatch, ...]
    specialty: SourcedFact
    provider_count: SourcedFact
    location: SourcedFact
    ehr: SourcedFact
    rationale: str
    warnings: tuple[str, ...]
    tool_calls: int

    def as_dict(self) -> dict:
        """Return the versionable, JSON-compatible result for the future shared lead state."""
        return {
            "status": self.status,
            "practice_id": self.practice.practice_id if self.practice else None,
            "match_confidence": self.match_confidence,
            "top_candidates": [asdict(candidate) for candidate in self.top_candidates],
            "facts": {
                "specialty": asdict(self.specialty),
                "provider_count": asdict(self.provider_count),
                "location": asdict(self.location),
                "ehr": asdict(self.ehr),
            },
            "rationale": self.rationale,
            "warnings": list(self.warnings),
            "tool_calls": self.tool_calls,
        }


@dataclass(frozen=True)
class AgentToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ModelTurn:
    calls: tuple[AgentToolCall, ...]
    assistant_content: tuple[object, ...]
    stop_reason: str


class ToolCallingModel(Protocol):
    def next_turn(self, messages: list[dict]) -> ModelTurn: ...


@dataclass
class AnthropicResearchModel:
    """Anthropic tool-use adapter. Inject `client` in tests; no API call happens at import time."""

    model: str = MODEL
    client: Any = field(default=None, repr=False)

    def _client(self):
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic(timeout=30.0, max_retries=1)
        return self.client

    def next_turn(self, messages: list[dict]) -> ModelTurn:
        import anthropic

        try:
            response = self._client().messages.create(
                model=self.model,
                max_tokens=2048,
                system=RESEARCH_SYSTEM,
                tools=RESEARCH_TOOLS,
                messages=messages,
            )
        except anthropic.AuthenticationError as error:
            raise ResearchModelError("Claude refused the credential; check ANTHROPIC_API_KEY") from error
        except anthropic.RateLimitError as error:
            raise ResearchModelError("Claude is rate limited; retry the research run later") from error
        except anthropic.APIStatusError as error:
            raise ResearchModelError(f"Claude answered with an error ({error.status_code})") from error
        except anthropic.APIConnectionError as error:
            raise ResearchModelError("Claude could not be reached") from error

        calls = tuple(
            AgentToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        )
        return ModelTurn(calls, tuple(response.content), response.stop_reason)


def _domain(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    candidate = value.strip()
    try:
        parsed = urlsplit(candidate if "://" in candidate else f"https://{candidate}")
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        return None
    try:
        if parsed.port not in (None, 443, 80):
            return None
    except ValueError:
        return None
    try:
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None
    if host.startswith("www."):
        host = host[4:]
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host if "." in host else None
    return None


def _normalized(value: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").casefold()).strip()


def rank_candidates(request: ResearchRequest, candidates: Sequence[PracticeCandidate]) -> tuple[CandidateMatch, ...]:
    """Rank candidates with transparent name/domain/location evidence; confidence is a heuristic, not a probability."""
    wanted_domain = _domain(request.domain_hint)
    wanted_name = _normalized(request.practice_name)
    wanted_location = _normalized(request.location_hint)
    ranked = []
    for candidate in candidates:
        candidate_domain = _domain(candidate.domain)
        candidate_name = _normalized(candidate.name)
        candidate_location = _normalized(candidate.location)
        name_score = difflib.SequenceMatcher(None, wanted_name, candidate_name).ratio()
        location_match = bool(wanted_location and candidate_location and
                              (wanted_location == candidate_location or wanted_location in candidate_location))

        if wanted_domain and candidate_domain and wanted_domain == candidate_domain:
            confidence, reason = 1.0, "exact domain"
        else:
            confidence = 0.90 * name_score + (0.08 if location_match else 0.0)
            if wanted_domain and candidate_domain and wanted_domain != candidate_domain:
                confidence -= 0.15
            confidence = max(0.0, min(0.99, confidence))
            reason = "name and location" if location_match else "name similarity"

        ranked.append(CandidateMatch(
            practice_id=candidate.practice_id,
            name=candidate.name,
            domain=candidate.domain,
            location=candidate.location,
            specialty=candidate.specialty,
            provider_count=candidate.provider_count,
            source=candidate.source,
            confidence=round(confidence, 3),
            match_reason=reason,
        ))
    return tuple(sorted(ranked, key=lambda candidate: (-candidate.confidence, candidate.practice_id)))


class ResearchAgent:
    """A tool-using research loop with deterministic match safeguards and an absolute call cap."""

    def __init__(
        self,
        directory: PracticeDirectory,
        website: PracticeWebsite,
        model: ToolCallingModel,
        *,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        if not 1 <= max_tool_calls <= MAX_TOOL_CALLS:
            raise ValueError(f"max_tool_calls must be between 1 and {MAX_TOOL_CALLS}")
        self.directory = directory
        self.website = website
        self.model = model
        self.max_tool_calls = max_tool_calls

    def research(self, request: ResearchRequest) -> ResearchResult:
        messages = [{"role": "user", "content": json.dumps(asdict(request), sort_keys=True)}]
        candidates: dict[str, PracticeCandidate] = {}
        websites: dict[str, WebsiteEvidence] = {}
        website_attempts: set[str] = set()
        used_strategies: set[str] = set()
        warnings: list[str] = []
        calls_made = 0
        lookup_failed = False
        chosen_id: Optional[str] = None
        rationale = ""
        complete = False

        while calls_made < self.max_tool_calls and not complete:
            turn = self.model.next_turn(messages)
            if not turn.calls:
                warnings.append(
                    f"research model stopped ({turn.stop_reason}) without a complete_research decision"
                )
                break
            messages.append({"role": "assistant", "content": list(turn.assistant_content)})
            results = []
            for call in turn.calls:
                if calls_made >= self.max_tool_calls:
                    warnings.append("research stopped at the tool-call limit")
                    break
                calls_made += 1
                result, warning = self._run_tool(call, request, candidates, websites, website_attempts,
                                                 used_strategies)
                if warning:
                    warnings.append(warning)
                    lookup_failed = lookup_failed or warning == "practice directory lookup failed"
                if call.name == "complete_research" and warning is None:
                    chosen_id = result["candidate_id"]
                    rationale = result["rationale"]
                    complete = True
                    results.append({"type": "tool_result", "tool_use_id": call.id, "content": "Research completed."})
                    break
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(result, sort_keys=True),
                })
            if results:
                messages.append({"role": "user", "content": results})

        if not complete and calls_made >= self.max_tool_calls:
            warnings.append("research stopped at the tool-call limit")
        return self._result(request, candidates, websites, chosen_id, rationale, warnings, calls_made, complete,
                            lookup_failed)

    def _run_tool(
        self,
        call: AgentToolCall,
        request: ResearchRequest,
        candidates: dict[str, PracticeCandidate],
        websites: dict[str, WebsiteEvidence],
        website_attempts: set[str],
        used_strategies: set[str],
    ) -> tuple[dict, Optional[str]]:
        if call.name == "practice_lookup":
            strategy = call.arguments.get("strategy")
            if strategy not in LOOKUP_STRATEGIES:
                return {"error": "Choose a supported lookup strategy."}, "model requested an invalid lookup strategy"
            if strategy in used_strategies:
                return {"error": "That lookup strategy has already been used."}, "model repeated a lookup strategy"
            used_strategies.add(strategy)
            lookup = PracticeLookupRequest(
                practice_name=request.practice_name,
                domain=_domain(request.domain_hint) if strategy != "name_only" else None,
                location=request.location_hint if strategy == "name_domain_location" else None,
                strategy=strategy,
            )
            try:
                found = self.directory.lookup(lookup)
            except PracticeLookupError:
                return {"error": "Practice directory lookup failed; try another supported strategy."}, \
                    "practice directory lookup failed"
            for candidate in found[:MAX_DIRECTORY_RESULTS]:
                candidates[candidate.practice_id] = candidate
            matches = rank_candidates(request, tuple(candidates.values()))[:MAX_DIRECTORY_RESULTS]
            return {"candidates": [asdict(match) for match in matches]}, None

        if call.name == "fetch_practice_website":
            practice_id = call.arguments.get("practice_id")
            candidate = candidates.get(practice_id) if isinstance(practice_id, str) else None
            if candidate is None:
                return {"error": "Website lookup requires a practice_id returned by practice_lookup."}, \
                    "model requested a website for an unreturned practice"
            domain = _domain(candidate.domain)
            if domain is None:
                website_attempts.add(candidate.practice_id)
                return {"error": "The candidate has no valid public website domain."}, \
                    f"no valid website domain for candidate {candidate.practice_id}"
            if candidate.practice_id not in websites:
                website_attempts.add(candidate.practice_id)
                try:
                    websites[candidate.practice_id] = self.website.inspect(domain)
                except WebsiteFetchError:
                    return {"error": "Website fetch failed; do not infer the EHR."}, \
                        f"website lookup failed for candidate {candidate.practice_id}"
            evidence = websites[candidate.practice_id]
            return {"practice_id": practice_id, "domain": domain, "ehr_mentions": list(evidence.ehr_mentions)}, None

        if call.name == "complete_research":
            candidate_id = call.arguments.get("candidate_id")
            rationale = call.arguments.get("rationale")
            if not isinstance(candidate_id, str) or not isinstance(rationale, str):
                return {"error": "candidate_id and rationale must be strings."}, "invalid completion payload"
            if not used_strategies:
                return {"error": "Run practice_lookup before completing research."}, \
                    "model completed before practice lookup"
            if candidate_id and candidate_id not in candidates:
                return {"error": "Select only a candidate returned by practice_lookup, or select none."}, \
                    "model selected an unreturned practice"
            candidate = candidates.get(candidate_id) if candidate_id else None
            if candidate and not candidate.ehr and _domain(candidate.domain) and candidate_id not in website_attempts:
                return {"error": "Inspect the selected candidate's website before completing research."}, \
                    "model completed without attempting available website research"
            return {"candidate_id": candidate_id or None, "rationale": rationale[:500]}, None

        return {"error": "Unknown research tool."}, f"model requested unsupported tool {call.name!r}"

    def _result(
        self,
        request: ResearchRequest,
        candidates: dict[str, PracticeCandidate],
        websites: dict[str, WebsiteEvidence],
        chosen_id: Optional[str],
        rationale: str,
        warnings: list[str],
        calls_made: int,
        complete: bool,
        lookup_failed: bool,
    ) -> ResearchResult:
        ranked = rank_candidates(request, tuple(candidates.values()))[:3]
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        high_confidence = bool(
            top and top.confidence >= MATCH_THRESHOLD and
            (second is None or top.confidence - second.confidence >= MATCH_MARGIN)
        )
        selected = candidates.get(chosen_id) if chosen_id else None

        status: Literal["matched", "ambiguous", "not_found", "partial", "failed"]
        practice: Optional[PracticeCandidate] = None
        match_confidence = top.confidence if top else None
        if not complete:
            status = "partial" if candidates else "failed"
            warnings.append("research did not reach a validated completion decision")
        elif chosen_id and selected and high_confidence and top and selected.practice_id == top.practice_id:
            status, practice = "matched", selected
        elif chosen_id and selected:
            status = "ambiguous"
            warnings.append("the model's selection did not meet the deterministic match threshold and margin")
        elif top and top.confidence >= AMBIGUOUS_THRESHOLD:
            status = "ambiguous"
        else:
            status = "failed" if lookup_failed else "not_found"

        if not rationale:
            rationale = top.match_reason if top else "No practice candidate was returned."
        if practice is None:
            unknown = SourcedFact(None, "unknown")
            return ResearchResult(
                status, None, match_confidence, ranked, unknown, unknown, unknown, unknown,
                rationale, tuple(dict.fromkeys(warnings)), calls_made,
            )

        ehr_value = practice.ehr
        ehr_source = practice.ehr_source or "unknown"
        if not ehr_value:
            evidence = websites.get(practice.practice_id)
            mentions = evidence.ehr_mentions if evidence else ()
            if len(mentions) == 1:
                ehr_value, ehr_source = mentions[0], f"website:https://{_domain(practice.domain)}/"
            elif len(mentions) > 1:
                warnings.append(f"website mentions multiple EHRs for candidate {practice.practice_id}; EHR remains unknown")
        else:
            evidence = websites.get(practice.practice_id)
            if evidence and evidence.ehr_mentions and practice.ehr.casefold() not in {
                mention.casefold() for mention in evidence.ehr_mentions
            }:
                warnings.append(f"website and directory EHR values differ for candidate {practice.practice_id}")

        return ResearchResult(
            status=status,
            practice=practice,
            match_confidence=match_confidence,
            top_candidates=ranked,
            specialty=SourcedFact(practice.specialty, practice.source),
            provider_count=SourcedFact(practice.provider_count, practice.source),
            location=SourcedFact(practice.location, practice.source),
            ehr=SourcedFact(ehr_value, ehr_source),
            rationale=rationale,
            warnings=tuple(dict.fromkeys(warnings)),
            tool_calls=calls_made,
        )


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() in ("script", "style", "noscript", "svg"):
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style", "noscript", "svg") and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


def extract_ehr_mentions(html: str) -> tuple[str, ...]:
    """Extract exact vendor names with an EHR/use cue; never return page text or surrounding snippets."""
    parser = _VisibleText()
    parser.feed(html)
    visible = re.sub(r"\s+", " ", " ".join(parser.parts))
    sentences = re.split(r"(?<=[.!?])\s+", visible)
    mentions = []
    for vendor in EHR_NAMES:
        for sentence in sentences:
            if not re.search(rf"(?<![A-Za-z0-9]){re.escape(vendor)}(?![A-Za-z0-9])", sentence, re.IGNORECASE):
                continue
            context = sentence.casefold()
            has_ehr_cue = re.search(r"\b(?:ehr|electronic health record)\b", context) is not None
            has_use_cue = re.search(
                r"\b(?:we|our (?:practice|office|team)|the practice)\s+(?:currently\s+)?"
                r"(?:use|uses|run|runs|work with|works with|are on|is on)\b|\bpowered by\b",
                context,
            ) is not None
            if has_ehr_cue or has_use_cue:
                mentions.append(vendor)
                break
    return tuple(mentions)


class HttpxPracticeWebsite:
    """Fetch one public HTTPS homepage without redirects, private-network targets, or unbounded HTML."""

    def __init__(
        self,
        client: Optional[httpx.Client] = None,
        resolver: Optional[Callable[[str], Sequence[str]]] = None,
    ) -> None:
        self.client = client
        self.resolver = resolver or self._resolve

    @staticmethod
    def _resolve(host: str) -> Sequence[str]:
        try:
            return tuple({entry[4][0] for entry in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
        except OSError as error:
            raise WebsiteFetchError("The practice website domain could not be resolved.") from error

    def inspect(self, domain: str) -> WebsiteEvidence:
        host = _domain(domain)
        if host is None:
            raise WebsiteFetchError("The practice website domain is invalid.")
        addresses = self.resolver(host)
        try:
            public_addresses = bool(addresses) and all(ipaddress.ip_address(address).is_global for address in addresses)
        except ValueError as error:
            raise WebsiteFetchError("The practice website domain resolved to an invalid IP address.") from error
        if not public_addresses:
            raise WebsiteFetchError("The practice website must resolve only to public IP addresses.")

        url = f"https://{host}/"
        owned_client = self.client is None
        client = self.client or httpx.Client(timeout=8.0, follow_redirects=False, trust_env=False,
                                              headers={"User-Agent": "LeadFinder-Research/1.0"})
        try:
            if client.follow_redirects:
                raise WebsiteFetchError("The practice website client must not follow redirects.")
            try:
                with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        if response.is_redirect:
                            raise WebsiteFetchError("The practice website redirected; redirects are not followed.")
                        raise WebsiteFetchError(f"The practice website returned HTTP {response.status_code}.")
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if content_type not in ("text/html", "application/xhtml+xml"):
                        raise WebsiteFetchError("The practice website did not return HTML.")
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > MAX_HTML_BYTES:
                            raise WebsiteFetchError("The practice website page exceeded the size limit.")
                        chunks.append(chunk)
            except httpx.HTTPError as error:
                raise WebsiteFetchError("The practice website could not be fetched.") from error
        finally:
            if owned_client:
                client.close()

        html = b"".join(chunks).decode("utf-8", errors="replace")
        return WebsiteEvidence(url, extract_ehr_mentions(html))
