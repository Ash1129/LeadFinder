"""Versioned deterministic visit features and the recall-oriented, pre-research lead gate."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Literal, Optional
from urllib.parse import unquote, urlsplit

from .outreach import KNOWN_EHRS

SCHEMA_VERSION = "visit-v3"
TAXONOMY_VERSION = "v2"
Tier = Literal["high", "medium", "low"]
RoutingDecision = Literal["research", "human_review", "stop"]
HIGH_TIER: Tier = "high"
MEDIUM_TIER: Tier = "medium"
LOW_TIER: Tier = "low"
TIER_RANK: dict[Tier, int] = {LOW_TIER: 0, MEDIUM_TIER: 1, HIGH_TIER: 2}
PAGE_CATEGORIES = (
    "demo_or_trial", "pricing", "ehr_integration", "security", "billing", "specialty_case_study", "usability",
    "blog", "careers", "about", "homepage", "unmapped",
)
CONTENT_SECTIONS = {
    "blog": "blog",
    "news": "blog",
    "articles": "blog",
    "webinars": "blog",
    "resources": "blog",
    "careers": "careers",
    "jobs": "careers",
}
DEMO_ALIASES = ("request-a-demo", "schedule-demo", "book-a-demo", "book-demo", "get-a-demo")

SPECIALTIES = (
    "Family Medicine", "Pediatrics", "Dermatology", "Internal Medicine", "Orthopedic Surgery",
    "Physical Therapy", "Cardiology", "Ophthalmology",
)
SPECIALTY_SLUGS = {
    "family-medicine": "Family Medicine",
    "family-practice": "Family Medicine",
    "pediatrics": "Pediatrics",
    "pediatric": "Pediatrics",
    "dermatology": "Dermatology",
    "internal-medicine": "Internal Medicine",
    "orthopedics": "Orthopedic Surgery",
    "orthopedic-surgery": "Orthopedic Surgery",
    "physical-therapy": "Physical Therapy",
    "cardiology": "Cardiology",
    "ophthalmology": "Ophthalmology",
}
EHR_NAMES = tuple(sorted(KNOWN_EHRS, key=len, reverse=True))
CRITERIA = ("integration", "usability", "security", "pricing")


@dataclass(frozen=True)
class ThresholdConfig:
    min_identification_confidence: float = 0.5
    high_tier_passes: bool = True
    min_criteria_breadth: int = 2
    repeat_min_sessions: int = 2
    repeat_min_pages_per_session: float = 3.0
    lookback_window_days: int = 30
    session_timeout_minutes: int = 30
    version: str = "v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_identification_confidence) or not 0.0 <= self.min_identification_confidence <= 1.0:
            raise ValueError("min_identification_confidence must be between 0 and 1")
        if self.min_criteria_breadth < 1:
            raise ValueError("min_criteria_breadth must be positive")
        if self.repeat_min_sessions < 1:
            raise ValueError("repeat_min_sessions must be positive")
        if not math.isfinite(self.repeat_min_pages_per_session) or self.repeat_min_pages_per_session <= 0:
            raise ValueError("repeat_min_pages_per_session must be positive")
        if self.lookback_window_days < 1 or self.session_timeout_minutes < 1:
            raise ValueError("lookback and session timeout must be positive")
        if not self.version.strip():
            raise ValueError("threshold version is required")


@dataclass(frozen=True)
class PageView:
    view_id: str
    page_url: str
    viewed_at: float
    seconds_on_page: float = 0.0

    def __post_init__(self) -> None:
        if not self.view_id.strip() or not self.page_url.strip():
            raise ValueError("view_id and page_url are required")
        if not math.isfinite(self.viewed_at):
            raise ValueError("viewed_at must be finite")
        if not math.isfinite(self.seconds_on_page) or self.seconds_on_page < 0:
            raise ValueError("seconds_on_page must be finite and nonnegative")


@dataclass(frozen=True)
class VisitAnalysisInput:
    """Allowlisted, scrubbed business hints and one visitor's page views for an analysis window."""

    lead_id: str
    practice_name_raw: Optional[str]
    domain_or_org_hint: Optional[str]
    location_hint: Optional[str]
    company_identified: bool
    identification_confidence: float
    page_views: tuple[PageView, ...]
    window_end: float
    patient_data_stripped: bool

    def __post_init__(self) -> None:
        if not self.lead_id.strip():
            raise ValueError("lead_id is required")
        if not 0.0 <= self.identification_confidence <= 1.0:
            raise ValueError("identification_confidence must be between 0 and 1")
        if not math.isfinite(self.window_end):
            raise ValueError("window_end must be finite")


@dataclass(frozen=True)
class PageFeature:
    view_id: str
    page: str
    category: str
    tier: Literal["high", "medium", "low"]
    viewed_at: float
    seconds_on_page: float
    session_id: str


@dataclass(frozen=True)
class VisitAnalysisResult:
    lead_id: str
    schema_version: str
    taxonomy_version: str
    threshold_config_version: str
    window_start: float
    window_end: float
    visit_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    practice_name_raw: Optional[str]
    domain_or_org_hint: Optional[str]
    location_hint: Optional[str]
    company_identified: bool
    identification_confidence: float
    session_count: int
    days_span: int
    recency_days: Optional[int]
    total_time_on_site: float
    pages_per_session: float
    avg_time_per_page: float
    page_category_counts: dict[str, int]
    page_category_flags: dict[str, bool]
    max_intent_tier: Literal["high", "medium", "low"]
    intent_level: Literal["high", "medium", "low"]
    visitor_role_guess: Literal["administrator", "owner", "unknown"]
    visitor_role_confidence: float
    criteria_breadth: int
    demo_without_depth: bool
    ehr_of_interest: Optional[str]
    specialty_interest_hint: Optional[str]
    product_line_interest: tuple[str, ...]
    pages_viewed: tuple[PageFeature, ...]
    passes_threshold: bool
    routing_decision: RoutingDecision
    threshold_reasons: tuple[str, ...]
    threshold_fail_reason: Optional[str]
    field_sources: dict[str, str]
    field_confidences: dict[str, float]
    warnings: tuple[str, ...]
    patient_data_stripped: bool
    session_timeout_minutes: int

    def as_dict(self) -> dict:
        """Return the deterministic feature section for the shared lead state."""
        result = asdict(self)
        result["visit_ids"] = list(self.visit_ids)
        result["session_ids"] = list(self.session_ids)
        result["product_line_interest"] = list(self.product_line_interest)
        result["pages_viewed"] = [asdict(page) for page in self.pages_viewed]
        result["threshold_reasons"] = list(self.threshold_reasons)
        result["warnings"] = list(self.warnings)
        return result


@dataclass(frozen=True)
class _PageClassification:
    page: str
    category: str
    tier: Literal["high", "medium", "low"]
    criteria: tuple[str, ...] = ()
    product_lines: tuple[str, ...] = ()
    ehr_hint: Optional[str] = None
    specialty_hint: Optional[str] = None
    is_demo: bool = False


def _safe_path(page_url: str) -> str:
    try:
        path = urlsplit(page_url).path or "/"
    except ValueError:
        return "/"
    path = unquote(path).split("?", 1)[0].split("#", 1)[0]
    path = re.sub(r"/+", "/", path.strip().lower()).strip("/")
    return f"/{path}" if path else "/"


def _path_has(path: str, *terms: str) -> bool:
    return any(re.search(rf"(?:^|/)(?:{term})(?:/|$|-)", path) for term in terms)


def _classify(page_url: str) -> _PageClassification:
    path = _safe_path(page_url)
    segments = [part for part in path.strip("/").split("/") if part]
    first_segment = segments[0] if segments else ""
    if first_segment in CONTENT_SECTIONS:
        category = CONTENT_SECTIONS[first_segment]
        return _PageClassification(category, category, LOW_TIER)
    if first_segment in DEMO_ALIASES:
        return _PageClassification("demo/trial", "demo_or_trial", HIGH_TIER, is_demo=True)
    if first_segment == "customers" and len(segments) > 1:
        specialty = SPECIALTY_SLUGS.get(segments[1])
        if specialty:
            return _PageClassification("specialty case study", "specialty_case_study", MEDIUM_TIER,
                                       specialty_hint=specialty)
    if _path_has(path, "demo", "request-demo", "demo-request", "trial", "free-trial"):
        return _PageClassification("demo/trial", "demo_or_trial", HIGH_TIER, is_demo=True)
    if _path_has(path, "pricing", "plans", "plans-and-pricing"):
        return _PageClassification("pricing", "pricing", HIGH_TIER, ("pricing",))
    if _path_has(path, "integration", "integrations", "ehr", "integrates-with"):
        ehr = next((name for name in EHR_NAMES if name.casefold().replace(" ", "-") in path), None)
        return _PageClassification("EHR integration", "ehr_integration", HIGH_TIER, ("integration",),
                                   ("integrations",), ehr_hint=ehr)
    if _path_has(path, "security", "hipaa", "privacy", "data-exit", "data-portability", "data-export"):
        return _PageClassification("security", "security", MEDIUM_TIER, ("security",))
    if _path_has(path, "billing", "revenue-cycle", "revenue-cycle-management"):
        return _PageClassification("billing", "billing", MEDIUM_TIER, product_lines=("billing",))

    parts = {part for part in path.split("/") if part}
    specialty = next((value for slug, value in SPECIALTY_SLUGS.items() if slug in parts), None)
    if _path_has(path, "case-study", "case-studies", "customer-stories") and specialty:
        return _PageClassification("specialty case study", "specialty_case_study", MEDIUM_TIER,
                                   specialty_hint=specialty)
    if _path_has(path, "features", "workflow", "usability", "booking", "scheduling", "appointments"):
        product = "scheduling" if _path_has(path, "booking", "scheduling", "appointments") else "workflow"
        return _PageClassification("workflow/features", "usability", MEDIUM_TIER, ("usability",), (product,))
    if _path_has(path, "blog", "news", "articles"):
        return _PageClassification("blog", "blog", LOW_TIER)
    if _path_has(path, "careers", "jobs", "employment"):
        return _PageClassification("careers", "careers", LOW_TIER)
    if _path_has(path, "about", "company"):
        return _PageClassification("about", "about", LOW_TIER)
    if path == "/":
        return _PageClassification("/", "homepage", LOW_TIER)
    return _PageClassification("unmapped", "unmapped", LOW_TIER)


def _sessionize(views: list[PageView], timeout_seconds: int) -> list[list[PageView]]:
    sessions: list[list[PageView]] = []
    for view in sorted(views, key=lambda item: (item.viewed_at, item.view_id)):
        if not sessions or view.viewed_at - sessions[-1][-1].viewed_at > timeout_seconds:
            sessions.append([])
        sessions[-1].append(view)
    return sessions


def _role_guess(counts: dict[str, int]) -> tuple[Literal["administrator", "owner", "unknown"], float]:
    admin = counts.get("billing", 0) + counts.get("ehr_integration", 0) + counts.get("usability", 0)
    owner = (counts.get("pricing", 0) + counts.get("demo_or_trial", 0) + counts.get("security", 0) +
             counts.get("specialty_case_study", 0))
    total = admin + owner
    if not total or admin == owner:
        return "unknown", 0.0
    return ("administrator", admin / total) if admin > owner else ("owner", owner / total)


def analyze_visits(
    data: VisitAnalysisInput,
    config: ThresholdConfig = ThresholdConfig(),
) -> VisitAnalysisResult:
    """Compute fixed-schema behavior features and apply the configured pre-research gate."""
    window_start = data.window_end - config.lookback_window_days * 86400
    views = sorted(
        (
            view for view in data.page_views
            if data.patient_data_stripped and window_start <= view.viewed_at <= data.window_end
        ),
        key=lambda item: (item.viewed_at, item.view_id),
    )
    sessions = _sessionize(views, config.session_timeout_minutes * 60)
    session_ids = tuple(f"{data.lead_id}:s{index + 1}" for index in range(len(sessions)))
    page_features: list[PageFeature] = []
    for session_id, session in zip(session_ids, sessions):
        for view in session:
            classification = _classify(view.page_url)
            page_features.append(PageFeature(
                view_id=view.view_id,
                page=classification.page,
                category=classification.category,
                tier=classification.tier,
                viewed_at=view.viewed_at,
                seconds_on_page=view.seconds_on_page,
                session_id=session_id,
            ))

    classifications = [_classify(view.page_url) for view in views]
    category_counts = {category: 0 for category in PAGE_CATEGORIES}
    criteria: set[str] = set()
    ehr_hints: set[str] = set()
    specialty_hints: set[str] = set()
    product_lines: set[str] = set()
    for classification in classifications:
        category_counts[classification.category] = category_counts.get(classification.category, 0) + 1
        criteria.update(classification.criteria)
        product_lines.update(classification.product_lines)
        if classification.ehr_hint:
            ehr_hints.add(classification.ehr_hint)
        if classification.specialty_hint:
            specialty_hints.add(classification.specialty_hint)

    max_intent_tier: Tier = max(
        (feature.tier for feature in page_features),
        key=lambda tier: TIER_RANK[tier],
        default=LOW_TIER,
    )
    intent_level: Tier = max_intent_tier
    session_count = len(sessions)
    pages_per_session = len(views) / session_count if session_count else 0.0
    total_time = sum(view.seconds_on_page for view in views)
    avg_time = total_time / len(views) if views else 0.0
    days_span = (int((views[-1].viewed_at - views[0].viewed_at) // 86400) if len(views) > 1 else 0)
    recency_days = int((data.window_end - views[-1].viewed_at) // 86400) if views else None
    role, role_confidence = _role_guess(category_counts)

    threshold_reasons: list[str] = []
    warnings: list[str] = []
    fail_reason: Optional[str] = None
    hard_requirements_met = True
    if not data.patient_data_stripped:
        hard_requirements_met = False
        fail_reason = "patient_data_not_scrubbed"
        warnings.append("visit data is not marked as scrubbed; downstream research must not run")
    elif not data.company_identified:
        hard_requirements_met = False
        fail_reason = "unidentified"
        warnings.append("practice identification is unresolved")
    elif data.identification_confidence < config.min_identification_confidence:
        hard_requirements_met = False
        fail_reason = "low_identification"
        warnings.append("practice identification confidence is below the gate minimum")

    if data.identification_confidence < config.min_identification_confidence:
        warnings.append("low_identification_confidence")
    if not views:
        warnings.append("no_visits_in_lookback_window")
    if category_counts.get("unmapped"):
        warnings.append("unmapped_pages_present")
    if len(ehr_hints) > 1:
        warnings.append("multiple_ehr_hints")
    if len(specialty_hints) > 1:
        warnings.append("multiple_specialty_hints")

    criteria_passes = len(criteria) >= config.min_criteria_breadth
    repeat_passes = (
        TIER_RANK[max_intent_tier] >= TIER_RANK[MEDIUM_TIER] and
        (session_count >= config.repeat_min_sessions or
         pages_per_session >= config.repeat_min_pages_per_session)
    )
    if config.high_tier_passes and max_intent_tier == HIGH_TIER:
        threshold_reasons.append("high_tier_page")
    if criteria_passes:
        threshold_reasons.append(f"criteria_breadth_{config.min_criteria_breadth}")
    if repeat_passes:
        threshold_reasons.append("repeat_interest")

    passes_threshold = hard_requirements_met and bool(threshold_reasons)
    if hard_requirements_met and not passes_threshold:
        fail_reason = "only_low_tier_pages" if views and max_intent_tier == LOW_TIER else "single_shallow_session"
    # Review identity-uncertain visits only when their behavior otherwise meets the intent rule.
    if passes_threshold:
        routing_decision: RoutingDecision = "research"
    elif fail_reason in ("unidentified", "low_identification") and threshold_reasons:
        routing_decision = "human_review"
    else:
        routing_decision = "stop"

    page_flags = {category: bool(count) for category, count in category_counts.items()}
    field_sources = {
        "practice_name_raw": "visit_log" if data.practice_name_raw else "unknown",
        "domain_or_org_hint": "visit_log" if data.domain_or_org_hint else "unknown",
        "location_hint": "visit_log" if data.location_hint else "unknown",
        "page_features": "visit_log",
        "visitor_role_guess": "inferred",
        "ehr_of_interest": "visit_log" if len(ehr_hints) == 1 else "unknown",
        "specialty_interest_hint": "visit_log" if len(specialty_hints) == 1 else "unknown",
    }
    field_confidences = {"visitor_role_guess": round(role_confidence, 3)}
    if len(ehr_hints) == 1:
        field_confidences["ehr_of_interest"] = 1.0
    if len(specialty_hints) == 1:
        field_confidences["specialty_interest_hint"] = 1.0

    return VisitAnalysisResult(
        lead_id=data.lead_id,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        threshold_config_version=config.version,
        window_start=window_start,
        window_end=data.window_end,
        visit_ids=tuple(view.view_id for view in views),
        session_ids=session_ids,
        practice_name_raw=data.practice_name_raw,
        domain_or_org_hint=data.domain_or_org_hint,
        location_hint=data.location_hint,
        company_identified=data.company_identified,
        identification_confidence=data.identification_confidence,
        session_count=session_count,
        days_span=days_span,
        recency_days=recency_days,
        total_time_on_site=total_time,
        pages_per_session=round(pages_per_session, 3),
        avg_time_per_page=round(avg_time, 3),
        page_category_counts=category_counts,
        page_category_flags=page_flags,
        max_intent_tier=max_intent_tier,
        intent_level=intent_level,
        visitor_role_guess=role,
        visitor_role_confidence=round(role_confidence, 3),
        criteria_breadth=len(criteria),
        demo_without_depth=(category_counts.get("demo_or_trial", 0) > 0 and
                            len(criteria) < config.min_criteria_breadth),
        ehr_of_interest=next(iter(ehr_hints)) if len(ehr_hints) == 1 else None,
        specialty_interest_hint=next(iter(specialty_hints)) if len(specialty_hints) == 1 else None,
        product_line_interest=tuple(sorted(product_lines)),
        pages_viewed=tuple(page_features),
        passes_threshold=passes_threshold,
        routing_decision=routing_decision,
        threshold_reasons=tuple(
            threshold_reasons if passes_threshold or routing_decision == "human_review" else ()
        ),
        threshold_fail_reason=None if passes_threshold else fail_reason,
        field_sources=field_sources,
        field_confidences=field_confidences,
        warnings=tuple(dict.fromkeys(warnings)),
        patient_data_stripped=data.patient_data_stripped,
        session_timeout_minutes=config.session_timeout_minutes,
    )
