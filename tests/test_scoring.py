import time

from leadfinder.scoring import band_of, score_practice
from leadfinder.seed import npi_check_digit, valid_npi

NOW = 1_800_000_000.0


def test_every_point_has_a_reason_and_source():
    s = score_practice("Family Medicine", 5, "athenahealth", [("/demo", NOW - 86400), ("/pricing", NOW - 3 * 86400)], now=NOW)
    assert s.total == sum(r.points for r in s.reasons)
    assert {r.source for r in s.reasons} == {"NPI record", "site visits"}
    assert s.reasons == sorted(s.reasons, key=lambda r: -r.points)


def test_good_fit_scores_hot_and_poor_fit_cool():
    hot = score_practice("Family Medicine", 5, "athenahealth", [("/demo", NOW), ("/pricing", NOW)], now=NOW)
    cool = score_practice("Cardiology", 40, "Epic", [], now=NOW)
    assert hot.band == "hot" and hot.total == 100            # 20 + 20 + 15 + 20 + 15 + 10, capped at 100
    assert cool.band == "cool" and cool.total == 4
    assert any("health system" in r.label for r in cool.reasons)


def test_page_points_count_once_and_are_capped():
    many = [("/demo", NOW - 30 * 86400)] * 5 + [("/pricing", NOW - 30 * 86400)] * 3 + [("/integrations", NOW - 30 * 86400)]
    s = score_practice("Pediatrics", 3, None, many, now=NOW)
    site = [r for r in s.reasons if r.source == "site visits"]
    assert sum(r.points for r in site) == 40                    # 20 + 15 + the 5 left of the cap; not recent
    assert any("(5 times)" in r.label for r in site)


def test_recent_visit_bonus():
    old = score_practice("Pediatrics", 3, None, [("/features", NOW - 20 * 86400)], now=NOW)
    new = score_practice("Pediatrics", 3, None, [("/features", NOW - 86400)], now=NOW)
    assert new.total - old.total == 10


def test_bands():
    assert (band_of(70), band_of(69), band_of(45), band_of(44), band_of(0)) == ("hot", "warm", "warm", "cool", "cool")


def test_npi_check_digit():
    assert npi_check_digit("123456789") == 3                    # CMS's published example: 1234567893
    assert valid_npi("1234567893") and not valid_npi("1234567890")


def test_default_now_is_the_clock():
    s = score_practice("Pediatrics", 3, None, [("/features", time.time())])
    assert any("last 7 days" in r.label for r in s.reasons)
