"""Tests for the sector map and portfolio sector concentration limits."""

import math

import pytest

from src.sectors import (
    UNKNOWN_SECTOR,
    load_sector_map,
    normalize_ticker,
    sector_capacity_inr,
    sector_exposure_inr,
    sector_of,
)


def _write_csv(tmp_path, text, name="sector_map.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestNormalizeTicker:
    def test_adds_nse_suffix_and_uppercases(self):
        assert normalize_ticker("reliance") == "RELIANCE.NS"
        assert normalize_ticker("TCS.NS") == "TCS.NS"
        assert normalize_ticker("  infy  ") == "INFY.NS"

    def test_empty_stays_empty(self):
        assert normalize_ticker("") == ""
        assert normalize_ticker(None) == ""


class TestLoadSectorMap:
    def test_loads_ticker_sector_pairs(self, tmp_path):
        path = _write_csv(tmp_path, "ticker,sector\nTCS,IT\nINFY.NS,IT\nHDFCBANK,Banking\n")

        mapping = load_sector_map(path)

        assert mapping == {"TCS.NS": "IT", "INFY.NS": "IT", "HDFCBANK.NS": "Banking"}

    def test_accepts_alternate_column_names_in_any_order(self, tmp_path):
        path = _write_csv(tmp_path, "industry,symbol\nPharma,SUNPHARMA\n")

        assert load_sector_map(path) == {"SUNPHARMA.NS": "Pharma"}

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_sector_map(str(tmp_path / "nope.csv")) == {}

    def test_unusable_header_is_ignored(self, tmp_path):
        path = _write_csv(tmp_path, "name,description\nfoo,bar\n")

        assert load_sector_map(path) == {}

    def test_skips_rows_missing_either_field(self, tmp_path):
        path = _write_csv(tmp_path, "ticker,sector\nTCS,IT\n,Banking\nINFY,\n")

        assert load_sector_map(path) == {"TCS.NS": "IT"}


class TestSectorOf:
    def test_unmapped_ticker_is_unknown(self):
        assert sector_of("WHO.NS", {"TCS.NS": "IT"}) == UNKNOWN_SECTOR

    def test_matches_regardless_of_input_form(self):
        assert sector_of("tcs", {"TCS.NS": "IT"}) == "IT"


class TestSectorExposure:
    def test_aggregates_by_sector(self):
        sector_map = {"TCS.NS": "IT", "INFY.NS": "IT", "HDFCBANK.NS": "Banking"}
        values = {"TCS.NS": 100.0, "INFY.NS": 50.0, "HDFCBANK.NS": 75.0}

        assert sector_exposure_inr(values, sector_map) == {"IT": 150.0, "Banking": 75.0}

    def test_pools_unmapped_tickers_together(self):
        """An unknown sector could be any sector — including one already at its
        limit — so unmapped names are capped as a single bucket."""
        values = {"A.NS": 100.0, "B.NS": 100.0}

        assert sector_exposure_inr(values, {}) == {UNKNOWN_SECTOR: 200.0}

    def test_ignores_zero_and_negative_values(self):
        assert sector_exposure_inr({"A.NS": 0.0, "B.NS": -5.0}, {}) == {}


class TestSectorCapacity:
    SECTOR_MAP = {"TCS.NS": "IT", "INFY.NS": "IT", "HDFCBANK.NS": "Banking"}

    def test_full_allowance_when_sector_is_empty(self):
        capacity = sector_capacity_inr(
            "TCS.NS", 1_000_000.0, {}, self.SECTOR_MAP, max_sector_pct=0.25
        )

        assert capacity == pytest.approx(250_000.0)

    def test_existing_exposure_reduces_capacity(self):
        capacity = sector_capacity_inr(
            "TCS.NS", 1_000_000.0, {"INFY.NS": 200_000.0}, self.SECTOR_MAP, 0.25
        )

        assert capacity == pytest.approx(50_000.0)

    def test_other_sectors_do_not_consume_capacity(self):
        capacity = sector_capacity_inr(
            "TCS.NS", 1_000_000.0, {"HDFCBANK.NS": 900_000.0}, self.SECTOR_MAP, 0.25
        )

        assert capacity == pytest.approx(250_000.0)

    def test_capacity_floors_at_zero_when_over_the_cap(self):
        capacity = sector_capacity_inr(
            "TCS.NS", 1_000_000.0, {"INFY.NS": 400_000.0}, self.SECTOR_MAP, 0.25
        )

        assert capacity == 0.0

    @pytest.mark.parametrize("max_pct", [0.0, -0.1, 1.0, 2.0])
    def test_disabled_cap_is_unbounded(self, max_pct):
        capacity = sector_capacity_inr(
            "TCS.NS", 1_000_000.0, {"INFY.NS": 999_999.0}, self.SECTOR_MAP, max_pct
        )

        assert math.isinf(capacity)

    def test_zero_portfolio_value_is_unbounded(self):
        assert math.isinf(sector_capacity_inr("TCS.NS", 0.0, {}, self.SECTOR_MAP, 0.25))


class TestUnknownSectorBudget:
    """Indian sector maps are chronically incomplete in exactly the small- and
    micro-cap segment where concentration risk is worst, so an exempt UNKNOWN
    pool would be a route around the cap."""

    PARTIAL_MAP = {"TCS.NS": "IT", "INFY.NS": "IT"}

    def test_unmapped_names_share_one_wider_budget(self):
        capacity = sector_capacity_inr(
            "MICRO.NS", 1_000_000.0, {}, self.PARTIAL_MAP, 0.25, max_unknown_pct=0.30
        )

        assert capacity == pytest.approx(300_000.0)

    def test_the_unmapped_pool_is_exhaustible(self):
        """200 unmapped micro-caps must not be able to become 100% of the book."""
        held = {f"MICRO{i}.NS": 100_000.0 for i in range(3)}

        capacity = sector_capacity_inr(
            "MICRO9.NS", 1_000_000.0, held, self.PARTIAL_MAP, 0.25, max_unknown_pct=0.30
        )

        assert capacity == 0.0

    def test_unmapped_exposure_does_not_consume_a_mapped_sectors_allowance(self):
        held = {"MICRO.NS": 300_000.0}

        capacity = sector_capacity_inr(
            "TCS.NS", 1_000_000.0, held, self.PARTIAL_MAP, 0.25, max_unknown_pct=0.30
        )

        assert capacity == pytest.approx(250_000.0)

    def test_with_no_map_at_all_the_cap_stays_inactive(self):
        """Otherwise a missing CSV freezes the whole portfolio at 30% invested."""
        capacity = sector_capacity_inr(
            "ANY.NS", 1_000_000.0, {}, {}, 0.25, max_unknown_pct=0.30
        )

        assert math.isinf(capacity)

    def test_unknown_budget_can_be_disabled(self):
        capacity = sector_capacity_inr(
            "MICRO.NS", 1_000_000.0, {}, self.PARTIAL_MAP, 0.25, max_unknown_pct=0.0
        )

        assert math.isinf(capacity)
