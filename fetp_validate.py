"""
Validation and uncertainty analysis for the African FETP workforce model.

The structural checks cover selected accounting identities, limiting cases,
analytic behavior, and unit relationships. They do not establish empirical
fit or predictive validity, and they need to be run in the target environment.

Sensitivity analysis uses Latin hypercube sampling with partial rank correlation
coefficients (Blower & Dowlatabadi 1994; Marino et al. 2008). IMPORTANT: every
input range below is an analyst-chosen uniform interval, not an elicited or
fitted distribution. The output is therefore a CONDITIONAL SENSITIVITY
ANALYSIS: it shows how results move across those ranges, and its percentiles
and "share of runs meeting the target" are NOT calibrated probabilities or
confidence intervals.

Every check returns a (passed, detail) pair for review in the app or from a
local script.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import fetp_model as M
from fetp_model import Params, TIERS


# ===========================================================================
# Structural tests.  Every test takes (row, p) where p is the ACTIVE parameter
# set (the sidebar values in the app), so results describe the scenario the
# user is looking at, not a default rebuilt from the country row.
# ===========================================================================

def _clone(p: Params) -> Params:
    return copy.deepcopy(p)


def prepare_params(row: pd.Series, p: Optional[Params] = None) -> Params:
    """Active params if given, else defaults with data-implied intake."""
    if p is not None:
        return _clone(p)
    q = M.default_params_for(row)
    q.annual_intake = M.implied_historical_intake(row, q)
    return q


def _all_loops(p: Params, on: bool) -> Params:
    p.mentor_feedback = p.burnout_feedback = p.emigration_feedback = p.post_feedback = on
    return p


def _isolated(row: pd.Series, p: Params) -> Tuple[pd.Series, Params]:
    """A synthetic single-tier, no-ageing-loss setting for analytic tests."""
    r = row.copy()
    for c in ("grads_adv", "grads_int", "grads_front"):
        r[c] = 0.0
    q = _clone(p)
    _all_loops(q, False)
    q.model_posts = False
    q.progression_from_below = {"advanced": 0.0, "intermediate": 0.0}
    q.retirement_age = 999.0
    q.attribute_pooled_to_host = True
    q.annual_intake = {t: 0.0 for t in TIERS}
    q.attrition = {"advanced": 0.05, "intermediate": 0.0, "frontline": 0.0}
    q.emigration = {"advanced": 0.03, "intermediate": 0.0, "frontline": 0.0}
    return r, q


class _NoMortality:
    def __enter__(self):
        self.saved = M.AGE_MORTALITY.copy()
        M.AGE_MORTALITY = np.zeros_like(self.saved)
    def __exit__(self, *a):
        M.AGE_MORTALITY = self.saved
        return False


def test_conservation(row: pd.Series, p: Params, horizon: int = 25,
                      cohort_history: Optional[pd.DataFrame] = None,
                      dt: float = M.DT_DEFAULT) -> Tuple[bool, str]:
    """People are neither created nor destroyed outside the named flows.

    trained(T) - trained(0) == tracked_inflow - (attrition + emigration +
                                                 mortality + retirement)

    `trained` is serving + unabsorbed pool. Progression remains enabled; it
    must transfer people between tiers without changing the all-tier total."""
    p2 = _clone(p)
    df = M.run(row, p2, horizon=horizon, cohort_history=cohort_history, dt=dt)
    first, last = df.iloc[0], df.iloc[-1]
    cols = ["heads_advanced", "heads_intermediate", "heads_frontline"]
    start, end = float(first[cols].sum()), float(last[cols].sum())
    inflow = float(df["graduates_total"].sum())
    outflow = float(sum(df[f"exit_{c}"].sum()
                        for c in ("attrition", "emigration", "mortality", "retirement")))
    resid = (end - start) - (inflow - outflow)
    scale = max(start, inflow, 1.0)
    ok = abs(resid) / scale < 1e-9
    return ok, (f"start={start:.2f} end={end:.2f} in={inflow:.2f} out={outflow:.2f} "
                f"residual={resid:.2e} ({abs(resid)/scale:.1e} of scale, tol 1e-9)")


def test_inflow_accounting(row: pd.Series, p: Params, horizon: int = 25,
                           cohort_history: Optional[pd.DataFrame] = None,
                           dt: float = M.DT_DEFAULT) -> Tuple[bool, str]:
    """Gross completions split into internal transfers, public entrants, and
    people who do not enter the tracked public-sector workforce."""
    df = M.run(row, _clone(p), horizon=horizon,
               cohort_history=cohort_history, dt=dt)
    gross = float(df.graduate_completions_total.sum())
    accounted = float(df.graduates_total.sum() + df.grads_nonpublic.sum()
                      + df.promoted_total.sum())
    resid = gross - accounted
    ok = abs(resid) <= 1e-8 * max(gross, 1.0)
    return ok, (f"gross completions={gross:.2f}; external public inflow="
                f"{df.graduates_total.sum():.2f}; non-public={df.grads_nonpublic.sum():.2f}; "
                f"internal promotions={df.promoted_total.sum():.2f}; residual={resid:.2e}")


def test_analytic_decay(row: pd.Series, p: Params, T: int = 10) -> Tuple[bool, str]:
    """One cohort, no inflow, constant hazards: the stock must equal N0 exp(-hT)
    to machine precision (survival is integrated exactly within each step)."""
    r, q = _isolated(row, p)
    r["grads_adv"] = 1000.0
    dt = 1.0 / 48.0
    with _NoMortality():
        df = M.run(r, q, horizon=T, dt=dt)
    n0 = float(df.iloc[0].compliance_heads)
    h = q.attrition["advanced"] + q.emigration["advanced"]
    cont = n0 * np.exp(-h * T)
    got = float(df.iloc[-1].compliance_heads)
    ok = abs(got - cont) / cont < 1e-9 and n0 > 0
    return ok, f"N0={n0:.3f} got={got:.6f} N0*exp(-hT)={cont:.6f}"


def test_pool_analytic_decay(row: pd.Series, p: Params, T: int = 10) -> Tuple[bool, str]:
    """The unabsorbed pool follows the same exact constant-hazard survival."""
    q = _clone(p)
    q.attrition = {"advanced": 0.05, "intermediate": 0.0, "frontline": 0.0}
    q.emigration = {"advanced": 0.03, "intermediate": 0.0, "frontline": 0.0}
    q.emigration_feedback = False
    dt = 1.0 / 48.0
    pool = {"advanced": 1000.0, "intermediate": 0.0, "frontline": 0.0}
    exits = {"advanced": 0.03, "intermediate": 0.0, "frontline": 0.0}
    for _ in range(int(round(T / dt))):
        pool, _ = M.step_pool(pool, q, dt, exits)
    h = q.attrition["advanced"] + exits["advanced"] + M._pool_mortality("advanced")
    expected = 1000.0 * np.exp(-h * T)
    got = pool["advanced"]
    ok = abs(got - expected) / expected < 1e-10
    return ok, f"pool 1000 -> {got:.6f}; exact 1000*exp(-hT)={expected:.6f}"


def test_progression_in_pool(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """A lower-tier pool member promoted to advanced remains one person."""
    q = _clone(p)
    q.model_posts = True
    q.progression_from_below = {"advanced": 0.50, "intermediate": 0.0}
    q.public_entry_share = {t: 1.0 for t in TIERS}
    stock = {t: np.zeros(len(M.AGES)) for t in TIERS}
    pool = {"advanced": 0.0, "intermediate": 100.0, "frontline": 0.0}
    grads = {"advanced": 10.0, "intermediate": 0.0, "frontline": 0.0}
    out, new_pool, info = M.enter_graduates(stock, pool, grads, free_posts=0.0, p=q)
    before = sum(float(stock[t].sum()) + pool[t] for t in TIERS)
    after = sum(float(out[t].sum()) + new_pool[t] for t in TIERS)
    ok = (abs(after - (before + info["tracked_inflow"])) < 1e-9
          and abs(new_pool["intermediate"] - 95.0) < 1e-9
          and abs(new_pool["advanced"] - 10.0) < 1e-9
          and abs(info["promoted"] - 5.0) < 1e-9)
    return ok, (f"total heads {before:.1f} -> {after:.1f}; intermediate pool="
                f"{new_pool['intermediate']:.1f}; advanced pool={new_pool['advanced']:.1f}; "
                f"promoted={info['promoted']:.1f}; new inflow={info['tracked_inflow']:.1f}")


def test_analytic_equilibrium(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """Constant intake I, hazard h, unlimited posts: stock -> I c s / h."""
    r, q = _isolated(row, p)
    q.annual_intake = {"advanced": 20.0, "intermediate": 0.0, "frontline": 0.0}
    with _NoMortality():
        df = M.run(r, q, horizon=220)
    h = q.attrition["advanced"] + q.emigration["advanced"]
    # per-step inflow I c s dt entering a stock with survival exp(-h dt):
    # equilibrium = I c s dt / (1 - exp(-h dt)) ~ I c s / h (exact as dt->0)
    dt = M.DT_DEFAULT
    expected = (20.0 * q.completion["advanced"] * q.public_entry_share["advanced"]
                * dt / (1.0 - np.exp(-h * dt)))
    got = float(df.iloc[-1].compliance_heads)
    ok = abs(got - expected) / expected < 1e-6
    cont = 20.0 * q.completion["advanced"] * q.public_entry_share["advanced"] / h
    return ok, (f"equilibrium {got:.4f} vs discrete-exact {expected:.4f} "
                f"(continuous I*c*s/h = {cont:.4f})")


def test_ageing_exact_and_dt_free(row: pd.Series, p: Params, T: int = 25) -> Tuple[bool, str]:
    """No hazards, no inflow: the people who retire by year T are exactly those
    whose age reaches the retirement age within T annual shifts, and the answer
    must not depend on dt (no numerical diffusion in the age chain)."""
    q = _clone(p)
    _all_loops(q, False)
    q.attrition = {t: 0.0 for t in TIERS}
    q.emigration = {t: 0.0 for t in TIERS}
    q.annual_intake = {t: 0.0 for t in TIERS}
    q.model_posts = False
    q.progression_from_below = {"advanced": 0.0, "intermediate": 0.0}
    with _NoMortality():
        st0 = M.initialize_stock(row, q)
        expected = float(sum(st0[t][M.AGES >= q.retirement_age - T].sum() for t in TIERS))
        got = {dt: float(M.run(row, q, horizon=T, dt=dt).exit_retirement.sum())
               for dt in (1 / 12, 1 / 48)}
    ok = all(abs(g - expected) < 1e-8 * max(expected, 1.0) for g in got.values())
    return ok, (f"retirees by year {T}: expected {expected:.4f}; dt=1/12 {got[1/12]:.4f}; "
                f"dt=1/48 {got[1/48]:.4f}")


def test_extreme_no_inflow_no_outflow(row: pd.Series, p: Params, horizon: int = 25) -> Tuple[bool, str]:
    """With every outflow and inflow switched off the stock must be constant."""
    q = _clone(p)
    _all_loops(q, False)
    q.attrition = {t: 0.0 for t in TIERS}
    q.emigration = {t: 0.0 for t in TIERS}
    q.annual_intake = {t: 0.0 for t in TIERS}
    q.retirement_age = 999.0
    with _NoMortality():
        df = M.run(row, q, horizon=horizon)
        a, b = df.iloc[0].compliance_heads, df.iloc[-1].compliance_heads
    ok = abs(a - b) < max(1e-6, 1e-6 * max(a, 1.0))
    return ok, f"heads {a:.6f} -> {b:.6f} (drift {b-a:+.2e})"


def test_extreme_total_attrition(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """100%/yr attrition empties the stock and never drives it negative."""
    q = _clone(p)
    q.attrition = {t: 1.0 for t in TIERS}
    q.annual_intake = {t: 0.0 for t in TIERS}
    df = M.run(row, q, horizon=10)
    end = df.iloc[-1].compliance_heads
    cols = ["heads_advanced", "heads_intermediate", "heads_frontline"]
    neg = bool((df[cols] < -1e-9).any().any())
    ok = end < 0.01 * max(df.iloc[0].compliance_heads, 1e-9) and not neg
    return ok, f"heads {df.iloc[0].compliance_heads:.1f} -> {end:.4f}; negative stock: {neg}"


def test_extreme_zero_population(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """Near-zero population implies near-zero JEE target and routine workload."""
    r = row.copy()
    for y in M.WPP_ANCHORS:
        r[f"pop_{y}"] = 1.0
    df = M.run(r, _clone(p), horizon=5)
    tgt, rout = df.iloc[0].jee_target_heads, df.iloc[0].routine_fte
    ok = tgt < 1e-4 and rout < 1e-3
    return ok, f"target={tgt:.3e} routine_fte={rout:.3e} (readiness FTE stays by design)"


def test_extreme_infinite_posts(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """If posts never bind there is no unabsorbed pool and deployed == trained."""
    q = _clone(p)
    q.model_posts = True
    q.initial_posts_ratio = 1000.0
    q.post_feedback = False
    q.post_creation_per_year = 0.0
    df = M.run(row, q, horizon=25)
    d = float((df.deployed_heads - df.compliance_heads).abs().max())
    u = float(df.unabsorbed_heads.max())
    return (d < 1e-6 and u < 1e-6), f"max |deployed - trained| = {d:.2e}; max pool = {u:.2e}"


def test_pool_identity(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """R3 acts on the identifiable pool only: with the loop on and a pool
    present, the pool hazard exceeds the base while post-holder emigration is
    untouched (checked via the loop-on/off difference in exits)."""
    q = _clone(p)
    _all_loops(q, False)
    st0 = M.initialize_stock(row, q)
    if sum(float(st0[t].sum()) for t in M.counted_tiers(q)) <= 0:
        return True, ("not applicable: no advanced/intermediate starting stock, so no "
                      "pool can exist at t0")
    q.model_posts = True
    q.initial_posts_ratio = 0.5            # force a pool at t0
    q.post_creation_per_year = 0.0
    q.emigration_feedback = True
    q.emigration_sensitivity = 2.0
    q.emigration_max_multiplier = 4.0
    on = M.run(row, q, horizon=3)
    q.emigration_feedback = False
    off = M.run(row, q, horizon=3)
    pool0 = float(on.iloc[0].unabsorbed_heads)
    rate_on, rate_off = float(on.iloc[0].emigration_rate_adv), float(off.iloc[0].emigration_rate_adv)
    serv_same = abs(on.iloc[0].deployed_heads - off.iloc[0].deployed_heads) < 1e-9
    ok = pool0 > 0 and rate_on > rate_off and serv_same
    return ok, (f"pool0={pool0:.1f}; advanced pool emigration hazard {rate_off:.3f} -> "
                f"{rate_on:.3f}; post-holder stock identical at t0: {serv_same}")


def test_integration_error(row: pd.Series, p: Params, horizon: int = 25,
                           tol: float = 0.01) -> Tuple[bool, str]:
    """Halving the time step must not change the trajectory materially."""
    q = _all_loops(_clone(p), True)
    out = {}
    for dt in (1 / 12, 1 / 24, 1 / 48):
        out[dt] = M.run(row, q, horizon=horizon, dt=dt).set_index("calendar_year").compliance_heads
    ref = out[1 / 48]
    worst, detail = 0.0, []
    for dt in (1 / 12, 1 / 24):
        s = out[dt]
        common = s.index.intersection(ref.index)
        rel = float(((s[common] - ref[common]).abs() / ref[common].clip(lower=1e-9)).max())
        if dt == 1 / 12:
            worst = rel
        detail.append(f"dt=1/{int(1/dt)}: max rel diff vs dt=1/48 = {rel:.4%}")
    return worst < tol, "; ".join(detail)


def test_switches_equal_zero_sensitivity(row: pd.Series, p: Params,
                                         horizon: int = 25) -> Tuple[bool, str]:
    """INTERNAL CONSISTENCY, not a v1 regression: with loops switched off the
    run must equal the run with loops on but sensitivities at zero and the
    mentor cap unbounded. (A true v1 comparison needs the v1 code and data,
    which are not in this bundle; see regression_report.py.)"""
    base = _clone(p)
    off = _all_loops(_clone(base), False)
    zero = _clone(base)
    # R1 is included only when there is an advanced stock to mentor from: with
    # none, capacity is 0 x anything = 0 and R1 legitimately halts training
    # (the mentor trap), so "unbounded capacity" cannot be constructed.
    st0 = M.initialize_stock(row, base)
    zero.mentor_feedback = bool(st0["advanced"].sum() > 0)
    zero.burnout_feedback = zero.emigration_feedback = True
    zero.post_feedback = False
    zero.burnout_sensitivity = 0.0
    zero.emigration_sensitivity = 0.0
    zero.mentor_share = 1e9
    a = M.run(row, off, horizon=horizon).compliance_heads.values
    b = M.run(row, zero, horizon=horizon).compliance_heads.values
    rel = float(np.max(np.abs(a - b) / np.clip(np.abs(a), 1e-9, None)))
    return rel < 1e-9, f"max relative difference = {rel:.3e}"


def test_monotone_in_intake(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """More training never reduces TRAINED stock (open loop). Deployed stock is
    reported separately because posts bind."""
    q = _all_loops(_clone(p), False)
    base = dict(p.annual_intake)
    vals = []
    for mult in (0.5, 1.0, 2.0, 4.0):
        q.annual_intake = {t: base[t] * mult for t in TIERS}
        vals.append(M.run(row, q, horizon=25).iloc[-1].compliance_heads)
    ok = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
    return ok, "trained heads at intake x(0.5,1,2,4) = " + ", ".join(f"{v:.1f}" for v in vals)


def test_demand_scaling(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """Numerical homogeneity checks on the REAL demand code: each output must
    scale exactly as its declared units imply."""
    pop, a2 = 1_000_000.0, 100.0
    b = M.demand_measures(pop, a2, p)
    checks = []
    q = _clone(p); q.people_per_fe *= 2
    checks.append(("target ~ 1/people_per_fe",
                   abs(M.demand_measures(pop, a2, q)["jee_target_heads"] / b["jee_target_heads"] - 0.5)))
    q = _clone(p); q.productive_hours_per_fte *= 2
    checks.append(("routine_fte ~ 1/hours_per_fte",
                   abs(M.demand_measures(pop, a2, q)["routine_fte"] / b["routine_fte"] - 0.5)))
    q = _clone(p); q.hours_per_reporting_unit_week *= 2
    checks.append(("routine_fte ~ hours_per_unit_week",
                   abs(M.demand_measures(pop, a2, q)["routine_fte"] / b["routine_fte"] - 2.0)))
    checks.append(("target ~ population",
                   abs(M.demand_measures(2 * pop, a2, p)["jee_target_heads"] / b["jee_target_heads"] - 2.0)))
    checks.append(("outbreak_fte ~ population",
                   abs(M.demand_measures(2 * pop, a2, p)["outbreak_fte"] / b["outbreak_fte"] - 2.0)))
    q = _clone(p); q.deployment_days *= 3
    checks.append(("outbreak_fte ~ deployment_days",
                   abs(M.demand_measures(pop, a2, q)["outbreak_fte"] / b["outbreak_fte"] - 3.0)))
    bad = [n for n, e in checks if e > 1e-9]
    return len(bad) == 0, f"{len(checks)} scaling relations; failing: {bad or 'none'}"


STRUCTURAL_TESTS: List[Tuple[str, Callable]] = [
    ("Mass conservation incl. pool and active progression (tol 1e-9)", test_conservation),
    ("Promotion while unabsorbed does not double-count heads", test_progression_in_pool),
    ("Inflow accounting: completions = promotions + public + non-public", test_inflow_accounting),
    ("Analytic: single cohort decay = N0 exp(-hT)", test_analytic_decay),
    ("Analytic: unabsorbed pool decay = N0 exp(-hT)", test_pool_analytic_decay),
    ("Analytic: constant intake equilibrium = discrete-exact I c s dt/(1-e^-h dt)", test_analytic_equilibrium),
    ("Analytic: retirement timing exact and independent of dt", test_ageing_exact_and_dt_free),
    ("Extreme condition: no inflow, no outflow -> constant stock", test_extreme_no_inflow_no_outflow),
    ("Extreme condition: 100%/yr attrition -> empties, stays non-negative", test_extreme_total_attrition),
    ("Extreme condition: population -> 0 -> demand -> 0", test_extreme_zero_population),
    ("Extreme condition: unlimited posts -> no pool, deployed == trained", test_extreme_infinite_posts),
    ("R3 acts on the unabsorbed pool only", test_pool_identity),
    ("Step-size convergence: dt = 1/12 and 1/24 vs 1/48", test_integration_error),
    ("Internal consistency: loops off == sensitivities zero (not a v1 regression)",
     test_switches_equal_zero_sensitivity),
    ("Monotonicity: trained stock non-decreasing in intake", test_monotone_in_intake),
    ("Demand block: numerical unit-scaling relations", test_demand_scaling),
]


# ---------------------------------------------------------------------------
# Dimensional analysis: a small unit algebra, not a label check.
# ---------------------------------------------------------------------------

class Dim:
    """Exponent vector over base dimensions. Addition requires equal dims."""
    def __init__(self, **e):
        self.e = {k: v for k, v in e.items() if v}
    def __mul__(self, o):
        r = dict(self.e)
        for k, v in o.e.items():
            r[k] = r.get(k, 0) + v
        return Dim(**r)
    def __truediv__(self, o):
        return self * Dim(**{k: -v for k, v in o.e.items()})
    def __add__(self, o):
        if self.e != o.e:
            raise TypeError(f"cannot add {self} and {o}")
        return Dim(**self.e)
    __sub__ = __add__
    def __eq__(self, o):
        return self.e == o.e
    def __repr__(self):
        return "*".join(f"{k}^{v}" for k, v in sorted(self.e.items())) or "1"


def _u(**e): return Dim(**e)


def test_dimensional_consistency() -> Tuple[bool, str]:
    """Verify the algebra of the demand and feedback blocks with a real unit
    system. Also proves the checker can fail by adding a rate to a stock."""
    person, head, hour, yr = _u(person=1), _u(head=1), _u(hour=1), _u(yr=1)
    week, unit, event, day, epi = _u(week=1), _u(unit=1), _u(event=1), _u(day=1), _u(epi=1)
    fte, mperson = _u(fte=1), _u(mperson=1)
    one = _u()
    pop = person
    ppfe = person / head
    admin2, unit_per_a2 = _u(admin2=1), unit / _u(admin2=1)
    hr_per_unit_week = hour / unit / week
    weeks_per_yr = week / yr
    prod_hours = hour / yr / fte
    out_per_m = event / yr / mperson
    mp_per_p = mperson / person
    team, dep_days, fh_per_day = epi / event, day, hour / day
    checks = [
        ("jee_target_heads = pop / people_per_fe", pop / ppfe, head),
        ("routine_hours = admin2 x unit/admin2 x week/yr x hour/unit/week",
         admin2 * unit_per_a2 * weeks_per_yr * hr_per_unit_week, hour / yr),
        ("routine_fte = routine_hours / productive_hours",
         (hour / yr) / prod_hours, fte),
        ("outbreak_hours = (pop x Mperson/person) x event/yr/Mperson x epi/event x day/event x hour/day",
         pop * mp_per_p * out_per_m * team * dep_days * fh_per_day, epi * hour / yr),
        ("training_intake_cap = (mentors x resident/mentor) / yr",
         _u(mentor=1) * (_u(resident=1) / _u(mentor=1)) / yr, _u(resident=1) / yr),
        ("post_creation = (head - head) / yr", (head - head) / yr, head / yr),
        ("burnout_multiplier = 1 + sens x excess", one + (one * one), one),
    ]
    bad = [n for n, got, want in checks if not (got == want)]
    try:
        _ = head + head / yr
        checker_can_fail = False
    except TypeError:
        checker_can_fail = True
    ok = (len(bad) == 0) and checker_can_fail
    return ok, (f"{len(checks)} expressions checked by unit algebra; mismatches: {bad or 'none'}; "
                f"rejects stock + rate: {checker_can_fail}. Note: outbreak_hours carries an "
                "epidemiologist factor (epi x hour), reconciled to FTE by the productive-hours "
                "denominator, as in the code.")


def run_structural_tests(row: pd.Series, p: Optional[Params] = None,
                         horizon: int = 25,
                         cohort_history: Optional[pd.DataFrame] = None,
                         dt: float = M.DT_DEFAULT) -> pd.DataFrame:
    """Run structural checks using the active parameters and starting cohort.

    The conservation and inflow checks also use the app's horizon, integration
    step, and optional cohort history. Analytic tests deliberately construct
    their own controlled inputs and step sizes.
    """
    p = prepare_params(row, p)
    recs = []
    for name, fn in STRUCTURAL_TESTS:
        try:
            if fn in (test_conservation, test_inflow_accounting):
                ok, detail = fn(row, p, horizon, cohort_history, dt)
            else:
                ok, detail = fn(row, p)
        except Exception as exc:                      # a crash is a failure
            ok, detail = False, f"EXCEPTION: {type(exc).__name__}: {exc}"
        recs.append({"test": name, "passed": bool(ok), "detail": detail})
    try:
        ok, detail = test_dimensional_consistency()
    except Exception as exc:
        ok, detail = False, f"EXCEPTION: {type(exc).__name__}: {exc}"
    recs.append({"test": "Dimensional consistency (unit algebra)", "passed": bool(ok),
                 "detail": detail})
    return pd.DataFrame(recs)


# ===========================================================================
# Uncertainty: Latin hypercube + PRCC
# ===========================================================================

# name -> (low, high, kind). 'kind' is only used for reporting.
# Every range is an ANALYST-CHOSEN UNIFORM INTERVAL with no empirical
# distribution behind it (see PSA_BASIS). Treat the output as conditional
# sensitivity analysis.
PSA_RANGES: Dict[str, Tuple[float, float, str]] = {
    "survival_advanced":      (0.50, 0.85, "share"),
    "survival_intermediate":  (0.55, 0.90, "share"),
    "public_share_advanced":  (0.55, 0.95, "share"),
    "attrition_advanced":     (0.015, 0.070, "rate/yr"),
    "emigration_advanced":    (0.010, 0.080, "rate/yr"),
    "retirement_age":         (55.0, 65.0, "years"),
    "overlap_adv_from_int":   (0.10, 0.60, "share"),
    "completion_advanced":    (0.65, 0.95, "share"),
    "protected_time_advanced": (0.40, 0.85, "share"),
    "skill_weight_intermediate": (0.40, 0.80, "weight"),
    "people_per_fe":          (100_000.0, 300_000.0, "person/head"),
    "annual_outbreaks_per_million": (0.2, 2.0, "event/yr/M"),
    "mentor_share":           (0.15, 0.60, "share"),
    "residents_per_mentor":   (1.0, 4.0, "resident/mentor"),
    "burnout_sensitivity":    (0.0, 0.80, "1/excess load"),
    "emigration_sensitivity": (0.0, 3.0, "1/unabsorbed share"),
    "political_ambition":     (0.20, 1.00, "share"),
    "post_adjust_years":      (2.0, 15.0, "years"),
    "intake_multiplier":      (0.5, 2.0, "multiplier"),
}


PSA_BASIS = {nm: "analyst range; no empirical distribution" for nm in PSA_RANGES}


def _apply_sample(p: Params, s: Dict[str, float], base_intake: Dict[str, float]) -> Params:
    q = copy.deepcopy(p)

    q.survival_to_2026["advanced"] = s["survival_advanced"]
    q.survival_to_2026["intermediate"] = s["survival_intermediate"]
    q.public_sector_share["advanced"] = s["public_share_advanced"]
    q.public_entry_share["advanced"] = s["public_share_advanced"]
    q.attrition["advanced"] = s["attrition_advanced"]
    q.emigration["advanced"] = s["emigration_advanced"]
    q.retirement_age = s["retirement_age"]
    q.overlap_adv_from_int = s["overlap_adv_from_int"]
    q.completion["advanced"] = s["completion_advanced"]
    q.protected_time["advanced"] = s["protected_time_advanced"]
    q.skill_weight["intermediate"] = s["skill_weight_intermediate"]
    q.people_per_fe = s["people_per_fe"]
    q.annual_outbreaks_per_million = s["annual_outbreaks_per_million"]
    q.mentor_share = s["mentor_share"]
    q.residents_per_mentor = s["residents_per_mentor"]
    q.burnout_sensitivity = s["burnout_sensitivity"]
    q.emigration_sensitivity = s["emigration_sensitivity"]
    q.political_ambition = s["political_ambition"]
    q.post_adjust_years = s["post_adjust_years"]
    intake_multiplier = float(s.get("intake_multiplier", 1.0))
    q.annual_intake = {t: base_intake[t] * intake_multiplier for t in TIERS}
    return q


def latin_hypercube(n: int, names: List[str], seed: int = 20261113) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k = len(names)
    cut = np.arange(n + 1) / n
    out = np.empty((n, k))
    for j in range(k):
        u = rng.uniform(cut[:n], cut[1:])
        out[:, j] = rng.permutation(u)
    cols = {}
    for j, nm in enumerate(names):
        lo, hi, _ = PSA_RANGES[nm]
        cols[nm] = lo + out[:, j] * (hi - lo)
    return pd.DataFrame(cols)


def psa(row: pd.Series, p: Optional[Params] = None, n: int = 500, horizon: int = 25,
        seed: int = 20261113, feedback: bool = True,
        cohort_history: Optional[pd.DataFrame] = None,
        dt: float = M.DT_DEFAULT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Latin hypercube CONDITIONAL SENSITIVITY ANALYSIS over PSA_RANGES.

    Returns (samples_with_outputs, trajectory_percentiles). The percentiles are
    over uniform analyst-chosen ranges; they are not credible intervals.
    """
    if p is None:
        p = prepare_params(row, None)
    p = copy.deepcopy(p)
    p.mentor_feedback = p.burnout_feedback = p.emigration_feedback = p.post_feedback = feedback
    # intake multiplier is applied to the ACTIVE intake, not a rebuilt default
    base_intake = dict(p.annual_intake)

    names = list(PSA_RANGES)
    X = latin_hypercube(n, names, seed)
    traj = []
    outs = []
    for i in range(n):
        q = _apply_sample(p, X.iloc[i].to_dict(), base_intake)
        try:
            df = M.run(row, q, horizon=horizon,
                       cohort_history=cohort_history, dt=dt)
        except Exception:
            outs.append({k: np.nan for k in
                         ("heads_final", "deployed_final", "gap_final", "ratio_final",
                          "years_to_target", "cum_emigration")})
            traj.append(np.full(int(round(horizon / dt)) + 1, np.nan))
            continue
        f = df.iloc[-1]
        met = df.index[df.jee_ratio >= 1.0]
        outs.append({
            "heads_final": f.compliance_heads,
            "deployed_final": f.deployed_heads,
            "gap_final": f.jee_gap_heads,
            "ratio_final": f.jee_ratio,
            "years_to_target": (df.loc[met[0], "time_years"] if len(met) else np.nan),
            "cum_emigration": f.cum_exit_emigration,
        })
        traj.append(df.compliance_heads.values)
    S = pd.concat([X, pd.DataFrame(outs)], axis=1)
    T = np.vstack([t for t in traj])
    years = M.BASE_YEAR + np.arange(T.shape[1]) * dt
    Q = pd.DataFrame({
        "calendar_year": years,
        "p2_5": np.nanpercentile(T, 2.5, axis=0),
        "p25": np.nanpercentile(T, 25, axis=0),
        "median": np.nanpercentile(T, 50, axis=0),
        "p75": np.nanpercentile(T, 75, axis=0),
        "p97_5": np.nanpercentile(T, 97.5, axis=0),
    })
    return S, Q


def required_intake_by_year(
    row: pd.Series,
    p: Optional[Params] = None,
    n: int = 100,
    horizon: int = 25,
    target_ratio: float = 1.0,
    outcome: str = "deployed_ratio",
    seed: int = 20261113,
    cohort_history: Optional[pd.DataFrame] = None,
    dt: float = M.DT_DEFAULT,
    max_multiplier: float = 32.0,
) -> pd.DataFrame:
    """Estimate annual intake run-rates needed by each target year.

    For each Latin-hypercube parameter draw, simulate a small grid of uniform
    multipliers on the active tier mix. Linearly interpolate the first crossing
    of the requested target. Initial trainee queues stay at the active intake;
    the candidate intake applies prospectively. The result is conditional on
    the analyst-selected ranges and the active tier mix, not a calibrated
    probability distribution or an optimized year-varying schedule.
    """
    if outcome not in ("deployed_ratio", "jee_ratio"):
        raise ValueError("outcome must be deployed_ratio or jee_ratio")
    if not 0.0 < target_ratio <= 2.0:
        raise ValueError("target_ratio must be greater than 0 and no greater than 2")
    if n < 2:
        raise ValueError("n must be at least 2")
    if max_multiplier <= 0:
        raise ValueError("max_multiplier must be positive")
    if not np.isfinite(dt) or dt <= 0 or dt > 1:
        raise ValueError("dt must be finite, positive, and no greater than one year")
    if not np.isclose(1.0 / dt, round(1.0 / dt), atol=1e-8):
        raise ValueError("annual target table requires a step size that divides one year")

    if p is None:
        p = prepare_params(row, None)
    p = copy.deepcopy(p)
    base_intake = {t: max(0.0, float(p.annual_intake.get(t, 0.0))) for t in TIERS}
    base_total = float(sum(base_intake.values()))
    if base_total <= 0.0:
        raise ValueError("Set a positive annual intake mix before estimating required intake")

    years = np.arange(M.BASE_YEAR + 1, M.BASE_YEAR + int(horizon) + 1, dtype=int)
    names = [nm for nm in PSA_RANGES if nm != "intake_multiplier"]
    X = latin_hypercube(n, names, seed)
    grid = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    multipliers = np.array(sorted({m for m in grid if m <= max_multiplier}
                                  | {float(max_multiplier)}))
    response = np.full((n, len(multipliers), len(years)), np.nan)
    for draw_index in range(n):
        sample = X.iloc[draw_index].to_dict()
        q = _apply_sample(p, sample, base_intake)
        for mult_index, multiplier in enumerate(multipliers):
            candidate = copy.deepcopy(q)
            candidate.annual_intake = {t: base_intake[t] * multiplier for t in TIERS}
            try:
                df = M.run(
                    row, candidate, horizon=int(horizon),
                    cohort_history=cohort_history, dt=dt,
                    initial_pipeline_intake=base_intake,
                )
                idx = [int(round((year - M.BASE_YEAR) / dt)) for year in years]
                response[draw_index, mult_index, :] = df.iloc[idx][outcome].to_numpy(dtype=float)
            except Exception:
                continue

    records = []
    for year_index, year in enumerate(years):
        required_mult = np.full(n, np.nan)
        nonmonotone = np.zeros(n, dtype=bool)
        for draw_index in range(n):
            values = response[draw_index, :, year_index]
            if not np.isfinite(values).all():
                continue
            nonmonotone[draw_index] = bool(np.any(np.diff(values) < -1e-6))
            if values[0] >= target_ratio:
                required_mult[draw_index] = 0.0
                continue
            crossings = np.flatnonzero(values[1:] >= target_ratio)
            if len(crossings) == 0:
                continue
            upper_index = int(crossings[0] + 1)
            lo_y, hi_y = values[upper_index - 1], values[upper_index]
            lo_m, hi_m = multipliers[upper_index - 1], multipliers[upper_index]
            if hi_y <= lo_y:
                required_mult[draw_index] = hi_m
            else:
                share = float(np.clip((target_ratio - lo_y) / (hi_y - lo_y), 0.0, 1.0))
                required_mult[draw_index] = lo_m + share * (hi_m - lo_m)

        feasible = np.isfinite(required_mult)
        feasible_mult = required_mult[feasible]
        total_values = feasible_mult * base_total
        tier_values = {t: feasible_mult * base_intake[t] for t in TIERS}

        def quantile(values: np.ndarray, q: float) -> float:
            return float(np.quantile(values, q)) if len(values) else np.nan

        records.append({
            "target_year": int(year),
            "target_ratio": float(target_ratio),
            "outcome": outcome,
            "total_intake_p05": quantile(total_values, 0.05),
            "total_intake_p50": quantile(total_values, 0.50),
            "total_intake_p95": quantile(total_values, 0.95),
            "advanced_intake_p50": quantile(tier_values["advanced"], 0.50),
            "intermediate_intake_p50": quantile(tier_values["intermediate"], 0.50),
            "frontline_intake_p50": quantile(tier_values["frontline"], 0.50),
            "share_sampled_sets_feasible": float(feasible.mean()),
            "share_sampled_sets_nonmonotone": float(nonmonotone.mean()),
            "feasible_draws": int(feasible.sum()),
            "sampled_draws": int(n),
            "max_multiplier_searched": float(max_multiplier),
        })
    return pd.DataFrame(records)


def prcc(samples: pd.DataFrame, outcome: str = "ratio_final") -> pd.DataFrame:
    """Partial rank correlation coefficients with two-sided p-values.

    PRCC is the right statistic here because the model is monotone but
    strongly non-linear in most inputs, and the inputs are sampled
    independently (Marino et al. 2008).
    """
    names = [c for c in PSA_RANGES if c in samples.columns]
    d = samples[names + [outcome]].dropna()
    n = len(d)
    R = d.rank().to_numpy(dtype=float)
    R = (R - R.mean(axis=0)) / R.std(axis=0, ddof=1)
    C = np.corrcoef(R, rowvar=False)
    try:
        P = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(C)
    k = len(names)
    recs = []
    for j, nm in enumerate(names):
        r = -P[j, k] / np.sqrt(P[j, j] * P[k, k])
        dof = n - k - 1
        if dof > 0 and abs(r) < 1:
            tstat = r * np.sqrt(dof / (1 - r ** 2))
            from math import erf, sqrt
            pval = 2 * (1 - 0.5 * (1 + erf(abs(tstat) / sqrt(2))))
        else:
            pval = np.nan
        recs.append({"parameter": nm, "prcc": r, "p_value": pval,
                     "units": PSA_RANGES[nm][2], "range_basis": PSA_BASIS[nm]})
    out = pd.DataFrame(recs)
    out["abs"] = out.prcc.abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def tornado(row: pd.Series, p: Optional[Params] = None, horizon: int = 25,
            outcome: str = "ratio_final") -> pd.DataFrame:
    """One-at-a-time low/high swings around the default, for the tornado plot."""
    if p is None:
        p = prepare_params(row, None)
    p = copy.deepcopy(p)
    p.mentor_feedback = p.burnout_feedback = p.emigration_feedback = p.post_feedback = True
    base_intake = dict(p.annual_intake)
    mid = {nm: (lo + hi) / 2 for nm, (lo, hi, _) in PSA_RANGES.items()}

    def outcome_of(sample):
        q = _apply_sample(p, sample, base_intake)
        f = M.run(row, q, horizon=horizon).iloc[-1]
        return {"ratio_final": f.jee_ratio, "heads_final": f.compliance_heads,
                "deployed_final": f.deployed_heads}[outcome]

    base = outcome_of(mid)
    recs = []
    for nm, (lo, hi, unit) in PSA_RANGES.items():
        s_lo, s_hi = dict(mid), dict(mid)
        s_lo[nm], s_hi[nm] = lo, hi
        v_lo, v_hi = outcome_of(s_lo), outcome_of(s_hi)
        recs.append({"parameter": nm, "units": unit, "low_value": lo, "high_value": hi,
                     "outcome_low": v_lo, "outcome_high": v_hi,
                     "swing": abs(v_hi - v_lo)})
    return (pd.DataFrame(recs).assign(baseline=base)
            .sort_values("swing", ascending=False).reset_index(drop=True))


def sensitivity_summary(S: pd.DataFrame) -> Dict[str, float]:
    """Labelled summary of a sensitivity run. Names say what these are."""
    good = S.dropna(subset=["ratio_final"])
    return {
        "n_runs": int(len(good)),
        "median_ratio_final": float(good.ratio_final.median()),
        "sampled_p2_5": float(good.ratio_final.quantile(.025)),
        "sampled_p97_5": float(good.ratio_final.quantile(.975)),
        "share_of_runs_meeting_target": float((good.ratio_final >= 1).mean()),
    }
