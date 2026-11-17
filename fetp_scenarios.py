"""Named conditional policy scenarios for the African FETP model.

Scenarios inherit the active application parameters and change only the
assumptions named in their descriptions. They are exploratory comparisons,
not causal estimates or probability-weighted forecasts.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import pandas as pd

import fetp_model as M
from fetp_model import Params, TIERS


SCENARIO_DOC = {
    "status_quo": "Active intake and model settings continue; feedback settings are toggled for comparison.",
    "donor_cliff": "To represent an immediate donor-cliff scenario, annual intake is set to 30% of the active value throughout the projection; financing itself is not modeled.",
    "train_harder": "Active intake doubles; other active settings remain in place.",
    "absorb_first": "With B1 active, intake is unchanged; the funded-post ceiling is 100% of the benchmark, adjustment time is 3 years, and annual post growth is capped at 80.",
    "domestic_transition": "Intake is unchanged; the funded-post ceiling is 90% of the benchmark and advanced-tier attrition and emigration are halved.",
}


def _base(row: pd.Series, base_params: Optional[Params] = None,
          feedback: bool = True) -> Params:
    if base_params is None:
        p = M.default_params_for(row)
        p.annual_intake = M.implied_historical_intake(row, p)
    else:
        p = copy.deepcopy(base_params)
    p.mentor_feedback = p.burnout_feedback = p.emigration_feedback = p.post_feedback = feedback
    return p


def make_scenarios(row: pd.Series, feedback: bool = True,
                   base_params: Optional[Params] = None) -> Dict[str, Params]:
    base = _base(row, base_params, feedback)
    out = {"status_quo": copy.deepcopy(base)}

    q = copy.deepcopy(base)
    q.annual_intake = {t: v * 0.30 for t, v in base.annual_intake.items()}
    out["donor_cliff"] = q

    q = copy.deepcopy(base)
    q.annual_intake = {t: v * 2.0 for t, v in base.annual_intake.items()}
    out["train_harder"] = q

    q = copy.deepcopy(base)
    q.political_ambition = 1.00
    q.post_adjust_years = 3.0
    q.max_post_growth_per_year = 80.0
    out["absorb_first"] = q

    q = copy.deepcopy(base)
    q.political_ambition = 0.90
    q.attrition["advanced"] *= 0.5
    q.emigration["advanced"] *= 0.5
    out["domestic_transition"] = q
    return out


def run_scenarios(row: pd.Series, horizon: int = 25, feedback: bool = True,
                  base_params: Optional[Params] = None,
                  cohort_history: Optional[pd.DataFrame] = None,
                  dt: float = M.DT_DEFAULT) -> Dict[str, pd.DataFrame]:
    return {
        name: M.run(row, p, horizon=horizon, cohort_history=cohort_history, dt=dt)
        for name, p in make_scenarios(row, feedback, base_params).items()
    }


def scenario_table(row: pd.Series, horizon: int = 25,
                   base_params: Optional[Params] = None,
                   cohort_history: Optional[pd.DataFrame] = None,
                   dt: float = M.DT_DEFAULT
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return detailed outcomes and the open-loop/feedback comparison."""
    recs = []
    for fb in (False, True):
        for name, df in run_scenarios(row, horizon, fb, base_params,
                                      cohort_history, dt).items():
            f = df.iloc[-1]
            recs.append({
                "scenario": name, "feedback": fb,
                "trained_heads": f.compliance_heads,
                "deployed_heads": f.deployed_heads,
                "unabsorbed": f.unabsorbed_heads,
                "posts": f.established_posts,
                "target": f.jee_target_heads,
                "ratio_deployed": f.deployed_heads / max(f.jee_target_heads, 1e-9),
                "cum_emigration": f.cum_exit_emigration,
                "cum_retirement": f.cum_exit_retirement,
                "binding": f.binding_constraint,
            })
    long = pd.DataFrame(recs)
    wide = long.pivot(index="scenario", columns="feedback", values="deployed_heads")
    wide.columns = ["deployed_open_loop", "deployed_with_feedback"]
    wide["feedback_effect_pct"] = 100 * (
        wide.deployed_with_feedback / wide.deployed_open_loop.clip(lower=1e-9) - 1
    )
    return long, wide.reset_index()


def decompose_decline(row: pd.Series, horizon: int = 25,
                      feedback: bool = True,
                      base_params: Optional[Params] = None,
                      cohort_history: Optional[pd.DataFrame] = None,
                      dt: float = M.DT_DEFAULT) -> pd.DataFrame:
    p = _base(row, base_params, feedback)
    df = M.run(row, p, horizon=horizon, cohort_history=cohort_history, dt=dt)
    f = df.iloc[-1]
    total = sum(f[f"cum_exit_{c}"] for c in
                ("attrition", "emigration", "mortality", "retirement"))
    return pd.DataFrame([
        {"channel": c, "people": float(f[f"cum_exit_{c}"]),
         "share_pct": 100 * float(f[f"cum_exit_{c}"]) / max(total, 1e-9),
         "addressable": c in ("attrition", "emigration")}
        for c in ("retirement", "emigration", "attrition", "mortality")
    ])


def policy_resistance_curve(row: pd.Series, horizon: int = 25,
                            multipliers=(0.5, 1, 2, 3, 4, 6, 8),
                            feedback: bool = True,
                            base_params: Optional[Params] = None,
                            cohort_history: Optional[pd.DataFrame] = None,
                            dt: float = M.DT_DEFAULT) -> pd.DataFrame:
    """Final-year outputs across intake multipliers, conditional on inputs."""
    base = _base(row, base_params, feedback)
    initial_intake = dict(base.annual_intake)
    recs = []
    for mult in multipliers:
        p = copy.deepcopy(base)
        p.annual_intake = {t: initial_intake[t] * mult for t in TIERS}
        f = M.run(row, p, horizon=horizon, cohort_history=cohort_history, dt=dt).iloc[-1]
        recs.append({
            "intake_multiplier": mult,
            "trained_heads": f.compliance_heads,
            "deployed_heads": f.deployed_heads,
            "unabsorbed": f.unabsorbed_heads,
            "cum_emigration": f.cum_exit_emigration,
            "cost_of_training_relative": mult,
        })
    return pd.DataFrame(recs)
