"""Build a portable structural-check and scenario summary for the v3 model.

This is not a v1 regression comparison. It uses the data file alongside this
script and writes outputs to the selected directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import fetp_model as M
import fetp_scenarios as SC
import fetp_validate as V


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name(M.DATA_FILE))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("report_outputs"))
    parser.add_argument("--horizon", type=int, default=M.HORIZON_DEFAULT)
    args = parser.parse_args()

    countries = M.prepare_countries(pd.read_csv(args.data), str(args.data))
    args.out.mkdir(parents=True, exist_ok=True)
    outcomes, checks, scenarios = [], [], []

    for _, row in countries.iterrows():
        p = M.default_params_for(row)
        p.annual_intake = M.implied_historical_intake(row, p)
        p.mentor_feedback = p.burnout_feedback = p.emigration_feedback = p.post_feedback = True
        projection = M.run(row, p, horizon=args.horizon)
        f = projection.iloc[-1]
        outcomes.append({
            "country": row["country"], "iso3": row["iso3"],
            "year": int(M.BASE_YEAR + args.horizon),
            "compliance_heads": f.compliance_heads,
            "deployed_heads": f.deployed_heads,
            "unabsorbed_heads": f.unabsorbed_heads,
            "jee_benchmark": f.jee_target_heads,
            "deployed_ratio": f.deployed_ratio,
            "cum_emigration": f.cum_exit_emigration,
            "binding_constraint": f.binding_constraint,
        })
        battery = V.run_structural_tests(row, p, horizon=args.horizon)
        battery.insert(0, "country", row["country"])
        checks.append(battery)
        long, _ = SC.scenario_table(row, args.horizon, p)
        long.insert(0, "country", row["country"])
        scenarios.append(long)

    pd.DataFrame(outcomes).to_csv(args.out / "cross_country_outcomes.csv", index=False)
    pd.concat(checks, ignore_index=True).to_csv(args.out / "structural_checks.csv", index=False)
    pd.concat(scenarios, ignore_index=True).to_csv(args.out / "scenario_outcomes.csv", index=False)
    print(f"Wrote v3 scenario and structural summaries to {args.out}")
    print("These outputs are conditional on model assumptions; they are not a v1 regression or empirical validation.")


if __name__ == "__main__":
    main()
