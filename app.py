"""Streamlit front end for the African Field Epidemiology Workforce Model.

The interface presents conditional scenarios, separates serving staff from
unabsorbed graduates, and labels structural checks and sampled sensitivities
according to what they establish.
"""

import io
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fetp_model as M
import fetp_scenarios as SC
import fetp_validate as V
from fetp_model import (
    BASE_YEAR, DATA_FILE, HORIZON_DEFAULT, POOLED_ADVANCED, TIERS, TIER_LABEL,
    TRAINING_COST_USD, Params, cost_block, cross_country_snapshot,
    data_completeness, default_params_for, implied_historical_intake,
    parse_growth, prepare_countries, run, supply_data_quality,
)

st.set_page_config(page_title="African Field Epidemiology Workforce Model",
                   layout="wide", initial_sidebar_state="expanded")

PAL = {"target": "#2b3a55", "trained": "#b23a2e", "deployed": "#2f6b4f",
       "posts": "#c08a2e", "floor": "#7a8595"}


# ===========================================================================
# cached helpers
# ===========================================================================

@st.cache_data(show_spinner=False)
def _prepare(raw, source, base_year, mult, floor, slope, closure,
             wpp=None, growth_df=None, decay=0.015, rate_meaning="auto"):
    growth = parse_growth(growth_df) if growth_df is not None else None
    return prepare_countries(raw, source, pop_base_year=base_year,
                             fert_multiplier=mult, growth_floor=floor,
                             growth_slope=slope, gap_closure=closure, wpp=wpp,
                             growth=growth, growth_decay=decay,
                             rate_meaning=rate_meaning)


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float):
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def _run(row_dict: dict, params_json: str, horizon: int, hist_json: str, dt: float):
    row = pd.Series(row_dict)
    p = Params(**json.loads(params_json))
    hist = pd.read_json(io.StringIO(hist_json)) if hist_json else None
    return run(row, p, horizon=horizon, cohort_history=hist, dt=dt)


@st.cache_data(show_spinner=False)
def _snapshot(countries, frontline, pooled, ppfe, overlap):
    return cross_country_snapshot(countries, frontline, pooled, ppfe, overlap)


@st.cache_data(show_spinner=True)
def _psa(row_dict: dict, params_json: str, n: int, horizon: int,
         feedback: bool, hist_json: str, dt: float):
    row = pd.Series(row_dict)
    p = Params(**json.loads(params_json))
    hist = pd.read_json(io.StringIO(hist_json)) if hist_json else None
    return V.psa(row, p, n=n, horizon=horizon, feedback=feedback,
                 cohort_history=hist, dt=dt)


@st.cache_data(show_spinner=True)
def _required_intake(row_dict: dict, params_json: str, n: int, horizon: int,
                     target_ratio: float, outcome: str,
                     hist_json: str, dt: float):
    row = pd.Series(row_dict)
    p = Params(**json.loads(params_json))
    hist = pd.read_json(io.StringIO(hist_json)) if hist_json else None
    return V.required_intake_by_year(
        row, p, n=n, horizon=horizon, target_ratio=target_ratio,
        outcome=outcome, cohort_history=hist, dt=dt,
    )


@st.cache_data(show_spinner=True)
def _validate(row_dict: dict, params_json: str, horizon: int,
              hist_json: str, dt: float):
    row = pd.Series(row_dict)
    p = Params(**json.loads(params_json))
    hist = pd.read_json(io.StringIO(hist_json)) if hist_json else None
    return V.run_structural_tests(row, p, horizon, hist, dt)


@st.cache_data(show_spinner=True)
def _scenarios(row_dict: dict, params_json: str, horizon: int,
               hist_json: str, dt: float):
    row = pd.Series(row_dict)
    p = Params(**json.loads(params_json))
    hist = pd.read_json(io.StringIO(hist_json)) if hist_json else None
    long, wide = SC.scenario_table(row, horizon, p, hist, dt)
    curves = {fb: SC.run_scenarios(row, horizon, fb, p, hist, dt)
              for fb in (True, False)}
    return long, wide, curves


def _pj(p: Params) -> str:
    return json.dumps(asdict(p), default=str, sort_keys=True)


# ===========================================================================
# sidebar input blocks
# ===========================================================================

def optional_overrides():
    wpp = growth_df = None
    with st.sidebar.expander("Population source (important)"):
        st.caption("For publication, population must come from UN WPP 2024. "
                   "Download the medium-variant CSV from "
                   "population.un.org/wpp (Demographic Indicators, Medium) "
                   "and place it next to the app as wpp*.csv, or upload it here.")
        local = sorted(Path(__file__).parent.glob("wpp*.csv"))
        if local:
            st.success(f"Using {local[0].name}")
            wpp = _read_csv(str(local[0]), local[0].stat().st_mtime)
        else:
            up = st.file_uploader("UN WPP 2024 CSV (medium variant)", type="csv",
                                  key="wpp_csv")
            wpp = pd.read_csv(up, low_memory=False) if up is not None else None
        up = st.file_uploader("Or a growth-rate CSV (iso3, growth_rate_pct)",
                              type="csv", key="growth_csv")
        growth_df = pd.read_csv(up) if up is not None else None
    return wpp, growth_df


def population_controls():
    with st.sidebar.expander("Population projection assumptions"):
        rm = st.selectbox(
            "Meaning of the rate column in the spreadsheet",
            ["auto", "growth_pct", "tfr"], index=0,
            help="The source column named 'fertility_rate' may represent annual "
                 "population growth or births per woman. 'auto' uses a "
                 "distribution-based heuristic; verify the source metadata. "
                 "Forcing either interpretation changes the population scenario.")
        return dict(
            rate_meaning=rm,
            decay=st.number_input("Annual decline in the growth rate", 0.0, 0.08,
                                  0.015, 0.005, format="%.3f"),
            base_year=int(st.number_input("Year of the Population column",
                                          2015, 2026, 2023, 1)),
            mult=st.number_input("Fertility multiplier (tfr mode only)", 0.5, 3.0,
                                 1.0, 0.05),
            floor=st.number_input("Growth at replacement fertility", 0.0, 0.03,
                                  0.008, 0.001, format="%.3f"),
            slope=st.number_input("Growth per fertility point above 2.1", 0.0,
                                  0.02, 0.0065, 0.0005, format="%.4f"),
            closure=st.number_input("Annual closure of the fertility gap", 0.0,
                                    0.08, 0.022, 0.002, format="%.3f"),
        )


def get_countries(pc, wpp):
    path = Path(__file__).parent / DATA_FILE
    args = (pc["base_year"], pc["mult"], pc["floor"], pc["slope"], pc["closure"],
            wpp, pc["growth_df"], pc["decay"], pc["rate_meaning"])
    if path.exists():
        return _prepare(pd.read_csv(path), str(path), *args)
    st.warning(f"{DATA_FILE} was not found next to the app. Upload it to continue.")
    up = st.file_uploader("Country reference table (CSV)", type="csv",
                          key="country_csv")
    if up is None:
        st.stop()
    return _prepare(pd.read_csv(up), up.name, *args)


def feedback_controls(p: Params, k):
    st.subheader("Feedback structure")
    st.caption("These are what make this a system dynamics model rather than a "
               "cohort projection. Switch them all off and the model reduces "
               "exactly to an open-loop projection.")
    all_on = st.checkbox("Enable all feedback loops", value=True, key=k("fb_all"))

    with st.expander("R1  Mentor capacity trap", expanded=False):
        st.caption("Advanced training needs advanced mentors, so the stock "
                   "limits its own replacement.")
        p.mentor_feedback = st.checkbox("Active", value=all_on, key=k("fb_r1"))
        p.mentor_share = st.slider("Share of advanced staff who mentor", 0.0, 1.0,
                                   0.35, 0.05, key=k("mshare"))
        p.residents_per_mentor = st.slider("Advanced residents per mentor", 0.5,
                                           6.0, 2.0, 0.5, key=k("rpm"))
        p.residents_per_mentor_int = st.slider("Intermediate trainees per mentor",
                                               1.0, 15.0, 6.0, 1.0, key=k("rpmi"))

    with st.expander("R2  Burnout spiral", expanded=False):
        st.caption("When capability falls short of workload, the people who "
                   "remain carry more and leave faster.")
        p.burnout_feedback = st.checkbox("Active", value=all_on, key=k("fb_r2"))
        p.burnout_sensitivity = st.slider(
            "Attrition increase per unit of excess load", 0.0, 1.0, 0.25, 0.05,
            key=k("bsens"),
            help="Analyst-selected scenario parameter. Read the sensitivity "
                 "analysis conditionally; it is not an estimated effect.")
        p.burnout_max_multiplier = st.slider("Cap on the attrition multiplier",
                                             1.0, 6.0, 3.0, 0.5, key=k("bmax"))

    with st.expander("R3  Unabsorbed-graduate emigration", expanded=False):
        st.caption("The emigration hazard applies to the explicit unabsorbed "
                   "pool. Its sensitivity is an analyst-selected assumption, "
                   "not an estimated effect.")
        p.emigration_feedback = st.checkbox("Active", value=all_on, key=k("fb_r3"))
        p.emigration_sensitivity = st.slider(
            "Emigration increase per unit unabsorbed share", 0.0, 4.0, 1.2, 0.1,
            key=k("esens"))
        p.emigration_max_multiplier = st.slider("Cap on the emigration multiplier",
                                                1.0, 8.0, 4.0, 0.5, key=k("emax"))

    with st.expander("B1  Establishment response", expanded=False):
        st.caption("Ministries fund posts in response to a gap they perceive "
                   "late. The delay can make the response overshoot.")
        p.post_feedback = st.checkbox("Active", value=all_on, key=k("fb_b1"))
        p.political_ambition = st.slider("Share of the JEE target eventually funded",
                                         0.0, 1.5, 0.60, 0.05, key=k("ambition"))
        p.post_adjust_years = st.slider("Years to close the posts gap", 1.0, 20.0,
                                        6.0, 1.0, key=k("padj"))
        p.post_perception_delay = st.slider("Years before the gap is perceived",
                                            0.5, 10.0, 3.0, 0.5, key=k("pdelay"))
        p.max_post_growth_per_year = st.number_input("Max new posts per year", 0.0,
                                                     500.0, 40.0, 5.0, key=k("pgrow"))
    return p


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    st.title("African Field Epidemiology Workforce Model")
    st.caption(f"Demand, supply and gap for FETP-trained field epidemiologists, "
               f"{BASE_YEAR}-{BASE_YEAR + HORIZON_DEFAULT}.")

    wpp, growth_df = optional_overrides()
    pc = population_controls()
    pc["growth_df"] = growth_df
    countries = get_countries(pc, wpp)

    src = countries["pop_source"].astype(str)
    n_wpp = int(src.str.startswith("UN WPP").sum())
    if n_wpp == len(src):
        st.success("Population: UN WPP 2024 medium variant.")
    elif n_wpp > 0:
        st.warning(f"UN WPP 2024 population is available for {n_wpp} of "
                   f"{len(src)} rows. Other rows use the listed fallback source; "
                   "review `pop_source` before interpreting the comparison.")
    elif src.str.startswith("INTERIM").any():
        st.warning(
            "**No UN WPP series is loaded for these rows. Population is projected "
            "from the spreadsheet rate column, interpreted by a heuristic.** "
            "Auto-detection says: "
            + str(countries["rate_detect_reason"].iloc[0])
            + ". This interim projection is not a substitute for a sourced "
              "population series; load UN WPP data before publication.")
    else:
        st.warning("No UN WPP series is loaded. Inspect each row's `pop_source`; "
                   "it may use an uploaded growth rate, a simplified fertility "
                   "formula, or a constant-population fallback.")

    all_hosts = bool((countries.hosts_fetp == 1).all())
    has_admin2 = bool(countries["admin2_count"].notna().any())

    # ----------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Country")
        hosts_only = False if all_hosts else st.checkbox("FETP host countries only", True)
        pool = countries[countries.hosts_fetp == 1] if hosts_only else countries
        region = st.selectbox("Region", ["All"] + sorted(pool.region.unique()))
        if region != "All":
            pool = pool[pool.region == region]
        name = st.selectbox("Country", pool.country.tolist())
        row = countries[countries.country == name].iloc[0]
        iso = row["iso3"]

        def k(s: str) -> str:
            return f"{s}_{iso}"

        horizon = st.slider("Horizon (years)", 5, 40, HORIZON_DEFAULT)
        dt = 1.0 / st.select_slider("Integration steps per year", [4, 12, 24, 48],
                                    value=12,
                                    help="Use smaller steps to inspect numerical "
                                         "step-size sensitivity. This does not test "
                                         "the model's empirical fit.")
        p = default_params_for(row)

        st.divider()
        st.subheader("Benchmark")
        p.people_per_fe = st.number_input("People per field epidemiologist",
                                          50_000.0, 1_000_000.0, 200_000.0,
                                          25_000.0, format="%.0f")
        p.frontline_counts_toward_target = st.checkbox(
            "Count frontline toward the target", value=False,
            help="Off by default. Williams et al. 2020 counts advanced and "
                 "intermediate only. Turning this on changes the benchmark "
                 "definition to include frontline graduates.")

        st.divider()
        st.subheader("Pipeline: annual intake")
        implied = implied_historical_intake(row, p)
        use_implied = st.checkbox(
            "Use the historical average implied by the data", value=True,
            key=k("useimp"),
            help="Approximate annual intake from cumulative graduates, programme "
                 "duration, and completion rate. It is an initialization based "
                 "on supplied data, not an observed annual intake series.")
        for t in TIERS:
            default = implied[t] if use_implied else 0.0
            p.annual_intake[t] = st.number_input(
                f"{TIER_LABEL[t]} intake/year", 0.0, 5000.0, float(round(default, 1)),
                1.0, key=k(f"intake_{t}_{use_implied}"))
        st.caption(f"Implied historical intake: " +
                   ", ".join(f"{TIER_LABEL[t]} {implied[t]:.0f}" for t in TIERS))

        st.divider()
        p = feedback_controls(p, k)

        st.divider()
        st.subheader("Supply: how much of the cumulative total is still active")
        for t in TIERS:
            p.survival_to_2026[t] = st.slider(
                f"Survival to {BASE_YEAR} - {TIER_LABEL[t]}", 0.0, 1.0,
                p.survival_to_2026[t], 0.05, key=k(f"surv_{t}"))
        for t in TIERS:
            p.public_sector_share[t] = st.slider(
                f"Still in public sector - {TIER_LABEL[t]}", 0.0, 1.0,
                float(p.public_sector_share[t]), 0.05, key=k(f"pss_{t}"))
        for t in TIERS:
            p.public_entry_share[t] = st.slider(
                f"Future graduates entering public service - {TIER_LABEL[t]}",
                0.0, 1.0, float(p.public_entry_share[t]), 0.05,
                key=k(f"entry_{t}"))

        with st.expander("Tier overlap in the published totals"):
            st.caption("Programmes count a person once per tier completed, so "
                       "summing advanced + intermediate counts progressors "
                       "more than once. The correction below is an assumption; "
                       "individual-level progression records would be needed "
                       "to confirm the distinct-person count.")
            p.correct_initial_overlap = st.checkbox("Correct the overlap", True,
                                                    key=k("ovl"))
            p.overlap_adv_from_int = st.slider(
                "Share of advanced graduates who also did intermediate", 0.0, 1.0,
                0.30, 0.05, key=k("ovl_ai"))
            p.overlap_int_from_front = st.slider(
                "Share of intermediate graduates who also did frontline", 0.0, 1.0,
                0.40, 0.05, key=k("ovl_if"))
            p.progression_from_below["advanced"] = st.slider(
                "Future advanced intake drawn from intermediate", 0.0, 1.0, 0.30,
                0.05, key=k("prog_a"))
            p.progression_from_below["intermediate"] = st.slider(
                "Future intermediate intake drawn from frontline", 0.0, 1.0, 0.40,
                0.05, key=k("prog_i"))

        st.divider()
        st.subheader("Establishment and absorption")
        p.model_posts = st.checkbox(
            "Model funded posts as a constraint", value=True,
            help="Jordan had 63 FETP graduates against a target of 63 but only "
                 "39 working as government epidemiologists.")
        p.initial_posts_ratio = st.slider("Initial posts / current active stock",
                                          0.5, 3.0, 1.0, 0.05, key=k("p0"))
        if not p.post_feedback:
            p.post_creation_per_year = st.number_input(
                "Net new funded posts per year (fixed)", -50.0, 200.0, 0.0, 1.0,
                key=k("pcpy"))

        with st.expander("Attrition and retirement"):
            for t in TIERS:
                p.attrition[t] = st.slider(f"Attrition - {TIER_LABEL[t]}", 0.0,
                                           0.15, p.attrition[t], 0.005,
                                           key=k(f"att_{t}"))
                p.emigration[t] = st.slider(f"Emigration - {TIER_LABEL[t]}", 0.0,
                                            0.15, p.emigration[t], 0.005,
                                            key=k(f"emi_{t}"))
            p.retirement_age = st.slider("Mandatory retirement age", 50.0, 70.0,
                                         60.0, 1.0, key=k("ret"))

        with st.expander("Capability conversion"):
            for t in TIERS:
                p.skill_weight[t] = st.slider(f"Skill weight - {TIER_LABEL[t]}",
                                              0.0, 1.0, p.skill_weight[t], 0.05,
                                              key=k(f"sw_{t}"))
                p.protected_time[t] = st.slider(
                    f"Protected epidemiology time - {TIER_LABEL[t]}", 0.0, 1.0,
                    p.protected_time[t], 0.05, key=k(f"pt_{t}"))

        with st.expander("Demand: workload and coverage"):
            if has_admin2:
                p.fe_per_admin2 = st.slider("Field epidemiologists per admin-2 unit",
                                            0.0, 5.0, 1.0, 0.25, key=k("fea2"))
                p.reporting_units_per_admin2 = st.number_input(
                    "Reporting units per admin-2 unit", 1.0, 500.0, 40.0, 5.0,
                    key=k("rua2"))
            else:
                st.caption("The spreadsheet has no admin-2 counts, so the "
                           "subnational coverage floor cannot be computed. "
                           "That measure is the harder test and is currently "
                           "unavailable for every country.")
            p.annual_outbreaks_per_million = st.slider(
                "Annual outbreak events per million", 0.0, 5.0, 0.5, 0.1,
                key=k("obk"))
            p.national_readiness_fte = st.number_input("National readiness FTE",
                                                       0.0, 200.0, 10.0, 1.0,
                                                       key=k("rdy"))

        if iso in POOLED_ADVANCED:
            st.divider()
            st.subheader("Regional hub attribution")
            p.attribute_pooled_to_host = st.checkbox(
                "Attribute pooled advanced graduates to the host country",
                value=p.attribute_pooled_to_host, key=k("pool"))
            st.caption(POOLED_ADVANCED[iso][0])

        st.divider()
        uploaded = st.file_uploader("Cohort history CSV (year, tier, intake)",
                                    type="csv")

    # ---------------------------------------------------- cohort history
    cohort_history = None
    if uploaded is not None:
        ch = pd.read_csv(uploaded)
        if not {"year", "tier", "intake"}.issubset(ch.columns):
            st.warning("Cohort CSV needs columns ['intake', 'tier', 'year']; ignoring it.")
        else:
            cohort_history = ch
            st.success(f"Using cohort history: {len(ch)} rows. Survival haircut bypassed.")

    # ---------------------------------------------------- data checks
    comp = data_completeness(row)
    if not comp["population"]:
        st.error(f"No population data for {row['country']}.")
        st.stop()

    dq = supply_data_quality(row, p.frontline_counts_toward_target)
    if not dq["interpretable"]:
        st.error("**The headline number for this country is not interpretable as "
                 "a workforce count.** " +
                 "; ".join(f"{TIER_LABEL[t]}: {dq['status'][t]}" for t in dq["unknown_tiers"]))
    if dq.get("zero_unconfirmed_tiers"):
        st.warning("Zero graduate counts are unconfirmed for " + ", ".join(
            TIER_LABEL[t] for t in dq["zero_unconfirmed_tiers"]
        ) + ". Results using these values may treat unpublished counts as zero.")

    # ---------------------------------------------------- simulate
    row_dict = {kk: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v)
                for kk, v in row.items()}
    hist_json = cohort_history.to_json() if cohort_history is not None else ""
    df = _run(row_dict, _pj(p), horizon, hist_json, dt)
    first, final = df.iloc[0], df.iloc[-1]

    c = st.columns(5)
    c[0].metric("Countable workforce now", f"{first['compliance_heads']:.0f}",
                help="Distinct advanced + intermediate graduates in serving "
                     "posts or the unabsorbed pool, after initial overlap correction.")
    c[1].metric(f"JEE target {BASE_YEAR}", f"{first['jee_target_heads']:.0f}",
                f"{first['jee_gap_heads']:+.0f} gap", delta_color="inverse")
    c[2].metric("Density per 200k now", f"{first['density_per_200k']:.2f}",
                "target is 1.00", delta_color="off")
    c[3].metric(f"JEE target {BASE_YEAR + horizon}", f"{final['jee_target_heads']:.0f}",
                f"{final['jee_gap_heads']:+.0f} gap", delta_color="inverse")
    c[4].metric("Binding constraint", final["binding_constraint"])

    tabs = st.tabs(["Compliance", "Capability", "Structure and exits",
                    "Scenarios", "Uncertainty", "Cross-country", "Cost",
                    "Validation", "Data"])

    # ---- compliance
    with tabs[0]:
        st.subheader("Heads against the 1:200,000 target")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.calendar_year, y=df.jee_target_heads,
                                 name="JEE target", line=dict(width=4, color=PAL["target"])))
        fig.add_trace(go.Scatter(x=df.calendar_year, y=df.compliance_heads,
                                 name="Trained (serving + unabsorbed)",
                                 line=dict(width=4, dash="dash", color=PAL["trained"])))
        if p.model_posts:
            fig.add_trace(go.Scatter(x=df.calendar_year, y=df.deployed_heads,
                                     name="Deployed (post-constrained)",
                                     line=dict(width=3, color=PAL["deployed"])))
            fig.add_trace(go.Scatter(x=df.calendar_year, y=df.established_posts,
                                     name="Funded posts",
                                     line=dict(width=2, dash="dot", color=PAL["posts"])))
        if np.isfinite(first["coverage_floor_heads"]):
            fig.add_trace(go.Scatter(x=df.calendar_year, y=df.coverage_floor_heads,
                                     name="Coverage floor",
                                     line=dict(width=2, dash="longdash", color=PAL["floor"])))
        fig.update_layout(template="plotly_white", height=470, hovermode="x unified",
                          yaxis_title="Headcount", xaxis_title="Year",
                          title=f"{row['country']}: compliance account")
        st.plotly_chart(fig, use_container_width=True)

        bc = df.binding_constraint.value_counts(normalize=True)
        st.caption("Binding constraint over the horizon: " +
                   ", ".join(f"{i} {v:.0%}" for i, v in bc.items()))
        if p.model_posts and final["unabsorbed_heads"] > 1:
            st.warning(
                f"{final['unabsorbed_heads']:.0f} trained epidemiologists are "
                f"unabsorbed by {BASE_YEAR + horizon}: trained, with no funded "
                "post. The emigration response shown by R3 depends on the "
                "selected sensitivity assumption.")

    # ---- capability
    with tabs[1]:
        st.subheader("Advanced-equivalent effective FTE against workload")
        fig = go.Figure()
        for col, lab in [("routine_fte", "Routine surveillance"),
                         ("outbreak_fte", "Outbreak response"),
                         ("readiness_fte", "Readiness"), ("analytics_fte", "Analytics")]:
            fig.add_trace(go.Scatter(x=df.calendar_year, y=df[col], name=lab,
                                     stackgroup="d", mode="lines"))
        fig.add_trace(go.Scatter(x=df.calendar_year, y=df.effective_fte,
                                 name="Effective capacity",
                                 line=dict(width=4, color="black")))
        fig.update_layout(template="plotly_white", height=440, hovermode="x unified",
                          yaxis_title="Advanced-equivalent FTE", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.calendar_year, y=df.load_ratio,
                                  name="Load ratio", line=dict(width=3, color=PAL["trained"])))
        fig2.add_hline(y=1.0, line_dash="dash",
                       annotation_text="workload equals capability")
        fig2.update_layout(template="plotly_white", height=280,
                           yaxis_title="Workload / capability", xaxis_title="Year",
                           title="Load ratio - the input to the burnout loop (R2)")
        st.plotly_chart(fig2, use_container_width=True)

    # ---- structure and exits
    with tabs[2]:
        st.subheader("Stock by tier")
        fig = go.Figure()
        for t in TIERS:
            fig.add_trace(go.Scatter(x=df.calendar_year, y=df[f"heads_{t}"],
                                     name=TIER_LABEL[t], stackgroup="h", mode="lines"))
        fig.update_layout(template="plotly_white", height=380, hovermode="x unified",
                          yaxis_title="Active headcount", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Where the workforce goes")
        st.caption("Retirement and mortality are demographic. Attrition and "
                   "emigration are treated as potentially policy-addressable "
                   "channels in these scenarios. The model does not estimate "
                   "the effects of policies on those rates.")
        ch = ["retirement", "emigration", "attrition", "mortality"]
        fig = go.Figure()
        for cname in ch:
            fig.add_trace(go.Scatter(x=df.calendar_year, y=df[f"cum_exit_{cname}"],
                                     name=cname.title(), stackgroup="x", mode="lines"))
        fig.update_layout(template="plotly_white", height=360, hovermode="x unified",
                          yaxis_title="Cumulative people lost", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)
        tot = sum(final[f"cum_exit_{cname}"] for cname in ch)
        if tot > 0:
            addressable = (final["cum_exit_attrition"] + final["cum_exit_emigration"]) / tot
            st.metric("Share of all losses that is policy-addressable",
                      f"{addressable:.0%}")

        st.dataframe(df[["calendar_year", "heads_advanced", "heads_intermediate",
                         "heads_frontline", "compliance_heads", "established_posts",
                         "unabsorbed_heads", "jee_target_heads", "density_per_200k",
                         "load_ratio", "binding_constraint"]]
                     .iloc[::int(round(1 / dt))].round(1),
                     use_container_width=True)

    # ---- scenarios
    with tabs[3]:
        st.subheader("Policy scenarios")
        for nm, doc in SC.SCENARIO_DOC.items():
            st.markdown(f"**{nm}** — {doc}")
        long, wide, curves = _scenarios(row_dict, _pj(p), horizon, hist_json, dt)

        fig = go.Figure()
        for nm, d in curves[True].items():
            fig.add_trace(go.Scatter(x=d.calendar_year, y=d.deployed_heads, name=nm,
                                     line=dict(width=3)))
        fig.add_trace(go.Scatter(x=df.calendar_year, y=df.jee_target_heads,
                                 name="JEE target", line=dict(width=4, dash="dash",
                                                              color="black")))
        fig.update_layout(template="plotly_white", height=440, hovermode="x unified",
                          yaxis_title="Deployed headcount", xaxis_title="Year",
                          title="Deployed workforce by scenario (feedback active)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**How do these conditional scenarios change when the "
                    "feedback switches are active?**")
        st.dataframe(wide.round(1), use_container_width=True, hide_index=True)

        st.markdown("**Intake sensitivity under the selected assumptions**")
        pr_fb = SC.policy_resistance_curve(row, horizon, feedback=True,
                                           base_params=p, cohort_history=cohort_history,
                                           dt=dt)
        pr_ol = SC.policy_resistance_curve(row, horizon, feedback=False,
                                           base_params=p, cohort_history=cohort_history,
                                           dt=dt)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr_ol.intake_multiplier, y=pr_ol.deployed_heads,
                                 name="Deployed, open loop", line=dict(width=3, dash="dot")))
        fig.add_trace(go.Scatter(x=pr_fb.intake_multiplier, y=pr_fb.deployed_heads,
                                 name="Deployed, with feedback", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=pr_fb.intake_multiplier, y=pr_fb.cum_emigration,
                                 name="Cumulative emigration", line=dict(width=3, dash="dash"),
                                 yaxis="y2"))
        fig.update_layout(template="plotly_white", height=400,
                          xaxis_title="Intake multiplier", yaxis_title="Deployed in final year",
                          yaxis2=dict(title="Cumulative emigration", overlaying="y",
                                      side="right"),
                          title="Deployed workforce and emigration across intake scenarios")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pr_fb.round(1), use_container_width=True, hide_index=True)

    # ---- uncertainty
    with tabs[4]:
        st.subheader("Conditional sensitivity analysis")
        st.caption("Latin hypercube sampling over analyst-chosen ranges. Percentiles "
                   "describe the sampled runs; they are not confidence intervals or "
                   "calibrated probabilities. Partial rank correlations summarize "
                   "sensitivity within these ranges.")
        n = st.select_slider("Samples", [100, 200, 400, 800, 1500], value=400)
        if st.button("Run uncertainty analysis", key=k("psa_go")):
            S, Q = _psa(row_dict, _pj(p), n, horizon, True, hist_json, dt)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=Q.calendar_year, y=Q.p97_5, line=dict(width=0),
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=Q.calendar_year, y=Q.p2_5, fill="tonexty",
                                     fillcolor="rgba(178,58,46,0.15)",
                                     line=dict(width=0), name="2.5–97.5th sampled percentile"))
            fig.add_trace(go.Scatter(x=Q.calendar_year, y=Q.p75, line=dict(width=0),
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=Q.calendar_year, y=Q.p25, fill="tonexty",
                                     fillcolor="rgba(178,58,46,0.30)",
                                     line=dict(width=0), name="25–75th sampled percentile"))
            fig.add_trace(go.Scatter(x=Q.calendar_year, y=Q["median"],
                                     line=dict(width=3, color=PAL["trained"]),
                                     name="Median"))
            fig.add_trace(go.Scatter(x=df.calendar_year, y=df.jee_target_heads,
                                     line=dict(width=3, dash="dash", color="black"),
                                     name="JEE target"))
            fig.update_layout(template="plotly_white", height=440,
                              yaxis_title="Trained (serving + unabsorbed) headcount",
                              xaxis_title="Year", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            good = S.dropna(subset=["ratio_final"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Median compliance ratio in final year",
                      f"{good.ratio_final.median():.2f}")
            m2.metric("2.5–97.5th sampled percentile",
                      f"{good.ratio_final.quantile(.025):.2f} – "
                      f"{good.ratio_final.quantile(.975):.2f}")
            m3.metric("Share of sampled runs meeting benchmark",
                      f"{(good.ratio_final >= 1).mean():.0%}")

            pr = V.prcc(S, "ratio_final").head(14)
            fig = go.Figure(go.Bar(x=pr.prcc, y=pr.parameter, orientation="h",
                                   marker_color=np.where(pr.prcc > 0, "#2a9d8f", "#e76f51")))
            fig.update_layout(template="plotly_white", height=460,
                              xaxis_title="Partial rank correlation coefficient",
                              yaxis=dict(autorange="reversed"),
                              title="What actually drives the result")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pr.round(3), use_container_width=True, hide_index=True)
            st.download_button("Download PSA samples", S.to_csv(index=False),
                               file_name=f"psa_{iso}.csv", mime="text/csv")
        else:
            st.info("Press the button; 400 samples takes a few seconds.")

        st.markdown("#### Annual trainee intake needed by target year")
        st.caption(
            "Each row estimates the annual trainee starts needed, using the "
            "current tier mix, if that intake rate is maintained from the current "
            "year through the listed target year. The existing trainee pipeline "
            "starts at the active intake level. This is a run-rate by deadline, "
            "not an optimized year-by-year intake schedule."
        )
        target_choice = st.selectbox(
            "Workforce measure to reach",
            ["Countable trained staff, including unabsorbed pool",
             "Deployed staff in funded posts"],
            key=k("required_intake_outcome"),
        )
        target_outcome = ("jee_ratio" if target_choice.startswith("Countable")
                          else "deployed_ratio")
        target_ratio = st.slider("Target fraction of the JEE benchmark",
                                 0.50, 1.00, 1.00, 0.05,
                                 key=k("required_intake_target"))
        required_n = st.select_slider("Parameter draws for intake table",
                                      [50, 100, 200], value=100,
                                      key=k("required_intake_n"))
        st.caption(
            "The table uses independent uniform samples over the model's "
            "listed analyst-selected ranges; it does not sample data-source or "
            "missing-count uncertainty. Percentile bands describe those sampled "
            "assumptions; they are not calibrated probabilities or confidence "
            "intervals. Intake is searched from 0 to 32× the active tier mix. "
            "A deployed-staff target may be unreachable in draws where funded "
            "posts bind."
        )
        if st.button("Estimate annual intake by target year",
                     key=k("required_intake_go")):
            try:
                required = _required_intake(
                    row_dict, _pj(p), required_n, horizon, target_ratio,
                    target_outcome, hist_json, dt,
                )
                view = required[[
                    "target_year", "total_intake_p50", "total_intake_p05",
                    "total_intake_p95", "advanced_intake_p50",
                    "intermediate_intake_p50", "frontline_intake_p50",
                    "share_sampled_sets_feasible", "share_sampled_sets_nonmonotone",
                    "feasible_draws", "sampled_draws",
                ]].rename(columns={
                    "target_year": "Target year",
                    "total_intake_p50": "Total starts/year (P50)",
                    "total_intake_p05": "Total starts/year (P05)",
                    "total_intake_p95": "Total starts/year (P95)",
                    "advanced_intake_p50": "Advanced starts/year (P50)",
                    "intermediate_intake_p50": "Intermediate starts/year (P50)",
                    "frontline_intake_p50": "Frontline starts/year (P50)",
                    "share_sampled_sets_feasible": "Sampled sets reaching target (%)",
                    "share_sampled_sets_nonmonotone": "Non-monotone response sets (%)",
                    "feasible_draws": "Feasible draws",
                    "sampled_draws": "Total draws",
                })
                for col in view.columns:
                    if "(%)" in col:
                        view[col] = (view[col] * 100).round(1)
                    elif col != "Target year":
                        view[col] = view[col].round(0)
                st.dataframe(view, use_container_width=True, hide_index=True)
                st.caption(
                    "P05–P95 and P50 are calculated among sampled parameter sets "
                    "that reached the target within the search range. The target "
                    "share and non-monotone share are descriptive fractions of "
                    "the sampled sets, not policy probabilities."
                )
                st.download_button(
                    "Download annual intake table",
                    required.to_csv(index=False),
                    file_name=f"required_intake_{iso}.csv",
                    mime="text/csv",
                    key=k("required_intake_download"),
                )
            except ValueError as exc:
                st.warning(str(exc))

    # ---- cross-country
    with tabs[5]:
        st.subheader("Cross-country snapshot")
        st.caption("Assumption-based estimate from cumulative training counts, "
                   "survival, public-sector shares, and tier-overlap adjustment. "
                   "It does not split the stock by funded-post occupancy. This "
                   "view applies the selected benchmark, pooled-host, and overlap "
                   "settings; other sidebar assumptions use defaults.")
        snap = _snapshot(countries, p.frontline_counts_toward_target,
                         p.attribute_pooled_to_host, p.people_per_fe,
                         p.correct_initial_overlap)
        if snap.empty:
            st.info("No countries have both population and graduate data.")
        else:
            if not snap.interpretable.all():
                st.warning("Some country rows have unconfirmed zero or unknown "
                           "graduate counts. Review the data-quality columns before "
                           "interpreting the comparison.")
            colors = np.where(snap.density_per_200k >= 1, "#2a9d8f", "#e76f51")
            fig = go.Figure(go.Bar(x=snap.country, y=snap.density_per_200k,
                                   marker_color=colors))
            fig.add_hline(y=1.0, line_dash="dash", annotation_text="JEE target")
            fig.update_layout(template="plotly_white", height=460,
                              yaxis_title="Density per 200,000", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            n_ok = int((snap.density_per_200k >= 1).sum())
            st.metric("Countries at or above the benchmark",
                      f"{n_ok} of {len(snap)}")
            st.dataframe(snap.round(2), use_container_width=True, hide_index=True)
            st.caption("This is an assumption-based estimate from cumulative "
                       "training counts, not an observed count of serving field "
                       "epidemiologists.")

    # ---- cost
    with tabs[6]:
        cb = cost_block(row, df, p)
        if cb is None:
            st.info(f"No country-specific unit cost is included for "
                    f"{row['country']}, so this bundle cannot calculate a wage "
                    "bill for this row.")
            st.markdown("**Training-cost notes included in this bundle:**")
            for t, (lo, hi, note) in TRAINING_COST_USD.items():
                rng = ("no published figure" if lo is None else
                       (f"USD {lo:,.0f}" if lo == hi else f"USD {lo:,.0f}-{hi:,.0f}"))
                st.markdown(f"- **{TIER_LABEL[t]}**: {rng}. {note}")
        else:
            st.metric("Unit cost per deployed epidemiologist (USD/yr)",
                      f"{cb['unit_cost_usd']:,.0f}")
            cc = st.columns(3)
            cc[0].metric("Annual wage bill now", f"${cb['annual_wage_bill_now']:,.0f}")
            cc[1].metric(f"Annual wage bill {BASE_YEAR + horizon}",
                         f"${cb['annual_wage_bill_final']:,.0f}")
            cc[2].metric("If fully at target", f"${cb['annual_wage_bill_at_target']:,.0f}")
            st.caption(cb["basis"])

    # ---- validation
    with tabs[7]:
        st.subheader("Structural validation")
        st.caption("These checks cover selected structural properties and the "
                   "active scenario's accounting. Passing them does not establish "
                   "empirical fit or predictive validity.")
        if st.button("Run the test battery", key=k("val_go")):
            T = _validate(row_dict, _pj(p), horizon, hist_json, dt)
            for _, r in T.iterrows():
                (st.success if r.passed else st.error)(
                    ("PASS  " if r.passed else "FAIL  ") + r.test)
                st.caption(str(r.detail))
            st.metric("Tests passed", f"{int(T.passed.sum())} of {len(T)}")
        st.warning("**Not tested: reproduction of historical workforce "
                   "trajectories.** The supplied data do not include annual "
                   "serving-workforce counts, so this bundle does not support "
                   "an empirical back-test.")

    # ---- data
    with tabs[8]:
        st.subheader(f"{row['country']}: data and caveats")
        if dq.get("zero_unconfirmed_tiers"):
            st.warning("Unconfirmed zero graduate counts: " + ", ".join(
                TIER_LABEL[t] for t in dq["zero_unconfirmed_tiers"]
            ))
        if not comp.get("provenance", False):
            st.warning("Source and source date are not recorded for this row.")
        for fld, fn in (("notes", st.warning), ("nso_divergence", st.info),
                        ("pop_source", st.info), ("rate_detect_reason", st.info)):
            v = str(row.get(fld, "")).strip()
            if v not in ("", "nan", "None"):
                fn(f"**{fld}**: {v}")
        meta = pd.DataFrame({"Field": list(row_dict.keys()),
                             "Value": ["-" if v is None else str(v)
                                       for v in row_dict.values()]})
        st.dataframe(meta, hide_index=True, use_container_width=False)
        d1, d2 = st.columns(2)
        d1.download_button("Download projection CSV", df.to_csv(index=False),
                           file_name=f"fetp_{iso}_{datetime.now():%Y%m%d}.csv",
                           mime="text/csv")
        d2.download_button("Download parameters JSON", _pj(p),
                           file_name=f"fetp_params_{iso}.json",
                           mime="application/json")

    st.markdown("---")
    st.caption(
        "The 1:200,000 target is US-derived (back-calculated from CSTE state "
        "surveys), adopted by GHSA in 2014 and the JEE in 2015, and its own "
        "authors call the evidence base thin. It counts advanced and "
        "intermediate graduates only. National compliance routinely masks "
        "subnational failure: Papua New Guinea met it nationally while over "
        "half its districts had no graduate. Treat the coverage floor, not the "
        "national ratio, as the harder test.")


main()
