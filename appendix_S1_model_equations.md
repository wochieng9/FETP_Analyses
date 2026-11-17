# Appendix S1. Model equations and implementation

This appendix describes the v3 implementation in `fetp_model.py`. The model
is a conditional scenario model; its structural checks do not establish
empirical fit or predictive validity. All headcount estimates depend on the
input data and analyst-selected parameters.

Time is measured in years. The default integration interval is one month
(`dt = 1/12`). The integration step must be positive and no greater than one
year. Age bins advance when the trajectory crosses an annual boundary, while
hazards are integrated over each time step. The app offers 4, 12, 24, or 48
steps per year.

## 1. Stocks, flows, and units

| Symbol | Meaning | Unit |
|---|---|---|
| `S[i,a]` | serving workforce by tier and age | person |
| `U[i]` | unabsorbed public-sector seekers by tier | person |
| `T[i]` | trainees in the completion pipeline | person |
| `P` | funded posts | post |
| `N` | national population | person |
| `D_jee` | headcount benchmark | head |
| `D_work`, `C` | workload demand and effective capability | FTE |
| `λ[i]`, `μ[i]`, `m(a)` | attrition, emigration, and mortality hazards | 1/year |

The benchmark account counts advanced and intermediate graduates by default.
Frontline graduates enter it only when `frontline_counts_toward_target=True`.
The capability account uses serving staff only.

## 2. Training and distinct people

The FIFO training queues are initialized at a steady-state level of
`intake_eff[i] × duration[i]`. Each queued cohort completes after the
specified tier duration, with completion probability `c[i]` applied at exit:

```text
graduates[i] = delayed_intake[i] × c[i]
duration[i]   = tier_months[i] / 12
intake_eff[i] = min(intake_desired[i], mentor_capacity[i])
```

The public-sector entry share applies to external completers. Completions are
accounted for as:

```text
gross completions = internal promotions
                  + new public-sector entrants
                  + non-public completers
```

An internal promotion transfers one person from the lower tier to the higher
tier at graduation. The transfer preserves whether the person was serving or
in the unabsorbed pool and preserves the serving person's age distribution.
If the requested progression share exceeds the available lower-tier stock,
the shortfall is treated as external recruitment. This is an aggregate
approximation; the model does not track person-level identities.

External public-sector seekers enter the unabsorbed pool if posts constrain
deployment. Available posts absorb pool members pro rata. A tier excluded from
the benchmark is not post-constrained. `graduates_total` in the simulation
output is the net public-sector workforce inflow used in mass balance;
`graduate_completions_total` records gross completions.

## 3. Serving and unabsorbed stocks

Serving headcount by tier and age evolves under competing hazards. For a
constant hazard within a step:

```text
survival fraction = exp(-(attrition + emigration + mortality) × dt)
```

Workers age one annual bin on the anniversary step. Workers reaching or
exceeding the retirement age leave the serving stock. The oldest age bin is
absorbing until the retirement rule removes its members.

The unabsorbed pool is not age-structured. Its mortality hazard is the
entry-age-weighted mean for that tier. Pool survival uses the same exact
constant-hazard expression:

```text
U_next[i] = U[i] × exp(-(attrition[i] + emigration_eff[i] + mean_mortality[i]) × dt)
```

The pool uses the base attrition hazard. The R3 feedback modifies emigration
for pool members only; post holders retain the base emigration hazard.

## 4. Initial workforce and overlap

The initial stock is estimated from cumulative graduates after regional-hub
attribution, tier-overlap correction, survival to the base year, and the
public-sector share:

```text
advanced*     = advanced
intermediate* = max(0, intermediate − overlap_AI × advanced)
frontline*    = max(0, frontline − overlap_IF × intermediate − overlap_AF × advanced)

initial serving stock = adjusted cumulative graduates
                      × survival-to-base-year share
                      × public-sector share
                      × shifted entry-age profile
```

When cohort-history data are supplied, the model reconstructs the initial
stock from those cohorts and applies the historical completion and
public-sector shares. In either initialization path, the age profile and
survival assumptions remain uncertain without observed active workforce
counts.

## 5. Demand and capability

Population is interpolated log-linearly across supplied anchor years. When
UN WPP data are absent, the simple country table uses an interim rate-based
projection; this is not a replacement for a sourced population series.

```text
D_jee = population / people_per_fe
routine_hours = admin2_count × reporting_units_per_admin2
              × 52 × hours_per_reporting_unit_week
outbreak_hours = population / 1,000,000 × annual_outbreaks_per_million
               × response_team_epi × deployment_days × field_hours_per_day
D_work = (routine_hours + outbreak_hours) / productive_hours_per_fte
       + national_readiness_fte + analytics_fte
```

If admin-2 counts are unavailable, the implementation uses population-based
reporting units and cannot calculate the subnational coverage floor.

```text
C = Σ_i serving_heads[i] × active_share[i]
                  × protected_time[i] × skill_weight[i]
load_ratio = D_work / max(C, epsilon)
```

The workload and capability parameters are scenarios; the supplied country
table does not contain operational workload measurements.

## 6. Funded posts and feedback

Initial posts are `initial_posts_ratio × initial counted workforce headcount`.
The perceived post gap accounts for current deployment and funded vacancies:

```text
service_gap = max(0, D_jee − deployed)
vacancies   = max(0, P − deployed)
post_gap    = max(0, service_gap − vacancies)
```

The gap is smoothed over `post_perception_delay`. Post creation is bounded by
the fiscal ceiling and annual growth/loss limits:

```text
ceiling = max(0, D_jee × political_ambition)
rate    = clip(min(perceived_post_gap, ceiling − P) / post_adjust_years,
               −max_post_loss_per_year, max_post_growth_per_year)
P_next  = max(0, P + rate × dt)
```

Existing serving staff are not laid off if the post ceiling later falls below
the number of incumbents; new hires wait until posts are available.

### R1 — Mentor capacity

```text
mentors = serving_advanced × mentor_share
cap_advanced = mentors × residents_per_mentor / advanced_duration
cap_intermediate = eligible_mentors × residents_per_mentor_int / intermediate_duration
```

### R2 — Workload-related attrition

```text
multiplier = min(burnout_max, 1 + burnout_sensitivity × max(0, load_ratio − 1))
attrition_eff[i] = attrition_base[i] × multiplier
```

This modifier is applied to serving stocks. Its sensitivity is analyst chosen.

### R3 — Emigration from the unabsorbed pool

```text
multiplier = min(emigration_max, 1 + emigration_sensitivity × unabsorbed_share)
emigration_eff[i] = emigration_base[i] × multiplier
```

The increased hazard applies to the unabsorbed pool only. The sensitivity is
not estimated from observed graduate migration data.

### B1 — Post response

Post creation responds to the delayed, vacancy-adjusted staffing gap, subject
to a ceiling of `political_ambition × D_jee`. This differs from a policy that
automatically funds the full benchmark.

## 7. Reported outcomes

```text
compliance_heads = Σ_{i∈counted} (serving_heads[i] + unabsorbed_heads[i])
deployed_heads   = Σ_{i∈counted} serving_heads[i]
effective_fte    = capability from serving heads in all tiers
```

Frontline staff can contribute to modeled capability while remaining outside
the benchmark account when `frontline_counts_toward_target=False`.

The 1:200,000 figure is an external benchmark, not a country-specific estimate
of required workforce. Report it separately from modeled workload FTE.

## 8. Annual intake requirements under uncertainty

The app's annual-intake table answers an inverse scenario question: for each
target year, what constant annual trainee-start rate, using the currently
selected tier mix, reaches the selected share of the benchmark? It defaults to
countable trained staff, including the unabsorbed pool; users can select
deployed staff in funded posts instead. The output is an annual intake
run-rate through the listed target year, not a year-varying optimized training
schedule.

The current trainee queue is held at the active baseline intake while the
candidate intake applies prospectively. This avoids assuming that a newly
selected higher intake already has a full pipeline in place. Completion
probabilities, program durations, mentor limits, public-sector entry, workforce
exits, and the other sampled parameters then determine when additional
graduates reach the workforce.

For each parameter draw, the implementation simulates a grid of intake
multipliers from zero through 32 times the active tier mix. It finds the first
grid crossing of the target and linearly interpolates between adjacent
points. Rows are reported for each calendar-year target deadline. The P05,
median, and P95 annual-intake values are conditional on draws that reach the
target within the searched range; the share of sampled draws that reach the
target is shown separately. Responses that decline anywhere across the intake
grid are flagged as non-monotone.

This is a conditional PSA-style analysis using independent uniform draws over
analyst-selected ranges. The intervals are not empirically elicited, and the
sampled shares are not calibrated probabilities. Results also hold the active
tier mix fixed and only vary parameters listed in `PSA_RANGES`; data-source
uncertainty and missing graduate counts are not sampled. A defensible
year-varying enrollment plan would require an explicit objective, tier-
allocation rules, and better information about the current trainee pipeline
and funded-post plans.

## 9. Structural checks and uncertainty

`fetp_validate.run_structural_tests()` defines 16 structural checks plus a
unit-algebra check. The checks cover mass balance with progression, pool
promotion accounting, public-sector inflow decomposition, analytic decay for
serving and pool stocks, equilibrium, retirement timing, limiting cases,
step-size sensitivity, and demand scaling. These checks verify selected
properties of the implementation; they do not establish calibration or
predictive accuracy.

The validation tab runs this battery for the selected country and active
scenario. A passing result is an implementation check, not evidence that the
assumptions match observed workforce trajectories.

The sensitivity module samples analyst-chosen ranges with Latin hypercube
sampling. Its percentiles and share of sampled runs meeting the benchmark are
conditional sensitivity summaries, not confidence intervals or calibrated
probabilities.

## References

- Barlas Y. Formal aspects of model validity and validation in system dynamics. *System Dynamics Review*. 1996;12(3):183–210.
- Forrester JW, Senge PM. Tests for building confidence in system dynamics models. *TIMS Studies in the Management Sciences*. 1980;14:209–28.
- Sterman JD. *Business Dynamics: Systems Thinking and Modeling for a Complex World*. McGraw-Hill; 2000.
- Williams SG, et al. One Field Epidemiologist per 200,000 Population: Lessons Learned from Implementing the Global Health Security Agenda Target. *Health Security*. 2020;18(S1):S-33–S-40.
