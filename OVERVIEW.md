# ICBMI — Technical Architecture Overview

## System summary

`ICBMI.py` is a single-file Python application (~6,500 lines) modelling the
defensive half of a ballistic missile engagement, paired with an in-file
verification suite of **92 checks** (`--selftest`).

It combines a Keplerian trajectory engine, a radar detection chain computed
from the range equation, a universal-variable Lambert fire-control solver, a
proportional-navigation terminal homing simulation with differential gravity,
Monte Carlo kill probability with confidence intervals, explicit salvo
correlation modelling, adaptive salvo escalation, launch-platform and
defended-footprint analysis, and an 11-mode interactive pygame visualisation
including a true-scale 3D vehicle, a physically simulated terminal collision
with 3D vehicle models at the intercept point, and a self-running demo tour.

**Design principle running through the codebase:** no headline number is stored
where it can be edited. Detection ranges are computed from the radar equation,
kill probability is simulated rather than asserted, the aimpoint ceiling is
cross-checked against a closed form, and the verification suite fails if a
fabricated figure reappears.

---

## Deliberate omission: no railgun model

The program assesses a railgun-launched interceptor and concludes it does not
work. That assessment lives entirely in **equations, margins and measured
correlations**. There is no geometry, no assembly, no dimensioned drawing and
no build description — for the launcher or the projectile.

This is a structural property of the code, not a stylistic one:

| Location | What it contains | What it does not |
|---|---|---|
| `RAILGUN_PROPOSAL` | Scalar inputs *under test* (mass, length, area, material) | Any assembly, tolerance or process |
| `gun_assessment()` | Four independent physics checks returning margins | Any geometry output |
| `report_railgun()` | Numeric critique + salvo mathematics | Any drawing |
| `_mode_gun` | Margin bars, pass/fail marks | **Zero polygon or circle draw calls** |

**Why.** A critique and a blueprint are different objects with different reach.
A critique tells a reader whether a concept closes; a model tells them how to
make one. Only the first answers the question this project asks, and only the
second keeps its value in hands the author never meant to help. Engineering
detail travels once written down — it does not check who is reading it, and
convergent designs across supposedly independent competitors are the visible
evidence of how readily it moves.

A negative result carries no such risk, because *"this fails, and here is the
number that kills it"* is not a manufacturing input. The salvo mathematics is
therefore present in full — and reused by the kill-vehicle analysis, so the
reasoning lives in one place — while the hardware is absent.

**Everything required to evaluate the concept is present. Nothing required to
build one is, and nothing of that kind should be added later.**

---

## Code structure

### Section 1 — Constants and specification
Earth model (μ, R, J2, ω), Vallado exponential atmosphere table, threat range
classes, `INTERCEPTOR` spec in millimetres and SI, `SENSORS`, `TRACK` error
model, `PLATFORMS`, `MATERIALS`. Values are marked `published` or `est.` at the
point of definition; the estimates drive Pk more than anything else and the
ABOUT view says so.

### Section 2 — Environment physics
`atmos_density`, `gravity` (with J2), `drag_accel` (rotating-atmosphere
relative velocity), `rk4_step`. Energy conserves to ~1e-11 over 3,000 s.

### Section 3 — Ballistic trajectory
`conic_range` (exact, handles burnout at altitude), `burnout_for_range`
(bisection on the minimum-energy solution), `build_threat`, and the
`Trajectory` class with **cubic Hermite interpolation** — position and velocity
are both stored, so Hermite is free and removes the ~0.9 m sagitta error linear
interpolation leaves on a gravitationally curved arc.

`lambert` / `lambert_best` — universal-variable Lambert with bisection on z,
trying both transfer directions. The interceptor site sits downrange for most
of an engagement, so the short way round is often retrograde; solving prograde
only silently returns a transfer most of the way around the Earth and makes
every intercept look infeasible.

### Section 4 — Sensors
`radar_max_range` (range equation with coherent integration gain),
`radar_horizon_m`, `first_detection`. Detection ranges are computed, never
asserted — they came out badly wrong when first guessed, which is the argument
for computing them.

### Section 5 — Interceptor and terminal homing
`platform_dv` (charges gravity and drag losses against the ideal burnout
velocity), `platform_state`, `solve_intercept` (Lambert to the earliest
feasible PIP, charging only the velocity actually *bought* — crediting Earth
co-rotation and orbital velocity).

`homing_run` — zero-effort-miss proportional navigation, the correct
exoatmospheric formulation. Adaptive step in `t_go`, terminating on the
analytic closest approach of the final coast. Carries a first-order guidance
lag and a per-vehicle aimpoint bias, optionally records its trajectory
(`record=True`), and optionally accepts an injected common track draw
(`shared_track=`) so a salvo can share one fire-control solution.
Differential gravity is computed from the full threat trajectory passed to
the solver, so terminal homing accounts for the gravitational divergence
between the kill vehicle and the threat over the closing interval.

`single_shot_pk`, `salvo_kill_probability`, `salvo_correlation`,
`salvo_outcome_split`, `adaptive_salvo_size`, `aimpoint_ceiling`,
`wilson_interval`, `pct_mc`.

### Section 6 — Engagement
The `Engagement` class runs the full chain: detection → fire control → track
degradation → homing → battlespace → inventory. Supports launch platform,
off-track site offset, in-flight target updates, salvo size, and object count.

### Section 7 — Analysis and reporting
Thirteen text reports (see README table), including `--escalate` (adaptive
salvo escalation on miss) and `--mkv` (multiple kill vehicle correlation).

### Section 8 — Visualisation
True-scale 3D geometry built parametrically from the spec dict, an orbit
camera, and 11 view modes. `build_kv_mesh` is factored out of
`build_interceptor_mesh` so the terminal view can place the kill vehicle alone
— the suite asserts the two are vertex-identical after re-basing, so they
cannot drift apart.

The ENGAGEMENT mode supports a **zoom-to-intercept** view (`Z` key) that
re-centres the camera on the predicted intercept point and renders both the
interceptor and threat as 3D meshes at true millimetre-to-metre scale. The
collision is driven by the same simulated terminal trajectory the reports
quote — a hit produces an explosion effect at the contact point, a miss
produces a flyby with measured clearance. A live intercept math panel (`M`
key) overlays the closing velocity, time-to-go, and miss distance. A
self-running **demo tour** (`D` key) cycles through every mode with
captions, and yields control when the user presses any key.

### Section 9 — Selftest and CLI

---

## Key modelling decisions

**The threat is an arc, not a vehicle.** Ground range alone fixes the
trajectory. This is what makes midcourse intercept a tractable kinematics
problem, and it is why no threat design is needed or present.

**Earliest feasible intercept.** `solve_intercept` takes the earliest feasible
PIP to preserve battlespace for a second shot. Side effect: reported energy
margin is always near zero, because the earliest feasible shot is by definition
the most demanding. Documented in `--footprint` rather than left to mislead.

**Track error grows since the last update, not since detection.** Modelling
growth across the whole flight assumes the sensor detects once then looks away,
which no real system does and which made the term ~30× too pessimistic.

**Salvo correlation is explicit.** Vehicles share a fire-control track
(common) and carry their own seekers (independent). Which dominates decides
whether extra vehicles multiply the odds or buy copies of the same miss.
Common random numbers pair the sampling across salvo size, so `P(kill)` is
monotone in vehicle count by construction — without pairing, a larger salvo
can report a *worse* result, which is impossible and is a sampling artefact.

**Certainty is never printed.** `_pct` cannot round to 100%, and `pct_mc`
reports an all-hit Monte Carlo run against its Wilson lower bound with the
trial count, because such a run establishes a bound set by the sample size, not
a demonstration of certainty.

---

## Verification philosophy

The suite tests physics against **independently known answers**, not agreement
with a hoped-for result:

- **Limit cases** — antipodal range must give circular orbital speed (7.90
  km/s); short range must give a 45° flight-path angle.
- **Closed-form cross-checks** — the Monte Carlo aimpoint ceiling must match
  `1 − exp(−R²/2σ²)`. Both are load-bearing; if they diverge, one is wrong.
- **Conservation and convergence** — energy drift, integrator step-size
  convergence, homing termination convergence, Lambert round-trip error.
- **Monotonicity** — Pk must fall with track error; salvo Pk must rise with
  vehicle count.
- **Structural agreement** — rendered 3D geometry must match the spec it claims
  to be drawn from; the standalone KV mesh must match the embedded one; the
  terminal collision view must render through approach, closest approach, and
  past-CPA frames for both hit and miss outcomes.
- **Anti-fabrication guards** — Pk may never be exactly 100%; the percentage
  formatter may never round to certainty; the terminal collision view must be
  driven by the same simulated trajectory the reports quote.
- **UI robustness** — every view mode must render without exception across
  window sizes down to 640×480; the demo tour must visit every mode and yield
  control on user input; the main loop must run in every mode without error.

The last of those exists because that failure already occurred: an earlier
collision view decided hit/miss with a fixed-seed coin flip while the render
sweep passed, so the check now verifies the rendered path *endpoint* equals the
recorded miss distance to sub-millimetre precision.

---

## Known limitations

- Every performance figure marked `est.` is an estimate, and they drive Pk more
  than anything else. Read Pk as a shape, not a figure.
- Single-threat engagements only; no raid modelling beyond object-count
  dilution.
- Boost-phase intercept is compared in prose (`--architectures`) but not
  simulated.
- Nothing here has been validated against a real system.
