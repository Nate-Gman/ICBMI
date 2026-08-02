# ICBMI — Ballistic Missile Defence Intercept Digital Twin

A standalone, single-file, interactive model of the **defensive** half of a
ballistic missile engagement: can a kinetic interceptor, launched from a known
site, be placed on a collision course with an incoming object before that
object reaches its aimpoint — and how often does that actually work?

`ICBMI.py` · ~6,500 lines · numpy + pygame · **92/92 verification checks**

---

> ## ⚠ There is no railgun model in this project, and the absence is the point
>
> This program assesses a railgun-launched interceptor (`--railgun`, `--mkv`,
> and the GUN view) and concludes that it does not work. It reaches that
> conclusion using **equations, margins and measured correlations only**.
>
> There is deliberately **no geometry, no assembly, no dimensioned drawing and
> no build description** — not for the launcher, not for the projectile. The
> GUN view renders margin bars and pass/fail marks and nothing that could be
> traced. This is enforced, not merely intended: the mode contains no polygon
> or circle drawing calls at all.
>
> **A critique and a blueprint are different objects with different reach.** A
> critique tells a reader whether a concept closes. A model tells them how to
> make one. Only the first is required to answer the question this project
> asks, and only the second keeps its value in hands the author never meant to
> help.
>
> Engineering detail travels once it is written down. It does not check who is
> reading it, and it does not stop working when it arrives somewhere
> unwelcome. Convergent designs appearing across supposedly independent
> competitors are the visible evidence of how readily it moves — a drawing
> that is good enough to build from is good enough for anyone to build from.
>
> A negative result carries no such risk, because *"this fails, and here is the
> number that kills it"* is not a manufacturing input. So the salvo
> mathematics is present in full and the hardware is absent.
>
> **Everything required to evaluate the concept is here. Nothing required to
> build one is — and nothing of that kind should be added later.**

---

## What is modelled

The defence. Sensor detection chain, track quality, predicted intercept point,
booster energy budget, kill-vehicle divert authority, proportional navigation
homing, miss distance, and single-shot kill probability.

The threat appears **only as a ballistic arc** — a mass on a Keplerian
trajectory, derived from ground range alone by the conic range equation. That
is all the defence knows about it and all this file models. Threat vehicle
design, propulsion, staging, warhead and reentry-vehicle physics, and
countermeasure/penetration-aid design are all absent, and none of it is needed:
an interceptor solves a kinematics problem against an object whose arc is set
by gravity.

## The honest headline

A single exoatmospheric interceptor against a single unaccompanied object, with
good track, is a solvable kinematics problem and the model closes it. Every
real-world complication then degrades it faster than interceptor performance
can recover — and the one that degrades it fastest, **object count**, is a
counting problem that better hardware does not touch.

Nothing here is validated against a real system. Every parameter is either an
open published figure or an estimate, and the file marks which.

## Running

```bash
python ICBMI.py                  # interactive viewer (11 modes)
python ICBMI.py --selftest       # 92 verification checks
python ICBMI.py --engage         # one engagement, full text report
```

| Report | Question it answers |
|---|---|
| `--feasibility` | What does the model actually close, and where does it stop? |
| `--railgun` | Does the gun-launched concept work? (No — four independent ways.) |
| `--mkv` | Does adding kill vehicles help? (Yes — until divert saturates.) |
| `--escalate` | On a miss, add one more round — does escalation close the gap? |
| `--levers` | Which knobs actually move Pk, with the others held ideal? |
| `--iftu` | What are in-flight target updates worth? |
| `--platforms` | Ground vs sea vs air vs space launch. |
| `--footprint` | How much area does one site defend? |
| `--layered` | Independence vs volume of fire. |
| `--discrimination` | How object count dilutes a fixed inventory. |
| `--battlespace` | Window and shot count vs threat range. |
| `--architectures` | Boost vs midcourse vs terminal. |

Options: `--threat-range KM`, `--loft F`, `--objects N`.

## Viewer

`TAB` or `1`–`9` cycle modes · `SPACE` play/pause · `D` demo tour ·
drag to orbit · wheel to zoom · `,`/`.` sim speed · `H` help · `ESC` quit

| Mode | Shows |
|---|---|
| ENGAGEMENT | Globe view of both arcs. **`Z`** drops into the terminal close-up — a real simulated collision with 3D vehicle models, explosion on hit, flyby on miss. `J`/`K` set salvo size. `A` auto-escalate. `M` live intercept math panel. |
| INTERCEPTOR | Booster and kill vehicle in true-scale 3D. `E` exploded, `X` section cut. |
| GEOMETRY | Closing velocity and the miss-distance distribution. |
| SENSORS | Radar range equation and horizon limits. |
| TIMELINE | Every delay in the engagement, to scale. |
| BATTLESPACE | Window and shot count vs threat range. |
| DISCRIMINATION | Object count against a fixed inventory. |
| GUN | Railgun assessment — **numbers only, no drawing**. |
| LAYERED | Independence vs volume of fire. |
| PHYSICS | The equations, with live values substituted. |
| ABOUT | Scope, provenance, and omissions. |

## Selected findings

**The interceptor is aimpoint-limited, not track-limited.** Sweeping each lever
with the others held ideal: aimpoint σ 0.1 → 0.4 m takes Pk from 100% to 57%,
while track error 500 → 20,000 m and divert 50 → 800 m/s both leave it at 100%.
Track error only bites *through* the divert budget — they are one joint
constraint, not two.

**The Monte Carlo reproduces an independent closed form.** Aimpoint bias is
Gaussian in two axes, so the miss is Rayleigh and
`Pk_max = 1 − exp(−R²/2σ²)`. At σ = 0.2 m the simulation gives 96% and the
formula gives 96%. Both are load-bearing and the suite fails if they diverge.

**A salvo multiplies the odds only across what the rounds do not share.**
Kill vehicles carry independent seekers, so at good track they fail
independently (measured correlation φ = −0.04) and extra vehicles genuinely
compound: 95.8% → 99.2%. Past divert saturation they land within ~2 m of each
other some 30 km off — φ = +0.96, and extra vehicles contribute *exactly
zero*. The unguided-rod salvo fails the same test for the opposite reason:
those rounds carry no seeker, so nothing about them is independent.

**In-flight target updates are insurance, not improvement.** Worth ~1 point at
nominal sensor quality; worth 76 points when the sensor degrades. They
decouple kill probability from radar quality within wide limits.

**Lateral reach is nearly free from the ground and expensive from orbit.**
Offsetting a ground site 2,000 km off-track costs +17 m/s — a booster starting
from rest simply aims. The same manoeuvre from a 7.35 km/s orbit costs
3,433 m/s at 27°. A silo aims; a satellite manoeuvres, and manoeuvring is what
costs.

## Verification

```bash
python ICBMI.py --selftest
```

92 checks across orbital mechanics, atmosphere, Lambert solving, sensors,
homing, salvo correlation, launch platforms, 3D geometry, terminal collision
visualization, demo tour, and every view mode.
They test **physics against independently known answers**, not agreement with
a hoped-for result — limit cases (antipodal range → circular orbital speed),
closed-form cross-checks, energy conservation, monotonicity, convergence, and
guards against fabricated certainty.

Two of those guards exist because the failure they catch already happened once:
the terminal collision view must be driven by the same simulated trajectory the
reports quote, and no probability may ever be printed as `100%` — an all-hit
Monte Carlo run establishes a bound set by the sample size, not certainty.

## Files

```
ICBMI.py            main program and verification suite
README.md           this file
OVERVIEW.md         technical architecture
INFORNMATIONAL.md   source transcript being critiqued (not documentation)
```

## Licence / status

Educational physics model of a defensive system. Hypothetical throughout,
validated against nothing, and not a design document for anything.
