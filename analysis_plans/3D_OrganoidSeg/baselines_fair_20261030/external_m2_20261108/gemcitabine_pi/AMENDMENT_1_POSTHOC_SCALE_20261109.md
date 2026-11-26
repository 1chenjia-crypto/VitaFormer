# Amendment 1 — post hoc scale sensitivity (2026-11-09, written AFTER the primary results were seen)

Status: **post hoc, descriptive only.** It was requested by the author after the primary results in RESULTS.md
(Dead class not validated at the pre-registered factor 0.25). It cannot change the primary gate or claim of the plan.
A factor that turns out "better" is not adopted.

Question: is the absence of Dead calls caused by the 0.25 downscaling?

Runs: factors 1.0, 0.75 and 0.5 (the other pre-listed candidates) × seeds 42 / 3407 / 2026.
Everything else is identical to the plan (same stretch, `infer_hard`, argmax, PI threshold T = 1347).
To save time, only frames t = 0, 52 and 72 h are run (t = 0 is needed for the area fold change).
Script `scripts/gem_pi_posthoc_scale_20261109.py`. Output goes to `posthoc_scale/` (new).

Reported per factor × seed at t52 and t72, with the same estimators and bootstrap as the plan:
- frames with Dead px, DF by dose;
- ρ(DF, PI) and partial ρ given M2 area fold change;
- ρ(DP, PI);
- pixel AUROC of P(Dead) vs PI+;
- Dead precision and recall vs PI+;
- ρ(area fold change, PI);
- foreground Dice vs OrganoID labels.

No new gate. Allowed wording if Dead fires at another scale: "Dead calls are scale-dependent; at factor X (post hoc)
the Dead fraction correlates with PI ρ = …". Validation must not be claimed.
