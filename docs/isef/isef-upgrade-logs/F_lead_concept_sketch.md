# Lead designer's concept sketch: "Lab Detective"

One-line pitch: a local AI lab partner that, when a student's measurements deviate from theory, works out *why*
(which physical or procedural cause), shows the evidence, and says which single extra measurement would settle it.

Core research question: When a student's lab data deviate from theory, can a system running entirely on a school
laptop identify the physical cause, and choose the most informative next measurement?

Components
1. Open-set experiment recognition with the local VLM (Qwen3.8-27B 4-bit via LM Studio if it accepts images).
   JSON-schema constrained output: experiment (6 registered + any other, named), configuration cues (incline present,
   hanging mass over a pulley, two carts facing, dynamometer horizontal, ...), apparatus list. YOLO boxes optionally
   drawn as numbered marks (Set-of-Mark). Candidate set with a conformal coverage guarantee (P1). Manual override kept.
2. Physics digital twin per registered experiment: forward model of the measurement protocol with instrument noise,
   plus a library of 5-8 fault models per experiment drawn from the curriculum's standard error sources (friction not
   balanced / over-balanced, hanging mass not << cart mass, slant length used as height, wheel rotational energy,
   photogate flag width error, stopwatch reaction time, scale zero offset, unit slip x1000, pulling at an angle,
   non-uniform pulling, spring beyond its elastic limit, measuring total length instead of extension ...).
3. Bayesian diagnosis: posterior over {random error only, fault_k} from marginal likelihoods (grid or Laplace over the
   fault magnitude) with priors (optionally shifted by the VLM's configuration cues). Out-of-model check (posterior
   predictive) so the system says "cause not in my list" instead of forcing a label.
4. Next-measurement advisor: expected information gain over a menu of feasible extra measurements (another height or
   angle or mass, weigh the hook, re-zero the scale, time over a longer distance) -> recommend the one that best
   separates the leading hypotheses.
5. Identifiability atlas (a scientific result in its own right): for each experiment and measurement design, the
   pairwise confusability of fault hypotheses from simulation; shows which faults a standard student dataset cannot
   tell apart and which extra measurement resolves them.
6. Local explanation: the same local LLM writes the explanation from a fact sheet (posterior, numbers); a verifier
   checks every number, unit and directional claim against the fact sheet; regenerate or fall back to a template.
   Counterfactual plot: the student's points against the curves each top hypothesis predicts.
7. The model upgrade ("operate on the model"): physics-engine-supervised fine-tuning. The simulator generates unlimited
   verifiable diagnosis tasks; rejection-sampling fine-tuning (STaR/RFT) of a small local model (~4B, school-laptop
   size) on its own verified reasoning; test generalisation to unseen fault magnitudes, to a held-out experiment, and to
   real classroom data; compare with the 27B with and without the engine as a tool, and with frontier APIs under the
   same prompt, candidate list and output schema.

Evaluation
- Synthetic benchmark with ground truth (e.g. 6 experiments x ~8 faults x 50 cases), held-out fault magnitudes, and a
  positive control: an unmodelled fault type must be flagged "unknown".
- Real classroom validation 11/9-11/12: each experiment run once as intended and once or twice with a deliberately
  introduced fault set up by the teacher; data typed into the system; diagnosis accuracy with Wilson CIs.
- Recognition: existing photos/frames (when they reach the Mac) plus CC-licensed Commons photos of other experiments;
  levelled comparison against the rule level and API VLMs.
- Explanations: verifier-caught error rate; teacher blind rating if the IRB rules allow it.
- Pre-registered acceptance criteria committed before the runs.

Feasibility notes
- Engine, simulator, benchmark and verifier are numpy code, CPU-light, can be built before 11/6.
- Heavy LLM runs after 11/6 overnight within 24 GB (27B ~19-20 GB inside LM Studio, never alongside another model).
- Small-model RFT locally with MLX (~8-10 GB) or on the 5090 with data generated on the server (only code uploaded).
