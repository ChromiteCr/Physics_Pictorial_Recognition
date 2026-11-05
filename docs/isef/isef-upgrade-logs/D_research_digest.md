# Verified research digest (2026-10-31; sources in research_results.md)

## Local model stack
- Qwen/Qwen3.8-27B exists (Apache-2.0, weights public ~2026-08-14). Dense 27.78B, natively multimodal (image AND video),
  262K context, Qwen3.5 hybrid architecture (Gated DeltaNet + gated attention). Thinking ON by default; disable with
  enable_thinking=false or LM Studio native /api/v1/chat reasoning:"off" (open bug: thinking-off ignored via the
  OpenAI-compatible path for some Qwen3.5 builds, so verify the reply has no reasoning content).
- The student's 16.08 GB file = lmstudio-community/Qwen3.8-27B-MLX-4bit (vision tower bf16). M4 Pro measured (same
  architecture): prefill ~115 tok/s, decode ~14 tok/s. Total memory with 8k context + one 1024 px image: ~18.5-21 GB, so
  only ~3-5 GB remain under the 24 GB cap for everything else. A 1024 px image = ~1,024 visual tokens (~9 s prefill);
  a raw 12 MP photo = ~12k tokens (overflows 8k context). Downscale every image.
- LM Studio: OpenAI-compatible + native REST, image input, JSON-schema structured output, tool use, lms CLI
  (get/load -c/--ttl/--estimate-only/unload), JIT + auto-evict. LM Studio CANNOT load LoRA adapters (fuse instead, or
  serve with mlx_vlm.server --adapter-path, which also offers JSON schema + token logprobs). The Mac has the Bionic app
  (LM Studio's agent app, uses the LM Studio runtime) and the lms CLI; no model downloaded yet.
- No official small Qwen3.8. Official small siblings with vision: Qwen3.5-0.8B/2B/4B/9B (MLX 4-bit: 4B 3.06 GB,
  9B 5.98 GB) - school-laptop tier candidates.
- Fine-tuning: mlx-vlm 0.7.x trainer supports SFT/DPO/GRPO with LoRA, grad checkpointing (Qwen3.5 arch). 27B QLoRA
  under 24 GB on the Mac is unproven (estimate 19-22 GB). 4B/9B fit easily. On the RTX 5090: Unsloth says 27B QLoRA fits
  in 24 GB (but Unsloth's Qwen3.5 guide warns against QLoRA for this family); bf16 LoRA of Qwen3.5-9B needs ~22 GB.
  ms-swift supports Qwen3.8. Adapters r=16: 27B ~160-235 MB, 9B ~87 MB, 4B ~65 MB. PEFT->MLX adapter conversion needs a
  small custom converter.
- VLMs read gauges badly (MeasureBench: best model 30.3% on real images); keep manual readings.

## ISEF rules that constrain the plan (official Society for Science pages, 2026-27 rules)
- ISEF 2027: May 8-14, Los Angeles. Entry only via an affiliated fair (fairs end by Apr 12, 2027). No Society deadline
  on Nov 15, so the student's 11-15 deadline belongs to the affiliated fair (unknown: registration only, or research
  plan + SRC/IRB paperwork, or full project).
- Category: best fit ROBO (Machine Learning or Cognitive Systems). SOFT no longer exists (now SFTD). PHYS only if the
  headline is a physics finding.
- Judging (science and engineering same weights): research question 10, design & methodology 15, execution 20,
  creativity & impact 20, presentation 35 (poster 10, interview 25). "Demonstrations / explanation models" are deemed
  inappropriate: need hypotheses or engineering criteria, quantitative tests, statistics.
- Human participants: ANY use/testing of the app by someone other than the student researcher, any surveys/opinions
  (teacher ratings included), any analysis of classmates' lab data, or classroom observation = human-participant
  research. Needs a 3-member school IRB (educator not the sponsor, administrator, medical/mental-health professional),
  Form 4 signed BEFORE recruitment, written minor assent + parental permission, adult consent. The Companion Guide says
  app results should not be disclosed to participants and strongly advises against AI chatbots/unscripted AI
  responses to participants. Exempt: the student researcher alone testing the app with no human variable.
- Forms 1, 1A, research plan (student-written), 1B (student + parent), Form 3 risk assessment must be done BEFORE
  experimentation; new Form 2A (support + AI disclosure with prompt log). After the affiliated fair the project cannot
  change; the plan must already list every phase. Research window: max 12 continuous months, nothing before Jan 2026.
- Continuation: work built on the June-Oct 2026 version (entered in other competitions/journal) should be declared on
  Form 7 as a substantive expansion (a new line of investigation); board and abstract show only the new work. The
  earlier five classroom sessions should not be presented as ISEF results unless IRB-approved.
- Generative AI (Rule 8, new 2026-27): may be used as a resource with citation and prompt logs; must NOT write the
  research plan, abstract, poster or paper first drafts, conclusions or future work, and must NOT supply citations or a
  starter bibliography. AI-written code must be identified (ledger) with prompt logs. Inappropriate AI use = grounds to
  fail to qualify.
- Booth: must work without internet (local model fits); precompute API baselines; no QR codes/URLs; credit figures.

## Prior art (novelty boundaries)
- The current YOLO + LLM lab assistant is nearly isomorphic to PhysicsAssistant (Latif et al., 2024: YOLOv8 + GPT-3.5
  robot for 8th-grade labs). Not novel by itself.
- Simulator-as-reward RLVR for physics already exists (Sim2Reason 2026: MuJoCo-generated QA, 5% tolerance reward,
  Qwen 3B-32B, IPhO +5-10 pp). LLM + physics engine grounding: Mind's Eye (ICLR 2023), SimLM, LLMPhy.
- Model-based diagnosis with entropy-minimising next measurement: GDE (de Kleer & Williams 1987); model
  discrimination design (Box & Hill 1967); modern BOED; BoxingGym (2025) shows LLMs struggle at EIG-style experiment
  design. In physics education: Bayesian model comparison in intro labs (Blais 2020), uncertainty revealing a missing
  assumption (Kok & Priemer 2023, rotational KE), Holmes/Wieman critical-thinking-with-data (PNAS 2015). LLMs spotting
  errors in student experiment plans (Bewersdorff 2023: 0.90 simple errors, 0.60 controlled-experiment judgement); LLM
  lab TAs give harmful answers >1/10 of the time (Kregear 2025); ~20% of LLM physics feedback has errors (Maus 2025).
- NOT found anywhere: a parametric library of systematic-error hypotheses for school mechanics experiments with
  literature noise models, Bayesian evidence ranking, EIG next-measurement advice, evaluated on real data; conformal
  prediction sets for experiment-type recognition; equipment-level minimal-pair (counterfactual) evaluation of VLMs
  for lab setups. Novelty must be framed as combination + domain + closed loop, not as any single new method.
- Literature noise values: manual start-stop timing ~N(0, 0.10 s^2) (single action bias 0.11 s, sd 0.07 s; Faux &
  Godolphin 2019); resolution u = d/sqrt(12); ruler +-1 mm; Newton-2 approximation relative error (m + m_e)/M; student
  pendulum g spread sd 0.28 m/s^2 (271 values); photogate calibration bias ~5.7%.
- Chinese curriculum standards require students to "analyse experimental errors and propose ways to reduce them".

## Data
- Existing: 809 annotated images (241 photos + 568 frames from 6 videos) + 25-photo intent set - not yet on the Mac;
  the public GitHub repo holds data/dataset.zip, so a GPU server can pull it directly from GitHub (no slow upload).
  ~65 images are on the Mac now.
- Wikimedia Commons: few thousand free-licensed apparatus files, mostly museum/diagrams; modern school-lab setups are a
  minority (estimate a few hundred; <100 for the six mechanics experiments). ~110 topic-labelled CC video lessons
  (Getaclass CC BY 3.0, Macedonian Wikiexperiments CC BY-SA 4.0) - keyframes usable for an open-set test.
- Open datasets: PhysiSens (5,185 lab-instrument images, CC BY 4.0), Figshare physics lab equipment (5,121 images, CC BY
  4.0); Physics-IQ (396 real videos, CC BY); PhysBench (mixed, 7.4 GB). No open dataset of raw school-student
  measurements for the six experiments.
- Synthetic: numpy simulation of the six experiments with injected faults is trivial on the Mac; MuJoCo has arm64
  wheels with off-screen rendering; Blender/Kubric too costly before 11-09.

## Current code (public repo ChromiteCr/Physics_Pictorial_Recognition, course version 2026-09-06)
- physics.py: EXPERIMENT_TYPES registry with 6 compute functions; inputs are mostly ONE trial (single reading), e.g.
  energy: mass, height, final speed; Newton: mass + force and/or acceleration; Hooke supports multi-point fit;
  "verified" if relative error < 15%.
- intent_classifier.py (rules + CONFIDENT_SCORE_GAP 0.3 fallback), explain_api.py (GLM via zhipuai), app.py
  (Streamlit, two tabs), animate.py (6 idealized animations), detect.py (YOLOv8).
