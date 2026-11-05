export const meta = {
  name: 'p2-stage2-compose-data-bench-ui',
  description: 'Physics component layer, training-data generators, physics & recognition benches, UI, real-run kit and student docs, then integration + full data generation + 5090 package',
  phases: [
    { title: 'Build', detail: 'components, prompts/generators, recognition bench, UI, kit+docs' },
    { title: 'Build2', detail: 'guide/recognize/card generators and physics bench (depend on prompts)' },
    { title: 'Integrate', detail: 'cross-module integration, review, full-size generation, packaging' },
  ],
}

const REPO = '/Users/billgao/Documents/Coding/Physics-Pictorial-Recognition'
const PY = `${REPO}/.venv/bin/python`
const DRAFT = '/private/tmp/claude-501/-Users-billgao-Documents-Coding-SCPKU-Proj/6e0639b7-50d1-4852-9fa9-a0a36681b6ab/scratchpad/physics_draft'

const COMMON = `
You are a senior engineer on a high-school ISEF project, phase P2 "实验侦探 Lab Detective": an OFFLINE AI lab partner for school mechanics
experiments (physics-heavy, physics + CS interdisciplinary). Photo -> identify the setup -> parametric physics twin -> pre-lab prediction &
uncertainty budget -> guided measurement with live checks -> Bayesian diagnosis of systematic-error causes -> most informative next measurement
-> explanation tied to knowledge points. The language model never computes physics; deterministic physics code produces every number (FactSheet).
Product must not resemble the "Gewu" paper (text-problem tutoring with a code sandbox, tool loops, voting): NO sandbox, NO tool-call loop, NO voting.
Distillation is the only borrowed idea (teacher answers blind; keep only outputs that agree with the physics engine; ≤ 5,248 distilled examples).
Wording rule for anything user-facing or in docs: never claim an absolute fixed set like "12 个实验"; say the curated experiments are a starting point and
other setups are composed from physics components.

Repo: ${REPO} (branch p2-lab-detective). Python: ${PY} (numpy, scipy, sklearn, pytest, torch, ultralytics, opencv, PIL, streamlit, pandas, matplotlib, yaml, jsonschema).
READ FIRST: docs/ARCHITECTURE.md, docs/isef/innovation-and-boundaries.md, lab_detective/labsim/core.py header, and the module docstrings/tests of
everything you build on (labsim/experiments/*, labsim/generic.py, engine.py, panel.py, eig.py, atlas.py, twin.py, uncertainty.py, facts.py,
knowledge/, tutor/, llm/, vision/, training/format.py, training/distill.py, training/package.py, bench/stats.py, bench/freeze.py, server/).
Rules:
- Only create/modify files you OWN (listed in your task). Need a change elsewhere? Make the smallest compatible change ONLY if it is a clear bug blocking you, and say so in your report.
- Old folders under /Users/billgao/Documents/Coding/SCPKU/ are READ-ONLY.
- Student-facing text in simplified Chinese. Code comments concise Chinese, matching existing style.
- No new third-party deps; no network in library code/tests; never call external APIs (API runners are written but only the user runs them).
- Memory: everything you run < 4 GB (parallel workers ≤ 6). Tests < 10 s each.
- Tests in tests/test_<module>.py; run: cd ${REPO} && ${PY} -m pytest -q tests/<files>; before committing also run the whole suite quickly.
- Commit ONLY your files: git add <paths> && git commit -m "<中文说明>" -- <paths> (no AI/tool names, no Co-Authored-By). If index.lock, sleep 3 and retry (≤20). Never git add -A, never push.
- Final answer: concise report (files, APIs, test results, open issues).`

const A = `TASK A (owner of lab_detective/physics/**, tests/test_physics_*.py): the PHYSICS COMPONENT LAYER — decompose the curated experiments into reusable components and compose new setups.
Start from the draft design in ${DRAFT}/base.py and ${DRAFT}/components.py (copy, then improve; they were written before the curated experiments existed). Read the finished curated experiments in lab_detective/labsim/experiments/ and EXTRACT their physics into:
- physics/base.py (Effect, Ctx, Component, AuxMeasurement, Setup), physics/components.py (bodies: cart with wheel inertia, block, ball/bob, hanger; connectors: spring, string, pulley; surfaces: track (friction balance), incline (angle, slant-as-height), lever; …),
- physics/instruments.py (ruler, spring scale incl. vertical/horizontal zero shift, stopwatch with reaction time, photogate speed & acceleration incl. flag-width error, ticker tape, phone timer with miscount, balance, protractor, caliper) — each maps a true quantity to the recorded reading (affine where possible so linear params survive),
- physics/templates.py (law templates: static spring balance, friction pull, uniform-acceleration system (cart/track/incline/pulley/hanging mass, generalised to effective mass M + Σ I/r² and net driving force), energy on an incline, system energy (hanging mass drives cart), pendulum period, spring-oscillator period, projectile from a ramp, torque balance, 1-D collision),
- physics/compose.py: Setup -> ComposedExperiment(Experiment) obeying labsim/core.py (predict → (c,U); causes = random_only + every component/instrument Effect; aux menu items from components; standard/single protocols; linearise from the template), so engine/panel/eig/atlas/facts work unchanged,
- physics/catalog.py: setups that mirror curated experiments (at least hooke, friction, newton2, energy, pendulum) AND ≥5 NEW compositions not in the curated set (e.g. 斜面上用悬挂重物拉小车的牛顿第二定律, 悬挂重物带动小车的系统机械能守恒, 弹簧振子周期与质量的关系, 光电门测自由落体 g, 斜面上滑块的匀加速运动), each marked held_out according to whether it appears in training (new compositions default to held_out=True so they can serve as transfer tests),
- physics/from_photo.py: map a vision RecognitionResult (apparatus list + cues, see vision/schema.py) to ranked candidate Setups (catalog match + composition rules), with the open questions the student must answer (masses, angle) listed.
VALIDATION (tests): (1) for each mirrored setup, composed predictions equal the curated experiment's predictions for random_only and for every shared cause (match within 1e-6 relative where the physics is the same; document any deliberate difference, e.g. exact vs approximate physics); (2) each new composition: zero-fault recovers its physical constant, engine identifies a large clearly-detectable fault in ≥70% of 20 sims, posterior < 0.5 s; (3) from_photo returns sensible candidates for cart+track+string, spring alone, dynamometer+wooden_block. Put derivations (2-6 lines) as comments.`

const B1 = `TASK B1 (owner of lab_detective/training/prompts.py, lab_detective/training/splits.py, lab_detective/training/tasks_diagnose.py, lab_detective/training/tasks_design.py, lab_detective/training/tasks_explain.py, lab_detective/training/tasks_knowledge.py, lab_detective/training/generate.py, tests/test_training_tasks.py):
Training-data generators (format: training/format.py Example; docs/ARCHITECTURE.md §2.5). prompts.py is the SINGLE source of prompts for training, distillation, benches and inference — every builder returns messages using tutor/context.py SYSTEM_PROMPT_ZH (import it) and task-specific user content; define JSON output schemas (dict) for diagnose and design.
- splits.py: training experiments vs held-out (registry flags), ONE withheld cause per training experiment (choose a physically interesting one; list them), seeds, train/val split rule (val = separate seeds, 5%).
- tasks_diagnose: from simulated datasets (training experiments + labsim/generic.py procedural family; protocol mix: single 15% / standard 60% / standard+1 extra measurement 25%; ~25% random_only), build the prompt (experiment card essentials from knowledge cards, data table, panel numbers, candidate causes as PLAIN descriptions with shuffled ids — no signature words, measurement menu) and the target JSON from the engine: data signature list, per-cause probability (rounded 0.05), not_in_list, next measurement = argmax EIG with predicted values under the top 2 causes, agreement verdict, ≤80-字 Chinese explanation from tutor/templates.py. 20% of examples are "propose causes" variants (candidate list withheld; target = engine-ranked causes in plain language).
- tasks_design: pre-lab planning from twin.py/uncertainty.py/atlas: target = expected readings table, dominant uncertainty source, recommended number/range of points, causes this protocol cannot separate and the measurement that would.
- tasks_explain: FactSheet -> explanation (templated with 3 paraphrase variants per slot); must pass tutor/verifier.py.
- tasks_knowledge: from knowledge/ graph: Q&A linking a cause/misconception to knowledge points and prerequisites.
- generate.py CLI: --n-diagnose --n-design --n-explain --n-knowledge --workers (multiprocessing, ≤6 workers, chunked, deterministic by seed) writing results/train/<task>.jsonl and results/val/<task>.jsonl; prints counts and throughput. Measure throughput at small n and report examples/sec per task (EIG is the expensive part — use n_samples≈64 and prune).
Tests: schemas valid, no held-out experiment or withheld cause appears in train, prompts contain no signature tags or engine probabilities, verifier passes on explanation targets, deterministic with seed.`

const B2 = (b1) => `TASK B2 (owner of lab_detective/training/tasks_guide.py, lab_detective/training/tasks_recognize.py, lab_detective/training/tasks_card.py, lab_detective/training/distill_jobs.py, tests/test_training_tasks2.py). B1 (prompts.py, splits.py, generate.py) is finished; its report:
---
${b1}
---
- tasks_guide: simulate whole lab sessions with tutor/session.py + tutor/policy.py ScriptedTutor over training experiments (and training-side composed setups if lab_detective/physics exists): student inputs synthesized (predictions sometimes wrong, readings from labsim), tutor turns as targets, context built with tutor/context.build_context (compact, never full history). Inquiry pedagogy: predict-first, one question at a time, hint ladder, no answer before prediction.
- tasks_recognize: from results/recognition/manifest.jsonl (built by vision/build_dataset.py) — ONLY items not reserved for the recognition bench (coordinate: use a deterministic split by photo session/EXIF time or file-name time stamp; put the split function in tasks_recognize.py and export it, the bench builder will import it); target = vision/schema RecognitionResult JSON; images referenced as images/<file> relative to results/recognition/; include "needs_review" items only if their label was confirmed (respect a review file results/recognition/review.json if present: {id: letter}).
- tasks_card: open-set experiment cards: procedural/generic experiments and training-side compositions → card JSON (schema knowledge/schema.py) targets.
- distill_jobs.py: build DistillJob lists (training/distill.py) for (a) diagnose — teacher sees exactly the B1 diagnose prompt; keep iff teacher JSON valid AND its top cause is in the engine's equivalence class of the engine top cause (use atlas/engine: causes whose posterior ratio to the top is > 0.5 count as tied) AND its next measurement is one of the top-2 EIG items AND the explanation passes tutor/verifier; (b) guide turns — keep iff verifier passes, exactly one question, no cause named before the student's prediction; (c) cards for named mechanics experiments outside the curated set — keep iff schema valid and units/relation family consistent. Provide CLI: python -m lab_detective.training.distill_jobs deepseek --task diagnose --n 2000 (needs key file) and hy4-export --task ... --n ... (writes results/distill/hy4_<task>/prompts.jsonl + README) and hy4-import. Total cap 5,248 across all teachers/tasks (read existing raw results).
Tests with fake teachers; no network.`

const C = (a, b1) => `TASK C (owner of lab_detective/labsim/hifi.py, lab_detective/bench/physics_bench.py, lab_detective/bench/score.py, lab_detective/bench/run_models.py, tests/test_physics_bench.py). Inputs: physics component layer report:
---
${a}
---
training prompts report:
---
${b1}
---
- labsim/hifi.py: a MISSPECIFIED "high-fidelity" world for testing only: wraps any Experiment and adds effects the engine never models (pulley/air drag where relevant, string mass, skewed (ex-Gaussian) timing noise, occasional ×10 transcription slips, Student-t reading noise) with documented magnitudes.
- bench/physics_bench.py build: splits written to results/bench/physics/<split>.jsonl in the training Example format (assistant message = gold JSON) + a gold sidecar with engine posteriors, equivalence classes (from atlas: causes the protocol cannot separate are merged), EIG of every menu item: in_distribution 600, held_out experiments (all held_out labsim experiments + held_out compositions; 100 each), noise_shift 300 (noise_scale 2, Student-t), withheld_cause 300, hifi 300, clean 150, unmodelled 150 (two causes at once; ×10 slip) split into detectable/degenerate, swapped_table 600 (K3 control), oracle_in_prompt 100 (K1), design 200, guide 200. Sizes configurable; a --small flag for tests. Use the SAME prompt builders as training (prompts.py). Ensure no overlap with training (leakage_report against results/train if present).
- bench/score.py: parse model outputs (tolerant JSON), metrics per split and per experiment: top-1 and equivalence-class top-1, log score, Brier, ECE, not-in-list AUROC, next-measurement EIG regret (normalised), JSON validity, verifier pass rate for explanation fields; paired comparisons between two model prediction files (exact McNemar, sign test across experiments, paired bootstrap) using bench/stats.py; markdown + JSON report.
- bench/run_models.py: run any OpenAICompat model (lab_detective/llm/client.py) on a bench split with identical prompts/budget (thinking off by default; flag for DeepSeek thinking=high as ceiling reference), JSONL predictions with model id/date/params; plus 'engine' and 'rule' reference predictors that run locally (engine = Bayes reference; rule = 20-line panel matcher). Never runs automatically; user launches.
Tests: tiny bench build (--small) + scoring of engine predictions gives top-1 = 1 on in-model splits' equivalence classes, scoring a deliberately wrong predictor gives lower scores, McNemar wiring.`

const D = `TASK D (owner of lab_detective/bench/recognition_bench.py, lab_detective/bench/review_gallery.py, tests/test_recognition_bench.py):
Uses vision/ (schema, letters, prompt, pipeline, build_dataset) and results/recognition/manifest.jsonl (already built). If tasks_recognize.py (another engineer) exports a split function use it; otherwise implement a deterministic session-based split (EXIF DateTime or file-name time stamps: photos within 30 min = one session; ≈25% of sessions → test) in recognition_bench.py and tell the other engineer via your report.
- review_gallery.py: generate results/recognition/review.html (offline, self-contained thumbnails base64 or relative paths) showing each photo with its provisional label and letter options, plus a tiny JS that exports review.json ({id: letter}) for the student to download — this is how the student confirms labels quickly tomorrow.
- recognition_bench.py: build results/bench/recognition/test.jsonl (Example format, image + vision/prompt.py messages, gold = reviewed label or provisional with flag), score predictions (top-1 over letters, per-source, open-set Z handling, AUROC of 'other' if logprobs absent use confidence field), and a 'rules' baseline using vision/cues.py on ground-truth YOLO boxes and on YOLO detections.
Tests on synthetic images.`

const E = `TASK E (owner of app.py (rewrite allowed; keep the P1 detection & canvas features working), ui/** , .streamlit/config.toml, tests/test_app.py):
Streamlit P2 app, fully offline, scripted mode by default (templates), optional "模型模式" talking to a local OpenAI-compatible server (mlx_vlm.server) via lab_detective/llm/client.py when available (detect by a quick GET with 0.5 s timeout; never required).
Flow (tabs or sidebar steps): ① 拍照识别（上传照片 → YOLO 框 + 画布修正（保留 P1 功能）→ 识别实验（vision/pipeline）→ 一个澄清问题 → 手动选择永远可用，含"其他实验"）; ② 实验设计（twin.py 预测表、uncertainty 预算、本方案分不开的原因、建议的测量范围）; ③ 测量（多行数据表 st.data_editor，按仪器配置，填一行就实时对照孪生：z 值和提示；先预测再测量）; ④ 诊断（嫌疑人看板：概率条、证据、幽灵曲线图、"下一次测量"卡含各原因预测值、修正后复测验证）; ⑤ 讲解与知识点（事实单讲解 + 校验器结果 + 相关知识点和前置知识学习路径）.
Use lab_detective.physics (component layer, being built in parallel — import lazily; if unavailable fall back to labsim registry) for "其他实验"/composed setups. Remove every import of explain_api (GLM) from the app; keep the chalkboard theme but make it fully offline (remove the Google Fonts @import, use system Chinese fonts: PingFang SC, Hiragino Sans GB, Noto Sans CJK, sans-serif). No absolute "12 个实验" wording.
Tests: streamlit.testing.v1 AppTest smoke tests (app loads; scripted flow for hooke with simulated data reaches diagnosis without exceptions; no network). Also run "streamlit run app.py --server.headless true --server.port 8765" briefly in the background and curl the health endpoint, then stop it.`

const F = `TASK F (owner of lab_detective/fieldkit/**, docs/student/**, tests/test_fieldkit.py):
1) fieldkit (real fault-injection runs on the student's own apparatus; only the student measures; an adult may set sealed faults; nothing about people is recorded):
   schedule.py: committed random schedule (seed + sha256) for N runs across chosen experiments (default hooke, friction or newton2, pendulum via the open-set/generic path) × conditions (clean + 3 causes incl. ≥1 pair the atlas says the standard protocol confounds) × repeats; each run lists the base protocol and ALL follow-ups (EIG-recommended, best fixed item, repeat) in randomised order; sealed_cards.html printable (offline) with cut lines, codes hiding the condition; csv templates per experiment; ingest.py validating & hashing the CSVs into Datasets; analysis.py producing the pre-specified analysis (engine diagnosis before/after each follow-up, paired Wilcoxon EIG vs best fixed, Wilson CIs) — analysis must be launched by the student.
   Exclude any destructive fault (no spring beyond elastic limit on real apparatus; simulate only).
2) docs/student/ (Chinese, plain, for the student; NO research-plan text, NO literature/citations):
   - 01_手推清单.md: every cause of every curated training experiment listed with its plain description ONLY (no signatures, no formulas of the effect) and a blank space; instructions: derive the expected data signature and the separating measurement yourself before reading code/cards.
   - 02_你要做的决定.md: checklist of decisions only the student makes (thresholds for each claim, which claims to keep, margins vs DeepSeek, sample sizes within compute) with the considerations, NOT answers.
   - 03_GPT审查包.md: a physics-only specification (math of the forward models for newton2, energy and one two-cause EIG example, written from the physics, no code) for the user's separate GPT agent to independently re-implement and review; plus instructions for what to send back.
   - 04_Hy4使用说明.md: how to run the Hy4 prompt packs and return responses.jsonl.
   - 05_明天的5090操作.md: short operational checklist that points to server/README.md (do not duplicate it).
Tests: schedule deterministic & balanced, cards contain no condition names, ingest round-trip.`

const G = (reports) => `TASK G — INTEGRATION, REVIEW, FULL-SIZE GENERATION, PACKAGE (you may modify any file to fix integration bugs; keep changes minimal and list them).
Reports from all stage-2 builders:
---
${reports}
---
1. Run the full test suite; fix failures. Add tests/test_integration.py: photo (one of the student's photos from results/recognition) → vision pipeline (rules only, scripted) → physics/from_photo candidates → choose setup → simulate a faulty dataset → facts.build_fact_sheet → tutor templates explanation → verifier ok; and an open-set composed setup end to end.
2. Review numerics: engine evidence vs brute-force numerical integration on a tiny case (1 linear param + small grid) must agree to < 1e-3 in log evidence; EIG non-negative and ≈0 for useless items; posterior timing per experiment (report table).
3. Generate the FULL training data (python -m lab_detective.training.generate with sizes targeting ~20,000 examples total across tasks: diagnose ~9,000 (incl. procedural), design ~1,500, explain ~2,000, knowledge ~1,000, guide ~3,000, recognize = all eligible photos ×2 augmentations (mild crop/brightness, keep labels), card ~800) using ≤6 workers; record throughput; also build the physics bench and recognition bench at full size, run leakage_report (must be clean), and freeze NOTHING (freezing is the student's job) — but write a FREEZE_INSTRUCTIONS.md next to the benches.
4. Package with lab_detective/training/package.py into results/package_<date>.tar.gz; report size and upload time at 100 KB/s; if > 150 MB, reduce image resolution to 640 px or drop augmentation and repackage.
5. Update README.md (P2 section, run instructions, version table rows P2a "物理孪生、贝叶斯引擎、信息增益与图谱", P2b "组件组合层、训练数据与评测基准、新界面", type feat; badge version → P2b) and docs/AI_USAGE.md (module ownership table + change log for today). Commit in small logical commits.
Final report: test counts, timings, data counts by task/source, package path/size/sha256, known gaps.`

phase('Build')
const a = agent(`${COMMON}\n${A}`, { label: 'build:physics-components', phase: 'Build' })
const b1 = agent(`${COMMON}\n${B1}`, { label: 'build:training-prompts', phase: 'Build' })
const d = agent(`${COMMON}\n${D}`, { label: 'build:recognition-bench', phase: 'Build', model: 'sonnet' })
const e = agent(`${COMMON}\n${E}`, { label: 'build:ui', phase: 'Build' })
const f = agent(`${COMMON}\n${F}`, { label: 'build:fieldkit-docs', phase: 'Build' })
const b2 = b1.then(r => agent(`${COMMON}\n${B2(r)}`, { label: 'build:training-tasks2', phase: 'Build2' }))
const c = Promise.all([a, b1]).then(([ra, rb]) => agent(`${COMMON}\n${C(ra, rb)}`, { label: 'build:physics-bench', phase: 'Build2' }))
const all = await Promise.all([a, b1, d, e, f, b2, c])
const names = ['A physics-components', 'B1 training-prompts', 'D recognition-bench', 'E ui', 'F fieldkit-docs', 'B2 training-tasks2', 'C physics-bench']
const reports = all.map((r, i) => `## ${names[i]}\n${(r || '(no report)').slice(0, 3500)}`).join('\n\n')
phase('Integrate')
const g = await agent(`${COMMON}\n${G(reports)}`, { label: 'integrate:generate-package', phase: 'Integrate' })

const H = `TASK H — REWRITE ALL MARKDOWN DOCS so they match the CURRENT direction (you own every *.md under docs/ and README.md; do not touch code).
Current direction (authoritative): a general photo-recognition AI lab partner for school MECHANICS (mechanics first; extensible), which guides students through
experiments with inquiry pedagogy; physics-heavy: error-physics twins, a physics COMPONENT layer that composes twins for setups beyond the curated ones,
identifiability atlas, Bayesian diagnosis, EIG next measurement, pre-lab uncertainty budget, real-apparatus closed loop; CS: on-device Qwen3.5-4B fine-tuned
mainly on physics-engine labels (+ ≤5,248 distilled examples filtered by engine agreement), recognition from photos (YOLO cues + VLM), phase-aware context
assembly, FactSheet + claim verifier, scripted fallback; evaluated on a physics bench and a recognition bench against the untuned 4B, Qwen3.8-27B, and DeepSeek V4.1 Flash
(the student pre-registers the targets; benches frozen by the student). Product never calls external models. No resemblance to the Gewu paper (no sandbox, tool loop, voting, text-problem tutoring).
Never claim a fixed absolute number of experiments; curated experiments are the starting point.
Integration report for facts about what now exists:
---
${'${INTEGRATION}'}
---
Do:
1. docs/isef/2026-10-31-isef-upgrade-design.md → rewrite in place as the current design (keep a short "版本沿革" section at the end summarising what changed from the 10-31 error-diagnosis-only draft and why). Keep ISEF compliance content (Rule 8, human participants, Form 7, forms timing) accurate; keep the student's decisions as open items, NOT filled answers.
2. docs/isef/2026-11-05-implementation-plan.md → update to what is built, what remains (model training on the 5090 tomorrow, distillation once the key is provided, student review of photo labels, freezing benches, real runs), with tomorrow's schedule.
3. docs/ARCHITECTURE.md → reflect the real module layout and interfaces (physics/ component layer, training/, bench/, fieldkit/, ui/).
4. README.md → P2 overview first (what it is, innovations, how to run: uv venv, streamlit run app.py, tests, data generation, packaging), then the P1 history; version table already updated by integration (verify the badge matches the top version row).
5. docs/AI_USAGE.md, docs/isef/innovation-and-boundaries.md, docs/isef/bench-comparators.md (keep facts; fix wording), docs/student/* (consistency only).
6. Search every .md for stale statements (GLM in the product, "12 个实验" absolutes, "只做误差诊断", old schedules with 10/31–11/5 coding, Proj/ paths, LM Studio as product server) and fix them.
Commit with a clear message. Report the list of files changed.`
const h = await agent(`${COMMON}\n${H.replace('${INTEGRATION}', (g || '').slice(0, 6000))}`, { label: 'docs:rewrite-md', phase: 'Integrate' })
return { reports, integration: g, docs: h }
