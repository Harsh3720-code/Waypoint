# Waypoint — Implementation Plan

Source: *PRD — Waypoint: AI Learner Support Portal* (Draft v0.1, 17 Nov 2026).
Status of this plan: **proposal, awaiting sign-off on the decisions in §2.**

---

## 1. What we are building (in one paragraph)

A web portal for learner-support staff in adult/FE colleges (LSAs, tutors, coordinators, admins)
to record session feedback, run the ILSP lifecycle (draft → awaiting approval → approved →
review due → reviewed), and log support given against ILSP strategies. On top of that core
sit three AI layers, all strictly *draft/suggest, human decides*:

1. **LLM drafting** (Claude, structured output) — rough notes → session log entry; support call
   → draft ILSP — grounded via **RAG** over admin-uploaded guidance (pgvector), wrapped in
   **guardrails** (PII redaction, tone, safeguarding banner, no diagnoses), and gated by an
   **eval suite in CI**.
2. **Check-in suggestion model** — XGBoost on engagement features, SHAP top-3 reasons in plain
   English, MLflow-tracked, nightly scoring, Evidently drift report.
3. **Weekly coordinator summary** per class.

Everything runs on synthetic data, ships in Docker, and deploys to Azure Container Apps via
GitHub Actions. It doubles as a portfolio piece, so code quality, docs, the model card and a
live demo are first-class deliverables.

---

## 2. Gaps, ambiguities and decisions needed

Things the PRD leaves open or that will bite during build. Each has a recommended default — I
will proceed with these unless told otherwise.

| # | Topic | Issue | Recommendation |
|---|---|---|---|
| D1 | **Embeddings** (PRD open Q4) | sentence-transformers pulls in PyTorch (~1–2 GB image), which hurts cold starts on a scale-to-zero Container App. Voyage AI needs an API key we don't have. | **Local embeddings via `fastembed`** (ONNX runtime, no PyTorch) with `BAAI/bge-small-en-v1.5` (384-dim, ~130 MB). No key, no cost, nothing leaves the box. Behind an `Embedder` interface so Voyage can be added later as a config switch. **Decided 18 Nov 2026.** |
| D2 | **Standalone vs TeachDesk** (open Q2) | Affects repo layout, auth, shared UI. | **Standalone.** TeachDesk is unrelated. **Decided 18 Nov 2026.** |
| D3 | **Learner agreement** | FR5 requires "learner agreement recorded", but the learner role is v2. | Staff record agreement on the learner's behalf (who, date, method: in person/verbal/signed). |
| D4 | **"Every view" audited** (FR10) | Auditing every read of everything is noisy. | Define *sensitive entities* = Learner profile/needs, SupportCall, ILSP, SessionNote, SupportLog, CheckInSuggestion. Audit create/update/delete/view/export on those only; list endpoints log one event per page, not per row. |
| D5 | **PII redaction approach** | Generic NER (Presidio/spaCy) misses names and adds weight. | We *know* every learner/staff name from the DB → **dictionary-based redaction** of known names + regex for emails, phones, DOBs, postcodes, ULNs; reversible placeholder map. Optional NER pass as a second line. Unit-tested with adversarial cases. |
| D6 | **"Sentiment of recent notes"** feature | Calling an LLM for sentiment in a nightly batch adds cost and non-determinism. | Compute sentiment **at note save time** with a small local scorer (VADER-style) and store it; the nightly job just aggregates. Document as a limitation. |
| D7 | **Synthetic labels for ML** | If labels are a deterministic function of features, the model "learns the generator" and metrics are meaningless. | Generator uses a **latent disengagement process** with noise; features are noisy observations; label = latent crosses threshold in next N weeks. Always report vs the PRD baseline ("missed 2+ in a row"). Synthetic group attribute (for fairness checks only, never a feature). |
| D8 | **LLM evals in CI** | Real Claude calls cost money and need a secret; can't run on forked PRs. | Unit tests use **recorded fixtures** (no network). The real eval suite runs on PRs touching `backend/app/ai/**` or `evals/**`, nightly, and before deploy. Fails CI below thresholds. |
| D9 | **MLflow on Azure** | A full always-on MLflow server costs money. | MLflow backend store = the app Postgres (separate schema), artifacts = Blob Storage; MLflow UI as a scale-to-zero Container App. Nightly scoring = **Container Apps Job** (cron). |
| D10 | **Auth details** | "JWT sessions" — storage unspecified. | Short-lived access JWT (in memory) + refresh token in **httpOnly, Secure, SameSite cookie**; argon2 password hashing; rate-limited login. |
| D11 | **Claude models** | PRD says "smaller model for simple tasks". | Drafting & weekly summary: Sonnet-class; tone check/classification: Haiku-class; eval judge: a different/stronger model than the drafter to reduce self-grading bias. Model IDs live in config, confirmed at M2. |
| D12 | **PDF export** (FR11) | Library choice. | HTML template → **WeasyPrint** (good typography, accessible). |
| D13 | **Retention period** | "Configurable" but no mechanism. | Admin setting + a scheduled purge job; soft-delete first, hard-delete after grace period; purge itself audited. |
| D14 | **Pilot vs portfolio** (open Q1) | A pilot needs a DPIA and security review. | Build *as if* real (per §9 of PRD) but scope = **portfolio**. Write a lightweight DPIA-style note in `docs/` anyway — good portfolio signal. |

Prerequisites, and how we cope without them:
- **Voyage key:** not needed (D1).
- **Anthropic API key, needed for M2.** Until we have one, M2 is built against a fake `LLMClient` and recorded responses, so the whole pipeline, the UI and the rule-based evals all work offline. The key is only required for live drafting and the LLM-as-judge evals. API usage is billed separately from a Claude.ai subscription; a few dollars of credit covers development and the eval suite.
- **Azure subscription, needed for M4 (around week 8).** A new Azure free account currently gives time-limited starter credit plus 12 months of some free services (check current terms when signing up). Until then: everything runs in `docker compose`, and the Bicep templates and deploy workflow are written and validated (`az bicep build`, what-if) but not applied. If Azure stays unavailable, the same images can deploy to any container host; Azure remains the documented target.

---

## 3. Architecture

```
┌──────────────┐  REST/JSON  ┌──────────────────────── FastAPI ─────────────────────────┐
│ React + Vite │ ──────────▶ │ api/ (routers, RBAC deps)                                │
│ TS, Tailwind │             │ services/ (ILSP workflow, audit, search, export)         │
└──────────────┘             │ ai/  redact → retrieve → Claude → guardrails → log       │
                             │ ml/  feature builder, scorer, SHAP explainer             │
                             └───────┬──────────────────────────────┬───────────────────┘
                                     │                              │
                          ┌──────────▼──────────┐        ┌──────────▼──────────┐
                          │ Postgres + pgvector │        │ MLflow (registry)   │
                          │ app data, chunks,   │        │ artifacts → Blob    │
                          │ audit, llm_calls    │        └─────────────────────┘
                          └─────────────────────┘
  Scheduled jobs (same backend image, different entrypoint):
    nightly_score · weekly_summary · weekly_drift_report · retention_purge
```

Key principles:
- **One backend image**, multiple entrypoints (API, jobs) — simpler CI and deploy.
- **AI behind interfaces** (`LLMClient`, `Embedder`, `Redactor`) so tests run offline with fakes.
- **RBAC enforced server-side** via FastAPI dependencies plus class-scoping (a tutor only sees
  learners in their classes; an LSA only their assigned classes).
- **Every AI output persisted with provenance**: model, prompt version, retrieved chunk IDs,
  guardrail results, and the staff decision (accepted / edited / rejected).

### Proposed repo layout

```
waypoint/
├── backend/
│   ├── app/
│   │   ├── api/            # routers per resource
│   │   ├── core/           # config, security, db session, rbac
│   │   ├── models/         # SQLAlchemy
│   │   ├── schemas/        # Pydantic (also the LLM structured-output schemas)
│   │   ├── services/       # ilsp_workflow, audit, export, search
│   │   ├── ai/             # redaction, rag, prompts/, guardrails, llm_client, summary
│   │   ├── ml/             # features, train, score, explain, drift
│   │   └── jobs/           # CLI entrypoints for scheduled jobs
│   ├── alembic/
│   └── tests/
├── frontend/               # React + Vite + TS + Tailwind
├── synthetic/              # Faker + scripted scenarios generator
├── evals/                  # ~50 cases, rule checks, LLM-judge, thresholds.yaml
├── infra/                  # Bicep for Azure resources
├── docs/                   # PLAN, architecture, model card, DPIA note, ADRs
├── docker-compose.yml
└── .github/workflows/      # ci.yml, evals.yml, deploy.yml
```

---

## 4. Milestone plan

Ordering follows the PRD (M0–M5), but each milestone ends in something demoable, and the AI work
is built against the real core from M1 rather than in isolation.

### M0 — Setup (week 1)
- Monorepo skeleton per §3; `docker-compose` with `db` (postgres+pgvector), `backend`, `frontend`, `mlflow`.
- Backend: FastAPI app, settings (pydantic-settings), SQLAlchemy 2.x + Alembic, `/health`.
- Frontend: Vite + React + TS + Tailwind, router, API client, **"Demo data" banner** on every page.
- Tooling: ruff + mypy + pytest (backend); eslint + prettier + vitest (frontend); pre-commit.
- CI (`ci.yml`): lint, type-check, test, build images.
- **Synthetic data generator v1**: users per role, courses, classes, learners, enrolments,
  attendance with the latent-engagement process (D7), sample support calls and notes.
- ADRs for D1–D14 decisions.

**Exit:** `docker compose up` → seeded DB, empty app shell with login page, green CI.

### M1 — Core portal (weeks 2–3)
- **FR1 Auth & roles** — login, refresh, logout; admin user management (create/deactivate); RBAC matrix tested exhaustively (role × endpoint).
- **FR2 Learner records** — profile, enrolments, attendance, disclosed needs.
- **FR3 Support call** form — screening questions, disclosures, consent to share; "Start ILSP" action.
- **FR4 Session feedback** — pick class/date/learners, notes + quick tags; versioned edits.
- **FR5 ILSP** — sections (needs, strategies, SMART targets, who/how often, review date), status state machine with guarded transitions, coordinator approval, version history, learner agreement (D3).
- **FR6 Support log** linked to strategies. **FR7 Reviews** — due list, per-strategy outcome.
- **FR9 Search/filters**. **FR10 Audit trail** (middleware + service, D4) with admin audit viewer.
- **FR11 PDF export** (D12).
- Basic coordinator dashboard (FR8) without ML tiles.

**Exit:** full manual ILSP lifecycle demo per role; backend coverage ≥ 80%; Playwright happy-path per role.

### M2 — AI drafting (weeks 4–5)
- `ai/` pipeline: **redact → retrieve → prompt → Claude (structured output against Pydantic schema) → validate → guardrails → un-redact → persist + log**.
- RAG: admin upload (PDF/DOCX/MD) → parse → chunk → embed → pgvector; citations shown per suggestion in UI.
- Guardrails: PII redaction (D5); tone check (flag + suggested rewording); safeguarding keyword+LLM classifier → banner only; no-diagnosis instruction + output check.
- UI: "Draft with AI" on session notes and ILSP; clear AI-generated labels; accept / edit / reject captured.
- LLM call log: latency, tokens, cost, guardrail outcomes, prompt version — no identifiers.
- **Evals**: ~50 synthetic cases; rule-based checks (SMART structure, PII leak, diagnosis terms, citation present) + LLM-as-judge (grounding, tone); `thresholds.yaml`; `evals.yml` workflow (D8).
- Seed guidance docs written from scratch (crib sheet, ILSP standards, strategy library) — original content, no real college material.

**Exit:** notes → draft in seconds with citations; eval suite green in CI with published scores.

### M3 — ML (weeks 6–7)
- Feature builder (shared by train and score — no train/serve skew): attendance trend, consecutive misses, support-log frequency, overdue reviews, recent-note sentiment (D6).
- Train XGBoost with time-based split; compare to baseline; PR-AUC, precision/recall at threshold; calibration check.
- SHAP → top-3 reasons → plain-English templates ("missed 3 of the last 4 sessions").
- MLflow tracking + registry; scorer loads `Production` alias.
- `nightly_score` job → `CheckInSuggestion` rows; admin-set threshold; staff decision recorded.
- Fairness: metrics by synthetic group → model card.
- Dashboard: check-in tile with reasons; **weekly AI summary** per class (§6.4).

**Exit:** dashboard shows explainable suggestions; model card draft with metrics vs baseline.

### M4 — Azure (week 8)
- Bicep: Container Apps env (API, frontend, MLflow UI), Container Apps Jobs (nightly/weekly), Postgres Flexible Server (Burstable, pgvector enabled), Blob, Key Vault, Container Registry, Log Analytics.
- GitHub Actions `deploy.yml`: OIDC login (no long-lived secrets) → build/push → Alembic migrate → deploy; target < 10 min.
- Managed identity for Key Vault and Blob access. Scale-to-zero on all apps.

**Exit:** merge to `main` → live demo URL updated automatically.

### M5 — Polish (weeks 9–10)
- Weekly Evidently drift report job (stored in Blob, linked from admin page).
- Accessibility pass to WCAG 2.2 AA (axe in Playwright, keyboard-only walkthrough).
- p95 < 300 ms check on non-AI endpoints (locust/k6 script).
- README with architecture diagram, model card, DPIA-style note, write-up, 3-min demo video script.

**Cut line if behind schedule** (drop in this order): Evidently report → weekly summary → PDF export → MLflow UI hosting (keep registry only).

---

## 5. Testing strategy

| Layer | Tooling | Notes |
|---|---|---|
| Unit | pytest | services, state machine, redaction, feature builder, SHAP templating |
| API | pytest + httpx + Postgres test container | RBAC matrix, audit events emitted |
| AI | pytest with recorded LLM fixtures | deterministic, offline, runs on every PR |
| Evals | custom runner, real Claude | gated workflow, thresholds fail CI |
| ML | pytest | feature parity train vs score, model beats baseline on test split |
| E2E | Playwright (`--trace off` by default) | one journey per role + axe accessibility checks |
| Perf | k6 or locust | p95 non-AI endpoints |

---

## 6. Immediate next step

On sign-off of §2, start **M0** on branch `claude/ecstatic-bardeen-1a05wx`: scaffold, compose,
CI skeleton, synthetic generator v1, ADRs.
