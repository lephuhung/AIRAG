# AIRAG harness — one entrypoint for the stack, tests and evals.
# Backend tests/evals run INSIDE the hrag-backend container (WORKDIR /app/backend),
# where the app deps + reachable providers (vLLM, Chroma, Postgres) already live.
# Install the test-only deps once:  make dev-deps
#
# Run `make help` for the target list.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# NOTE: the two vLLM engines (docker-compose.vllm.yml: hrag-vllm-ocr :8001,
# hrag-vllm-memory :8088) are GPU-heavy, minutes to cold-start, and shared by
# OCR + the classifier/memory agent. NEVER restart them from the harness — see
# docs/vllm.md. To flip an A/B arm, restart ONLY hrag-backend (code is bind-mounted).
DC        := docker compose -f docker-compose.services.yml
BK        := docker exec hrag-backend
BK_EVAL   := docker exec -e PROMPT_EVAL=1 hrag-backend
# A/B forwards auth env (JWT-gated debug-chat): export AB_TOKEN, or AB_USER+AB_PASSWORD.
BK_AB     := docker exec -e AB_TOKEN -e AB_USER -e AB_PASSWORD -e AB_TOTP hrag-backend
WORKSPACE ?=             # workspace UUID (required for eval-rag / ab)
TS        := $(shell date +%Y%m%dT%H%M%S)

# ── Stack lifecycle ──────────────────────────────────────────────────────────
.PHONY: up down logs ps restart-backend
up:              ## Start the full app+infra stack (services compose) — does NOT touch vLLM
	$(DC) up -d
down:            ## Stop the stack (keeps volumes)
	$(DC) down
logs:            ## Tail backend logs
	docker logs -f hrag-backend
ps:              ## Show stack containers
	$(DC) ps
restart-backend: ## Reload backend code (bind-mounted, no rebuild)
	docker restart hrag-backend

# ── Test / eval deps ─────────────────────────────────────────────────────────
.PHONY: dev-deps
dev-deps:        ## Install pytest + PyYAML into the backend container
	$(BK) pip install -r requirements-dev.txt

# ── Offline unit tests (no LLM, no live corpus) ──────────────────────────────
.PHONY: test
test:            ## Fast pure-unit tests (chunker + validity extractor)
	$(BK) pytest tests/services -v

# ── Retrieval golden set (needs Chroma + real corpus, no LLM) ────────────────
.PHONY: test-recall test-section test-validity
test-recall:     ## Recall@k over the golden retrieval set (soft-gated vs baseline)
	$(BK) pytest tests/retrieval/test_recall.py -v
test-section:    ## Section-by-Dieu retrieval regression
	$(BK) pytest tests/retrieval/test_section_retrieval.py -v
test-validity:   ## Validity-layer retrieval behaviour
	$(BK) pytest tests/retrieval/test_validity.py -v

# ── Prompt-eval suite (LIVE LLM — needs reachable providers) ─────────────────
.PHONY: eval-prompts
eval-prompts:    ## Live LLM-judge prompt suite → JSON report
	$(BK_EVAL) pytest tests/prompts -m prompt_eval -v \
		--prompt-report tests/prompts/reports/run_$(TS).json

# ── End-to-end RAG evals (drive the live agent over HTTP) ────────────────────
.PHONY: eval-rag eval-ragas
eval-rag:        ## DeepEval RAG quality suite against /api/v1/rag (WORKSPACE=$(WORKSPACE))
	$(BK) python scripts/eval_rag.py --workspace $(WORKSPACE)
eval-ragas:      ## RAGAS synthetic-testset eval
	$(BK) python scripts/eval_ragas_synthetic.py

# ── Agent A/B harness (compare two configs on one query set) ─────────────────
.PHONY: ab ab-compare
ab:              ## Run one A/B arm: make ab ARM=react QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=<uuid>
	$(BK_AB) python -m scripts.ab_eval run --arm $(ARM) --queries $(QUERIES) \
		--workspace $(WORKSPACE) --out tests/prompts/reports/ab_$(ARM)_$(TS).json
ab-compare:      ## Diff two arm reports: make ab-compare A=ab_base.json B=ab_react.json
	$(BK) python -m scripts.ab_eval compare $(A) $(B)

# ── Prompt report diff ───────────────────────────────────────────────────────
.PHONY: compare-prompts
compare-prompts: ## Diff two prompt-eval reports: make compare-prompts A=old.json B=new.json
	$(BK) python scripts/compare_prompt_reports.py $(A) $(B)

# ── Frontend ─────────────────────────────────────────────────────────────────
.PHONY: fe-lint fe-build
fe-lint:         ## ESLint the frontend
	cd frontend && pnpm lint
fe-build:        ## Type-check + production build
	cd frontend && pnpm build

# ── Aggregate ────────────────────────────────────────────────────────────────
.PHONY: check
check: test fe-lint  ## CI-parity offline checks (unit tests + FE lint)

.PHONY: help
help:            ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
