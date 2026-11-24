# TableHop-PPO. `make help` lists targets. PY defaults to the CS769-tablehop env (see scripts/setup.sh).
ENV ?= $(HOME)/CS769-work/CS769-tablehop
PY ?= $(if $(wildcard $(ENV)/bin/python),$(ENV)/bin/python,$(if $(wildcard .venv/bin/python),.venv/bin/python,python3))
.PHONY: help setup data probe test lint smoke tables figures paper report slides demo demo-ask webapp release reproduce-smoke reproduce-all clean-pyc

help:
	@echo "setup          create the CS769-tablehop venv from requirements-lock.txt"
	@echo "data           download HybridQA + tables, build processed splits, run leakage audit"
	@echo "test / smoke   unit tests (CPU, synthetic) / tests + tiny training runs (~1-2 min)"
	@echo "reproduce-all  full study: cache, answer head, tuning, 3 seeds, eval, tables, figures (~12-15 h)"
	@echo "tables figures results/tables/*.csv|tex and results/figures/*.png|pdf from results/runs"
	@echo "demo / demo-ask  query -> hops -> evidence -> answer with the trained model (terminal)"
	@echo "webapp         same in the browser: http://127.0.0.1:8769"
	@echo "paper report slides  deliverables/paper.pdf, WORK_REPORT.docx|pdf, final_presentation.pptx"
	@echo "release        clean GitHub-ready copy in ../tablehop-ppo-release"


setup:
	bash scripts/setup.sh $(ENV)

data:
	bash scripts/download_data.sh
	$(PY) scripts/prepare_data.py
	$(PY) scripts/audit_leakage.py

probe:
	bash scripts/probe_env.sh

test:
	$(PY) -m pytest -x -q tests

lint:
	$(PY) -m ruff check src scripts tests

smoke:
	PY=$(PY) bash scripts/smoke.sh

tables:
	$(PY) scripts/summarize.py

figures:
	$(PY) scripts/make_figures.py

paper: tables figures
	cd paper && for J in paper_final:final paper_anonymous:review; do N=$${J%%:*}; M=$${J##*:}; \
	  pdflatex -interaction=nonstopmode -jobname=$$N "\\def\\ACLMODE{$$M}\\input{main.tex}" >/dev/null; bibtex $$N >/dev/null; \
	  pdflatex -interaction=nonstopmode -jobname=$$N "\\def\\ACLMODE{$$M}\\input{main.tex}" >/dev/null; \
	  pdflatex -interaction=nonstopmode -jobname=$$N "\\def\\ACLMODE{$$M}\\input{main.tex}" >/dev/null; done; \
	cp paper_final.pdf ../deliverables/paper.pdf && cp paper_anonymous.pdf ../deliverables/paper_anonymous.pdf

report: tables figures
	$(PY) scripts/make_report.py

slides: tables figures
	$(PY) scripts/slides/build_final_deck.py

demo:
	PY=$(PY) bash scripts/demo.sh

demo-ask:
	PY=$(PY) bash scripts/demo.sh ask

webapp:
	@test -f results/demo/showcase.json || $(PY) scripts/pick_showcase.py
	$(PY) scripts/webapp.py $(if $(RUN),--run $(RUN),)

release:
	$(PY) scripts/export_release.py --with-h1-checkpoints

reproduce-smoke: smoke

reproduce-all:
	PY=$(PY) bash scripts/reproduce_all.sh

clean-pyc:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
