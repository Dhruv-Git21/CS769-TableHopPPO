#!/usr/bin/env bash
# Confirms the orchestration pack is intact before starting Claude Code.
set -e
ok=1
need=(CLAUDE.md START_PROMPT.md STATUS.md Makefile requirements.txt .claude/settings.json
 .claude/agents/implementer.md .claude/agents/lit-researcher.md .claude/agents/experimenter.md .claude/agents/writer.md .claude/agents/reviewer.md
 .claude/commands/resume.md .claude/commands/gate.md .claude/commands/handoff.md
 docs/PROPOSAL.md docs/INTERIM_SUMMARY.md docs/CLAIMS.md
 specs/00_MASTER_PLAN.md specs/01_ARCHITECTURE.md specs/02_ENV_REWARD.md specs/03_BASELINES_EXPERIMENTS.md specs/04_METRICS_EVAL.md specs/05_PAPER_SPEC.md specs/06_DELIVERABLES.md
 tasks/BACKLOG.md tasks/TEMPLATE.md scripts/probe_env.sh)
for f in "${need[@]}"; do [ -f "$f" ] || { echo "MISSING $f"; ok=0; }; done
for a in .claude/agents/*.md; do grep -q '^model: sonnet' "$a" || { echo "agent not pinned to sonnet: $a"; ok=0; }; done
command -v claude >/dev/null || echo "WARN: claude CLI not on PATH"
[ $ok = 1 ] && echo "Setup OK" || { echo "Setup INCOMPLETE"; exit 1; }
