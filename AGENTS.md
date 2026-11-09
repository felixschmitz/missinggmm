First read `.ai-instructions/AGENTS.md`.

@.ai-instructions/profiles/tier-a.md

# missinggmm

## Overview

Installable Python package implementing the Abrevaya and Donald (2017) GMM estimator for
linear regression with missing regressors. The GMM estimator is the main contribution.
Complete-case, dummy-variable and Dagenais FGLS estimators are comparison baselines.
Depends only on `numpy` at runtime. The replication project (Monte Carlo study, paper)
lives in a separate repository and may later depend on this package.

## Build & Test

- `pixi run tests` — run tests
- `pixi run prek run --all-files` — lint, format, and type-check

## Architecture

`src/missinggmm/` is the importable package. Public API to be defined; keep helpers
private (leading underscore) until stable.

## Commits

Every commit message ends with the co-author trailer given in the session's attribution
instructions, e.g. `Co-Authored-By: Claude <model name> <noreply@anthropic.com>`. For
squash merges, put the trailer in the PR description so it lands in the squashed commit.
The `PreToolUse` hook in `.claude/hooks/` blocks Claude Code commits and PR descriptions
that lack it.
