#!/usr/bin/env bash
# PreToolUse hook (Bash): block agent-authored commits and PR descriptions whose command
# text lacks a `Co-Authored-By: Claude` trailer. Requires `jq`; without it the hook
# blocks rather than silently skipping the check.
#
# Only commands that start a shell line or follow `;`, `&`, `|` or `(` are inspected, so
# quoted text such as `echo "git commit"` is ignored. `gh pr merge` is checked only when
# it passes an explicit body, since otherwise the merge commit takes the PR description.
if ! command -v jq >/dev/null; then
  echo "Blocked: jq is required by .claude/hooks/require-coauthor-trailer.sh." >&2
  exit 2
fi

command=$(jq -r '.tool_input.command // ""')
start='(^|[;&|(])[[:space:]]*'
git_global='([[:space:]]+(-C|-c)[[:space:]]+[^[:space:]]+|[[:space:]]+--[^[:space:]]+)*'

if grep -qE "${start}git${git_global}[[:space:]]+commit([[:space:]]|\$)" <<<"$command"; then
  # An amend that keeps the existing message keeps its trailer.
  if grep -qE -- '--amend' <<<"$command" && grep -qE -- '--no-edit' <<<"$command"; then
    exit 0
  fi
  kind="commit message"
elif grep -qE "${start}gh[[:space:]]+pr[[:space:]]+create([[:space:]]|\$)" <<<"$command"; then
  kind="PR description"
elif grep -qE "${start}gh[[:space:]]+pr[[:space:]]+merge([[:space:]]|\$)" <<<"$command" \
  && grep -qE -- '(^|[[:space:]])(-b|--body|-F|--body-file)([[:space:]=]|$)' <<<"$command"; then
  kind="merge commit body"
else
  exit 0
fi

if ! grep -qE 'Co-Authored-By: Claude [^<]*<noreply@anthropic\.com>' <<<"$command"; then
  echo "Blocked: the $kind must include 'Co-Authored-By: Claude <model name> <noreply@anthropic.com>' (see AGENTS.md, Commits). Pass the message inline with -m/--body so the trailer is visible." >&2
  exit 2
fi
