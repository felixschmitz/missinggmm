#!/usr/bin/env bash
# PreToolUse hook (Bash): block agent-authored commits and PR descriptions that lack
# the `Co-Authored-By: Claude` trailer. `gh pr merge` is checked only when it passes an
# explicit body, since otherwise the squash commit takes the PR description.
command=$(jq -r '.tool_input.command // ""')

if grep -qE '(^|[;&|[:space:]])git[[:space:]]+commit([[:space:]]|$)' <<<"$command"; then
  kind="commit message"
elif grep -qE '(^|[;&|[:space:]])gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$)' <<<"$command"; then
  kind="PR description"
elif grep -qE '(^|[;&|[:space:]])gh[[:space:]]+pr[[:space:]]+merge([[:space:]]|$)' <<<"$command" \
  && grep -qE '(^|[[:space:]])(-b|--body|-F|--body-file)([[:space:]=]|$)' <<<"$command"; then
  kind="merge commit body"
else
  exit 0
fi

if ! grep -qE 'Co-Authored-By: Claude [^<]*<noreply@anthropic\.com>' <<<"$command"; then
  echo "Blocked: the $kind must end with 'Co-Authored-By: Claude <model name> <noreply@anthropic.com>' (see AGENTS.md, Commits). Pass the message inline with -m/--body so the trailer is visible." >&2
  exit 2
fi
