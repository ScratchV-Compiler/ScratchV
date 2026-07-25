---
slug: sync-zmz-debug-workflow
status: drafting
intent: clear
review_required: false
pending-action: write .omo/plans/sync-zmz-debug-workflow.md
approach: Two-phase: (1) Git branch sync — reset zmz_dev to match origin/main; (2) Debug workflow — fix identified issues in ai-code-review.yml, then trigger a workflow_dispatch test run.

---

# Draft: sync-zmz-debug-workflow

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
1. Git branch sync: zmz_dev reset to origin/main, force-pushed | active
2. Workflow fix: apply robustness improvements to ai-code-review.yml | active
3. Workflow test: trigger workflow_dispatch and verify | active

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
- Git sync strategy: `git reset --hard origin/main` on zmz_dev, then `git push --force-with-lease` → explicitly chosen by user
- Default merge vs rebase question was bypassed by user's explicit choice of reset

## Findings (cited - path:lines)

### Git state
- Current branch: `main`, worktree clean
- `origin/zmz_dev` exists; local `zmz_dev` does NOT exist
- main is 31 commits ahead, 1 commit behind origin/zmz_dev
- The 1 unique zmz_dev commit is `1f0da9b add qemu install doc`

### Workflow issues (ai-code-review.yml — 208 lines)
All findings from exploration:
1. **No curl timeout/retry** (line 89): `curl -s` without `--max-time` or `--retry` — API hang could exceed 15-min timeout
2. **Non-standard API endpoint** (line 89-93): `token.sensenova.cn` with model `deepseek-v4-flash` — not standard DeepSeek API. Must verify reachability.
3. **Silent error suppression** (line 58): `2>/dev/null || true` masks git errors silently
4. **workflow_dispatch no comment** (lines 169-173): Manual dispatch runs produce no PR comment — expected design, but review output only in runner logs
5. **BASE_REF fallback** (line 25): For workflow_dispatch, falls back to `main` — `origin/main` may be stale if not fetched

Referenced files verified:
- `.github/prompts/code-review-prompt.md` ✅ exists (63 lines)
- External actions: `actions/checkout@v4` ✅, `actions/github-script@v7` ✅

## Decisions (with rationale)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | `git reset --hard origin/main` on zmz_dev | User explicitly chose this. Discards the 1 unique commit, makes zmz_dev byte-identical to main. |
| D2 | `git push --force-with-lease` instead of `--force` | Safer: aborts if remote has new commits since fetch. |
| D3 | Add curl `--max-time 60 --retry 3` | Prevents runner hangs on slow API; retries transient failures. |
| D4 | Improve error logging: preserve stderr from git diff | Gives debug info when git operation fails. |
| D5 | Test via workflow_dispatch after fixes | Full end-to-end validation in actual GitHub Actions environment. |

## Scope IN

1. **Git phase**: Create local `zmz_dev` tracking branch → `git reset --hard origin/main` → `git push --force-with-lease origin zmz_dev`
2. **Workflow fixes**:
   - Add `--max-time 60 --retry 3` to curl call
   - Improve error handling/logging for git diff failures
   - Add `SENSENOVA_API_KEY` presence check before calling API
   - Any other robustness improvements uncovered during fix
3. **Test run**: Commit fixes → push → trigger `workflow_dispatch` on GitHub → verify run logs

## Scope OUT (Must NOT have)

- Do NOT change workflow trigger conditions (keep workflow_dispatch + pull_request)
- Do NOT modify the AI prompt (`.github/prompts/code-review-prompt.md`)
- Do NOT change the model or API endpoint unless it proves unreachable
- Do NOT modify any other workflows (ci.yml etc.)
- Do NOT change CLAUDE.md or any project rules

## Open questions

None — all forks resolved by user input or exploration.

## Approval gate
status: awaiting-approval
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
