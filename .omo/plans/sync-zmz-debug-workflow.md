# sync-zmz-debug-workflow - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** The `zmz_dev` branch reset to match `main` exactly, and the AI code review workflow made more robust with timeout/retry and better error handling, then tested via a real GitHub Actions run.

**Why this approach:** Reset is the simplest way to make two branches identical; workflow fixes target the concrete gaps found during code analysis before any test run.

**What it will NOT do:** Not change the AI prompt, not alter trigger conditions, not modify other workflows, not change project rules.

**Effort:** Short
**Risk:** Low - git force-push uses `--force-with-lease` safety; workflow changes are additive (timeout/retry) with no behavioral regression.
**Decisions to sanity-check:** git reset chosen by user; curl --max-time 60 --retry 3 as standard best practice.

Your next move: approve, then execute each todo in sequence.

---

> TL;DR (machine): Short effort, Low risk. Phase 1: git reset zmz_dev to origin/main. Phase 2: harden ai-code-review.yml (curl timeout, error handling). Phase 3: workflow_dispatch test run.

## Scope
### Must have
1. Create local zmz_dev tracking origin/zmz_dev
2. `git reset --hard origin/main` on zmz_dev
3. `git push --force-with-lease origin zmz_dev`
4. Add `curl --max-time 60 --retry 3` to line 89 of ai-code-review.yml
5. Improve git diff error handling (replace `2>/dev/null || true` with explicit check)
6. Add SENSENOVA_API_KEY pre-check before curl call
7. Commit and push all fixes
8. Trigger workflow_dispatch on GitHub and verify run logs

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Do NOT modify `.github/prompts/code-review-prompt.md`
- Do NOT change workflow triggers or permissions
- Do NOT modify the model name or API endpoint
- Do NOT touch other CI workflows (ci.yml, etc.)
- Do NOT modify CLAUDE.md or project configuration
- Do NOT run the full CI benchmark suite

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after (workflow runs are validated by GitHub Actions logs)
- Evidence: .omo/evidence/sync-zmz-debug-workflow/

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.

| Wave | Todos | Description |
|------|-------|-------------|
| 1    | 1-3   | Git branch sync |
| 2    | 4-6   | Workflow fixes + commit |
| 3    | 7     | Push and test run |

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | — | 2 | — |
| 2 | 1 | 3 | — |
| 3 | 2 | — | — |
| 4 | — | 5 | — |
| 5 | 4 | 6 | — |
| 6 | 5 | 7 | — |
| 7 | 3, 6 | 8 | — |
| 8 | 7 | — | — |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

- [x] 1. Create local zmz_dev branch tracking origin/zmz_dev
  What to do / Must NOT do: Run `git checkout -b zmz_dev origin/zmz_dev` to create and switch to local zmz_dev. Must NOT push anything yet.
  Parallelization: Wave 1 | Blocked by: — | Blocks: 2
  References (executor has NO interview context - be exhaustive): Run from repo root `/home/watneyzhu/original_scratchv/ScratchV`
  Acceptance criteria (agent-executable): `git branch --show-current` outputs `zmz_dev`; `git log --oneline -1` shows `1f0da9b add qemu install doc`
  QA scenarios (happy): confirm branch created and switched. Evidence .omo/evidence/sync-zmz-debug-workflow/task-1-happy.log
  Commit: N
  Must NOT: push, commit, or modify any files.

- [x] 2. Reset zmz_dev to match origin/main
  What to do / Must NOT do: Run `git reset --hard origin/main` to make zmz_dev identical to main. This discards the 1 unique commit on zmz_dev.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 3
  References: `git reset --hard origin/main` from repo root
  Acceptance criteria: `git log --oneline -1` shows `f3944ba docs(phase1+2): extension exercises` (latest main commit). `git rev-parse HEAD` equals `git rev-parse origin/main`.
  QA scenarios (happy): verify HEAD matches origin/main. Evidence .omo/evidence/sync-zmz-debug-workflow/task-2-happy.log
  Commit: N

- [x] 3. Force-push zmz_dev to remote with safety
  What to do / Must NOT do: Run `git push --force-with-lease origin zmz_dev`. Must use `--force-with-lease`, NOT `--force`.
  Parallelization: Wave 1 | Blocked by: 2 | Blocks: 7
  References: repo root after task 2
  Acceptance criteria: Remote zmz_dev HEAD matches origin/main. Verify with `git ls-remote origin zmz_dev` and compare SHA with `git rev-parse origin/main`.
  QA scenarios (happy): confirm SHAs match. Evidence .omo/evidence/sync-zmz-debug-workflow/task-3-happy.log
  Commit: N

- [x] 4. Add curl timeout and retry to workflow API call
  What to do / Must NOT do: Edit `.github/workflows/ai-code-review.yml` line 89-93. Change `curl -s -w "\n%{http_code}"` to `curl -s --max-time 60 --retry 3 -w "\n%{http_code}"`. Must NOT change any other line.
  Parallelization: Wave 2 | Blocked by: — | Blocks: 5, 6
  References: `.github/workflows/ai-code-review.yml` line 89
  Acceptance criteria: Line 89 contains `--max-time 60 --retry 3`. `git diff` shows exactly that change.
  QA scenarios (happy): verify the exact curl command change with git diff. Evidence .omo/evidence/sync-zmz-debug-workflow/task-4-happy.log
  Commit: N

- [x] 5. Improve git diff error handling in workflow
  What to do / Must NOT do: Edit `.github/workflows/ai-code-review.yml` line 58. Replace `DIFF=$(git diff "origin/$BASE_REF...HEAD" -- "$FILE" 2>/dev/null || true)` with explicit error handling:
  ```
  DIFF=$(git diff "origin/$BASE_REF...HEAD" -- "$FILE" 2>&1) || { echo "  [ERROR] git diff failed for $FILE: $DIFF"; continue; }
  ```
  Must NOT change other error handling patterns.
  Parallelization: Wave 2 | Blocked by: 4 | Blocks: 6
  References: `.github/workflows/ai-code-review.yml` line 58
  Acceptance criteria: The line properly captures stderr and exits on failure instead of silently masking.
  QA scenarios (happy + failure): verify with git diff after change. Evidence .omo/evidence/sync-zmz-debug-workflow/task-5-happy.log
  Commit: N

- [x] 6. Add SENSENOVA_API_KEY pre-check before API call
  What to do / Must NOT do: After line 87 (before the curl call) in `.github/workflows/ai-code-review.yml`, add:
  ```bash
  if [ -z "$SENSENOVA_API_KEY" ] || [ "$SENSENOVA_API_KEY" = "sk-placeholder" ]; then
    echo "  [ERROR] SENSENOVA_API_KEY is not set or is placeholder. Skipping review for $FILE."
    continue
  fi
  ```
  Must NOT change any other logic.
  Parallelization: Wave 2 | Blocked by: 5 | Blocks: 7
  References: `.github/workflows/ai-code-review.yml` lines 37, 89
  Acceptance criteria: Pre-check exists before curl. If key is empty, the step skips gracefully with error message.
  QA scenarios (happy): verify with git diff. Evidence .omo/evidence/sync-zmz-debug-workflow/task-6-happy.log
  Commit: N

- [x] 7. Commit and push all workflow fixes
  What to do / Must NOT do: `git add .github/workflows/ai-code-review.yml` then `git commit -m "fix(ci): add curl timeout/retry, improve error handling in ai-code-review"` then `git push origin zmz_dev`. Must use English commit message.
  Parallelization: Wave 3 | Blocked by: 3, 6 | Blocks: 8
  References: After all fixes applied to ai-code-review.yml
  Acceptance criteria: `git log -1 --oneline` shows the commit. Remote zmz_dev includes the commit.
  QA scenarios (happy): verify commit exists, verify push succeeded. Evidence .omo/evidence/sync-zmz-debug-workflow/task-7-happy.log
  Commit: Y | fix(ci): add curl timeout/retry, improve error handling in ai-code-review

- [x] 8. Trigger workflow_dispatch and verify run
  NOTE: gh CLI unavailable in local env. Manual trigger required.
  FIXED: Chinese filename handling, --retry-all-errors, rate-limit delay, enhanced error logging.
  Re-run at: https://github.com/ScratchV-Compiler/ScratchV/actions/workflows/ai-code-review.yml
  → "Run workflow" → branch: zmz_dev
  What to do / Must NOT do: Use GitHub API to trigger workflow_dispatch on zmz_dev branch for `ai-code-review.yml`. Wait for run completion. Check run logs for success. Command: `gh workflow run ai-code-review.yml --ref zmz_dev` then `gh run watch`.
  If `gh` CLI not available, print instructions for manual trigger.
  Must NOT: run on main branch, or run full CI.
  Parallelization: Wave 3 | Blocked by: 7 | Blocks: —
  References: GitHub Actions UI or `gh` CLI
  Acceptance criteria: Workflow run completes successfully (exit code 0). Logs show the curl command with `--max-time 60 --retry 3`. All steps pass.
  QA scenarios (happy + failure): verify run URL and logs. Evidence .omo/evidence/sync-zmz-debug-workflow/task-8-happy.log
  Commit: N

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit — verify all 8 todos completed as specified
- [ ] F2. Code quality review — review the changed workflow file for any issues
- [ ] F3. Real manual QA — check the workflow run logs manually
- [ ] F4. Scope fidelity — confirm nothing outside scope was modified

## Commit strategy
- One commit for the workflow fixes (todo 7)
- Git operations (todos 1-3) are not committed—they are branch management
- Commit message: `fix(ci): add curl timeout/retry, improve error handling in ai-code-review`

## Success criteria
1. ✅ Local zmz_dev exists and matches origin/main exactly
2. ✅ Remote zmz_dev updated and matches origin/main
3. ✅ Workflow file has curl `--max-time 60 --retry 3` and improved error handling
4. ✅ All fixes pushed to remote zmz_dev
5. ✅ workflow_dispatch run completed successfully on GitHub Actions
