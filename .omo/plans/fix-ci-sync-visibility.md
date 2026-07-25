# Plan: Fix CI Sync Visibility

## Intent

**CLEAR** — 修复 `.github/workflows/ci.yml` 中 sync 步骤的 `2>/dev/null` 问题，让 checkout 结果可观测。

## Background

CI 自托管 runner 使用 `/opt/ScratchV` 本地 mirror（不更新），配合 `git fetch origin main --depth=1` 尝试拉取最新代码。但：

1. `git fetch` 网络不稳定，有时失败
2. `2>/dev/null` 吞掉了 checkout 成功/失败的所有输出
3. 无法判断 CI 实际测试的是 merge commit 还是 fallback 到 origin/main

## Changes

### File: `.github/workflows/ci.yml`

两处（test job + benchmark job）sync 步骤，L34-35 和 L82-83。

**Before:**
```yaml
git fetch origin main --depth=1 2>/dev/null || echo "WARNING: git fetch failed, using local mirror"
git checkout -f "$GITHUB_SHA" 2>/dev/null || git checkout -f origin/main 2>/dev/null || true
```

**After:**
```yaml
echo "=== CI Sync: checkout target ==="
echo "GITHUB_SHA=$GITHUB_SHA"
echo "GITHUB_EVENT_NAME=$GITHUB_EVENT_NAME"
MIRROR=/opt/ScratchV
WORKSPACE=/opt/actions-runner/_work/ScratchV/ScratchV
rm -rf "$WORKSPACE"
cp -a "$MIRROR" "$WORKSPACE"
cd "$WORKSPACE"
echo "Fetching origin main..."
git fetch origin main --depth=1 || echo "::warning::git fetch origin main failed"
echo "Checkout target: $GITHUB_SHA"
if git checkout -f "$GITHUB_SHA"; then
  echo "::notice::Checkout successful: $(git log -1 --format='%h %ai %s')"
else
  echo "::warning::GITHUB_SHA ($GITHUB_SHA) not found in local repo — falling back to origin/main"
  git checkout -f origin/main
  echo "::warning::FALLBACK: CI is testing origin/main: $(git log -1 --format='%h %ai %s')"
fi
```

### Key differences

| Before | After |
|--------|-------|
| `git fetch ... 2>/dev/null` | `git fetch ... \|\| echo "::warning::..."` — fetch 失败时高亮告警 |
| `checkout ... 2>/dev/null \|\| ... 2>/dev/null` | `if checkout; then notice; else warning + fallback; fi` — 成功/失败都有日志 |
| fallback 无声无息 | fallback 时输出 `FALLBACK:` 并显示实际 checkout 的 commit |
| 无 checkout commit 信息 | checkout 成功后用 `git log -1` 显示 commit hash + 时间 + message |

## Verification

提交后触发一次 CI（如 push 空 commit 到 zmz_dev），日志应看到：

```
=== CI Sync: checkout target ===
GITHUB_SHA=abc123..
GITHUB_EVENT_NAME=pull_request
Fetching origin main...
Checkout target: abc123..
::notice:: Checkout successful: ad8f7ca 2026-07-25 11:56:28 +0800 fix(ci): improve...
```

若 fallback：

```
::warning::GITHUB_SHA (abc123..) not found in local repo — falling back to origin/main
::warning::FALLBACK: CI is testing origin/main: 997d2aa 2026-06-28 20:51:00 +0800 [FIX] Test...
```

## Commit

```
fix(ci): make sync step checkout result visible in logs

- Remove 2>/dev/null from git fetch and git checkout
- Add ::notice:: for successful checkout with commit info
- Add ::warning:: for fetch failure and fallback
- Applies to both test and benchmark jobs
```
