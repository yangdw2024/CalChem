---
name: package-release-pipeline
description: >-
  Standardized end-to-end workflow to verify, freeze, tag, and publish CalChem
  (pure Rust + optional CUDA sm_120 native CLI and zero-Python npm CLI wrapper)
  across GitHub (Repositories, Tags, Releases, Actions) and npm. Use whenever
  releasing, publishing, deploying, or packaging CalChem to GitHub or npm.
---

# CalChem Package Release Pipeline (GitHub & npm)

This skill defines the standardized, zero-drift workflow for releasing **CalChem** — a pure Rust + optional CUDA (`sm_120`) native binary (`calchem_cli.exe`) paired with a zero-Python Node.js CLI wrapper (`@yangdw2024/calchem` in `npm/`).

> **Distribution Status**: PyPI (`calchem-py`) was permanently retired in `v0.2.3`. Do **not** build wheels, `sdist`, or Python extension packages unless explicitly resurrected by architecture decision. Active channels are **GitHub Releases** and **npm**.

---

## 1. Quick Decision Matrix

| Target Channel | Artifacts | Core Tools | Auth / Secret Handling |
|---|---|---|---|
| **GitHub** | Clean Git commit, annotated tag `v<ver>`, Windows x86_64 `.zip` + `SHA256SUMS.txt`, Release Notes | `git`, `gh` (`C:\Program Files\GitHub CLI\gh.exe` fallback on Windows), [`scripts/create_github_release.py`](./scripts/create_github_release.py) | `gh auth` or out-of-repo token (see [`references/github_api_secrets.md`](./references/github_api_secrets.md)) |
| **npm** | Scoped zero-Python CLI wrapper (`npm/bin/calchem.js`, `npm/index.js`, `npm/index.d.ts`, `npm/package.json`) | `npm`, `npx`, [`scripts/wait_npm_propagation.py`](./scripts/wait_npm_propagation.py) | Out-of-repo local private file + ephemeral `--userconfig` destroyed in `finally` |

---

## 2. Pre-Release Integrity Gate (Mandatory Before Tagging)

All checks below must pass **sequentially** (never run concurrent Cargo builds/tests) with `RAYON_NUM_THREADS=4`, `OMP_NUM_THREADS=1`:

1. **Single Frozen Release Binary & Zero Drift**:
   - Confirm no file under `crates/` has `LastWriteTimeUtc` newer than `build_target/release/calchem_cli.exe` (`NewerThanExeCount == 0`).
   - Record the exact SHA256 of `calchem_cli.exe` (and `.cubin` if CUDA is built) and verify all final benchmark/audit outputs share that exact SHA256.
   - Freeze `crates/` once final audit completes until release tagging and publishing finish.
2. **Test & Lint Gates**:
   - `cargo test --workspace --no-fail-fast` (explicitly list any `#[ignore]` tests in release records).
   - `cargo clippy --workspace --all-targets` (`0` errors).
   - `python -m unittest discover -s E:\YDW\Trae_Gaussian\scripts -p "test_*.py" -v` (`verify_calchem_job.py` contract suite).
3. **Version Parity Check**:
   - Align version across workspace `Cargo.toml`, `crates/calchem-cli/Cargo.toml`, `npm/package.json`, and `README.md` / `npm/README.md`.
4. **Honesty Gate for Release Notes**:
   - Slow items, unsupported combinations, GPU small-molecule launch overhead, CPU/GPU thread counts, and accuracy tolerances must be listed **side-by-side** with verified speedups. Never omit failures, `#[ignore]` tests, or method boundaries.

---

## 3. GitHub Distribution Workflow (Windows-Safe)

### Step 3.1: Commit, Tag & Push
```bash
git status --short
git tag -a v<version> -m "Release v<version>"
git push origin master
git push origin v<version>
```

### Step 3.2: Package Release Archive & Checksums
Stage release archive (`calchem_cli-v<version>-windows-x86_64.zip`) and `SHA256SUMS.txt` cleanly without polluting the repository working tree:
- Package the audited `calchem_cli.exe` (and required runtime files/docs if applicable).
- Generate `SHA256SUMS.txt` matching exact uppercase/lowercase hex digests and byte sizes.

### Step 3.3: Create GitHub Release (UTF-8 File Notes, Safe `gh.exe` Lookup)
On Windows:
- `gh.exe` may not be in default `PATH`; check `Get-Command gh` and fall back to `C:\Program Files\GitHub CLI\gh.exe`.
- Always write release notes to a **UTF-8 without BOM** file and pass `--notes-file <path>` (never pass multi-line Chinese/Markdown `--notes` strings inline on PowerShell CLI, which risks mojibake or quoting corruption).
- Alternatively use [`scripts/create_github_release.py`](./scripts/create_github_release.py) and verify uploaded asset names, sizes, and digests via `gh release view v<version>`.

---

## 4. npm Publishing Workflow (Zero-Python CLI Wrapper)

For native Rust/CUDA engines, npm distributes a lightweight Node.js wrapper that locates the user's native binary cleanly via `CALCHEM_BIN` or `PATH`. See [`references/npm_cli_wrapper_pattern.md`](./references/npm_cli_wrapper_pattern.md) for layout guidance.

### Step 4.1: Manifest & Strict Binary Resolution Rules
1. **Scoped Package Visibility & Bin Entry (`npm/package.json`)**:
   ```json
   {
     "bin": {
       "calchem": "bin/calchem.js"
     },
     "publishConfig": {
       "access": "public"
     }
   }
   ```
   *(No `./` prefix on `bin/calchem.js`, preventing npm normalization warnings.)*
2. **Fail-Fast `CALCHEM_BIN` Contract (`npm/index.js` & `npm/bin/calchem.js`)**:
   - No Python probing or `pip install` fallback exists.
   - If the caller explicitly sets `CALCHEM_BIN`, verify that exact binary exists/executes and **fail fast with a clear non-zero error** if invalid — never silently fall back to another binary on `PATH` or stale paths.
3. **Dry-Run Tarball Inspection**:
   ```bash
   cd npm && npm pack --dry-run
   ```
   Confirm zero stray artifacts, logs, `.env`, or secret files appear in the packed tarball.

### Step 4.2: Credential Isolation & Ephemeral `--userconfig` Publish
**Security Red Line**: Never store publish tokens inside `E:\` repository files, never hardcode secret paths or token values in docs/scripts, and never persist `_authToken` in `~/.npmrc`.

Follow this exact ephemeral publish-and-purge sequence (PowerShell):
1. Read the token from an **out-of-repo local private file** (stored outside any Git repo).
2. Write a temporary config file (`$tmpNpmrc` in temp storage) containing `//registry.npmjs.org/:_authToken=<TOKEN>` with newline encoding.
3. Execute inside `try { ... } finally { ... }`:
   - Verify auth: `npm --userconfig $tmpNpmrc whoami`
   - Publish: `npm --userconfig $tmpNpmrc publish --access public` (inside `npm/`)
4. In the mandatory `finally` block:
   - Delete `$tmpNpmrc` (`Remove-Item -Force -ErrorAction SilentlyContinue`).
   - Clear any accidental user-level token: `npm config delete //registry.npmjs.org/:_authToken` (suppress harmless missing-key exit if unset).
   - Verify default state is unauthenticated again: run `npm whoami` (without `--userconfig`) and confirm it **exits non-zero** (`POST_CLEAR_WHOAMI_RC != 0`).

### Step 4.3: Post-Publish Verification & PowerShell Stderr Pitfall
1. Wait for npm edge CDN propagation (`npm view <pkg>@<ver> dist.shasum` or [`scripts/wait_npm_propagation.py`](./scripts/wait_npm_propagation.py)).
2. Smoke-test end-user execution: `npx --yes @yangdw2024/calchem@<ver> --version`.
3. **PowerShell 5.1 `stderr` Trap**: Native CLIs like `calchem_cli.exe --help` write to `stderr` even when exiting `0`, which causes PowerShell with `$ErrorActionPreference = 'Stop'` to throw `NativeCommandError`. Wrap CLI verification checks via `cmd /c "... >nul 2>&1"` and inspect `$LASTEXITCODE`.

---

## 5. Post-Release Verification Checklist

- [ ] **Git**: Working tree clean (`git status --short` empty); `master` and `v<ver>` pushed to `origin`.
- [ ] **GitHub Release**: Tag `v<ver>` release published with Windows x86_64 `.zip` + `SHA256SUMS.txt` and honest release notes (both gains and known boundaries/slow items).
- [ ] **npm**: `@yangdw2024/calchem@<ver>` live with `latest` tag, clean tarball verified, `npx` smoke test exits `0`.
- [ ] **Secret Hygiene**: Temporary `--userconfig` removed, `~/.npmrc` clean, default `npm whoami` confirmed non-zero exit.
