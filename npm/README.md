# @yangdw2024/calchem

> Cross-platform Node.js CLI wrapper for **CalChem**, a pure Rust (+ optional CUDA) quantum chemistry engine (`calchem_cli`).

[![npm version](https://img.shields.io/npm/v/@yangdw2024/calchem.svg)](https://www.npmjs.com/package/@yangdw2024/calchem)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

`@yangdw2024/calchem` locates and invokes the native `calchem_cli` binary (`calchem_cli.exe` on Windows) with zero Python dependency.

Binary lookup order:
1. `process.env.CALCHEM_BIN`
2. `./bin/calchem_cli[.exe]`, `../build_target/release/calchem_cli[.exe]`, `../target/release/calchem_cli[.exe]`
3. `calchem_cli[.exe]` on `PATH`

## Prerequisites

Build the native CLI binary from the repository root (or provide `CALCHEM_BIN`):

```bash
cargo build --release -p calchem-cli
```

## Quick Start

```bash
# Show CLI help
npx @yangdw2024/calchem --help

# Run a single-point calculation with structured JSON output
npx @yangdw2024/calchem molecule.xyz --task energy --method rks --xc b3lyp --basis "6-31g(d,p)" --grid fine --disp d3bj --json --out result.json
```

### Node.js API Usage

```javascript
import { findBinary, isInstalled, runCalChem } from '@yangdw2024/calchem';

console.log('Binary path:', findBinary(), 'installed:', isInstalled());

const child = runCalChem([
  'water.xyz',
  '--task', 'energy',
  '--method', 'rks',
  '--xc', 'b3lyp',
  '--basis', '6-31g(d,p)',
  '--json',
]);
child.on('exit', (code) => {
  console.log('Process exited with code:', code);
});
```

## Implemented Native Capabilities & Boundaries

- **SCF**: RHF, UHF, RKS, UKS (`SVWN5`, `BLYP`, `PBE`, `B3LYP`, `B3LYP5`, `PBE0`, `CAM-B3LYP`, `wB97X`, `wB97X-D`), Stored/Direct/RI-J (`auto-jfit` / `def2-jfit-f`).
- **GPU**: Optional CUDA acceleration covers 3-center RI-J integrals (`L <= 2`, falling back to CPU when polarized auxiliary F-shells are present), RI-J contractions, and DFT XC grid quadrature. Hybrid exact exchange `K` and analytical gradients run on CPU.
- **Properties & Solvation**: Gas-phase analytic gradients, Cartesian BFGS optimization, numerical harmonic frequencies, AO-Direct TDA/Full-LR TD-DFT, and SCRF single-point solvation (`C-PCM`, `IEF-PCM`, `COSMO`, `SMD`) using `Bondi (alpha=1.20)` defaults for PCM.

See the root [README.md](../README.md) and [docs/NATIVE_METHOD_BOUNDARIES.md](../docs/NATIVE_METHOD_BOUNDARIES.md) for verified benchmark accuracy and method boundaries.

## Links

- **GitHub Repository**: [https://github.com/yangdw2024/CalChem](https://github.com/yangdw2024/CalChem)
- **License**: MIT
