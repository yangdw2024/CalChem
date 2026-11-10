# @yangdw2024/calchem

> Cross-platform CLI & Node.js wrapper for **CalChem**: Unified quantum chemistry computational engine with Rust numerical core.

[![npm version](https://img.shields.io/npm/v/@yangdw2024/calchem.svg)](https://www.npmjs.com/package/@yangdw2024/calchem)
[![PyPI version](https://img.shields.io/pypi/v/calchem-qc.svg)](https://pypi.org/project/calchem-qc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

`@yangdw2024/calchem` allows developers, automated CI pipelines, and computational workflows to invoke **CalChem** directly from the terminal or Node.js without manual environment configuration.

If the underlying `calchem-qc` Python engine is not installed, the wrapper automatically installs it from PyPI.

## Quick Start

### Run directly with `npx` (No installation needed)

```bash
# View help
npx @yangdw2024/calchem --help

# Check environment & backends
npx @yangdw2024/calchem doctor
npx @yangdw2024/calchem backends

# Run calculation
npx @yangdw2024/calchem run input.json
```

### Global Installation

```bash
npm install -g @yangdw2024/calchem

calchem doctor
calchem run input.json
```

### Node.js API Usage

```javascript
import { runCalChem, isInstalled } from '@yangdw2024/calchem';

console.log('CalChem installed:', isInstalled());

const child = runCalChem(['run', 'water.json']);
child.on('exit', (code) => {
  console.log('Process exited with code:', code);
});
```

## Features

- **Hartree-Fock (RHF / UHF)**: Obara-Saika recurrence integrals, Schwarz screening, DIIS convergence.
- **Density Functional Theory (DFT)**: LDA, GGA (PBE, BLYP), Hybrids (B3LYP, PBE0), Range-Separated Hybrids (CAM-B3LYP, ωB97X-D).
- **Dispersion Corrections**: Standard Grimme D3(BJ) with fractional coordination number interpolation.
- **Solvation Models**: CPCM / SMD self-consistent reaction field with analytic gradients.
- **Geometry Optimization & Frequencies**: BFGS / RFO optimization, TS search, IRC verification, IR vibrational intensities.

## Links

- **GitHub Repository**: [https://github.com/yangdw2024/CalChem](https://github.com/yangdw2024/CalChem)
- **PyPI Package**: [https://pypi.org/project/calchem-qc/](https://pypi.org/project/calchem-qc/)
- **License**: MIT
