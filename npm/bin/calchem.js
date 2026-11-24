#!/usr/bin/env node

import process from 'node:process';
import { findBinary, runCalChem } from '../index.js';

const args = process.argv.slice(2);
const binPath = findBinary();

if (!binPath) {
  const exe = process.platform === 'win32' ? 'calchem_cli.exe' : 'calchem_cli';
  console.error(`Error: Native CalChem binary (${exe}) was not found.`);
  console.error('Set CALCHEM_BIN to its full path, place it in PATH, or build from source:');
  console.error('  cargo build --release -p calchem-cli');
  process.exit(1);
}

const child = runCalChem(args);
child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 0);
  }
});
