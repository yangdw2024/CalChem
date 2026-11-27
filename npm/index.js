import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function binaryName() {
  return process.platform === 'win32' ? 'calchem_cli.exe' : 'calchem_cli';
}

function isRunnableFile(filePath) {
  try {
    if (!filePath) return false;
    const stat = fs.statSync(filePath);
    if (!stat.isFile()) return false;
    if (process.platform !== 'win32') {
      fs.accessSync(filePath, fs.constants.X_OK);
    }
    return true;
  } catch {
    return false;
  }
}

export function findBinary() {
  const exe = binaryName();

  const rawExplicit = process.env.CALCHEM_BIN?.trim();
  if (rawExplicit) {
    const explicit = path.resolve(rawExplicit);
    return isRunnableFile(explicit) ? explicit : null;
  }

  const localCandidates = [
    path.resolve(__dirname, 'bin', exe),
    path.resolve(__dirname, '..', 'build_target', 'release', exe),
    path.resolve(__dirname, '..', 'target', 'release', exe),
  ];
  for (const candidate of localCandidates) {
    if (isRunnableFile(candidate)) {
      return candidate;
    }
  }

  const pathEnv = process.env.PATH || '';
  for (const dir of pathEnv.split(path.delimiter)) {
    if (!dir) continue;
    const candidate = path.join(dir, exe);
    if (isRunnableFile(candidate)) {
      return candidate;
    }
  }

  return null;
}

export function isInstalled() {
  return findBinary() !== null;
}

export function runCalChem(args = [], options = {}) {
  const binPath = findBinary();
  if (!binPath) {
    throw new Error(
      `Native CalChem binary (${binaryName()}) was not found. Set CALCHEM_BIN or build with: cargo build --release -p calchem-cli`
    );
  }
  return spawn(binPath, args, {
    stdio: 'inherit',
    ...options,
  });
}

export default {
  findBinary,
  isInstalled,
  runCalChem,
};
