#!/usr/bin/env node

import { spawn, spawnSync } from 'node:child_process';
import process from 'node:process';
import path from 'node:path';

function isHealthyPython(cmd, args = []) {
  try {
    const res = spawnSync(cmd, [...args, '-c', 'import sys, ctypes; sys.exit(0)'], {
      stdio: 'ignore',
      timeout: 3000
    });
    return res.status === 0;
  } catch {
    return false;
  }
}

function findPython() {
  if (process.env.CALCHEM_PYTHON) {
    return { cmd: process.env.CALCHEM_PYTHON, args: [] };
  }

  const localAppData = process.env.LOCALAPPDATA || 'C:\\Users\\Administrator\\AppData\\Local';
  const py312 = path.join(localAppData, 'Programs', 'Python', 'Python312', 'python.exe');

  if (process.platform === 'win32') {
    if (isHealthyPython(py312)) {
      return { cmd: py312, args: [] };
    }
    if (isHealthyPython('py', ['-3.12'])) {
      return { cmd: 'py', args: ['-3.12'] };
    }
    if (isHealthyPython('python3')) {
      return { cmd: 'python3', args: [] };
    }
    if (isHealthyPython('python')) {
      return { cmd: 'python', args: [] };
    }
    if (isHealthyPython('py')) {
      return { cmd: 'py', args: [] };
    }
  } else {
    for (const cmd of ['python3', 'python']) {
      if (isHealthyPython(cmd)) {
        return { cmd, args: [] };
      }
    }
  }
  return null;
}

function checkCalChemInstalled(pyObj) {
  try {
    const res = spawnSync(pyObj.cmd, [...pyObj.args, '-c', 'import sys, calchem; sys.exit(0)'], {
      stdio: 'ignore',
      timeout: 5000
    });
    return res.status === 0;
  } catch {
    return false;
  }
}

const args = process.argv.slice(2);
const pyObj = findPython();

if (!pyObj) {
  console.error('Error: A working Python 3.10+ runtime was not found.');
  console.error('Please ensure Python is installed or set the CALCHEM_PYTHON environment variable.');
  process.exit(1);
}

// Handle top-level install shortcut: npx @yangdw2024/calchem install
if (args[0] === 'install') {
  console.log(`[CalChem] Installing or updating calchem-qc via pip (${pyObj.cmd})...`);
  const child = spawn(pyObj.cmd, [...pyObj.args, '-m', 'pip', 'install', '-U', 'calchem-qc'], {
    stdio: 'inherit'
  });
  child.on('exit', (code) => process.exit(code || 0));
} else {
  // Check if calchem-qc is installed, if not, prompt or auto-install
  if (!checkCalChemInstalled(pyObj)) {
    console.log('[CalChem] calchem-qc is not currently installed in this Python environment.');
    console.log('[CalChem] Automatically running: pip install calchem-qc ...');
    const inst = spawnSync(pyObj.cmd, [...pyObj.args, '-m', 'pip', 'install', 'calchem-qc'], {
      stdio: 'inherit'
    });
    if (inst.status !== 0) {
      console.error('[CalChem] Failed to install calchem-qc automatically.');
      console.error('Please run manually: pip install calchem-qc');
      process.exit(inst.status || 1);
    }
  }

  // Forward command directly to python -m calchem
  const child = spawn(pyObj.cmd, [...pyObj.args, '-m', 'calchem', ...args], {
    stdio: 'inherit'
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exit(code || 0);
    }
  });
}
