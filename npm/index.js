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

export function findPython() {
  if (process.env.CALCHEM_PYTHON) {
    return { cmd: process.env.CALCHEM_PYTHON, args: [] };
  }

  const localAppData = process.env.LOCALAPPDATA || 'C:\\Users\\Administrator\\AppData\\Local';
  const py312 = path.join(localAppData, 'Programs', 'Python', 'Python312', 'python.exe');

  if (process.platform === 'win32') {
    if (isHealthyPython(py312)) return { cmd: py312, args: [] };
    if (isHealthyPython('py', ['-3.12'])) return { cmd: 'py', args: ['-3.12'] };
    if (isHealthyPython('python3')) return { cmd: 'python3', args: [] };
    if (isHealthyPython('python')) return { cmd: 'python', args: [] };
    if (isHealthyPython('py')) return { cmd: 'py', args: [] };
  } else {
    for (const cmd of ['python3', 'python']) {
      if (isHealthyPython(cmd)) return { cmd, args: [] };
    }
  }
  return null;
}

export function isInstalled() {
  const pyObj = findPython();
  if (!pyObj) return false;
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

export function runCalChem(args = [], options = {}) {
  const pyObj = findPython();
  if (!pyObj) throw new Error('A working Python 3.10+ runtime is required.');
  return spawn(pyObj.cmd, [...pyObj.args, '-m', 'calchem', ...args], {
    stdio: 'inherit',
    ...options
  });
}

export default {
  findPython,
  isInstalled,
  runCalChem
};
