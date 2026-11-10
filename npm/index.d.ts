import { ChildProcess, SpawnOptions } from 'node:child_process';

export declare function getPythonPath(): string | null;
export declare function isInstalled(): boolean;
export declare function runCalChem(args?: string[], options?: SpawnOptions): ChildProcess;

declare const _default: {
  getPythonPath: typeof getPythonPath;
  isInstalled: typeof isInstalled;
  runCalChem: typeof runCalChem;
};

export default _default;
