import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
import psutil

d = Path(__file__).resolve().parent
root = d.parents[2]
def run(label, geom, xc, exe, threads):
    inp = Path('E:/YDW/ChemDatas/Gjf') / (geom + '.gjf')
    args = [str(exe), str(inp), '--rij','--gpu','--grid','coarse','--xc',xc,'--basis','sto-3g']
    env = dict(os.environ, CALCHEM_PROFILE='1', RAYON_NUM_THREADS=str(threads))
    start = time.monotonic()
    peak = 0
    with (d / (label+'.log')).open('w') as output:
        p = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT, env=env)
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                mem = proc.memory_info()
                peak = max(peak, mem.peak_wset)
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.25)
    result = dict(args=args, threads=threads, exit=p.returncode, wall_s=time.monotonic()-start, peak_working_set_bytes=peak,
                  input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(), binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (d / (label+'.json')).write_text(json.dumps(result, indent=2))
    return p.returncode
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('geom')
    parser.add_argument('xc')
    parser.add_argument('exe', nargs='?', type=Path, default=root / 'build_target/release/calchem_cli.exe')
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    raise SystemExit(run(args.label, args.geom, args.xc, args.exe, args.threads))
