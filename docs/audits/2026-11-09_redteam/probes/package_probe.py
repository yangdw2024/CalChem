"""Use installed setuptools in isolated E: staging trees; no install, dependency resolution, or publish."""
import contextlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from setuptools import build_meta

ROOT=Path(r'E:\YDW\Trae_Gaussian');tempfile.tempdir=os.environ['TEMP']
OUT=Path(os.environ['TEMP']).parent/'package_probe';OUT.mkdir(exist_ok=True)
for include_binary in [False,True]:
    name='with_existing_binary' if include_binary else 'fresh_source'
    stage=OUT/name;stage.mkdir(exist_ok=True)
    for f in ['pyproject.toml', 'README.md', 'LICENSE', 'setup.py']:
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, stage / f)
    ignored=['__pycache__','*.pyc','*.pdb']+([] if include_binary else ['*.pyd','*.so','*.dylib'])
    shutil.copytree(ROOT/'calchem',stage/'calchem',ignore=shutil.ignore_patterns(*ignored),dirs_exist_ok=True)
    dist=stage/'dist';dist.mkdir(exist_ok=True)
    previous=Path.cwd()
    try:
        os.chdir(stage)
        try:
            with (stage/'build.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                artifact=build_meta.build_wheel(str(dist))
            with zipfile.ZipFile(dist/artifact) as z:
                metadata=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/WHEEL'))).decode()
                native=[n for n in z.namelist() if n.endswith(('.pyd','.so','.dylib'))]
            print(json.dumps({'stage':name,'wheel':str(dist/artifact),'native_modules':native,'wheel_metadata':metadata}),flush=True)
        except Exception as e:
            import traceback
            print(f"Error in {name}: {e}")
            traceback.print_exc()
    finally:
        os.chdir(previous)
