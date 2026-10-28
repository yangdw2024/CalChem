# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['calchem.backends.native', 'calchem.backends.gaussian', 'calchem.backends.orca', 'calchem.backends.pyscf', 'calchem.backends.xtb', 'calchem.backends.rest', 'calchem.core', 'calchem.jobs', 'calchem.api.service', 'calchem.config', 'calchem.io']
hiddenimports += collect_submodules('calchem')


a = Analysis(
    ['calchem/gui/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[('calchem', 'calchem')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CalChem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CalChem',
)
