# PyInstaller spec for CalChem GUI (one-folder)
# Build: pyinstaller calchem.spec

block_cipher = None

a = Analysis(
    ['calchem/gui/__main__.py'],
    pathex=['E:\\YDW\\Trae_Gaussian'],
    binaries=[],
    datas=[
        ('calchem', 'calchem'),
    ],
    hiddenimports=[
        'calchem.backends.native',
        'calchem.backends.gaussian',
        'calchem.backends.orca',
        'calchem.backends.pyscf',
        'calchem.backends.xtb',
        'calchem.backends.rest',
        'calchem.core',
        'calchem.jobs',
        'calchem.api.service',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CalChem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='CalChem',
)
