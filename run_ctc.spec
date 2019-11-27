# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(
    ["run_ctc.py"],
    pathex=["C:\\Users\\Martin\\Documents\\GitHub\\CTC_CytoCensus"],
    binaries=[],
    datas=[("C:\\Users\\Martin\\Documents\\GitHub\\CTC_CytoCensus\models", "models")],
    hiddenimports=[
        "pywt._extensions._cwt",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors",
        "sklearn.tree",
        "tifffile._tifffile",
        "sklearn.tree._utils",
        "sklearn.neighbors.quad_tree",
        "sklearn.neighbors.typedefs",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["pyqt", "qt", "PyQt5", "PyQT4"],
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
    name="run_ctc_cytocensus",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="run_ctc_cytocensus",
)
