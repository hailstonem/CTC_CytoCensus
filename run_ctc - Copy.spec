# -*- mode: python -*-

block_cipher = None

# pylint: disable=E0603,E0602
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="run_ctc_cytocensus",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=True,
)
