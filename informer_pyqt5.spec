# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['informer_pyqt5.py','cnn_ui0831.py',"C:/Users/31498/Desktop/Informer2020-main_20230906/data/data_loader.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/data/utils/masking.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/data/utils/metrics.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/data/utils/timefeatures.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/data/utils/tools.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/exp/exp_basic.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/exp/exp_informer_pyqt.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/models/attn.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/models/decoder.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/models/embed.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/models/encoder.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/models/model.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/utils/masking.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/utils/metrics.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/utils/timefeatures.py",
    "C:/Users/31498/Desktop/Informer2020-main_20230906/utils/tools.py"],
    pathex=["C:/Users/31498/Desktop/Informer2020-main_20230906"],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='卫星异常检测软件HIT(cpu测试版)',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="C:/Users/31498/Desktop/Informer2020-main_20230906/HIT.ico"
)
