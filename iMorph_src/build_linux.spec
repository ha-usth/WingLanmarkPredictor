# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['../main.py'],
             pathex=[''],
             binaries=[],
             datas=[],
             hiddenimports=['greenlet', 'sklearn.utils.sparsetools._graph_validation', 'sklearn.utils.sparsetools._graph_tools', 'scipy.special._ufuncs_cxx', 'sklearn.utils.lgamma', 'sklearn.utils.weight_vector', 'sklearn.neighbors._typedefs', 'sklearn.utils._cython_blas', 'sklearn.neighbors._quad_tree', 'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
