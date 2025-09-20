{
  pkgs,
  lib,
  stdenv,
  ...
}:
let
  pythonPackages = pkgs.python312Packages;
in
pkgs.mkShell {
  buildInputs = [
    pythonPackages.python
    pythonPackages.venvShellHook
    pkgs.autoPatchelfHook
    pythonPackages.pandas
    pythonPackages.matplotlib
    pythonPackages.seaborn
    pythonPackages.openpyxl
    pythonPackages.scikit-learn
    pythonPackages.lightgbm
    pythonPackages.optuna
    pythonPackages.numba
    pkgs.pandoc
    pkgs.texliveFull

  ];
  venvDir = "./.venv";
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -U jupyter
    autoPatchelf ./.venv
  '';
  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH=${lib.makeLibraryPath [ stdenv.cc.cc ]}
  '';
}
