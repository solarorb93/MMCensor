@echo off
if not exist .\mmc-home.txt ( echo "Do not run this directly, only run bat files in the main MMC folder" && pause && exit 2 )
@echo on
call bat_files/check-prereqs.bat
call bat_files/uninstall-pytorch-cuda.bat
powershell -Command "Invoke-WebRequest https://github.com/winpython/winpython/releases/download/8.2.20240618final/Winpython64-3.12.4.1dot.exe -OutFile Winpython64-3.12.4.1dot.exe"
Winpython64-3.12.4.1dot.exe --help -o"winpython-pytorch-cuda" -y
del Winpython64-3.12.4.1dot.exe
call ./winpython-pytorch-cuda/WPy64-31241/scripts/env_for_icons.bat
pip install -r installation_helpers/requirements_pytorch-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.nvidia.com --cache-dir pip-cache/ --prefer-binary --log install-log-pytorch-cuda-verbose.txt --verbose
copy bat_files\run-pytorch-cuda.bat .\run-pytorch-cuda.bat
copy bat_files\uninstall-pytorch-cuda.bat .\uninstall-pytorch-cuda.bat
pause
