@echo off
if not exist .\mmc-home.txt ( echo "Do not run this directly, only run bat files in the main MMC folder" && pause && exit 2 )
@echo on
call bat_files/check-prereqs.bat
call bat_files/uninstall-openvino.bat
powershell -Command "Invoke-WebRequest https://github.com/winpython/winpython/releases/download/8.2.20240618final/Winpython64-3.12.4.1dot.exe -OutFile Winpython64-3.12.4.1dot.exe"
Winpython64-3.12.4.1dot.exe --help -o"winpython-openvino" -y
del Winpython64-3.12.4.1dot.exe
call ./winpython-openvino/WPy64-31241/scripts/env_for_icons.bat
pip install -r installation_helpers/requirements_openvino.txt --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.nvidia.com --cache-dir pip-cache/ --prefer-binary --log install-log-openvino-verbose.txt --verbose
cd mmc_code
python mmcensor\setup\make_openvino.py
copy ..\bat_files\run-openvino.bat ..\run-openvino.bat
copy ..\bat_files\uninstall-openvino.bat ..\uninstall-openvino.bat
pause
