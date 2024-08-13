@echo off
if not exist .\mmc-home.txt ( echo "Do not run this directly, only run bat files in the main MMC folder" && pause && exit 2 )
@echo on
call bat_files/check-prereqs.bat
call bat_files/uninstall-pytorch-cpu.bat
powershell -Command "Invoke-WebRequest https://github.com/winpython/winpython/releases/download/8.2.20240618final/Winpython64-3.12.4.1dot.exe -OutFile Winpython64-3.12.4.1dot.exe"
Winpython64-3.12.4.1dot.exe --help -o"winpython-pytorch-cpu" -y
del Winpython64-3.12.4.1dot.exe
call ./winpython-pytorch-cpu/WPy64-31241/scripts/env_for_icons.bat
pip install -r installation_helpers/requirements_pytorch-cpu.txt --cache-dir pip-cache/ --prefer-binary --log install-log-pytorch-cpu-verbose.txt --verbose
copy bat_files\run-pytorch-cpu.bat run-pytorch-cpu.bat
copy bat_files\uninstall-pytorch-cpu.bat uninstall-pytorch-cpu.bat
pause
