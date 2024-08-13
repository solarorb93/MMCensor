@echo off
if not exist .\mmc-home.txt ( echo "Do not run this directly, only run bat files in the main MMC folder" && pause && exit 2 )
@echo on
call ./winpython-pytorch-cuda/WPy64-31241/scripts/env_for_icons.bat
set mmcNNenv=pytorch-cuda
cd mmc_code
python mmcensor_realtime.py
pause
