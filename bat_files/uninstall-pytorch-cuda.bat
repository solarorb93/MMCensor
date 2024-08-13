@echo off
if not exist .\mmc-home.txt ( echo "Do not run this directly, only run bat files in the main MMC folder" && pause && exit 2 )
@echo on
if exist .\winpython-pytorch-cuda\ rmdir .\winpython-pytorch-cuda\ /q /s
if exist run-pytorch-cuda.bat del /q run-pytorch-cuda.bat
REM taken from https://stackoverflow.com/questions/20329355/how-to-make-a-batch-file-delete-itself to delete the uninstall file as well
(goto) 2>nul & if exist uninstall-pytorch-cuda.bat del /q uninstall-pytorch-cuda.bat
