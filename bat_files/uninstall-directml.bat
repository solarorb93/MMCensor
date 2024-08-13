@echo off
if not exist .\mmc-home.txt ( echo "Do not run this directly, only run bat files in the main MMC folder" && pause && exit 2 )
@echo on
if exist .\winpython-directml\ rmdir .\winpython-directml\ /q /s
if exist .\neuralnet_models\640m.onnx del /q .\neuralnet_models\640m.onnx
if exist run-directml.bat del /q run-directml.bat
REM taken from https://stackoverflow.com/questions/20329355/how-to-make-a-batch-file-delete-itself to delete the uninstall file as well
(goto) 2>nul & if exist uninstall-directml.bat del /q uninstall-directml.bat
