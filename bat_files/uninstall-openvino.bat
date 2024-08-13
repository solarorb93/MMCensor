@echo off
if not exist .\mmc-home.txt ( echo "Do not run this directly, only run bat files in the main MMC folder" && pause && exit 2 )
@echo on
if exist .\winpython-openvino\ rmdir .\winpython-openvino\ /q /s
if exist .\neuralnet_models\640m_openvino_model\ rmdir .\neuralnet_models\640m_openvino_model\ /q /s
if exist run-openvino.bat del /q run-openvino.bat
REM taken from https://stackoverflow.com/questions/20329355/how-to-make-a-batch-file-delete-itself to delete the uninstall file as well
(goto) 2>nul & if exist uninstall-openvino.bat del /q uninstall-openvino.bat
