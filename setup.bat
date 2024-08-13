@echo off
ECHO Welcome to the setup program for MMCensor
ECHO You can install this program with various neural-net backends,
ECHO depending on your system.  These are listed here, from fastest
ECHO to slowest.  Install the best option for your system.  If you
ECHO have issues, you can always go to a slower option and see if
ECHO that works.
echo.
ECHO 1. If you have a modern NVidia Graphics Card (16xx, 20xx, or later),
ECHO enter 1 to install the tensorrt backend.
echo.
ECHO 2. If you have a NVidia 9xx or 10xx Graphics Card, enter 2 to install
ECHO a vanilla PyTorch CUDA backend.
echo.
ECHO 3. If you have a modern AMD Radeon Card (probably anything with RX
ECHO in the name, but I'm not sure), enter 3 to install a DirectML backend.
echo.
ECHO 4. Otherwise, enter 4 to install an OpenVino CPU backend.  Openvino
ECHO is optimized for Intel processors, but shouldn't be any downside on
ECHO AMD processors.
echo.
ECHO 5. If for some reason none of the above options work, enter 5 to install
ECHO a vanilla PyTorch CPU backend, which is the slowest but most compatible
ECHO backend.
echo.
set okay=0
set /p installChoice=Enter a number (1, 2, 3, 4, or 5):
echo.
if %installChoice%==1 set okay=1 && call "./bat_files/setup-tensorrt.bat" 2>&1 | "./bat_files/tee.bat" install-log-tensorrt.txt
if %installChoice%==2 set okay=1 && call "./bat_files/setup-pytorch-cuda.bat" 2>&1 | "./bat_files/tee.bat" install-log-pytorch-cuda.txt
if %installChoice%==3 set okay=1 && call "./bat_files/setup-directml.bat" 2>&1 | "./bat_files/tee.bat" install-log-directml.txt
if %installChoice%==4 set okay=1 && call "./bat_files/setup-openvino.bat" 2>&1 | "./bat_files/tee.bat" install-log-openvino.txt
if %installChoice%==5 set okay=1 && call "./bat_files/setup-pytorch-cpu.bat" 2>&1 | "./bat_files/tee.bat" install-log-pytorch-cpu.txt
if %okay%==0 echo Please enter 1, 2, 3, 4, or 5.  You did not enter a valid choice.
echo.
set /p dummy="Press enter to continue..."
