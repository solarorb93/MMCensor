@echo off
if not exist .\mmc-home.txt ( echo "Do not run this directly" && pause && exit 2 )
if not exist .\neuralnet_models\640m.pt (
	echo.
	echo ***********************************************************************************
	echo ******************************** MISSING FILE *************************************
	echo ###################################################################################
	echo #                                                                                 #
	echo # You are missing the NudeNet neural net weights file.                            #
	echo # This file is called 640m.pt and must be placed                                  #
	echo # in the /neuralnet_models/ folder.                                               #
	echo #                                                                                 #
	echo # Download this file from the NudeNet github at                                   #
	echo # https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.pt    #
	echo #                                                                                 #
	echo # If that link doesn't work, look for the "pytorch link"                          #
	echo # for the "640m" model on the main NudeNet github site:                           #
	echo # https://github.com/notAI-tech/NudeNet                                           #
	echo #                                                                                 #
	echo # Download the 640m.pt file, which should be approximately                        #
	echo # 50MB, and save it in the /neuralnet_models/ folder                              #
	echo #                                                                                 #
	echo # THIS PROGRAM WILL NOT WORK UNTIL YOU DO THIS.                                   #
	echo #                                                                                 #
	echo ###################################################################################
	echo.
	pause
	exit 2
)

