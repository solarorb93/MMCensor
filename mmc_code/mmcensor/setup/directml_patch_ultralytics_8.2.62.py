import ultralytics
import os

ultralytics_version = ultralytics.__version__

if ultralytics_version != '8.2.62':
    print( "**************************************************" )
    print( "**************************************************" )
    print( "**************************************************" )
    print( "**************************************************" )
    print( "Can't patch ultralytics for directml support." )
    print( "Expected to find ultralytics 8.2.62, but found %s"%ultralytics_version )
    print( "THIS WILL NOT WORK" )
    print( "**************************************************" )
    print( "**************************************************" )
    print( "**************************************************" )
    assert ultralytics_version == '8.2.62', "Incorrect ultralytics version"

patch_file = "../winpython-directml/WPy64-31241/python-3.12.4.amd64/Lib/site-packages/ultralytics/nn/autobackend.py"
backup_file = patch_file + ".backup"

os.rename( patch_file, backup_file )

in_file = open( backup_file, "r", encoding='utf-8' )
lines = in_file.readlines()
assert lines[184] == '            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))\n'
assert lines[190] == '            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]\n'

lines[184] = '            check_requirements(("onnx", "onnxruntime-directml"))\n'
lines[190] = '            providers = ["DmlExecutionProvider", "CPUExecutionProvider" ]\n'

out_file = open( patch_file, "w", encoding='utf-8' )
out_file.writelines(lines)
out_file.close

