import os

file_names = []
for file in os.listdir("./input/pla"):
    file_names.append("./input/pla/" + file)

os.system(".\\focus-stack\\focus-stack.exe ./input/pla/* --output=output3.png --consistency=0 --align-keep-size --no-whitebalance --no-contrast")
#os.system(".\\focus-stack\\focus-stack.exe ./input/pla2/* --output=output3.png --consistency=0 --align-keep-size --no-whitebalance --no-contrast")
#os.system(".\\focus-stack\\focus-stack.exe output2.png output3.png --output=output4.png --consistency=0 --align-keep-size --no-whitebalance --no-contrast")