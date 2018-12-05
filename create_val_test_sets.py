# This script can be run to move about 10 of our samples to each validation and test sets
from random import randint
import os.path
import shutil

for i in range(62):
    string_i = str(i)
    if len(string_i) == 1:
        string_i = "0" + string_i
    os.mkdir("data/val/" + string_i)
    os.mkdir("data/test/" + string_i)
    # for j in range(20):
    #     file = "data/train/" + string_i + "/" + str(i*110+randint(0, 110)) + ".png"
    #     if os.path.isfile(file):
    #         if j < 10:
    #             new_file = "data/val/" + string_i + "/" + str(i*110+randint(0, 110)) + ".png"
    #         else:
    #             new_file = "data/test/" + string_i + "/" + str(i*110+randint(0, 110)) + ".png"
    #         shutil.move(file, new_file)
    #     else:
    #         print(i)
