import os

new_file = []

current_path = os.getcwd()

with open("./tmp/test_koreantravel.csv", "r+") as f:
    for line in f:
        listed = line.split(",")
        dir_path = f"{current_path}/tmp/data/{listed[4]}/{listed[5]}/{listed[6][:-1]}"
        
        #os.makedirs(dir_path)
        print(dir_path)

