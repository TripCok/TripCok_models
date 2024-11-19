new_file = []


with open("./tmp/test2.csv", "r+") as f:
    for line in f:
        
        line = "".join(line.split())
        listed = line.split(",")
        #print(listed)
        listed.pop(1)
        listed.pop(1)
        listed.pop(1)
        #print(listed)

        listed = ','.join(listed)
        print(listed)
        new_file.append(listed)

with open("changed.csv", "w") as f2:
    f2.writelines(new_file)
