with open("category.csv", "r") as f:
    for line in f:
        l = line.split(",")
        print(l[-1])
