import csv

with open("t3.csv", "r") as f:
    reader = csv.reader(f)
    cnt = 0
    for row in reader: cnt+=1
    print(cnt)
