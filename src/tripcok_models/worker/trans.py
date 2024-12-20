import csv

CSV_FILE = "./translate.csv"
CSV_WRITE = "category.csv"
with open(CSV_FILE) as csvfile:
    reader = csv.reader(csvfile)
    csv_list = list(reader)

for col in range(len(csv_list)):
    for i in range(4):
        del csv_list[col][0]

csv.writer(open(CSV_WRITE, "w")).writerows(csv_list)
