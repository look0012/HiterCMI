import csv

def StorFile(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["True Label", "Prediction"])
        writer.writerows(data)
    print(f"Results stored in {filename}")
