import pandas as pd

file_path = "./traffic_status.csv"

street_data = pd.read_csv(file_path)

street_data = street_data.rename(columns={
    "polyline.coordinates.0.0": "coor.0.0",
    "polyline.coordinates.0.1": "coor.0.1",
    "polyline.coordinates.1.0": "coor.1.0",
    "polyline.coordinates.1.1": "coor.1.1"
})

street_data = street_data.drop_duplicates()


def convert_str_to_fl(str):
    street_data[str] = [float(coor.replace(",", ".")) for coor in street_data[str]]


convert_str_to_fl("coor.0.0")
convert_str_to_fl("coor.0.1")
convert_str_to_fl("coor.1.0")
convert_str_to_fl("coor.1.1")


# function to check if los and vel are completely related
def check_same(los, vel):
    same = 0
    not_same = 0
    for i in range(len(street_data)):
        if street_data["los"][i] == los and street_data["velocity"][i] == vel:
            same += 1
        elif street_data["los"][i] == los and street_data["velocity"][i] != vel:
            not_same += 1
        elif street_data["los"][i] != los and street_data["velocity"][i] == vel:
            not_same += 1
    return same / (same + not_same)

# print(check_same("B", 30))


times = [d.split(" ")[1] for d in street_data["time"]]


# time gap = 5
def convert_time_to_int(str):
    parts = str.split(":")
    return (int(parts[0]) * 60 + int(parts[1]))//5


street_data["time"] = [convert_time_to_int(d) for d in times]


feature_names = ["coor.0.0", "coor.0.1",
                 "coor.1.0", "coor.1.1",
                 "time"]

################## Balance Data ##############
los = street_data.los
los_count = list(los.astype("category").value_counts())
print("----------------Data before balancing----------------")
print(los.astype("category").value_counts())
min_los_count = min(los_count)

street_data = street_data.sample(frac=1, random_state=1)

los_labels = list(set(los.astype("category").values))
all_sub_data = []
for label in los_labels:
    sub_data = street_data[street_data.los == label][:min_los_count]
    all_sub_data.append(sub_data)

new_street_data = pd.concat(all_sub_data)
new_street_data = new_street_data.sample(frac=1, random_state=1)

#################### Balance Done ############
X = new_street_data[feature_names]
y = new_street_data.los
##############################################

los_count = y.astype("category").value_counts()
print("----------------Data after balancing----------------")
print(los_count)
