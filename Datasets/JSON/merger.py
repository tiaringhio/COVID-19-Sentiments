import os  # for manipulates files and subdirectories
import json  # handle json files

json_folder_path = os.getcwd()
# In order to get the list of all files that ends with ".json"
# we will get list of all files, and take only the ones that ends with "json"
json_files = [x for x in os.listdir(json_folder_path) if x.endswith("json")]
json_data = list()
for json_file in json_files:
    json_file_path = os.path.join(json_folder_path, json_file)
    with open(json_file_path, "r") as f:
        json_data.append(json.load(f))

# now after iterate all the files, we have the data in the array, so we can just write it to file
output_path = os.path.join(json_folder_path, "data.json")
with open(output_path, "w") as f:
    json.dump(json_data, f)
