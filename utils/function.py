import os
import csv

def make_datapath_list(root_path):
    video_list = list()

    for folder_name in os.listdir(root_path):
        video_img_directory_path = os.path.join(root_path, folder_name)
        video_list.append(video_img_directory_path)

    return video_list

def get_label_id_dictionary(label_dicitionary_path='./data/label_dictionary.csv'):
    label_id_dict = {}
    id_label_dict = {}

    with open(label_dicitionary_path, encoding="utf-8") as f:

        reader = csv.DictReader(f, delimiter=",", quotechar='"')

        for row in reader:
            label_id_dict.setdefault(row["label"], int(row["label_id"]))
            id_label_dict.setdefault(int(row["label_id"]), row["label"])

    return label_id_dict,  id_label_dict