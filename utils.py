import csv
import yaml

KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]


def create_csv_file(filename: str = "results.csv"):
    """
    creates csv file with keypoints names as column names
    """
    keypoint_names = KEYPOINT_NAMES
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["FRAME#", *keypoint_names, ';'])


def save_results_to_csv(results: dict, new_file: bool = True, filename: str = None):
    """
    saves results in csv file
    """
    if filename is None:
        filename = "results.csv"

    if new_file:
        create_csv_file(filename)

    with open(filename, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            all_coords = []
            for keypoint, coords in value.items():
                all_coords.append([coord.flatten().tolist()[0] for coord in coords])
            writer.writerow([key, *all_coords, ';'])


def read_cfg_file(cfg_file):
    with open(cfg_file) as info:
        info_dict = yaml.load(info, Loader=yaml.FullLoader)
    return info_dict
