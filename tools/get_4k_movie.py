import glob
import os
import json

json_root = "/vepfs-zulution/qiufeng/documents/code/MoviiDB/GeminiVideoCaption/temp/movies_4k"
video_root = "/cv/hanjinru/data/moviiDB/movies_4k/clips"
json_files = glob.glob(os.path.join(json_root, "**/*.json"), recursive=True)

data_list = []
for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)
        data['file_path'] = json_file
        data_list.append(data)

train_set_list = []
for data in data_list:
    info_dict = {}
    info_dict['file_path'] = data['file_path'].replace(json_root, video_root).replace(".json", ".mp4")
    info_dict['text'] = data['precise_prompt']
    info_dict['type'] = 'video'
    train_set_list.append(info_dict)

print(len(train_set_list))
if not os.path.exists("datasets/movie_4k"):
    os.makedirs("datasets/movie_4k")
with open("datasets/movie_4k/train_set.json", "w") as f:
    json.dump(train_set_list, f, indent=4)


