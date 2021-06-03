import numpy as np
import os, csv

# read client and video files
with open("meta/uploaderId.txt") as f:
    client_list = [x.strip() for x in f.readlines()]
with open("meta/manifest.txt") as f:
    video_list = [x.strip() for x in f.readlines()]

# read labels
labels = ["bedroom", "kitchen", "bathroom", "living-room", "dining-room", "none-of-the-above"]
label_list = np.load("meta/scene_category/scene_category_full.npy")

# video to client+label mapping
client_to_videos = {}
for client, video, label in zip(client_list, video_list, label_list):
    if client not in client_to_videos:
        client_to_videos[client] = []
    client_to_videos[client].append((video, label))


# Remove clients with less than 7 samples
client_dict = {}
for c in client_list:
    count = client_dict.get(c, 0)
    client_dict[c] = count + 1
selected_clients = [k for k, v in client_dict.items() if v >=7]




ratios = [(0.8, 'train'), (0.05, 'val'), (0.15, 'test')]

base_index = 0
video_count = 0
for ratio, folder in ratios:
    temp_index = base_index

    results = [['client_id', 'data_path', 'label_name', 'label_id']]
    client_idx = 0
    for i in range(temp_index, temp_index+int(ratio*len(selected_clients))):
        for sample in client_to_videos[selected_clients[i]]:
            results.append([client_idx, sample[0], labels[sample[1]], sample[1]])
            video_count += 1
                # move data

        client_idx += 1
        base_index += 1
    csv_output = 'client_data_mapping/'+folder+'.csv'
    with open(csv_output, "w") as fout:
        writer = csv.writer(fout, delimiter=',')
        writer.writerow(['client_id' , 'data_path' , 'label_name', 'label_id'])
        for r in results:
            writer.writerow(r)
print(len(selected_clients))
print(video_count)