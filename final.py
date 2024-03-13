import os
import cv2
import json
import pandas as pd
from tqdm import tqdm
from deepface_match import predict_one, verify2


BASE_PATH = '/home/asaeedi/Desktop/bankwork/'
CSV_NAME = 'ehraz.csv'
DATA_PATH = 'data'
DATA_PATH = os.path.join(BASE_PATH, DATA_PATH)

# data
df = pd.read_csv(os.path.join(BASE_PATH, CSV_NAME))
# df = df[df['liveness'].notna()]
print(df.shape)
uids = list(df['r_id'].unique())
verification_results = {}

for uid in tqdm(uids):
    # do stm here
    person_df = df[df['r_id'] == uid]
    person_dir = os.path.join(DATA_PATH, uid)
    # find video
    image_path = [os.path.join(person_dir, item) for item in os.listdir(person_dir) if not item.endswith('.mp4')][0]
    video_path = [os.path.join(person_dir, item) for item in os.listdir(person_dir) if item.endswith('.mp4')][0]
    image = cv2.imread(image_path)
    # embedding 1
    emb1 = predict_one(image)
    # read vid
    cap = cv2.VideoCapture(video_path)
    # add skipping value here.
    # process every 10 frames
    process_this_frame = 59
    ret, frame = cap.read()
    verification_percentage = 0
    total_processed_frames = 0
    while ret:
        process_this_frame += 1
        ret, frame = cap.read()
        if process_this_frame % 60 == 0:
            # do the compare
            try:
                res = verify2(emb1, frame)
                if res['verified']:
                    verification_percentage += 1
                total_processed_frames += 1
            except Exception as e:
                print('no faces!')
    cap.release()
    try:
        verification_results[uid] = verification_percentage / total_processed_frames
    except ZeroDivisionError:
        print('video was not opened!')

with open("verification.json", "w") as outfile: 
    json.dump(verification_results, outfile)
