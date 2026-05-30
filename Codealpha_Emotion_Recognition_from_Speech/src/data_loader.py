import os
import pandas as pd

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def load_dataset(dataset_path):

    paths = []
    emotions = []

    for actor in os.listdir(dataset_path):

        actor_path = os.path.join(dataset_path, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):

            emotion_code = file.split("-")[2]

            emotion = emotion_dict[emotion_code]

            paths.append(
                os.path.join(actor_path, file)
            )

            emotions.append(emotion)

    return pd.DataFrame({
        "path": paths,
        "emotion": emotions
    })