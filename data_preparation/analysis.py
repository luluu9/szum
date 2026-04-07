import glob

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns


# To tak średnio działało, brak informacji o dynamice utworu itp.
def extract_features_mfcc(file):
    y, sr = librosa.load(file)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=17), axis=1)
    return mfcc


def extract_features(file):
    target_duration = 10

    y, sr = librosa.load(file)

    target_length = target_duration * sr

    if len(y) > target_length:
        y = y[:target_length]
    else:
        # Padding zerami, może to ma wpływ na pliki typu noise, bo one są krótkie, ale z badania podobieństwa na to nie wychodzi
        y = np.pad(y, (0, target_length - len(y)), "constant", constant_values=(0, 0))

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    return mel_spec


def get_top_similar(files, features):
    sim_matrix = cosine_similarity(features)
    results = {}

    for i in range(len(files)):
        row = sim_matrix[i].copy()
        row[i] = -1

        top_indices = np.argsort(row)[-5:][::-1]

        results[files[i]] = [(files[j], sim_matrix[i, j]) for j in top_indices]

    return results


# Robienie heatmapy robione za pomocą gpt, nie uważam, żeby to było jakieś krytyczne, żeby samemu trzeba to było pisać ~ Kacper Chodubski


def create_heatmap(files, features):
    sim_matrix = cosine_similarity(features)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, vmin=0, vmax=1, cmap="viridis")

    plt.title("Macierz podobieństwa")
    plt.show()


# def create_heatmap(files, features):
#     sim_matrix = cosine_similarity(features)
#
#     plt.figure(figsize=(8, 6))
#     im = plt.imshow(sim_matrix, aspect="auto")
#     plt.colorbar(im, label="Cosine similarity")
#
#     plt.title("Macierz podobieństwa (cosine similarity)")
#     plt.xlabel("Utwory")
#     plt.ylabel("Utwory")
#
#     labels = [f.split("/")[-1] for f in files]
#     if len(labels) <= 20:  # żeby nie zrobić chaosu
#         plt.xticks(range(len(labels)), labels, rotation=90)
#         plt.yticks(range(len(labels)), labels)
#
#     plt.tight_layout()
#     plt.show()


def main():
    print("Start")

    dir_path = "./MERGED/Music/"
    files = glob.glob(dir_path + "*.wav")

    features = [extract_features(f) for f in files]
    features = np.array(features)
    features_flat = features.reshape(features.shape[0], -1)

    create_heatmap(files=files, features=features_flat)

    # similar_songs = get_top_similar(files, features_flat)
    #
    # seen_pairs = set()
    #
    # # gpt pomoc przy deduplikacji podobnych piosenek, problem symetryczności macierzy, a == b i b == a
    # for song, similar_list in similar_songs.items():
    #     for similar_song, score in similar_list:
    #         if score <= 0.9:
    #             continue
    #         pair = tuple(sorted([song, similar_song]))
    #
    #         if pair in seen_pairs:
    #             continue
    #         seen_pairs.add(pair)
    #         print(f"{pair[0]} == {pair[1]} -> {score:.4f}\n")
    print("End")


if __name__ == "__main__":
    main()
