from pathlib import Path
from subprocess import run
import numpy as np
import matplotlib.pyplot as plt
from mir_eval.beat import f_measure

# read audio with torch audio
import torchaudio


# load files in test_results/ folder

results_dir = Path("../debug_beattracking_test_results_uzvlxyrn")


ids_file = results_dir / "ids.txt"

with open(ids_file, "r") as f:
    ids = f.readlines()
    stems = [Path(i.strip()).stem for i in ids]


# execute this scp <host>:/data0/<user>/ssl-mtg/downstream_datasets/beattracking/genre_tzanetakis/audio/22kmono/pop/pop.00089.wav ../debug_beattracking_test_results_uzvlxyrn

audio_base_path = Path(
    "/data0/<user>/ssl-mtg/downstream_datasets/beattracking/genre_tzanetakis/audio/22kmono"
)

for k in range(4):
    genre, tid = stems[k].split(".")
    audio_name = f"{genre}.{tid}.wav"
    audio_file = results_dir / audio_name

    if not audio_file.exists():
        audio_path = audio_base_path / genre / audio_name
        run(["scp", f"<host>:{audio_path}", str(results_dir)])

    x_file = results_dir / f"input_{k}.npy"

    y_file = results_dir / f"y_true_sample_{k}.npy"
    y_estp_file = results_dir / f"y_proc_{k}.npy"

    y_est_file = results_dir / f"activations_{k}.npy"

    x = np.load(x_file)

    y = np.load(y_file)
    y_estp = np.load(y_estp_file)

    y_est = np.load(y_est_file)

    fm = f_measure(y, y_estp)

    audio, sr = torchaudio.load(str(audio_file))
    audio = audio.squeeze().numpy()

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"y_est shape: {y_est.shape}")
    print(f"y_estp shape: {y_estp.shape}")
    print(f"audio shape: {audio.shape}")
    print(f"F-measure: {fm:.3f}")

    # plot the results
    f, axs = plt.subplots(3, 1, figsize=(20, 9))
    axs[0].matshow(x.T, aspect="auto", cmap="viridis")

    fs = 22050
    ts = np.arange(len(audio)) / fs
    axs[1].plot(ts, audio, label="Audio")
    axs[1].set_xlim(0, ts[-1])  # Force x=0 alignment

    fs = 75 / 4
    ts = np.arange(len(y_est)) / fs
    axs[2].plot(ts, y_est, label="Activations")
    axs[2].set_xlim(0, ts[-1])  # Force x=0 alignment

    for i in range(1, 3):
        for j in y:
            axs[i].axvline(x=j, color="green", linestyle="--")
        for j in y_estp:
            axs[i].axvline(x=j, color="red", linestyle="--")

    # # Eliminate margins by setting limits and removing spacing
    # for ax in axs:
    #     # ax.margins(x=0, y=0)  # Remove any auto margins
    #     ax.spines["left"].set_position("zero")  # Align y-axis at x=0
    #     # ax.spines["bottom"].set_position("zero")  # Align x-axis at y=0

    # # # Remove subplot spacing
    # # plt.subplots_adjust(hspace=0)

    plt.savefig(results_dir / "figure.png", dpi=300)
    plt.close()
