from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import torch
import asteroid
from asteroid.models import BaseModel
from asteroid.data import LibriMix
from scipy.io import wavfile
import numpy as np
from IPython.display import display, Audio
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display

def main():
    train_loader, val_loader = LibriMix.loaders_from_mini(
        task = 'sep_clean', batch_size = 4)

    train_set, val_set = LibriMix.mini_from_download(task='sep_clean')
    #print(train_set.df.values[799][2]) # path mix
    #print(train_set.df.values[799][3]) # path s1
    #print(train_set.df.values[799][4]) # path s2
    print(train_set.df.values[0][2])

    sepformer = separator.from_hparams(source="speechbrain/sepformer-wsj02mix",
                                   savedir='pretrained_models/sepformer-wsj02mix')

    ConvTasNet = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    DPRNNTasNet= BaseModel.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean")

    est_sources_sep = sepformer.separate_file(train_set.df.values[0][2])

    torchaudio.save("sepformer1.wav", est_sources_sep[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("sepformer2.wav", est_sources_sep[:, :, 1].detach().cpu(), 8000)

    ConvTasNet.separate(train_set.df.values[0][2], output_dir="C:\\Users\\miari\\Desktop\\repositories\\SourceSeparation\\codebase\\ConvTasNet" , force_overwrite=True)
    DPRNNTasNet.separate(train_set.df.values[0][2],output_dir="C:\\Users\\miari\\Desktop\\repositories\\SourceSeparation\\codebase\\DPRNNTasNet" ,  force_overwrite=True)

    est1 = sf.read("ConvTasNet/5400-34479-0005_4973-24515-0007_est1.wav")[0]
    est2 = sf.read("ConvTasNet/5400-34479-0005_4973-24515-0007_est2.wav")[0]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    show_magspec(est1, sr=8000, ax=ax[0])
    show_magspec(est2, sr=8000, ax=ax[1])
    plt.show()

    anechoic_sampled_mixture, _ = torchaudio.load("ConvTasNet/5400-34479-0005_4973-24515-0007_est1.wav")
    waveform = anechoic_sampled_mixture.detach().numpy()[0]
    plt.plot(waveform)
    plt.title("Anechoic Mixture")
    plt.show()
    plt.close()
    plt.specgram(waveform)
    display(Audio(waveform, rate=8000))
    plt.show()
def show_magspec(waveform, **kw):
    return librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(waveform))),
        y_axis="log", x_axis="time",
        **kw
    )

if __name__ == '__main__':
    main()


