from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import torch
import asteroid
from asteroid.models import sudormrf
from asteroid.models import BaseModel
from asteroid.data import LibriMix
from scipy.io import wavfile
import numpy as np
from IPython.display import display, Audio
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
#from sudormrf import  SuDORMRFImprovedNet,SuDORMRFNet
# from asteroid.masknn import SuDORMRF, SuDORMRFImproved
# from asteroid.utils.torch_utils import script_if_tracing
# from asteroid.models.base_models import BaseEncoderMaskerDecoder replace imports with these in codebase\sudormrf from asteroid librira



def main():
    train_loader, val_loader = LibriMix.loaders_from_mini(
        task = 'sep_clean', batch_size = 4)

    train_set, val_set = LibriMix.mini_from_download(task='sep_clean')
    #print(train_set.df.values[799][2]) # path mix
    #print(train_set.df.values[799][3]) # path s1
    #print(train_set.df.values[799][4]) # path s2

    print(train_set.df.values[0][2])

    #load models
    sepformer = separator.from_hparams(source="speechbrain/sepformer-wsj02mix",
                                   savedir='pretrained_models/sepformer-wsj02mix')

    ConvTasNet = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    DPRNNTasNet= BaseModel.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean")
    Sudoimprovednet=BaseModel.from_pretrained('pretrained_models/best_model.pth')
    path="C:\\Users\\miari\\Desktop\\repositories\\SourceSeparation\\codebase\\"

    #separate
    est_sources_sep = sepformer.separate_file(val_set.df.values[0][2])
    torchaudio.save("sepformer1.wav", est_sources_sep[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("sepformer2.wav", est_sources_sep[:, :, 1].detach().cpu(), 8000)
    ConvTasNet.separate(val_set.df.values[0][2], output_dir=path+"ConvTasNet" , force_overwrite=True)
    DPRNNTasNet.separate(val_set.df.values[0][2],output_dir=path+"DPRNNTasNet" ,  force_overwrite=True)
    Sudoimprovednet.separate(val_set.df.values[0][2],output_dir=path+"Sudo" , resample=True,  force_overwrite=True)
    name="3752-4944-0000_5694-64025-0004_est1.wav"


    #view waveforms
    estse1 = sf.read("sepformer1.wav")[0]
    estse2 = sf.read("sepformer2.wav")[0]

    estc1 = sf.read("ConvTasNet/"+name)[0]
    estc2 = sf.read("ConvTasNet/"+name)[0]

    estd1 = sf.read("DPRNNTasNet/"+name)[0]
    estd2 = sf.read("DPRNNTasNet/"+name)[0]

    ests1 = sf.read("Sudo/"+name)[0]
    ests2 = sf.read("Sudo/"+name)[0]



    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    show_magspec(estse1, sr=8000, ax=ax[0])
    show_magspec(estse2, sr=8000, ax=ax[1])
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    show_magspec(estc1, sr=8000, ax=ax[0])
    show_magspec(estc2, sr=8000, ax=ax[1])
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    show_magspec(estd1, sr=8000, ax=ax[0])
    show_magspec(estd2, sr=8000, ax=ax[1])
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    show_magspec(ests1, sr=8000, ax=ax[0])
    show_magspec(ests2, sr=8000, ax=ax[1])
    plt.show()


    anechoic_sampled_mixture, _ = torchaudio.load("sepformer1.wav")
    waveform = anechoic_sampled_mixture.detach().numpy()[0]
    plt.plot(waveform)
    plt.title("Anechoic Mixture")
    plt.show()
    plt.close()
    plt.specgram(waveform)
    display(Audio(waveform, rate=8000))
    plt.show()

    anechoic_sampled_mixture, _ = torchaudio.load("ConvTasNet/"+name)
    waveform = anechoic_sampled_mixture.detach().numpy()[0]
    plt.plot(waveform)
    plt.title("Anechoic Mixture")
    plt.show()
    plt.close()
    plt.specgram(waveform)
    display(Audio(waveform, rate=8000))
    plt.show()

    anechoic_sampled_mixture, _ = torchaudio.load("DPRNNTasNet/"+name)
    waveform = anechoic_sampled_mixture.detach().numpy()[0]
    plt.plot(waveform)
    plt.title("Anechoic Mixture")
    plt.show()
    plt.close()
    plt.specgram(waveform)
    display(Audio(waveform, rate=8000))
    plt.show()

    anechoic_sampled_mixture, _ = torchaudio.load("Sudo/"+name)
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


