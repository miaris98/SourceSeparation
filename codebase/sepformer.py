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

"""
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
"""

def main():
    train_loader, val_loader = LibriMix.loaders_from_mini(
        task = 'sep_clean', batch_size = 4)

    train_set, val_set = LibriMix.mini_from_download(task='sep_clean')
    print(train_set.df.values[799][2]) # path mix
    print(train_set.df.values[799][3]) # path s1
    print(train_set.df.values[799][4]) # path s2

    sampling_rate, samples = wavfile.read(train_set.df.values[799][2])
    mix_tensor = torch.tensor(samples.astype(float)).float()

    #mix_tensor =mix_tensor.type(torch.LongTensor)

    sepformer = separator.from_hparams(source="speechbrain/sepformer-wsj02mix",
                                   savedir='pretrained_models/sepformer-wsj02mix')

    #ConvTasNet = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k")

    ConvTasNet = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    #Awais = BaseModel.from_pretrained("Awais/Audio_Source_Separation")

    # for custom file, change path
    #s1,s2=ConvTasNet.forward(mix_tensor)
    est_sources_sep = sepformer.separate_file(train_set.df.values[0][2])


    torchaudio.save("sepformer1.wav", est_sources_sep[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("sepformer2.wav", est_sources_sep[:, :, 1].detach().cpu(), 8000)
    #display(Audio("sepformer1.wav")) works only on ipyd

    mix=ConvTasNet.separate(train_set.df.values[0][2], force_overwrite=True)
    print(mix)
   # Awais.separate("Awais-mixture.wav", force_overwrite=True)




if __name__ == '__main__':
    main()