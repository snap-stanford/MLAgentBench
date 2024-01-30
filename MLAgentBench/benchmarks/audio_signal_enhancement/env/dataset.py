import torch, torchaudio

class MixIT_train_data(torch.utils.data.Dataset):
    
    """
    A dataset class for the training set for MixIT, where each example is a triplet consisting of  
    noisy speech, noise, and their sum. 

    Args:
        frame_len: The frame length for the short-time Fourier transform.
        wav_len: The waveform length of each audio clip.
        fname: The path to the configuration file for the training set.
        clean_path: The directory path of the clean speech dataset. (not used)
        noise_path: The directory path of the noise dataset. (not used)

    Returns:
        feat: The power-law compressed magnitude spectrogram of the sum of a noisy speech example and 
            a noise example in the short-time Fourier transform domain. (NB: This is unnecessary as it
            can be computed from mixture_stft and is to be removed in future updates.)
        mixture_stft: The sum of the noisy speech example and the noise example in the short-time 
            Fourier transform domain.
        noise_stft: The noise example in the short-time Fourier transform domain.
        noisy_stft: The noisy speech example in the short-time Fourier transform domain.
    """
    
    # Note for future updates: This is unnecessary and can be merged with PU_train_data.

    def __init__(self, frame_len, wav_len, fname, clean_path, noise_path):

        super().__init__()
        self.frame_len = frame_len
        self.wav_len = wav_len
        self.clean_path = clean_path
        self.noise_path = noise_path

        self.clean_fnames = []
        self.noise_fnames = []
        self.start_times = []
        self.SNRs = []
        self.noise_fnames2 = []
        self.start_times2 = []
        with open(fname, 'r') as f:
            for line in f.readlines():
                toks = line.split(' ')
                self.clean_fnames.append(toks[0])
                self.noise_fnames.append(toks[1])
                self.start_times.append(float(toks[2]))
                self.SNRs.append(float(toks[3]))
                self.noise_fnames2.append(toks[4])
                self.start_times2.append(float(toks[5]))

    def __len__(self):

        return len(self.clean_fnames)

    def __getitem__(self, idx):
        noise_wav, _ = torchaudio.load(self.noise_fnames[idx], frame_offset = int(16000 * self.start_times[idx]), 
                                       num_frames = self.wav_len)
        noise_wav = (noise_wav - torch.mean(noise_wav)) / torch.std(noise_wav)
        noise_wav = torch.squeeze(noise_wav)

        noise_wav2, _ = torchaudio.load(self.noise_fnames2[idx], frame_offset = int(16000 * self.start_times2[idx]), 
                                        num_frames = self.wav_len)
        noise_wav2 = (noise_wav2 - torch.mean(noise_wav2)) / torch.std(noise_wav2)
        noise_wav2 = torch.squeeze(noise_wav2)

        clean_wav, sfreq = torchaudio.load(self.clean_fnames[idx])
        if sfreq != 16000:
            resampler = torchaudio.transforms.Resample(sfreq, 16000)
            clean_wav = resampler(clean_wav)
        clean_wav = torch.squeeze(clean_wav)
        if clean_wav.size(0) < self.wav_len:
            clean_wav = torch.cat([clean_wav, torch.zeros(self.wav_len - clean_wav.size(0))], dim = 0)
        elif clean_wav.size(0) > self.wav_len:
            clean_wav = clean_wav[:self.wav_len]
        clean_wav = clean_wav - torch.mean(clean_wav)

        factor = 10. ** (self.SNRs[idx] / 20.) / torch.sqrt(torch.sum(clean_wav ** 2) / torch.sum(noise_wav2 ** 2))
        clean_wav = clean_wav * factor

        noisy_wav = clean_wav + noise_wav2
        noisy_wav /= torch.std(noisy_wav)

        mixture_wav = noisy_wav + noise_wav
        mixture_wav /= torch.std(mixture_wav)

        noise_stft = torch.stft(input = noise_wav,
                                n_fft = self.frame_len,
                                hop_length = self.frame_len // 4,
                                window = torch.hamming_window(self.frame_len),
                                return_complex = True)
        noisy_stft = torch.stft(input = noisy_wav,
                                n_fft = self.frame_len,
                                hop_length = self.frame_len // 4,
                                window = torch.hamming_window(self.frame_len),
                                return_complex = True)
        mixture_stft = torch.stft(input = mixture_wav,
                                  n_fft = self.frame_len,
                                  hop_length = self.frame_len // 4,
                                  window = torch.hamming_window(self.frame_len),
                                  return_complex = True)

        feat = torch.abs(mixture_stft) ** 1/15
        feat = feat[None, :, :].to(torch.float32)

        return feat, mixture_stft[None, :, :], noise_stft[None, :, :], noisy_stft[None, :, :]


class PU_train_data(torch.utils.data.Dataset):
        
    """
    A dataset class for the training set for PULSE, where each example is either noisy speech or 
    noise. The examples with even indices are noisy speech examples, which are considered to be 
    unlabelled data. The examples with odd indices are noise examples, which are considered to be 
    positive examples.

    Args:
        frame_len: The frame length for the short-time Fourier transform.
        wav_len: The waveform length of each audio clip.
        fname: The path to the configuration file for the training set.
        clean_path: The directory path of the clean speech dataset. (not used)
        noise_path: The directory path of the noise dataset. (not used)

    Returns:
        feat: The power-law compressed magnitude spectrogram of mixture_stft. (NB: This is unnecessary 
            as it can be computed from mixture_stft and is to be removed in future updates.)
        mask: A binary mask indicating whether each time-frequency component is a positive example (1)
            or an unlabelled example (0). For the even indices, this is a matrix of all zeros. For the 
            odd indices, this is a matrix of all ones. 
        mixture_stft: Noisy speech or noise in the short-time Fourier transform domain.
        mixture_stft: This is a dummy variable and not used. (This is used just to make sure that 
            there are four outputs for the sake of consistency with other dataset classes.)
    """

    def __init__(self, frame_len, wav_len, fname, clean_path, noise_path):

        super().__init__()
        self.frame_len = frame_len
        self.wav_len = wav_len
        self.clean_path = clean_path
        self.noise_path = noise_path

        self.clean_fnames = []
        self.noise_fnames = []
        self.start_times = []
        self.SNRs = []
        self.noise_fnames2 = []
        self.start_times2 = []
        with open(fname, 'r') as f:
            for line in f.readlines():
                toks = line.split(' ')
                self.clean_fnames.append(toks[0])
                self.noise_fnames.append(toks[1])
                self.start_times.append(float(toks[2]))
                self.SNRs.append(float(toks[3]))
                self.noise_fnames2.append(toks[4])
                self.start_times2.append(float(toks[5]))
                
    def __len__(self):

        return len(self.clean_fnames) * 2

    def __getitem__(self, idx):

        if idx % 2 == 1: # The examples with odd indices are noises (considered to be positive examples)
            noise_wav, _ = torchaudio.load(self.noise_fnames[idx//2], frame_offset = int(16000 * self.start_times[idx//2]), 
                                           num_frames = self.wav_len)
            noise_wav = (noise_wav - torch.mean(noise_wav)) / torch.std(noise_wav)
            noise_wav = torch.squeeze(noise_wav)

            # extract features 
            noise_stft = torch.stft(input = noise_wav,
                                    n_fft = self.frame_len,
                                    hop_length = self.frame_len // 4,
                                    window = torch.hamming_window(self.frame_len),
                                    return_complex = True)
            mixture_stft = noise_stft
            feat = torch.abs(noise_stft) ** 1/15
            feat = feat[None, :, :].to(torch.float32)

            # compute mask consisting of all 1's
            mask = torch.ones(feat.size(), dtype = torch.float32)
            mixture_wav = noise_wav
            clean_wav = torch.zeros(mixture_wav.size(), dtype = torch.float32)

        else: # The examples with even indices are noisy speeches (considered to be unlabelled examples)
            noise_wav, _ = torchaudio.load(self.noise_fnames2[idx//2], frame_offset = int(16000 * self.start_times2[idx//2]), 
                                            num_frames = self.wav_len)
            noise_wav = (noise_wav - torch.mean(noise_wav)) / torch.std(noise_wav)
            noise_wav = torch.squeeze(noise_wav)

            # load clean speech 
            clean_wav, sfreq = torchaudio.load(self.clean_fnames[idx//2])
            if sfreq != 16000:
                resampler = torchaudio.transforms.Resample(sfreq, 16000)
                clean_wav = resampler(clean_wav)
            clean_wav = torch.squeeze(clean_wav)
            if clean_wav.size(0) < self.wav_len:
                clean_wav = torch.cat([clean_wav, torch.zeros(self.wav_len - clean_wav.size(0))], dim = 0)
            elif clean_wav.size(0) > self.wav_len:
                clean_wav = clean_wav[:self.wav_len]
            clean_wav = clean_wav - torch.mean(clean_wav)

            # scale clean speech
            factor = 10. ** (self.SNRs[idx//2] / 20.) / torch.sqrt(torch.sum(clean_wav ** 2) / torch.sum(noise_wav ** 2))
            clean_wav = clean_wav * factor

            # mix clean speech and noise
            mixture_wav = clean_wav + noise_wav
            mixture_wav /= torch.std(mixture_wav)
 
            # extract features
            mixture_stft = torch.stft(input = mixture_wav,
                                      n_fft = self.frame_len,
                                      hop_length = self.frame_len // 4,
                                      window = torch.hamming_window(self.frame_len),
                                      return_complex = True)
            feat = torch.abs(mixture_stft) ** 1/15
            feat = feat[None, :, :].to(torch.float32)

            # compute mask consisting of all 0's (indicating unlabelled data)
            mask = torch.zeros(feat.size(), dtype = torch.float32)

        return feat, mask, mixture_stft[None, :, :], mixture_stft[None, :, :]


class PN_data(torch.utils.data.Dataset):

    """
    A dataset class for parallel data consisting of noisy speech and the corresponding clean speech. 
    This is used in supervised learning or for the validation/test set for PULSE and MixIT.

    Args:
        partition: The data partition. The returns differ depending on whether partition == 'train'.
        frame_len: The frame length for the short-time Fourier transform.
        wav_len: The waveform length of each audio clip.
        fname: The path to the configuration file.
        clean_path: The directory path of the clean speech dataset. (not used)
        noise_path: The directory path of the noise dataset. (not used)

    Returns:
        feat: The power-law compressed magnitude spectrogram of mixture_stft. (NB: This is unnecessary 
            as it can be computed from mixture_stft and is to be removed in future updates.)
        mask (only in the case "partition == 'train'"): This is a dummy variable and not used. (This 
            is used just to make sure that there are four outputs for the sake of consistency with 
            other dataset classes.)
        mixture_stft: Noisy speech in the short-time Fourier transform domain.
        clean_stft (only in the case "partition == 'train'"): Clean speech in the short-time Fourier 
            transform domain.
        mixture_wav (only in the case "partition != 'train'"): Noisy speech in the time domain.
        clean_wav (only in the case "partition != 'train'"): Clean speech in the time domain.
    """
        
    def __init__(self, partition, frame_len, wav_len, fname, clean_path, noise_path):

        super().__init__()
        self.partition = partition
        self.frame_len = frame_len
        self.wav_len = wav_len
        self.clean_path = clean_path
        self.noise_path = noise_path

        self.clean_fnames = []
        self.noise_fnames = []
        self.start_times = []
        self.SNRs = []
        with open(fname, 'r') as f:
            for line in f.readlines():
                toks = line.split(' ')
                self.clean_fnames.append(toks[0])
                self.noise_fnames.append(toks[1])
                self.start_times.append(float(toks[2]))
                self.SNRs.append(float(toks[3]))

    def __len__(self):

        return len(self.clean_fnames)

    def __getitem__(self, idx):

        # load noise
        noise_wav, _ = torchaudio.load(self.noise_fnames[idx], frame_offset = int(16000 * self.start_times[idx]), 
                                       num_frames = self.wav_len)
        noise_wav = (noise_wav - torch.mean(noise_wav)) / torch.std(noise_wav)
        noise_wav = torch.squeeze(noise_wav)

        clean_wav, sfreq = torchaudio.load(self.clean_fnames[idx])
        if sfreq != 16000:
            resampler = torchaudio.transforms.Resample(sfreq, 16000)
            clean_wav = resampler(clean_wav)
        clean_wav = torch.squeeze(clean_wav)
        if clean_wav.size(0) < self.wav_len:
            clean_wav = torch.cat([clean_wav, torch.zeros(self.wav_len - clean_wav.size(0))], dim = 0)
        elif clean_wav.size(0) > self.wav_len:
            clean_wav = clean_wav[:self.wav_len]
        clean_wav = clean_wav - torch.mean(clean_wav)

        # scale clean speech
        factor = 10. ** (self.SNRs[idx] / 20.) / torch.sqrt(torch.sum(clean_wav ** 2) / torch.sum(noise_wav ** 2))
        clean_wav = clean_wav * factor

        # mix clean speech and noise
        mixture_wav = clean_wav + noise_wav
        dnm = torch.std(mixture_wav)
        mixture_wav /= dnm
        clean_wav /= dnm
        noise_wav /= dnm

        # extract feature
        mixture_stft = torch.stft(input = mixture_wav,
                                  n_fft = self.frame_len,
                                  hop_length = self.frame_len // 4,
                                  window = torch.hamming_window(self.frame_len),
                                  return_complex = True)
        mixture_stft = mixture_stft[None, :, :]
        feat = torch.abs(mixture_stft) ** 1/15.
        feat = feat.to(torch.float32)

        # compute mask
        clean_stft = torch.stft(input = clean_wav,
                                n_fft = self.frame_len,
                                hop_length = self.frame_len // 4,
                                window = torch.hamming_window(self.frame_len),
                                return_complex = True)
        noise_stft = torch.stft(input = noise_wav,
                                n_fft = self.frame_len,
                                hop_length = self.frame_len // 4,
                                window = torch.hamming_window(self.frame_len),
                                return_complex = True)
        mask = torch.abs(clean_stft) > torch.abs(noise_stft)
        mask = mask[None, :, :].to(torch.float32)

        if self.partition == 'train':
            return feat, mask, mixture_stft, clean_stft[None, :, :]
        else:
            return feat, mixture_stft, mixture_wav, clean_wav


def load_data(max_batch_size, world_size, rank, method, frame_len, wav_len, 
              train_fname, val_fname, test_fname, clean_path, noise_path):

    """
    Creates data loaders for the training, the validation, and the test sets.

    Args:
        max_batch_size: The batch size.
        world_size: The world size. (For single-GPU training, world_size == 1.)
        rank: The rank of the device. (For single GPU training, rank == 0.)
        method: The method for speech enhancement, which is 'PU' for PULSE, 'PN' for supervised 
            learning, and 'MixIT' for MixIT.
        frame_len: The frame length for the short-time Fourier transform.
        wav_len: The waveform length of each audio clip.
        train_fname: The path to the configuration file for the training set.
        val_fname: The path to the configuration file for the validation set.
        test_fname: The path to the configuration file for the test set.
        clean_path: The directory path of the clean speech dataset. (not used)
        noise_path: The directory path of the noise dataset. (not used)
        
    Returns:
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        test_loader: The data loader for the test set.
        train_batch_size: This is unnecessary and is to be removed in future updates.
    """
    
    if method == 'PN':
        train_data = PN_data('train', frame_len, wav_len, train_fname, clean_path, noise_path)
    elif method == 'MixIT':
        train_data = MixIT_train_data(frame_len, wav_len, train_fname, clean_path, noise_path)
    else:
        train_data = PU_train_data(frame_len, wav_len, train_fname, clean_path, noise_path)
    val_data = PN_data('val', frame_len, wav_len, val_fname, clean_path, noise_path)
    test_data = PN_data('test', frame_len, wav_len, test_fname, clean_path, noise_path)

    train_batch_size = min(len(train_data) // world_size, max_batch_size)
    val_batch_size = min(len(val_data) // world_size, max_batch_size)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data,
        num_replicas = world_size,
        rank = rank,
        shuffle = True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_data,
        num_replicas = world_size,
        rank = rank,
        shuffle = True)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size = train_batch_size,
                                               shuffle = False,
                                               num_workers = 2,
                                               pin_memory = True,
                                               sampler = train_sampler)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size = val_batch_size,
                                             shuffle = False,
                                             num_workers = 2,
                                             pin_memory = True,
                                             sampler = val_sampler)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size = 1,
                                              shuffle = False,
                                              num_workers = 2,
                                              pin_memory = True)
 
    return train_loader, val_loader, test_loader, train_batch_size
