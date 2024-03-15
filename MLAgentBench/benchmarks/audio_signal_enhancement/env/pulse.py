from xml.etree.ElementTree import QName
import torch, time, argparse, os
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from model import make_CNN
from dataset import load_data
from loss import sigmoid_loss, sa_loss, weighted_pu_loss, MixIT_loss
from metric import calc_si_sdri
if os.path.isfile('distributed.py'):
    from distributed import setup_cluster
    
@torch.no_grad()
def enhance(mixture_stft, estmask, length, gpu, frame_len):
    
    """
    Masking-based speech enhancement.
    
    Args:
        mixture_stft: The observed noisy speech in the short-time Fourier transform domain.
        estmask: The estimated mask to be used for masking-based speech enhancement.
        length: The waveform length of the enhanced speech.
        gpu: The GPU used.
        frame_len: The frame length for the short-time Fourier transform.
        
    Returns:
        estwav: The enhanced speech in the time domain.
        est_stft: The enhanced speech in the short-time Fourier transform domain.
    """
    
    est_stft = mixture_stft * estmask
    shape = est_stft.size()
    est_stft = est_stft.reshape(-1, shape[-2], shape[-1])
    estwav = torch.istft(input = est_stft,
                         n_fft = frame_len,
                         hop_length = frame_len // 4,
                         window = torch.hamming_window(frame_len).to(gpu),
                         length = length)
    return estwav, est_stft


@torch.no_grad()
def evaluate(model, test_loader, gpu, method, frame_len):

    """
    Evaluate a speech enhancement model on the test set in terms of the improvement in the 
    scale-invariant signal-to-noise ratio (SI-SNRi).
    
    Args:
        model: A speech enhancement model to be evaluated.
        test_loader: The data loader for the test set.
        gpu: The GPU used.
        method: The speech enhancement method, which is 'PU' for PULSE, 'PN' for supervised 
            learning, and 'MixIT' for mixture invariant training (MixIT).
        frame_len: The frame length for the short-time Fourier transform.
        
    Returns:
        si_sdri_ave: The SI-SNRi of the model on the test set, averaged over the whole test set.
    """
    
    si_sdri_ave = torch.Tensor([0.]).to(gpu)    
    data_size = torch.Tensor([0.]).to(gpu)

    for x, mixture_stft, mixture_wav, clean_wav in test_loader:
        x = x.to(gpu)
        mixture_stft = mixture_stft.to(gpu)
        mixture_wav = mixture_wav.to(gpu)
        clean_wav = clean_wav.to(gpu)

        with torch.no_grad():
            yhat = model(x)
            if method == 'PN':
                mask = F.sigmoid(yhat)
            elif method == 'MixIT':
                mask = F.sigmoid(yhat[:, :1, :, :])
            else:
                mask = yhat < 0
            est_wav, _ = enhance(mixture_stft, mask, mixture_wav.shape[-1], gpu, frame_len)
            si_sdri = calc_si_sdri(clean_wav, est_wav, mixture_wav)
            si_sdri_ave += torch.sum(si_sdri)
            data_size += 1

            print(str(data_size.to(torch.int).item()) + '| si_sdri:' + str(si_sdri.item()) + 'dB')
            
    si_sdri_ave /= data_size

    return si_sdri_ave


def train_MixIT(model, optimizer, train_loader, val_loader, epochs, gpu, world_size, rank, frame_len):

    """
    Train a speech enhancement model using mixture invariant training (MixIT). The model checkpoint 
    with the maximum validation SI-SNRi is stored in 'pulse/model_MixIT.pth'.
    
    Args:
        model: The speech enhancement model to be trained.
        optimizer: The optimiser used.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        epochs: The number of epochs.
        gpu: The gpu used.
        world_size: The world size (world_size == 1 for single-GPU training).
        rank: The rank of the device (rank == 0 for single-GPU training).
        frame_len: The frame length for the short-time Fourier transform.
        
    Returns:
        max_si_sdri: The maximum validation SI-SNRi.
    """
    
    history = {}
    history['train_loss'], history['val_si_sdri'] = [], []
    max_si_sdri = torch.tensor(float('-inf')).to(gpu)

    for epoch in range(epochs):
        # training phase
        data_loader = train_loader
        model.train()

        ave_loss = torch.Tensor([0.]).to(gpu)
        data_size = torch.Tensor([0.]).to(gpu)
        
        for x, mix_stft, noise_stft, noisy_stft in data_loader:
            x = x.to(gpu, non_blocking = True)
            mix_stft = mix_stft.to(gpu, non_blocking = True)
            noise_stft = noise_stft.to(gpu, non_blocking = True)
            noisy_stft = noisy_stft.to(gpu, non_blocking = True)

            with torch.set_grad_enabled(True):
                yhat = model(x)
                loss = MixIT_loss(yhat, mix_stft, noise_stft, noisy_stft)
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                optimizer.step()
                ave_loss += loss * x.size(dim = 0)
                data_size += x.size(dim = 0)

        if world_size > 1:
            dist.all_reduce(ave_loss, op = dist.ReduceOp.SUM)
            dist.all_reduce(data_size, op = dist.ReduceOp.SUM)
        ave_loss /= data_size
        history['train_loss'].append(ave_loss)
                
        # validation phase
        data_loader = val_loader
        model.eval()

        data_size = torch.Tensor([0.]).to(gpu)
        ave_si_sdri = torch.Tensor([0.]).to(gpu)
        
        for feat2, noisy_stft, noisy_wav, cln_wav in data_loader:
            noisy_stft = noisy_stft.to(gpu, non_blocking = True)
            cln_wav = cln_wav.to(gpu, non_blocking = True)
            feat2 = feat2.to(gpu, non_blocking = True)
            noisy_wav = noisy_wav.to(gpu, non_blocking = True)
            with torch.set_grad_enabled(False):
                yhat = model(feat2)
                est_wav, _ = enhance(noisy_stft, F.sigmoid(yhat[:,:1,:,:]), cln_wav.shape[-1], gpu, frame_len)
                si_sdri = calc_si_sdri(cln_wav, est_wav, noisy_wav)
                data_size += feat2.size(dim = 0)
                ave_si_sdri += torch.sum(si_sdri)

        if world_size > 1:
            dist.all_reduce(ave_si_sdri, op = dist.ReduceOp.SUM)
            dist.all_reduce(data_size, op = dist.ReduceOp.SUM)
        ave_si_sdri /= data_size
        if ave_si_sdri > max_si_sdri:
            max_si_sdri = ave_si_sdri
            if rank == 0:
                torch.save(model.state_dict(), 'model_MixIT.pth')

        history['val_si_sdri'].append(ave_si_sdri)
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {history['train_loss'][-1].item():.5f} - val_si_sdri: {history['val_si_sdri'][-1].item():.5f}")

    return max_si_sdri


def train(model, optimizer, train_loader, val_loader, epochs, gpu, world_size, rank, beta, gamma, method, 
          frame_len, p=1.0, mode = 'nn', prior=.7):

    """
    Train a speech enhancement model. The model checkpoint with the maximum validation SI-SNRi is stored 
    in 'model_*.pth'.
    
    Args:
        model: The speech enhancement model to be trained.
        optimizer: The optimiser used.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        epochs: The number of epochs.
        gpu: The gpu used.
        world_size: The world size (world_size == 1 for single-GPU training).
        rank: The rank of the device (rank == 0 for single-GPU training).
        beta: The beta parameter for learning from positive and unlabelled data (PU learning) using a 
            non-negative empirical risk
        gamma: The gamma parameter for PU learning using a non-negative empirical risk
        method: The speech enhancement method, which is 'PU' for PULSE and 'PN' for supervised 
            learning.
        frame_len: The frame length for the short-time Fourier transform.
        p: The exponent in the weight for the weighted sigmoid loss, which equals 1.0 for weighting
            by the magnitude spectrogram and 0.0 for no weighting.
        mode: The type of the empirical risk for PU learning, which is 'nn' for the non-negative
            empirical risk and 'unbiased' for the unbiased one.
        prior: The class prior for the positive class.
        
    Returns:
        max_si_sdri: The maximum validation SI-SNRi.
    """
    
    history = {}
    history['train_loss'], history['val_si_sdri'] = [], []
    max_si_sdri = torch.tensor(float('-inf')).to(gpu)

    for epoch in range(epochs):
        # training phase
        data_loader = train_loader
        model.train()

        ave_loss = torch.Tensor([0.]).to(gpu)
        data_size = torch.Tensor([0.]).to(gpu)

        for x, y, mix_stft, cln_stft in data_loader:
            x = x.to(gpu, non_blocking = True)
            y = y.to(gpu, non_blocking = True)
            mix_stft = mix_stft.to(gpu, non_blocking = True)
            cln_stft = cln_stft.to(gpu, non_blocking = True)

            with torch.set_grad_enabled(True):
                yhat = model(x)
                    
                if method == 'PU':
                    loss = weighted_pu_loss(y, yhat, mix_stft, beta, gamma, prior, ell = sigmoid_loss, p=p, mode = mode)
                elif method == 'PU_no_weight':
                    loss = weighted_pu_loss(y, yhat, mix_stft, beta, gamma, prior, ell = sigmoid_loss, p=0, mode = mode)
                elif method == 'PN':
                    loss = sa_loss(yhat, mix_stft, cln_stft)
                elif method == 'MixIT':
                    loss = MixIT_loss(yhat, y, cln_stft, mix_stft)
                    
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                optimizer.step()

                ave_loss += loss * x.size(dim = 0)
                data_size += x.size(dim = 0)

        if world_size > 1:
            dist.all_reduce(ave_loss, op = dist.ReduceOp.SUM)
            dist.all_reduce(data_size, op = dist.ReduceOp.SUM)
        ave_loss /= data_size
        history['train_loss'].append(ave_loss)

        # validation phase
        data_loader = val_loader
        model.eval()

        data_size = torch.Tensor([0.]).to(gpu)
        ave_si_sdri = torch.Tensor([0.]).to(gpu)

        for x, mix_stft, mix_wav, cln_wav in data_loader:
            x = x.to(gpu, non_blocking = True)
            mix_stft = mix_stft.to(gpu, non_blocking = True)
            mix_wav = mix_wav.to(gpu, non_blocking = True)
            cln_wav = cln_wav.to(gpu, non_blocking = True)
            with torch.set_grad_enabled(False):
                if method == 'MixIT':
                    yhat = model((torch.abs(mix_stft) ** 1/15).to(torch.float32))
                else:
                    yhat = model(x)
                    
                if method == 'PN':
                    mask = F.sigmoid(yhat)
                elif method == 'MixIT':
                    mask = F.sigmoid(yhat[:, :1, :, :])
                else:
                    mask = yhat < 0.0
                                       
                est_wav, _ = enhance(mix_stft, mask, mix_wav.shape[-1], gpu, frame_len)
                si_sdri = calc_si_sdri(cln_wav, est_wav, mix_wav)
                data_size += x.size(dim = 0)    
                ave_si_sdri += torch.sum(si_sdri)

        if world_size > 1:
            dist.all_reduce(ave_si_sdri, op = dist.ReduceOp.SUM)
            dist.all_reduce(data_size, op = dist.ReduceOp.SUM)
        ave_si_sdri /= data_size
        if ave_si_sdri > max_si_sdri:
            max_si_sdri = ave_si_sdri
            if rank == 0:
                if method == 'PN':
                    torch.save(model.state_dict(), 'model_PN.pth')
                else:
                    torch.save(model.state_dict(), 'model_PU_' + mode + '_' + str(p) + '.pth')

        history['val_si_sdri'].append(ave_si_sdri)
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {history['train_loss'][-1].item():.5f} - val_si_sdri: {history['val_si_sdri'][-1].item():.5f}")

    return max_si_sdri

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 100, help = 'The number of epochs.')
    parser.add_argument("--batch_size", type = int, default = 16, help = 'The batch size per GPU.')
    parser.add_argument("--beta", type = float, default = 0.0, help = 'The hyperparameter beta in PU learning with a non-negative empirical risk.')
    parser.add_argument('--gamma', type = float, default = 0.096, help = 'The hyperparameter gamma in PU learning with a non-negative empirical risk.')
    parser.add_argument('--p', type = float, default = 1.0, help = 'The exponent for the weight in the weighted sigmoid loss. ')
    parser.add_argument('--mode', type = str, default = 'nn', help = 'The type of the empirical risk in PU learning ("nn" or "unbiased"). ')
    parser.add_argument("--prior", type = float, default = 0.6, help = 'The class prior for the positive class in PU learning.')
    parser.add_argument("--frame_len", type = int, default = 1024, help = 'The frame length for the short-time Fourier transform.')
    parser.add_argument('--dist', action = 'store_true', help = 'A flag indicating that distributed training is activated.')
    parser.add_argument('--prefix', type = str, help = 'The name of the partition in distributed training.')
    parser.add_argument('--fix_seed', action = 'store_true', help = 'A flag indicating that the seeds for random number generators are fixed (for debugging).')
    parser.add_argument("--wav_len", type = int, default = 50000, help = 'The waveform length of each audio clip.')
    parser.add_argument("--lr_per_GPU", type = float, default = 0.000037, help = 'The learning rate per GPU.')
    parser.add_argument("--blocks", type = int, default = 2, help = 'A hyperparameter related to the depth in the convolutional neural network.')
    parser.add_argument("--channels", type = int, default = 8, help = 'A hyperparameter related to the number of channels in the convolutional neural network.')
    parser.add_argument("--droprate", type = float, default = 0.2, help = 'The dropout rate.')
    parser.add_argument("--fcblocks", type = int, default = 0, help = 'A hyperparameter related to the kernel size in the convolutional neural network.')
    parser.add_argument("--method", type = str, default = 'PU', help = 'The audio signal enhancement method ("PU", "PN", or "MixIT").')
    parser.add_argument("--train_fname", type = str, default = 'voicebank_train_set.txt', help = 'The path to the configuration file for the training set.')
    parser.add_argument("--val_fname", type = str, default = 'voicebank_val_set.txt', help = 'The path to the configuration file for the validation set.')
    parser.add_argument("--test_fname", type = str, default = 'voicebank_test_set.txt', help = 'The path to the configuration file for the test set.')
    parser.add_argument("--clean_path", type = str, default = 'voicebank', help = 'The directory path for the clean speech dataset.')
    parser.add_argument("--noise_path", type = str, default = 'DEMAND', help = 'The directory path for the noise dataset.')
    
    return parser.parse_args()


def main():

    args = get_args()

    # Fix seeds (for debugging only)
    if args.fix_seed:
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        else:
            torch.mps.manual_seed(seed)


    
    # Set up a distributed training environment
    if args.dist:
        rank, local_rank, world_size, gpu = setup_cluster(args.prefix)
    else:
        if torch.cuda.is_available():
            rank, local_rank, world_size, gpu = 0, 0, 1, torch.device('cuda')
        else:
            rank, local_rank, world_size, gpu = 0, 0, 1, torch.device('mps')

    if args.dist:
        if rank == 0:
            print(args)
    else:
        print(args)
    
    # Load_data
    if rank == 0:
        print('loading data ...')
    train_loader, val_loader, test_loader, _\
    = load_data(args.batch_size, world_size, rank, args.method, args.frame_len, args.wav_len, 
                args.train_fname, args.val_fname, args.test_fname, args.clean_path, args.noise_path)

    # Construct the CNN
    model = make_CNN(args.blocks, args.channels, args.droprate, args.fcblocks, args.method).to(gpu)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [local_rank])

    # Define the optimizer
    lr = args.lr_per_GPU * world_size
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Training
    if rank == 0:
        print('training ...')
    if args.method == 'MixIT':
        val_si_sdri = train_MixIT(model, optimizer, train_loader, val_loader, 
                                  args.epochs, gpu, world_size, rank, args.frame_len)
    else:
        val_si_sdri = train(model, optimizer, train_loader, val_loader, 
                            args.epochs, gpu, world_size, rank, args.beta, args.gamma, args.method, 
                            args.frame_len, p=args.p, mode=args.mode, prior=args.prior)
    
    # Evaluation
    if rank == 0:
        print('evaluating ...')
        if args.method == 'PU':
            model.load_state_dict(torch.load('model_PU_' + args.mode + '_' + str(args.p) + '.pth'))
        else:
            model.load_state_dict(torch.load('model_' + args.method + '.pth'))
        test_si_sdri = evaluate(model, test_loader, gpu, args.method, args.frame_len)

    
    # Display results
    if rank == 0:
        print('========================================')
        print(args)
        print('validation SI-SNRi: ' + str(val_si_sdri.item()) + 'dB')
        print('test SI-SNRi: ' + str(test_si_sdri.item()) + 'dB')
        print('========================================')

    # Kill distributed training environment (for the multi-GPU case only)
    if world_size > 1:
        time.sleep(10); dist.destroy_process_group()


if __name__ == '__main__':
    main()
