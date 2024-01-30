import torch

@torch.no_grad()
def si_sdr(true, est):

    para = (torch.sum(true * est, 1, keepdim = True) / torch.sum(true ** 2, 1, keepdim = True)) * true
    num = torch.sum(para ** 2, 1)
    dnm = torch.sum((est - para) ** 2, 1)

    return 10 * torch.log10(num / dnm)


@torch.no_grad()
def calc_si_sdri(true, est, mix):

    return si_sdr(true, est) - si_sdr(true, mix)




