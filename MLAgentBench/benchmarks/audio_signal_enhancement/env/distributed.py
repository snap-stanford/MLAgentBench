import torch, os
import torch.distributed as dist

def setup_cluster(prefix):
    slurm_job_nodelist = os.environ['SLURM_JOB_NODELIST']
    if '[' not in slurm_job_nodelist:
        os.environ['MASTER_ADDR'] = slurm_job_nodelist
    else:
        postfix = slurm_job_nodelist[len(prefix)+1:-1].split(',')[0]
        os.environ['MASTER_ADDR'] = prefix + min(postfix.split('-'))

    os.environ['MASTER_PORT'] = str(12345 + int(min(os.environ['SLURM_STEP_GPUS'].split(","))))

    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    torch.cuda.set_device(local_rank)
    gpu = torch.device('cuda')

    if world_size > 1:
        dist.init_process_group('nccl', init_method = 'env://', rank = rank, world_size = world_size)

    return rank, local_rank, world_size, gpu