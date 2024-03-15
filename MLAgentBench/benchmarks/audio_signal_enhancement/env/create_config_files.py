import random, glob

def one_corpus(min_snr, max_snr, val_size, noise_len, train_val_noise_len, clip_len, \
    noise_dpath, trainval_speech_dpath, test_speech_dpath, train_fname, val_fname, \
    test_fname, seed):

    # Fix the seed for the random number generator for reproducibility
    random.seed(seed)

    # noise file names
    noise_fnames = glob.glob(noise_dpath + '**/*.[wW][aA][vV]', recursive=True)
    noise_fnames = random.sample(noise_fnames, len(noise_fnames))

    # speech file names for the training and the validation sets
    speech_fnames = glob.glob(trainval_speech_dpath + '**/*.[wW][aA][vV]', recursive=True)
    speech_fnames = random.sample(speech_fnames, len(speech_fnames))

    # validation set
    with open(val_fname, 'w') as f:
        for i in range(val_size):
            f.write(speech_fnames[i] + ' ' \
                + noise_fnames[random.randint(0, len(noise_fnames) - 1)] + ' '\
                    + str(random.uniform(0., train_val_noise_len - clip_len)) + ' '\
                        + str(random.uniform(min_snr, max_snr)) + '\n')

    # training set
    with open(train_fname, 'w') as f:
        for i in range(val_size, len(speech_fnames)):
            f.write(speech_fnames[i] + ' ' \
                + noise_fnames[random.randint(0, len(noise_fnames) - 1)] + ' '\
                    + str(random.uniform(0., train_val_noise_len - clip_len)) + ' '\
                        + str(random.uniform(min_snr, max_snr)) + ' '\
                            + noise_fnames[random.randint(0, len(noise_fnames) - 1)] + ' '\
                                + str(random.uniform(0., train_val_noise_len - clip_len)) + '\n')

    # speech file names for the test set
    speech_fnames = glob.glob(test_speech_dpath + '**/*.[wW][aA][vV]', recursive=True)
    speech_fnames = random.sample(speech_fnames, len(speech_fnames))

    # test set
    with open(test_fname, 'w') as f:
        for i in range(len(speech_fnames)):
            f.write(speech_fnames[i] + ' ' \
                + noise_fnames[random.randint(0, len(noise_fnames) - 1)] + ' '\
                    + str(random.uniform(train_val_noise_len, noise_len - clip_len)) + ' '\
                        + str(random.uniform(min_snr, max_snr)) + '\n')


def main():

    """
    Create configuration files. Two sets of configuration files are created, which
    correspond to TIMIT and voicebank. For TIMIT, TIMIT_train_set.txt, TIMIT_val_set.txt, 
    and TIMIT_test_set.txt are created for the training, the validation, and the test sets. 
    For voicebank, voicebank_train_set.txt, voicebank_val_set.txt, and voicebank_test_set.txt 
    are created.
    """

    # TIMIT
    min_snr = -5.
    max_snr = 10.
    val_size = 402
    noise_len = 300.
    train_val_noise_len = 270.
    clip_len = 3.125 + 1e-7
    noise_dpath = 'DEMAND/'
    trainval_speech_dpath = 'TIMIT/TRAIN/'
    test_speech_dpath = 'TIMIT/TEST/'
    train_fname = 'TIMIT_train_set.txt'
    val_fname = 'TIMIT_val_set.txt'
    test_fname = 'TIMIT_test_set.txt'
    seed = 10

    one_corpus(min_snr, max_snr, val_size, noise_len, train_val_noise_len, clip_len, \
        noise_dpath, trainval_speech_dpath, test_speech_dpath, train_fname, val_fname, \
        test_fname, seed)

    # voicebank
    min_snr = -5.
    max_snr = 10.
    val_size = 1157
    noise_len = 300.
    train_val_noise_len = 270.
    clip_len = 3.125 + 1e-7
    noise_dpath = 'DEMAND/'
    trainval_speech_dpath = 'voicebank/clean_trainset_wav'
    test_speech_dpath = 'voicebank/clean_testset_wav'
    train_fname = 'voicebank_train_set.txt'
    val_fname = 'voicebank_val_set.txt'
    test_fname = 'voicebank_test_set.txt'
    seed = 20

    one_corpus(min_snr, max_snr, val_size, noise_len, train_val_noise_len, clip_len, \
        noise_dpath, trainval_speech_dpath, test_speech_dpath, train_fname, val_fname, \
        test_fname, seed)


if __name__ == '__main__':

    main()