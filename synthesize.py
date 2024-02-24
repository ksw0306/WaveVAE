import time
import torch
from torch.utils.data import DataLoader
from data import LJspeechDataset, collate_fn_synthesize
from model import WaveVAE
from torch.distributions.normal import Normal
import os
import argparse
from tqdm import tqdm
import soundfile as sf

def build_model():
    model = WaveVAE()
    return model


def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        print("INFO: this model is trained with DataParallel. Creating new state_dict without module...")
        state_dict = checkpoint["state_dict"]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize WaveVAE of LJSpeech',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='./DATASETS/ljspeech/', help='Dataset Path')
    parser.add_argument('--sample_path', type=str, default='./samples', help='Sample Path')
    parser.add_argument('--save', '-s', type=str, default='./params', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, default='./params', help='Checkpoint path to resume / test.')
    parser.add_argument('--loss', type=str, default='./loss', help='Folder to save loss')
    parser.add_argument('--log', type=str, default='./log', help='Log folder.')

    parser.add_argument('--model_name', type=str, default='wavevae_01', help='Model Name')
    parser.add_argument('--load_step', type=int, default=0, help='Load Step')

    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')


    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir(args.sample_path):
        os.makedirs(args.sample_path)
    if not os.path.isdir(os.path.join(args.sample_path, args.model_name)):
        os.makedirs(os.path.join(args.sample_path, args.model_name))

    # LOAD DATASETS
    test_dataset = LJspeechDataset(args.data_path, False, 0.1)

    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_synthesize,
                            num_workers=args.num_workers, pin_memory=True)

    step = args.load_step
    path = os.path.join(args.load, args.model_name, "checkpoint_step{:09d}_ema.pth".format(step))
    model = build_model()
    model = load_checkpoint(path, model)
    model.to(device)

    model.eval()
    # print(model)
    # exit()
    print('remove_weight_norm')
    model.remove_weight_norm()

    for i, (x, _, c, _) in enumerate(test_loader):
        if i < args.num_samples:
            x, c = x.to(device), c.to(device)
            print(x.size())
            q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
            z = q_0.sample()

            wav_truth_name = '{}/{}/generate_{}_{}_truth.wav'.format(args.sample_path,
                                                                        args.model_name,
                                                                        args.load_step,
                                                                        i)
            sf.write(wav_truth_name, x.squeeze().to(torch.device("cpu")).numpy(), samplerate=22050)
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                x_sample = model.generate(z, c).squeeze()
            torch.cuda.synchronize()
            print('{} seconds'.format(time.time() - start_time))
            wav = x_sample.to(torch.device("cpu")).data.numpy()
            wav_name = '{}/{}/generate_{}_{}.wav'.format(args.sample_path,
                                                            args.model_name,
                                                            args.load_step,
                                                            i)
            sf.write(wav_name, wav, samplerate=22050)
            print('{} Saved!'.format(wav_name))