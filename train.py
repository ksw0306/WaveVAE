import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from data import LJspeechDataset, collate_fn, collate_fn_synthesize
from modules import ExponentialMovingAverage, stft
from model import WaveVAE
import numpy as np
import soundfile as sf
import os
import argparse
import json
import time
from tqdm import tqdm

def build_model():
    model = WaveVAE()
    return model


def clone_as_averaged_model(model, ema):
    assert ema is not None
    averaged_model = build_model()
    averaged_model.to(device)
    if args.num_gpu > 1:
        averaged_model = torch.nn.DataParallel(averaged_model)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param = ema.shadow[name].clone()
    return averaged_model


def train(epoch, model, optimizer, scheduler, ema):
    global global_step
    epoch_loss = 0.
    running_loss = [0., 0., 0., 0., 0.]
    model.train()
    start_time = time.time()
    display_step = 1
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for batch_idx, (x, _, c, _) in progress_bar:
        global_step += 1

        x, c = x.to(device), c.to(device)

        optimizer.zero_grad()
        x_rec, x_prior, loss_rec, loss_kl = model(x, c)

        stft_rec, stft_rec_log = stft(x_rec[:, 0, 1:])
        stft_truth, stft_truth_log = stft(x[:, 0, 1:])
        stft_prior, stft_prior_log = stft(x_prior[:, 0, 1:])

        loss_frame_rec = criterion_l2(stft_rec, stft_truth) + criterion_l1(stft_rec_log, stft_truth_log)
        loss_frame_prior = criterion_l2(stft_prior, stft_truth) + criterion_l1(stft_prior_log, stft_truth_log)

        # KL annealing coefficient
        alpha = 1 / (1 + np.exp(-5e-5 * (global_step - 5e+5)))
        loss_rec, loss_kl = loss_rec.mean(), loss_kl.mean()
        loss_tot = loss_rec + loss_kl * alpha + loss_frame_rec + loss_frame_prior
        loss_tot.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)

        running_loss[0] += loss_tot.item() / display_step
        running_loss[1] += loss_rec.item() / display_step
        running_loss[2] += loss_kl.item() / display_step
        running_loss[3] += loss_frame_rec.item() / display_step
        running_loss[4] += loss_frame_prior.item() / display_step
        epoch_loss += loss_tot.item()
        if (batch_idx + 1) % display_step == 0:
            # print('Global Step : {}, [{}, {}] [Total Loss, Rec Loss, KL Loss, STFT Recon, STFT Prior)] : {}'
            #       .format(global_step, epoch, batch_idx + 1, np.array(running_loss)))
            # print('{} Step Time : {}'.format(display_step, end_time - start_time))
            progress_bar.set_postfix(
                Total_Loss="{:.2f}".format(np.array(running_loss)[0]), 
                Rec_Loss="{:.2f}".format(np.array(running_loss)[1]), 
                KL_Loss="{:.2f}".format(np.array(running_loss)[2]), 
                STFT_Recon="{:.2f}".format(np.array(running_loss)[3]), 
                STFT_Prior="{:.2f}".format(np.array(running_loss)[4]),
            )
            running_loss = [0., 0., 0., 0., 0.]
        del loss_tot, loss_frame_rec, loss_frame_prior, loss_kl, loss_rec, x, c, x_rec, x_prior
        del stft_rec, stft_truth, stft_prior, stft_truth_log
    print('{} Epoch Training Loss : {:.4f}'.format(epoch, epoch_loss / (len(train_loader))))
    return epoch_loss / len(train_loader)


def evaluate(model, ema=None):
    if ema is not None:
        model_ema = clone_as_averaged_model(model, ema)
    model_ema.eval()
    running_loss = [0., 0., 0., 0., 0.]
    epoch_loss = 0.

    display_step = 1
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, (x, _, c, _) in progress_bar:
        x, c = x.to(device), c.to(device)

        x_rec, x_prior, loss_rec, loss_kl = model(x, c)

        stft_rec, stft_rec_log = stft(x_rec[:, 0, 1:])
        stft_truth, stft_truth_log = stft(x[:, 0, 1:])
        stft_prior, stft_prior_log = stft(x_prior[:, 0, 1:])

        loss_frame_rec = criterion_l2(stft_rec, stft_truth) + criterion_l1(stft_rec_log, stft_truth_log)
        loss_frame_prior = criterion_l2(stft_prior, stft_truth) + criterion_l1(stft_prior_log, stft_truth_log)

        # KL annealing coefficient
        alpha = 1 / (1 + np.exp(-5e-5 * (global_step - 5e+5)))
        loss_rec, loss_kl = loss_rec.mean(), loss_kl.mean()
        loss_tot = loss_rec + loss_kl * alpha + loss_frame_rec + loss_frame_prior

        if ema is not None:
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)

        running_loss[0] += loss_tot.item() / display_step
        running_loss[1] += loss_rec.item() / display_step
        running_loss[2] += loss_kl.item() / display_step
        running_loss[3] += loss_frame_rec.item() / display_step
        running_loss[4] += loss_frame_prior.item() / display_step
        epoch_loss += loss_tot.item()

        if (batch_idx + 1) % display_step == 0:
            # print('Global Step : {}, [{}, {}] [Total Loss, Rec Loss, KL Loss, STFT Recon, STFT Prior)] : {}'
            #       .format(global_step, epoch, batch_idx + 1, np.array(running_loss)))
            progress_bar.set_postfix(
                Total_Loss="{:.2f}".format(np.array(running_loss)[0]), 
                Rec_Loss="{:.2f}".format(np.array(running_loss)[1]), 
                KL_Loss="{:.2f}".format(np.array(running_loss)[2]), 
                STFT_Recon="{:.2f}".format(np.array(running_loss)[3]), 
                STFT_Prior="{:.2f}".format(np.array(running_loss)[4]),
            )
            running_loss = [0., 0., 0., 0., 0.]
        del loss_tot, loss_frame_rec, loss_frame_prior, loss_kl, loss_rec, x, c, x_rec, x_prior
        del stft_rec, stft_truth, stft_prior, stft_truth_log
    epoch_loss /= len(test_loader)
    print('Evaluation Loss : {:.4f}'.format(epoch_loss))
    del model_ema
    return epoch_loss


def synthesize(model, ema=None):
    global global_step
    if ema is not None:
        model_ema = clone_as_averaged_model(model, ema)
    model_ema.eval()
    for batch_idx, (x, _, c, _) in enumerate(synth_loader):
        if batch_idx == 0:
            x, c = x.to(device), c.to(device)

            q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
            z = q_0.sample()
            wav_truth_name = '{}/{}/generate_{}_{}_truth.wav'.format(args.sample_path, args.model_name, global_step, batch_idx)
            sf.write(wav_truth_name, x.to(torch.device("cpu")).squeeze().numpy(), samplerate=22050)
            print('{} Saved!'.format(wav_truth_name))

            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                if args.num_gpu == 1:
                    x_prior = model_ema.generate(z, c).squeeze()
                else:
                    x_prior = model_ema.module.generate(z, c).squeeze()
            torch.cuda.synchronize()
            print('{} seconds'.format(time.time() - start_time))
            wav = x_prior.to(torch.device("cpu")).data.numpy()
            wav_name = '{}/{}/generate_{}_{}.wav'.format(args.sample_path, args.model_name, global_step, batch_idx)
            sf.write(wav_name, wav, samplerate=22050)
            print('{} Saved!'.format(wav_name))
            del x_prior, wav, x, c, z, q_0
    del model_ema


def save_checkpoint(model, optimizer, scheduler, global_step, global_epoch, ema=None):
    checkpoint_path = os.path.join(args.save, args.model_name, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    torch.save({"state_dict": model.state_dict(),
                "optimizer": optimizer_state,
                "scheduler": scheduler_state,
                "global_step": global_step,
                "global_epoch": global_epoch}, checkpoint_path)
    if ema is not None:
        averaged_model = clone_as_averaged_model(model, ema)
        checkpoint_path = os.path.join(args.save, args.model_name, "checkpoint_step{:09d}_ema.pth".format(global_step))
        torch.save({"state_dict": averaged_model.state_dict(),
                    "optimizer": optimizer_state,
                    "scheduler": scheduler_state,
                    "global_step": global_step,
                    "global_epoch": global_epoch}, checkpoint_path)


def load_checkpoint(step, model, optimizer, scheduler, ema=None):
    global global_step
    global global_epoch

    checkpoint_path = os.path.join(args.save, args.model_name, "checkpoint_step{:09d}.pth".format(step))
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    if ema is not None:
        checkpoint_path = os.path.join(args.save, args.model_name, "checkpoint_step{:09d}_ema.pth".format(step))
        checkpoint = torch.load(checkpoint_path)
        averaged_model = build_model()
        averaged_model.to(device)
        try:
            averaged_model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            print("INFO: this model is trained with DataParallel. Creating new state_dict without module...")
            state_dict = checkpoint["state_dict"]
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            averaged_model.load_state_dict(new_state_dict)
        for name, param in averaged_model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    return model, optimizer, scheduler, ema

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    np.set_printoptions(precision=4)

    parser = argparse.ArgumentParser(description='Train WaveVAE of LJSpeech',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='./DATASETS/ljspeech/', help='Dataset Path')
    parser.add_argument('--sample_path', type=str, default='./samples', help='Sample Path')
    parser.add_argument('--save', '-s', type=str, default='./params', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, default='./params', help='Checkpoint path to resume / test.')
    parser.add_argument('--loss', type=str, default='./loss', help='Folder to save loss')
    parser.add_argument('--log', type=str, default='./log', help='Log folder.')

    parser.add_argument('--model_name', type=str, default='wavevae_01', help='Model Name')
    parser.add_argument('--load_step', type=int, default=0, help='Load Step')

    parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='The Learning Rate.')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='Exponential Moving Average Decay')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of workers')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use. >1 uses DataParallel')

    args = parser.parse_args()

    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)

    # Checkpoint dir
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.loss):
        os.makedirs(args.loss)
    if not os.path.isdir(args.sample_path):
        os.makedirs(args.sample_path)
    if not os.path.isdir(os.path.join(args.sample_path, args.model_name)):
        os.makedirs(os.path.join(args.sample_path, args.model_name))
    if not os.path.isdir(os.path.join(args.save, args.model_name)):
        os.makedirs(os.path.join(args.save, args.model_name))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # LOAD DATASETS
    train_dataset = LJspeechDataset(args.data_path, True, 0.1)
    test_dataset = LJspeechDataset(args.data_path, False, 0.1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                            num_workers=args.num_workers)
    synth_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_synthesize,
                            num_workers=args.num_workers, pin_memory=True)
    
    path = os.path.join(args.load, args.model_name, "checkpoint_step{:09d}_ema.pth".format(args.load_step))
    
    model = build_model()

    model.to(device)
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200000, gamma=0.5)
    criterion_l2 = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    ema = ExponentialMovingAverage(args.ema_decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    global_step, global_epoch = 0, 0
    load_step = args.load_step

    log = open(os.path.join(args.log, '{}.txt'.format(args.model_name)), 'w')
    state = {k: v for k, v in args._get_kwargs()}

    if load_step == 0:
        list_train_loss, list_loss = [], []
        log.write(json.dumps(state) + '\n')
        test_loss = 100.0
    else:
        model, optimizer, scheduler, ema = load_checkpoint(load_step, model, optimizer, scheduler, ema)
        list_train_loss = np.load('{}/{}_train.npy'.format(args.loss, args.model_name)).tolist()
        list_loss = np.load('{}/{}.npy'.format(args.loss, args.model_name)).tolist()
        list_train_loss = list_train_loss[:global_epoch]
        list_loss = list_loss[:global_epoch]
        test_loss = np.min(list_loss)

    for epoch in range(global_epoch + 1, args.epochs + 1):
        training_epoch_loss = train(epoch, model, optimizer, scheduler, ema)
        with torch.no_grad():
            test_epoch_loss = evaluate(model, ema)

        state['training_loss'] = training_epoch_loss
        state['eval_loss'] = test_epoch_loss
        state['epoch'] = epoch
        list_train_loss.append(training_epoch_loss)
        list_loss.append(test_epoch_loss)

        if test_loss > test_epoch_loss:
            test_loss = test_epoch_loss
            save_checkpoint(model, optimizer, scheduler, global_step, epoch, ema)
            print('Epoch {} Model Saved! Loss : {:.4f}'.format(epoch, test_loss))
            synthesize(model, ema)
        np.save('{}/{}_train.npy'.format(args.loss, args.model_name), list_train_loss)
        np.save('{}/{}.npy'.format(args.loss, args.model_name), list_loss)

        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)

    log.close()