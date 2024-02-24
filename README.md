# WaveVAE

work in progress

Note that my implementation isn't stable yet. 

A Pytorch Implementation of WaveVAE (Mel Spectrogram --> Waveform)

part of "Parallel Neural Text-to-Speech"


# Requirements

1. **Install Required Packages**
```bash
pip install -r requirements.txt
```
2. **For Pytorch**
https://pytorch.org/get-started/locally/

### Note : Tested on Python 3.9.16, Pytorch 2.2.0 (CUDA 11.8) on Win 10


# Examples

#### Step 1. Download Dataset

- LJSpeech : [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)

#### Step 2. Preprocessing (Preparing Mel Spectrogram)

`python preprocessing.py --in_dir ljspeech --out_dir DATASETS/ljspeech`

#### Step 3. Train Model

`python train.py --model_name wavevae_1 --batch_size 4 --num_gpu 2`

#### Step 4. Synthesize

`--load_step CHECKPOINT` : the # of the model's global training step (also depicted in the trained weight file)

`python synthesize.py --model_name wavevae_1 --load_step 10000 --num_samples 5`

# References

- WaveNet vocoder : [https://github.com/r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- Parallel Neural Text-to-Speech : [https://arxiv.org/abs/1905.08459](https://arxiv.org/abs/1905.08459)
