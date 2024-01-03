# An official implementation of the ICASSP 2023 paper: [SG-VAD: Stochastic Gates Based Speech Activity Detection](https://ieeexplore.ieee.org/abstract/document/10096938)

# Evaluation results (published checkpoint)
### AVA-speech test
* EER=**10.40%**
* TPR@FPR=0.315 is **0.96**
* ROCAUC=**0.95**
### HAVIC test
* EER=**23.29%**
* TPR@FPR=0.315 is **0.91**
* ROCAUC=**0.83**


# Training

In order to train a new model you need to prepare the next dataset:
    1. The manifest file with noise audio files paths (the same duration as the audios with words), the label should be "background"
    2. The manifest file with spoken words file paths, each word should be labeled with one category

Please note that, as described in the paper, the main VAD model is trained as mask/filter as presented in the next schema:

<img src="https://github.com/jsvir/vad/tree/main/imgs/train.png" width="500"/>

Once the training is finished, the final model architecture is:

<img src="https://github.com/jsvir/vad/tree/main/imgs/inference.png" width="200"/>



1. Prepare your dataset in manifest format supported by [NeMo](https://github.com/NVIDIA/NeMo)
2. Update config file with your paths and hyper-params
3. Install NeMo requirements
4. Run `train.py` script.

# Inference:
1. We publish a pre-trained Pytorch checkpoint (`sgvad.pth`)
2. To use the published checkpoint as-is you need to calibrate the threshold for model output. All values under the threshold are predicted as Non-speech.
3. The default value for the threshold is **3.5**, but it may be too aggressive for your application.
4. To try it on test audios run `python sgvad.py`


# Aknowledgements:

We thank [NeMo](https://github.com/NVIDIA/NeMo) team for their great open source repo.

# Citation:

```bibtex
@inproceedings{svirsky2023sg,
  title={SG-VAD: Stochastic Gates Based Speech Activity Detection},
  author={Svirsky, Jonathan and Lindenbaum, Ofir},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
