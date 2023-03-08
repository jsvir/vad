# An official implementation of the ICASSP 2023 paper: [SG-VAD: Stochastic Gates Based Speech Activity Detection](https://arxiv.org/abs/2210.16022)

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
@article{svirsky2022sg,
  title={SG-VAD: Stochastic Gates Based Speech Activity Detection},
  author={Svirsky, Jonathan and Lindenbaum, Ofir},
  journal={arXiv preprint arXiv:2210.16022},
  year={2022}
} 
```