import glob
import torch
from omegaconf import OmegaConf, DictConfig
from nemo.collections.asr.modules import AudioToMFCCPreprocessor, ConvASREncoder
import librosa


class SGVAD:
    def __init__(self, preprocessor: AudioToMFCCPreprocessor,
                 model: ConvASREncoder,
                 cfg: DictConfig):
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.preprocessor.featurizer.dither = 0.0
        self.preprocessor.featurizer.pad_to = 0
        self.model = model
        self.model.eval()
        self.model.freeze()
        self.preprocessor.to(self.cfg.device)
        self.model.to(self.cfg.device)

    def predict(self, wave):
        if isinstance(wave, str):
            wave = self.load_audio(wave)
            wave = torch.tensor(wave)
        if not isinstance(wave, torch.Tensor):
            wave = torch.tensor(wave)
        wave = wave.reshape(1, -1)
        wave_len = torch.tensor([wave.size(-1)]).reshape(1)
        processed_signal, processed_signal_len = self.preprocessor(input_signal=wave, length=wave_len)
        with torch.no_grad():
            mu, _ = self.model(audio_signal=processed_signal, length=processed_signal_len)
            binary_gates = torch.clamp(mu + 0.5, 0.0, 1.0)
            score = binary_gates.sum(dim=1).mean().item()
        return score >= self.cfg.threshold

    def load_audio(self, fpath):
        return librosa.load(fpath, sr=self.cfg.sample_rate)[0]

    @classmethod
    def init_from_ckpt(cls):
        cfg = OmegaConf.load("./cfg.yaml")
        ckpt = torch.load(cfg.ckpt, map_location='cpu')
        preprocessor = AudioToMFCCPreprocessor(**cfg.preprocessor)
        preprocessor.load_state_dict(ckpt['preprocessor'], strict=True)
        vad = ConvASREncoder(**cfg.vad)
        vad.load_state_dict(ckpt['vad'], strict=True)
        return cls(preprocessor, vad, cfg)

    def save_ckpt(self):
        ckpt_dict = {"preprocessor": self.preprocessor.state_dict(), "vad": self.model.state_dict()}
        torch.save(ckpt_dict, './sgvad.pth')


if __name__ == "__main__":
    sgvad = SGVAD.init_from_ckpt()
    for fpath in glob.glob("test/*.wav"):
        label = {0: "Non Speech", 1: "Speech"}
        print(f"Audio: {fpath} is {label[sgvad.predict(fpath)]}")
