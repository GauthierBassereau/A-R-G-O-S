import torch
from typing import Iterable
import torch.nn.functional as F

class TextEncoderDinov3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model, self.tokenizer = torch.hub.load(
            "facebookresearch/dinov3",
            "dinov3_vitl16_dinotxt_tet1280d20h24l",
            source="github",
        )
        self.text_model = model.text_model
        
        # Free up memory
        del model
        import gc
        gc.collect()

    def tokenize(self, texts: Iterable[str]) -> torch.LongTensor:
        return self.tokenizer.tokenize(list(texts))

    def forward(self, texts: Iterable[str], normalize: bool = True) -> torch.Tensor:
        tokens = self.tokenize(texts)
        # put the tokens on the device on which this nn module is
        tokens = tokens.to(next(self.text_model.parameters()).device)
        feats = self.text_model(tokens)
        return F.normalize(feats, dim=-1) if normalize else feats


def _assert_same_tokens(dino_tok, texts):
    import clip
    d = dino_tok.tokenize(texts)
    c = clip.tokenize(texts)
    if d.shape != c.shape:
        raise AssertionError(f"token shapes differ: {tuple(d.shape)} vs {tuple(c.shape)}")
    if not torch.equal(d, c):
        raise AssertionError("token ids differ")


if __name__ == "__main__":
    model = TextEncoderDinov3()
    model.to("mps")
    texts = ["robot arm resting", "robot arm grasping a dinosaur"]
    _assert_same_tokens(model.tokenizer, texts)
    txt = model(texts, normalize=True)
    print("text_feats:", tuple(txt.shape))
