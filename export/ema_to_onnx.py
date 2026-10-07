"""Export EMA Lightning to three ONNX graphs that only use plain tensor ops.

The index bookkeeping (word of each letter/frame, positions, the attention
window) moves out of the graphs: the Rust engine (src/ema.rs) computes it and
passes it in. Each graph is checked against the PyTorch model on a real
sentence, through PyTorch and through ONNX Runtime.

Usage: pip install -r export/requirements.txt
       python export/ema_to_onnx.py [out dir]      (default: ema_onnx)

  text.onnx    ids[1,L] i64, mask[1,L] f32          -> h[1,L,224], dur[1,L]
  sound.onnx   h, cg[1,L], fg[1,T], allow[1,T,L] f32,
               fmask[1,T] f32, noise[1,4,T,64]       -> latents[1,T,64]
  decoder.onnx z[1,64,T]                             -> audio[1,T*1920]
"""

import hashlib
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from ema_lightning.api import REPO, EMA
from ema_lightning.model import apply_rope, masked_group_norm, rope

# The weights the published ONNX files were made from
# (huggingface.co/canberkkkkkk/ema-lightning, revision 7a6ba1ad216bb2f1da9863f80ac8770a6a807632).
WEIGHTS = {
    "ema.pt": "95aec03dafbe0e1d69bca774ab597c779464729a14bc99bfcb52480090c7dfe6",
    "decoder.pt": "9595819b173f411340f63d11332695121a97f8bf1f6d8b6fef0b21cf99c7ad67",
}

OUT = sys.argv[1] if len(sys.argv) > 1 else "ema_onnx"
os.makedirs(OUT, exist_ok=True)
for name, sha in WEIGHTS.items():
    with open(hf_hub_download(REPO, name), "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != sha:
        sys.exit(f"{name}: the weights changed upstream (sha256 {got}); these files are for {sha}")
tts = EMA(device="cpu")
model, decoder, engine, frontend = tts._engine.model, tts._engine.decoder, tts._engine, tts._frontend


def attention(attn, x, cos, sin, keep):
    """The model's Attention with an additive float mask (no bool SDPA)."""
    B, T, D = x.shape
    q, k, v = (t.view(B, T, attn.h, attn.dh).transpose(1, 2) for t in attn.qkv(x).chunk(3, -1))
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    s = (q @ k.transpose(-2, -1)) / math.sqrt(attn.dh) + (1.0 - keep[:, None, None, :]) * -1e9
    return attn.proj((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D))


class Text(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, mask):
        enc = self.m.text
        x = enc.emb(ids)
        for block in enc.conv:
            x = block(x, mask)
        h = enc.norm(x)
        return h, self.duration(h, mask)

    def duration(self, h, mask):
        """Duration.forward with exp(x) - 1 for expm1 (not in ONNX) and float masks."""
        net = self.m.chardur.net
        m = mask[:, None]
        x = masked_group_norm(F.silu(net[0](h.transpose(1, 2) * m)), net[2], m) * m
        x = masked_group_norm(F.silu(net[4](x)), net[6], m)
        log_d = self.m.chardur.out(x.transpose(1, 2)).squeeze(-1) * mask
        return (torch.exp(log_d.clamp(max=6.0)) - 1.0).clamp(min=1e-3) * mask


class Sound(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, h, cg, fg, allow, fmask, noise, fpos):
        m, al = self.m, self.m.aligner
        B, L, D = h.shape
        T = fg.shape[1]
        # One learned query for every frame; broadcast instead of expand (keeps T symbolic).
        q = (al.q(al.frame_q) + fg[:, :, None] * 0.0).view(B, T, al.h, al.dh).transpose(1, 2)
        k = al.k(h).view(B, L, al.h, al.dh).transpose(1, 2)
        v = al.v(h).view(B, L, al.h, al.dh).transpose(1, 2)
        q = apply_rope(q, *(t[:, None] for t in rope(fg * al.pos_scale, al.dh)))
        k = apply_rope(k, *(t[:, None] for t in rope(cg * al.pos_scale, al.dh)))
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(al.dh) * al.log_temp.exp()
        sig2 = (al.log_sigma.exp() ** 2).view(1, al.h, 1, 1)
        dist2 = ((fg[:, :, None] - cg[:, None, :]) ** 2)[:, None]
        logits = logits - al.bias_w.view(1, al.h, 1, 1) * dist2 / (2 * sig2)
        a = allow[:, None]
        attn = (logits * a + (1.0 - a) * -1e4).softmax(-1) * a
        cond = al.o((attn @ v).transpose(1, 2).reshape(B, T, D))

        cos, sin = rope(fpos[0], m.dh)  # frame index 0..T-1, passed in: no Range op
        x = noise[:, 0]
        times = m.times
        for i, t in enumerate(times):
            step = torch.full((B,), t)
            c = m.t_embed(step)
            bc = m.ada_shared(c).view(-1, 6, m.d)
            y = m.in_proj(x) + cond
            for block in m.blocks:
                p = (bc + block.ada_offset[None]).unbind(1)
                sa, ga, aa, sf, gf, af = (z.unsqueeze(1) for z in p)
                y = y + aa * attention(block.attn, block.n1(y) * (1 + ga) + sa, cos, sin, fmask)
                y = y + af * block.ff(block.n2(y) * (1 + gf) + sf)
            s, g = m.ada_out(c).chunk(2, -1)
            v_pred = m.out_proj(m.norm_out(y) * (1 + g.unsqueeze(1)) + s.unsqueeze(1))
            x1 = x + (1 - t) * v_pred
            if i + 1 < len(times):
                x = (1 - times[i + 1]) * noise[:, i + 1] + times[i + 1] * x1
        return x1


class Dec(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, z):
        return self.d(z, None)


def timeline(text):
    """Everything the graphs no longer compute, from the reference engine's own code."""
    p = engine.piece(text, 0.0, 0)
    engine.plan([p], 1.0)
    L, T = p.letters, p.frames
    dur = p.dur[None]
    mask = torch.ones(1, L, dtype=torch.bool)
    cw, wstart = p.cw[None], p.wstart[None]
    c = dur.clamp(min=1e-4) * mask
    done = c.cumsum(-1)
    before = done - c
    word = cw.clamp(min=0)
    total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
    cp = ((done - before.gather(1, wstart) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * mask
    fw, fp = p.fw[None], p.fp[None]
    n_words = int(cw.max()) + 1
    cwc, fwc = cw.clamp(min=0), fw.clamp(min=0)
    wlen = torch.zeros(1, n_words).scatter_add_(1, cwc, (cw >= 0).float()).clamp(min=1.0)
    woff = wlen.cumsum(-1) - wlen
    cg = woff.gather(1, cwc) + cp.float() * wlen.gather(1, cwc)
    fg = woff.gather(1, fwc) + fp.float() * wlen.gather(1, fwc)
    rel = cw[:, None, :] - fw[:, :, None]
    allow = ((rel >= -1) & (rel <= 1)).float()
    return p, cg, fg, allow


text = frontend("Toplantının amacı yeni sürümün planını belirlemekti.")
p, cg, fg, allow = timeline(text)
L, T = p.letters, p.frames
ids = p.ids[None]
mask = torch.ones(1, L)
fmask = torch.ones(1, T)
noise = engine.noise(p)[None]

with torch.no_grad():
    h_ref, dur_ref = model.text_stage(ids, mask > 0.5)
    lat_ref = model.sound_stage(h_ref, dur_ref, mask > 0.5, p.cw[None], p.wstart[None], p.fw[None], p.fp[None],
                                fmask > 0.5, noise)
    wav_ref = decoder(lat_ref.transpose(1, 2), None)
    h, dur = Text(model)(ids, mask)
    fpos = torch.arange(T, dtype=torch.float32)[None]
    lat = Sound(model)(h, cg, fg, allow, fmask, noise, fpos)
    wav = Dec(decoder)(lat.transpose(1, 2))
print("wrappers vs model: h", float((h - h_ref).abs().max()), "dur", float((dur - dur_ref).abs().max()),
      "latents", float((lat - lat_ref).abs().max()), "audio", float((wav - wav_ref).abs().max()))

kw = dict(opset_version=17, dynamo=False, do_constant_folding=True)
torch.onnx.export(Text(model), (ids, mask), f"{OUT}/text.onnx", input_names=["ids", "mask"],
                  output_names=["h", "dur"], dynamic_axes={"ids": {1: "L"}, "mask": {1: "L"}, "h": {1: "L"},
                                                           "dur": {1: "L"}}, **kw)
torch.onnx.export(Sound(model), (h, cg, fg, allow, fmask, noise, fpos), f"{OUT}/sound.onnx",
                  input_names=["h", "cg", "fg", "allow", "fmask", "noise", "fpos"], output_names=["latents"],
                  dynamic_axes={"h": {1: "L"}, "cg": {1: "L"}, "fg": {1: "T"}, "allow": {1: "T", 2: "L"},
                                "fmask": {1: "T"}, "noise": {2: "T"}, "fpos": {1: "T"}, "latents": {1: "T"}}, **kw)
torch.onnx.export(Dec(decoder), (lat.transpose(1, 2),), f"{OUT}/decoder.onnx", input_names=["z"],
                  output_names=["audio"], dynamic_axes={"z": {2: "T"}, "audio": {1: "S"}}, **kw)

import onnxruntime as ort  # noqa: E402

def run(name, feeds):
    return ort.InferenceSession(f"{OUT}/{name}.onnx").run(None, {k: v.numpy() for k, v in feeds.items()})

oh, od = run("text", {"ids": ids, "mask": mask})
ol, = run("sound", {"h": torch.from_numpy(oh), "cg": cg.float(), "fg": fg.float(), "allow": allow,
                    "fmask": fmask, "noise": noise, "fpos": fpos})
ow, = run("decoder", {"z": torch.from_numpy(ol).transpose(1, 2).contiguous()})
print("onnx vs model: h", float(np.abs(oh - h_ref.numpy()).max()), "dur", float(np.abs(od - dur_ref.numpy()).max()),
      "latents", float(np.abs(ol - lat_ref.numpy()).max()), "audio", float(np.abs(ow - wav_ref.numpy()).max()))
for f in ("text", "sound", "decoder"):
    print(f, os.path.getsize(f"{OUT}/{f}.onnx") // 1024, "KB")
