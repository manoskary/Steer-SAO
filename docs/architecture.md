# Architecture

Steer-SAO keeps the Stable Audio 3 runtime intact and inserts lightweight trainable adapter
branches into DiT cross-attention blocks.

## Runtime Path

1. `stable_audio_3.StableAudioModel` loads `small-music`.
2. `MuseControlAdapterManager` locates transformer blocks that expose `cross_attn`.
3. Each cross-attention module is wrapped with `MuseControlCrossAttention`.
4. The wrapper returns the original cross-attention output plus optional decoupled adapter
   outputs for attribute and audio-reference controls.
5. `ControlledDiTModel` performs MuseControlLite-style multi-branch guidance:
   unconditional, text, text+attributes, and text+attributes+audio.

## Control Geometry

SA3 Small uses 44.1 kHz stereo audio with SAME downsampling ratio 4096. Control features are
interpolated to the requested latent length, so 5-second and 120-second generations do not share
a hard-coded token count.

## Checkpoints

Only adapter and control-conditioner weights are saved. Base SA3 weights remain in Hugging Face
cache and are loaded by `stable-audio-3`.

