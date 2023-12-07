# Created by Scalers AI for Dell Inc.
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import openvino as ov
import torch
import whisper

model = whisper.load_model("base")
model.to("cpu")
model.eval()
WHISPER_ENCODER_OV = Path("whisper-base-ov-model/whisper_encoder.xml")
WHISPER_DECODER_OV = Path("whisper-base-ov-model/whisper_decoder.xml")
mel = torch.zeros((1, 80, 3000))
audio_features = model.encoder(mel)
encoder_model = ov.convert_model(model.encoder, example_input=mel)
ov.save_model(encoder_model, WHISPER_ENCODER_OV)


def attention_forward(
    attention_module,
    x: torch.Tensor,
    xa: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """
    Override for forward method of decoder attention module with storing cache values explicitly.
    Parameters:
      attention_module: current attention module
      x: input token ids.
      xa: input audio features (Optional).
      mask: mask for applying attention (Optional).
      kv_cache: dictionary with cached key values for attention modules.
      idx: idx for search in kv_cache.
    Returns:
      attention module output tensor
      updated kv_cache
    """
    q = attention_module.query(x)

    if xa is None:
        # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
        # otherwise, perform key/value projections for self- or cross-attention as usual.
        k = attention_module.key(x)
        v = attention_module.value(x)
        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=1)
            v = torch.cat((kv_cache[1], v), dim=1)

    else:
        if kv_cache is None or kv_cache[0].shape[1] == 0:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = attention_module.key(xa)
            v = attention_module.value(xa)
        else:
            k, v = kv_cache

    kv_cache_new = (k, v)

    wv, qk = attention_module.qkv_attention(q, k, v, mask)
    return attention_module.out(wv), kv_cache_new


def block_forward(
    residual_block,
    x: torch.Tensor,
    xa: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """
    Override for residual block forward method for providing kv_cache to attention module.
      Parameters:
        residual_block: current residual block.
        x: input token_ids.
        xa: input audio features (Optional).
        mask: attention mask (Optional).
        kv_cache: cache for storing attention key values.
      Returns:
        x: residual block output
        kv_cache: updated kv_cache

    """
    x0, kv_cache_self = residual_block.attn(
        residual_block.attn_ln(x), mask=mask, kv_cache=kv_cache[0]
    )
    x = x + x0
    if residual_block.cross_attn:
        x1, kv_cache_cross = residual_block.cross_attn(
            residual_block.cross_attn_ln(x), xa, kv_cache=kv_cache[1]
        )
        x = x + x1
    x = x + residual_block.mlp(residual_block.mlp_ln(x))
    return x, (kv_cache_self, kv_cache_cross)


class CrossAttnKVGetter(torch.nn.Module):
    """
    Helper class for scripting approach of caching cross attention key values.
    The main idea that they should be calculated once and reused for next steps.
    Tracing can not correctly catch condition for that, that is why we need to use scripting for this part of model.
    """

    def __init__(self, attn):
        super().__init__()
        self.attn_key = attn.key
        self.attn_value = attn.value

    def forward(
        self, xa: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor]
    ):
        if kv_cache is None or kv_cache[0].shape[1] == 0:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = self.attn_key(xa)
            v = self.attn_value(xa)
        else:
            k, v = kv_cache
        return k, v


def crossattention_forward(
    attention_module,
    x: torch.Tensor,
    xa: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """
    Override for forward method of decoder cross attention module with storing cache values explicitly.
    Parameters:
      attention_module: current attention module
      x: input token ids.
      xa: input audio features (Optional).
      mask: mask for applying attention (Optional).
      kv_cache: dictionary with cached key values for attention modules.
      idx: idx for search in kv_cache.
    Returns:
      attention module output tensor
      updated kv_cache
    """
    q = attention_module.query(x)

    if xa is None:
        # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
        # otherwise, perform key/value projections for self- or cross-attention as usual.
        k = attention_module.key(x)
        v = attention_module.value(x)
    else:
        k, v = attention_module.kv_getter(xa, kv_cache)
    kv_cache_new = (k, v)

    wv, qk = attention_module.qkv_attention(q, k, v, mask)
    return attention_module.out(wv), kv_cache_new


# update forward functions
for _, block in enumerate(model.decoder.blocks):
    block.forward = partial(block_forward, block)
    block.attn.forward = partial(attention_forward, block.attn)
    if block.cross_attn:
        kv_getter = CrossAttnKVGetter(block.cross_attn)
        block.cross_attn.kv_getter = torch.jit.script(kv_getter)
        block.cross_attn.forward = partial(
            crossattention_forward, block.cross_attn
        )


def decoder_forward(
    decoder,
    x: torch.Tensor,
    xa: torch.Tensor,
    kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
):
    """
    Override for decoder forward method.
    Parameters:
      x: torch.LongTensor, shape = (batch_size, <= n_ctx) the text tokens
      xa: torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
           the encoded audio features to be attended on
      kv_cache: Dict[str, torch.Tensor], attention modules hidden states cache from previous steps
    """
    if kv_cache is not None:
        offset = kv_cache[0][0][0].shape[1]
    else:
        offset = 0
        kv_cache = [(None, None) for _ in range(len(decoder.blocks))]
    x = (
        decoder.token_embedding(x)
        + decoder.positional_embedding[offset : offset + x.shape[-1]]
    )
    x = x.to(xa.dtype)
    kv_cache_upd = []

    for block, kv_block_cache in zip(decoder.blocks, kv_cache):
        x, kv_block_cache_upd = block(
            x, xa, mask=decoder.mask, kv_cache=kv_block_cache
        )
        kv_cache_upd.append(tuple(kv_block_cache_upd))

    x = decoder.ln(x)
    logits = (
        x @ torch.transpose(decoder.token_embedding.weight.to(x.dtype), 1, 0)
    ).float()

    return logits, tuple(kv_cache_upd)


# override decoder forward
model.decoder.forward = partial(decoder_forward, model.decoder)

encoder_hidden_size = audio_features.shape[2]
kv_cache_init = [
    (
        (
            torch.zeros((5, 0, encoder_hidden_size)),
            torch.zeros((5, 0, encoder_hidden_size)),
        ),
        (
            torch.zeros((1, 0, encoder_hidden_size)),
            torch.zeros((1, 0, encoder_hidden_size)),
        ),
    )
    for _ in range(len(model.decoder.blocks))
]

tokens = torch.ones((5, 3), dtype=torch.int64)
logits, kv_cache = model.decoder(
    tokens, audio_features, kv_cache=kv_cache_init
)

tokens = torch.ones((5, 1), dtype=torch.int64)
decoder_model = ov.convert_model(
    model.decoder, example_input=(tokens, audio_features, kv_cache)
)
decoder_cache_input = decoder_model.inputs[2:]
for i in range(2, len(decoder_cache_input), 4):
    decoder_cache_input[i].get_node().set_partial_shape(
        ov.PartialShape([-1, -1, encoder_hidden_size])
    )
    decoder_cache_input[i + 1].get_node().set_partial_shape(
        ov.PartialShape([-1, -1, encoder_hidden_size])
    )

decoder_model.validate_nodes_and_infer_types()
ov.save_model(decoder_model, WHISPER_DECODER_OV)
del decoder_model
