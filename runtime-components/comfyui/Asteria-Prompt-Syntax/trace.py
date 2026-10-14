"""Observe the tokens supplied to the original encoder without extra encoding."""
from __future__ import annotations

from hashlib import sha256
import json
import math

MAX_TOKENS = 16384
MAX_BYTES = 4 * 1024 * 1024
ENCODERS = ("CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeSDXLRefiner", "CLIPTextEncodeFlux", "TextEncodeQwenImageEdit")


class _ClipRecorder:
    def __init__(self, clip, source="clip", budget=None, capture_words=False):
        self.clip = clip
        self.source = source
        self.calls = []
        self.partial = False
        self.budget = budget if budget is not None else {"remaining": MAX_TOKENS}
        self.capture_words = capture_words

    def __getattr__(self, name):
        return getattr(self.clip, name)

    def _stream(self, name, chunks):
        tokenizer = self.clip.tokenizer
        decoder = getattr(tokenizer, name, None) or getattr(tokenizer, f"clip_{name}", None)
        result = {"encoder": name, "chunk_lengths": [len(chunk) for chunk in chunks], "chunks": []}
        for chunk in chunks:
            rows = []
            ended = False
            for position, item in enumerate(chunk):
                if self.budget["remaining"] <= 0:
                    self.partial = True
                    break
                self.budget["remaining"] -= 1
                token, weight = item[:2]
                row = {"position": position, "weight": float(weight)}
                if not math.isfinite(row["weight"]):
                    row["weight"] = None
                    self.partial = True
                if isinstance(token, int):
                    row.update(kind="token", id=token)
                    if ended and token == getattr(decoder, "pad_token", None):
                        row["role"] = "padding"
                    elif token == getattr(decoder, "end_token", None):
                        row["role"] = "end"
                        ended = True
                    elif position == 0 and token == getattr(decoder, "start_token", None):
                        row["role"] = "start"
                    try:
                        row["text"] = decoder.decode([token], skip_special_tokens=False)[:512]
                    except (AttributeError, KeyError, TypeError, ValueError):
                        row["text"] = ""
                elif hasattr(token, "detach"):
                    # Embedding rows are small; never retain a model tensor in the receipt.
                    row.update(kind="embedding", shape=list(token.shape), dtype=str(token.dtype))
                    if token.numel() * token.element_size() <= 1024 * 1024:
                        import torch
                        data = token.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
                        row["sha256"] = sha256(data).hexdigest()
                    else:
                        self.partial = True
                else:
                    row["kind"] = "unknown"
                    self.partial = True
                rows.append(row)
            result["chunks"].append(rows)
            if self.budget["remaining"] <= 0:
                break
        return result

    def _record(self, tokens, options):
        try:
            self.calls.append({
                "source": self.source,
                "streams": [self._stream(name, chunks) for name, chunks in tokens.items()],
                "options": json.loads(json.dumps(options, allow_nan=False)),
            })
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            self.partial = True

    def encode_from_tokens_scheduled(self, tokens, **kwargs):
        # Observe the final token dictionary, after L/G replacement and padding.
        self._record(tokens, kwargs)
        result = self.clip.encode_from_tokens_scheduled(tokens, **kwargs)
        if self.capture_words and not self.partial and self.calls:
            streams = self.calls[-1]['streams']
            result = [[tensor, {**metadata, '_asteria_word_streams': streams}] for tensor, metadata in result]
            for _, metadata in result:
                ids = metadata.get('t5xxl_ids')
                t5 = next((s for s in streams if s['encoder'] == 't5xxl'), None)
                if ids is not None and t5 is not None:
                    actual = ids.flatten().tolist()
                    if actual == [row.get('id') for chunk in t5['chunks'] for row in chunk]:
                        metadata['_asteria_word_t5_ids'] = actual
        return result

    def load_model(self, tokens, *args, **kwargs):
        # Anima 3.8B's semantic encoder consumes these exact tokens in raw_hidden_states.
        self._record(tokens, {"entry": "load_model"})
        return self.clip.load_model(tokens, *args, **kwargs)


def _output(result, encoder, inputs, observers):
    receipt = {"version": 1, "encoder_class": encoder, "inputs": inputs,
               "calls": [call for observer in observers for call in observer.calls],
               "partial": any(observer.partial for observer in observers)}
    if not receipt["calls"]:
        receipt["partial"] = True
    if len(json.dumps(receipt, ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
        receipt.update(calls=[], partial=True)
    if len(json.dumps(receipt, ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
        receipt.update(inputs={})
    return {"result": result, "ui": {"asteria_prompt_trace": [receipt]}}


class AsteriaPromptEncodeTrace:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "encoder": (ENCODERS,),
                             "parameters": ("STRING", {"default": "{}"})},
                "optional": {"vae": ("VAE",), "image": ("IMAGE",), "capture_words": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("CONDITIONING",)
    # Ask Comfy to retain the cached UI receipt without executing the encoder again.
    HAS_INTERMEDIATE_OUTPUT = True
    FUNCTION = "encode"
    CATEGORY = "Asteria/conditioning"

    def encode(self, clip, encoder, parameters, capture_words=False, **connections):
        if encoder not in ENCODERS:
            raise ValueError("prompt_trace_encoder_not_supported")
        import nodes
        source = nodes.NODE_CLASS_MAPPINGS[encoder]
        inputs = json.loads(parameters)
        observer = _ClipRecorder(clip, capture_words=capture_words)
        # V3 uses execute; legacy nodes expose the function named by FUNCTION.
        if callable(getattr(source, "execute", None)):
            result = source.execute(clip=observer, **inputs, **connections)
            result = result.result
        else:
            result = getattr(source(), source.FUNCTION)(clip=observer, **inputs, **connections)
        return _output(result, encoder, inputs, [observer])


class AsteriaAnimaPromptTrace:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "native_clip": ("CLIP",), "qwen35_clip": ("CLIP",),
                             "parameters": ("STRING", {"default": "{}"})},
                "optional": {"capture_words": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    HAS_INTERMEDIATE_OUTPUT = True
    FUNCTION = "encode"
    CATEGORY = "Asteria/conditioning"

    def encode(self, model, native_clip, qwen35_clip, parameters, capture_words=False):
        import nodes
        inputs = json.loads(parameters)
        budget = {"remaining": MAX_TOKENS}
        native = _ClipRecorder(native_clip, "native_clip", budget, capture_words=capture_words)
        semantic = _ClipRecorder(qwen35_clip, "qwen35_clip", budget)
        source = nodes.NODE_CLASS_MAPPINGS["AnimaQwen35UnifiedPrompt"]
        result = getattr(source(), source.FUNCTION)(model=model, native_clip=native, qwen35_clip=semantic, **inputs)
        return _output(result, "AnimaQwen35UnifiedPrompt", inputs, [native, semantic])
