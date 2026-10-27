"""GB10 Model Attention: pick the attention backend per model in a graph (incl. SageAttention).

Core ComfyUI's ModelAttentionBackend offers only "pytorch" and "comfy kitchen" attention; Sage is
otherwise a process-wide --use-sage-attention flag, which would change every stock graph on the
server. This node attaches any registered attention function to one model, the same way
ModelAttentionBackend does (set_model_optimized_attention).
"""
import logging

import comfy.ldm.modules.attention as attention


class GB10ModelAttention:
    @classmethod
    def INPUT_TYPES(cls):
        names = [n for n in ("sage", "pytorch", "flash", "sub_quad", "split")
                 if attention.get_attention_function(n, None) is not None]
        return {"required": {"model": ("MODEL",), "backend": (names, {"default": names[0]})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model, backend):
        func = attention.get_attention_function(backend, None)
        if func is None:
            logging.warning("GB10ModelAttention: backend %s unavailable, using pytorch", backend)
            func = attention.get_attention_function("pytorch")
        m = model.clone()
        m.set_model_optimized_attention(func)
        return (m,)


NODE_CLASS_MAPPINGS = {"GB10ModelAttention": GB10ModelAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"GB10ModelAttention": "GB10 Model Attention"}
