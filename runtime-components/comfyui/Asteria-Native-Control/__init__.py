from .nodes import AsteriaKreaDepthControl, AsteriaMossControlLoader, AsteriaMossLoraLoader, AsteriaMossControlMap, AsteriaMossControlApply

NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    AsteriaKreaDepthControl, AsteriaMossControlLoader, AsteriaMossLoraLoader, AsteriaMossControlMap, AsteriaMossControlApply)}
