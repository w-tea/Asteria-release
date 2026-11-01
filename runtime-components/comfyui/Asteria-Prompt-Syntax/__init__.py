from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .trace import AsteriaPromptEncodeTrace, AsteriaAnimaPromptTrace

NODE_CLASS_MAPPINGS["AsteriaPromptEncodeTrace"] = AsteriaPromptEncodeTrace
NODE_DISPLAY_NAME_MAPPINGS["AsteriaPromptEncodeTrace"] = "Asteria Prompt Input Record"
NODE_CLASS_MAPPINGS["AsteriaAnimaPromptTrace"] = AsteriaAnimaPromptTrace
NODE_DISPLAY_NAME_MAPPINGS["AsteriaAnimaPromptTrace"] = "Asteria Anima Prompt Input Record"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
