import streamlit.components.v1 as components
from pathlib import Path

_component_func = components.declare_component(
    name="reactflow_fmu_editor",
    path=str(Path(__file__).parent.parent / "streamlit-reactflow" / "frontend" / "dist")
)
import os
print(os.listdir(Path(__file__).parent.parent / "streamlit-reactflow" / "frontend" / "dist"))

def reactflow_editor(*, args=None, key=None):
    return _component_func(args=args, key=key)
