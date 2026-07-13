"""
Gradio UI для OpenManus — под РЕАЛЬНЫЙ select_action.

SECURITY (S-18 §1.2): launch(share=False, server_name="127.0.0.1") — НИКАКОГО публичного
gradio.live-туннеля. Uncensored Legion-движок не выставляется в интернет.
"""
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr  # noqa: E402

from ui.core import run_query  # noqa: E402


def _handle(text: str, actions_csv: str, priority: float) -> Tuple[str, float, str]:
    actions = [a.strip() for a in (actions_csv or "").split(",") if a.strip()] or ["proceed", "wait"]
    result = run_query(text, actions, float(priority))
    return str(result.get("action")), float(result.get("confidence", 0.0)), str(result)


def create_ui() -> "gr.Blocks":
    with gr.Blocks(title="OpenManus") as demo:
        gr.Markdown("# OpenManus — Decision Agent\nПриватный контур (localhost, share=False).")
        text = gr.Textbox(label="Запрос", placeholder="What is the risk of BTC?", lines=3)
        actions = gr.Textbox(label="Действия (через запятую)", value="buy, sell, wait")
        priority = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Priority")
        btn = gr.Button("Отправить", variant="primary")
        out_action = gr.Textbox(label="Решение")
        out_conf = gr.Number(label="Confidence")
        out_raw = gr.Textbox(label="Полный результат", lines=6)
        btn.click(_handle, inputs=[text, actions, priority], outputs=[out_action, out_conf, out_raw])
    return demo


if __name__ == "__main__":
    # Красная линия: share=False, только localhost.
    create_ui().launch(share=False, server_name="127.0.0.1", server_port=7860)
