"""
Streamlit UI для OpenManus — под РЕАЛЬНЫЙ select_action.

Запуск: streamlit run ui/streamlit_app.py  (Streamlit по умолчанию слушает localhost).
Приватный Legion-контур: не публиковать наружу.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st  # noqa: E402

from ui.core import run_query  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="OpenManus", page_icon="🧭")
    st.title("OpenManus — Decision Agent")
    st.caption("Реальный select_action: OSINT-обогащение → remizov-решение. Приватный контур (localhost).")

    text = st.text_area("Запрос:", height=120, placeholder="What is the risk of BTC?")
    actions_raw = st.text_input("Возможные действия (через запятую):", value="buy, sell, wait")
    priority = st.slider("Priority", 0.0, 1.0, 0.5, 0.1)

    if st.button("Отправить", type="primary"):
        if not text.strip():
            st.error("Введите запрос")
            return
        actions = [a.strip() for a in actions_raw.split(",") if a.strip()] or ["proceed", "wait"]
        with st.spinner("Обработка…"):
            try:
                result = run_query(text, actions, priority)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ошибка: {exc}")
                return
        st.success(f"Решение: {result.get('action')} (confidence={result.get('confidence')})")
        st.json(result)


if __name__ == "__main__":
    main()
