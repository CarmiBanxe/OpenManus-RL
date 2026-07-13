"""Sphinx config для OpenManus (приватный Legion-контур)."""
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "OpenManus"
author = "OpenManus / Legion"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Тяжёлые/внешние зависимости мокаем — autodoc не должен их импортировать при сборке.
autodoc_mock_imports = [
    "torch", "torchaudio", "torchvision", "transformers", "numpy",
    "gradio", "streamlit", "psutil", "httpx", "fastapi", "pydantic",
    "jwt", "bcrypt", "scipy", "sympy", "yaml", "prometheus_client",
]

html_theme = "alabaster"  # встроенный, без доп. зависимостей
exclude_patterns = ["_build"]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
