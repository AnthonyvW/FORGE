from __future__ import annotations

from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QGroupBox,
)
    
def camera_page() ->QWidget:
    w = QWidget()
    layout = QVBoxLayout(w)

    top = QGroupBox("Camera Device")
    form = QFormLayout(top)
    layout.addWidget(top)

    return w