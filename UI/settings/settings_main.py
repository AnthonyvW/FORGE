from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
)

from .pages.camera_settings import camera_page
from .pages.automation_settings import automation_page
from .pages.machine_vision_settings import machine_vision_page
from .pages.navigation_settings import navigation_page

class SettingsButton(QToolButton):
    def __init__(self, tooltip: str = "Settings", parent: QWidget | None = None)-> None:
        super().__init__(parent)
        self.setToolTip(tooltip)
        self.setText("⚙")

        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(34)
        self.setFixedHeight(26)

class SettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(860, 580)

        root = QHBoxLayout(self)

        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(220)

        self.pages = QStackedWidget()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        left = QVBoxLayout()
        left.addWidget(QLabel("Categories"))
        left.addWidget(self.sidebar)

        right = QVBoxLayout()
        right.addWidget(self.pages)
        right.addWidget(buttons)

        root.addLayout(left)
        root.addLayout(right)

        self._add_page("Camera", camera_page())
        self._add_page("Navigation", navigation_page())
        self._add_page("Automation", automation_page())
        self._add_page("Machine Vision", machine_vision_page())

        self.sidebar.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.sidebar.setCurrentRow(0)

    def open_to(self, category: str) -> None:
        for i in range(self.sidebar.count()):
            item = self.sidebar.item(i)
            if item and item.text() == category:
                self.sidebar.setCurrentRow(i)
                return

    def _add_page(self, name: str, page: QWidget) -> None:
        self.pages.addWidget(page)
        self.sidebar.addItem(QListWidgetItem(name))