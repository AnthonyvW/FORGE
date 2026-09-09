from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class UpdateAvailableDialog(QDialog):
    """Renders a GitHub release's title/description as formatted notes, mirroring ChangelogDialog."""

    def __init__(self, version: str, title: str, notes: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Update Available")
        self.resize(560, 640)

        layout = QVBoxLayout(self)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(f"# FieldWeave {version}\n\n### {title}\n\n{notes}")
        layout.addWidget(browser)

        buttons = QHBoxLayout()
        buttons.addStretch()

        not_now_button = QPushButton("Not Now", self)
        not_now_button.clicked.connect(self.reject)
        buttons.addWidget(not_now_button)

        update_button = QPushButton("Update Now", self)
        update_button.setDefault(True)
        update_button.clicked.connect(self.accept)
        buttons.addWidget(update_button)

        layout.addLayout(buttons)
