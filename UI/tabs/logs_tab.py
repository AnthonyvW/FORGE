from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from logger import get_logger


class LogsTab(QWidget):
    """Logs tab showing application logs with controls"""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Log display
        self._log_display = QTextEdit()
        self._log_display.setReadOnly(True)
        self._log_display.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #cccccc;
            }
        """)
        
        # Buttons
        self._clear_btn = QPushButton("Clear Display")
        self._clear_btn.clicked.connect(self._clear_display)
        
        self._open_folder_btn = QPushButton("Open Log Folder")
        self._open_folder_btn.clicked.connect(self._open_log_folder)
        
        # Auto-scroll checkbox
        from PySide6.QtWidgets import QCheckBox
        self._auto_scroll_check = QCheckBox("Auto-scroll")
        self._auto_scroll_check.setChecked(True)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self._clear_btn)
        button_layout.addWidget(self._open_folder_btn)
        button_layout.addStretch()
        button_layout.addWidget(self._auto_scroll_check)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(self._log_display, 1)
        layout.addLayout(button_layout)
        
        # Register with logger
        self._logger = get_logger()
        self._logger.register_callback(self._on_log_message)
        
        # Add initial message
        self._log_display.append(f"Logs directory: {self._logger.get_log_directory()}")
        self._log_display.append("=" * 80)
    
    def _on_log_message(self, level: str, message: str):
        """
        Handle incoming log message.
        This is called from the logger for each message.
        """
        # Format with color based on level
        color = self._get_level_color(level)
        formatted = f'<span style="color: {color};">[{level}]</span> {self._escape_html(message)}'
        
        self._log_display.append(formatted)
        
        # Auto-scroll to bottom if enabled
        if self._auto_scroll_check.isChecked():
            scrollbar = self._log_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def _get_level_color(self, level: str) -> str:
        """Get color for log level"""
        colors = {
            'DEBUG': '#666666',
            'INFO': '#0066cc',
            'WARNING': '#cc6600',
            'ERROR': '#cc0000',
            'CRITICAL': '#990000',
        }
        return colors.get(level, '#000000')
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
    
    def _clear_display(self):
        """Clear the log display (doesn't delete log files)"""
        self._log_display.clear()
        self._log_display.append(f"Logs directory: {self._logger.get_log_directory()}")
        self._log_display.append("=" * 80)
        self._log_display.append("Display cleared")
    
    def _open_log_folder(self):
        """Open the log folder in file explorer"""
        log_dir = self._logger.get_log_directory()
        
        try:
            if sys.platform == 'win32':
                # Windows
                subprocess.Popen(['explorer', str(log_dir)])
            elif sys.platform == 'darwin':
                # macOS
                subprocess.Popen(['open', str(log_dir)])
            else:
                # Linux
                subprocess.Popen(['xdg-open', str(log_dir)])
            
            self._logger.info(f"Opened log folder: {log_dir}")
        except Exception as e:
            self._logger.error(f"Failed to open log folder: {e}")
    
    def closeEvent(self, event):
        """Unregister from logger when widget closes"""
        self._logger.unregister_callback(self._on_log_message)
        super().closeEvent(event)
