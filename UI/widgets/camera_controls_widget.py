from __future__ import annotations

from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox, QComboBox
)
from PySide6.QtCore import Qt
from logger import info, error, warning
from app_context import get_app_context


class CameraControlsWidget(QWidget):
    """
    Widget for camera controls including photo capture and file management.
    """
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        # Default values
        self._default_folder = Path("./output")
        self._current_folder = self._default_folder
        
        # Supported image formats
        self._image_formats = {
            "TIFF": ".tiff",
            "JPEG": ".jpg",
            "PNG": ".png"
        }
        
        # Ensure output folder exists
        self._ensure_output_folder()
        
        # Setup UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Photo capture group
        capture_group = self._create_capture_group()
        layout.addWidget(capture_group)
        
        layout.addStretch()
        
    def _create_capture_group(self) -> QGroupBox:
        """Create the photo capture control group"""
        group = QGroupBox("Photo Capture")
        layout = QVBoxLayout(group)
        
        # Folder selection row
        folder_layout = QHBoxLayout()
        folder_label = QLabel("Output Folder:")
        folder_label.setMinimumWidth(100)
        
        self._folder_edit = QLineEdit()
        self._folder_edit.setText(str(self._current_folder))
        self._folder_edit.setPlaceholderText("Select output folder...")
        
        self._browse_button = QPushButton("Browse...")
        self._browse_button.clicked.connect(self._browse_folder)
        
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self._folder_edit, 1)
        folder_layout.addWidget(self._browse_button)
        
        # Filename row
        filename_layout = QHBoxLayout()
        filename_label = QLabel("Filename:")
        filename_label.setMinimumWidth(100)
        
        self._filename_edit = QLineEdit()
        self._filename_edit.setPlaceholderText("Leave empty for auto-generated name")
        
        filename_layout.addWidget(filename_label)
        filename_layout.addWidget(self._filename_edit, 1)
        
        # Image format row
        format_layout = QHBoxLayout()
        format_label = QLabel("Image Format:")
        format_label.setMinimumWidth(100)
        
        self._format_combo = QComboBox()
        self._format_combo.addItems(self._image_formats.keys())
        self._format_combo.setCurrentText("TIFF")  # Default to TIFF
        
        self._open_folder_button = QPushButton("Browse Output")
        self._open_folder_button.clicked.connect(self._open_folder)
        
        format_layout.addWidget(format_label)
        format_layout.addWidget(self._format_combo)
        format_layout.addWidget(self._open_folder_button)
        format_layout.addStretch()
        
        # Capture button row
        buttons_layout = QHBoxLayout()
        
        self._capture_button = QPushButton("Take Photo")
        self._capture_button.setMinimumHeight(40)
        self._capture_button.clicked.connect(self._take_photo)
        
        buttons_layout.addWidget(self._capture_button)
        
        # Add all to group layout
        layout.addLayout(folder_layout)
        layout.addLayout(filename_layout)
        layout.addLayout(format_layout)
        layout.addLayout(buttons_layout)
        
        return group
    
    def _ensure_output_folder(self):
        """Ensure the output folder exists"""
        try:
            self._current_folder.mkdir(parents=True, exist_ok=True)
            info(f"Output folder ready: {self._current_folder}")
        except Exception as e:
            error(f"Failed to create output folder: {e}")
            # Show toast for error
            ctx = get_app_context()
            if ctx.toast:
                ctx.toast.error(f"{str(e)}", title="Folder Creation Failed")
    
    def _browse_folder(self):
        """Open folder selection dialog"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(self._current_folder),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self._current_folder = Path(folder)
            self._folder_edit.setText(str(self._current_folder))
            self._ensure_output_folder()
            info(f"Output folder changed to: {self._current_folder}")
            
            # Show toast notification
            ctx = get_app_context()
            if ctx.toast:
                ctx.toast.success(f"{self._current_folder.name}", title="Output Folder Changed")
    
    def _open_folder(self):
        """Open the output folder in the system file manager"""
        import subprocess
        import sys
        
        ctx = get_app_context()
        toast = ctx.toast
        
        try:
            folder_path = str(self._current_folder.resolve())
            
            if sys.platform == 'win32':
                # Windows
                subprocess.run(['explorer', folder_path])
            elif sys.platform == 'darwin':
                # macOS
                subprocess.run(['open', folder_path])
            else:
                # Linux
                subprocess.run(['xdg-open', folder_path])
            
            info(f"Opened folder: {folder_path}")
            if toast:
                toast.info("Opening in file explorer...", title="Opening Folder", duration=10000)
        except Exception as e:
            error(f"Failed to open folder: {e}")
            if toast:
                toast.error(f"{str(e)}", title="Failed to Open Folder")
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open folder: {e}"
            )
    
    def _generate_filename(self) -> str:
        """Generate a filename based on current timestamp and selected format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        format_name = self._format_combo.currentText()
        extension = self._image_formats[format_name]
        return f"image_{timestamp}{extension}"
    
    def _get_filepath(self) -> Path:
        """Get the complete filepath for saving"""
        # Get selected format
        format_name = self._format_combo.currentText()
        extension = self._image_formats[format_name]
        
        # Use custom filename if provided, otherwise generate one
        filename = self._filename_edit.text().strip()
        if not filename:
            filename = self._generate_filename()
        else:
            # Remove any existing extension
            filename_path = Path(filename)
            filename_base = filename_path.stem
            
            # Add the selected extension
            filename = f"{filename_base}{extension}"
        
        return self._current_folder / filename
    
    def _take_photo(self):
        """Capture a still photo from the camera"""
        ctx = get_app_context()
        toast = ctx.toast
        
        try:
            # Get camera from app context
            camera = ctx.camera
            
            if camera is None:
                warning("Attempted to capture photo but camera is not available")
                if toast:
                    toast.warning("Camera not available", title="Camera Error")
                return
            
            if not camera.is_open:
                warning("Attempted to capture photo but camera is not open")
                if toast:
                    toast.warning("Please open the camera first", title="Camera Not Open")
                return
            
            # Get filepath
            filepath = self._get_filepath()
            
            # Ensure folder exists
            self._ensure_output_folder()
            
            info(f"Capturing photo to: {filepath}")
            if toast:
                toast.info("Please wait while the image is captured...", title="Capturing Image")
            
            # Capture still image (highest resolution)
            success = camera.capture_and_save_still(
                filepath=filepath,
                resolution_index=0,  # Highest resolution
                additional_metadata={"timestamp": datetime.now().isoformat()},
                timeout_ms=5000
            )
            
            if success:
                info(f"Photo saved successfully: {filepath}")
                if toast:
                    toast.success(f"Saved to: {filepath.name}", title="Image Captured", duration=10000)
                # Clear custom filename after successful capture
                self._filename_edit.clear()
            else:
                error(f"Failed to capture photo to: {filepath}")
                if toast:
                    toast.error("Unable to capture image from camera", title="Capture Failed")
                
        except Exception as e:
            error(f"Error capturing photo: {e}")
            if toast:
                toast.error(f"{str(e)}", title="Capture Error")
            import traceback
            error(traceback.format_exc())