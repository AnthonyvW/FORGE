from __future__ import annotations

from typing import Optional, Any
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from app_context import get_app_context
from camera.amscope_camera import AmscopeCamera
from camera.base_camera import BaseCamera, CameraInfo
from logger import get_logger


class CameraPreview(QFrame):
    """Camera Preview Area with live streaming"""
    
    # Signal for camera events (thread-safe)
    camera_event = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Camera state
        self._camera: Optional[BaseCamera] = None
        self._camera_info: Optional[CameraInfo] = None
        self._img_width = 0
        self._img_height = 0
        self._img_buffer: Optional[bytes] = None
        self._is_streaming = False
        self._no_camera_logged = False  # Track if we've already logged no camera message
        
        # UI elements
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setScaledContents(False)
        self._video_label.setMinimumSize(1, 1)  # Allow shrinking
        from PySide6.QtWidgets import QSizePolicy
        self._video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._video_label.setStyleSheet("color: #888; font-size: 16px;")
        self._video_label.setText("Initializing camera...")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._video_label, 1)
        
        self.setStyleSheet("QFrame { background: #000000; }")
        
        # Timer for checking camera availability
        self._init_timer = QTimer(self)
        self._init_timer.timeout.connect(self._try_initialize_camera)
        
        # Connect camera event signal
        self.camera_event.connect(self._on_camera_event)
        
        # Start initialization
        self._init_timer.start(500)
    
    def _try_initialize_camera(self):
        """Try to initialize and connect to camera"""
        self._init_timer.stop()
        
        logger = get_logger()
        
        # Get camera from app context
        ctx = get_app_context()
        self._camera = ctx.camera
        
        if self._camera is None:
            self._video_label.setText("No camera available - SDK not loaded")
            logger.error("No camera available - SDK not loaded")
            return
        
        # Try to enumerate and connect to first camera
        try:
            cameras = AmscopeCamera.enumerate_cameras()
            
            if len(cameras) == 0:
                self._video_label.setText("No camera detected")
                if not self._no_camera_logged:
                    logger.warning("No camera connected")
                    self._no_camera_logged = True
                # Retry in a few seconds
                self._init_timer.start(3000)
                return
            
            # Camera found, reset flag
            self._no_camera_logged = False
            
            # Use first camera
            self._camera_info = cameras[0]
            self._open_camera()
            
        except Exception as e:
            self._video_label.setText(f"Camera error: {str(e)}")
            logger.error(f"Camera initialization error: {e}")
    
    def _open_camera(self):
        """Open and start streaming from camera"""
        if not self._camera or not self._camera_info:
            return
        
        # Don't re-open if already streaming
        if self._is_streaming and self._camera.is_open:
            return
        
        logger = get_logger()
        
        try:
            # Set camera info for Amscope camera
            if isinstance(self._camera, AmscopeCamera):
                self._camera.set_camera_info(self._camera_info)
            
            # Open camera
            if not self._camera.open(self._camera_info.id):
                self._video_label.setText("Failed to open camera")
                logger.error("Failed to open camera")
                return
            
            # Get current resolution
            res_index, width, height = self._camera.get_current_resolution()
            self._img_width = width
            self._img_height = height
            
            # Allocate image buffer
            if isinstance(self._camera, AmscopeCamera):
                buffer_size = AmscopeCamera.calculate_buffer_size(width, height, 24)
                self._img_buffer = bytes(buffer_size)
            
            # Enable auto exposure by default
            self._camera.set_auto_exposure(True)
            
            # Start capture
            if not self._camera.start_capture(self._camera_callback, self):
                self._camera.close()
                self._video_label.setText("Failed to start camera stream")
                logger.error("Failed to start camera stream")
                return
            
            self._is_streaming = True
            # Clear text when streaming starts - video will show instead
            self._video_label.setText("")
            logger.info(f"Streaming: {self._camera_info.displayname} ({width}x{height})")
            
        except Exception as e:
            self._video_label.setText(f"Error: {str(e)}")
            logger.error(f"Camera open error: {e}")
    
    @staticmethod
    def _camera_callback(event: int, context: Any):
        """
        Camera event callback (called from camera thread).
        Forward to UI thread via signal.
        """
        if isinstance(context, CameraPreview):
            context.camera_event.emit(event)
    
    def _on_camera_event(self, event: int):
        """Handle camera events in UI thread"""
        if not self._camera or not self._camera.is_open:
            return
        
        # Get event constants
        if isinstance(self._camera, AmscopeCamera):
            events = AmscopeCamera.get_event_constants()
            
            if event == events.IMAGE:
                self._handle_image_event()
            elif event == events.ERROR:
                self._handle_error()
            elif event == events.DISCONNECTED:
                self._handle_disconnected()
    
    def _handle_image_event(self):
        """Handle new image from camera"""
        if not self._camera or not self._img_buffer:
            return
        
        try:
            # Pull image into buffer
            if self._camera.pull_image(self._img_buffer, 24):
                # Create QImage from buffer
                if isinstance(self._camera, AmscopeCamera):
                    stride = AmscopeCamera.calculate_stride(self._img_width, 24)
                    image = QImage(
                        self._img_buffer,
                        self._img_width,
                        self._img_height,
                        stride,
                        QImage.Format.Format_RGB888
                    )
                    
                    # Make a deep copy to avoid keeping reference to buffer
                    image = image.copy()
                    
                    # Scale to fit label while maintaining aspect ratio
                    if self._video_label.width() > 0 and self._video_label.height() > 0:
                        scaled_image = image.scaled(
                            self._video_label.width(),
                            self._video_label.height(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.FastTransformation  # Use fast transformation to reduce memory
                        )
                        self._video_label.setPixmap(QPixmap.fromImage(scaled_image))
        except Exception as e:
            get_logger().error(f"Error handling image: {e}")
    
    def _handle_error(self):
        """Handle camera error"""
        self._video_label.setText("Camera error occurred")
        get_logger().error("Camera error occurred")
        self._close_camera()
        # Try to reconnect
        self._init_timer.start(3000)
    
    def _handle_disconnected(self):
        """Handle camera disconnection"""
        self._video_label.setText("Camera disconnected")
        get_logger().warning("Camera disconnected")
        self._close_camera()
        # Try to reconnect
        self._init_timer.start(3000)
    
    def _close_camera(self):
        """Close camera and cleanup"""
        self._is_streaming = False
        if self._camera:
            self._camera.close()
        self._img_buffer = None
        # Don't clear the label here - let error messages show
    
    def closeEvent(self, event):
        """Handle widget close event"""
        self._close_camera()
        super().closeEvent(event)
    
    def cleanup(self):
        """Cleanup resources when widget is being destroyed"""
        self._init_timer.stop()
        self._close_camera()
