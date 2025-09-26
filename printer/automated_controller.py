from typing import Tuple
import time
from .models import Position, FocusScore
from .config import AutomationConfig
from .base_controller import BasePrinterController
from image_processing.analyzers import ImageAnalyzer

from forgeConfig import (
    ForgeSettings,
)

from .base_controller import command

class AutomatedPrinter(BasePrinterController):
    """Extended printer controller with automation capabilities"""
    def __init__(self, forgeConfig: ForgeSettings, automation_config: AutomationConfig, camera):
        super().__init__(forgeConfig)
        self.automation_config = automation_config
        self.camera = camera
        self.is_automated = False

        self.register_handler("AUTOMATION", self.automation_macro)

    def _evaluate_focus(self, focus_score: float) -> FocusScore:
        """Evaluate focus score and return corresponding enum"""
        if focus_score > self.automation_config.good_focus_threshold:
            return FocusScore.GOOD
        elif focus_score > self.automation_config.moderate_focus_threshold:
            return FocusScore.MODERATE
        return FocusScore.POOR

    def automation_macro(self, cmd: command) -> None:
        self._handle_status(self.status_cmd(cmd.message), True),
        for i in range(50):
            self._handle_status(self.status_cmd(f"Starting {i}"), True),
            
            if self.pause_point():
                return  # stop requested; do not requeue
            self._exec_gcode("G0 X10 Y10", wait=True)
            
            if self.pause_point():
                return  # stop requested; do not requeue
            self._exec_gcode("G0 X40 Y10", wait=True)

            if self.pause_point():
                return  # stop requested; do not requeue
            self._exec_gcode("G0 X40 Y40", wait=True)

            if self.pause_point():
                return  # stop requested; do not requeue
            self._exec_gcode("G0 X10 Y40", wait=True)


    def start_automation(self) -> None:
        """Start the automation process"""

        self.reset_after_stop()
        
        # Enqueue the macro like any other command
        self.enqueue_cmd(command(
            kind="AUTOMATION",
            value="",
            message= "Beginning Square Macro",
            log=True
        ))

    def setPosition1(self) -> None:
        self.automation_config.x_start = self.position.x
        self.automation_config.y_start = self.position.y
        self.automation_config.z_start = self.position.z

    def setPosition2(self) -> None:
        self.automation_config.x_end = self.position.x
        self.automation_config.y_end = self.position.y
        self.automation_config.z_end = self.position.z

    def _get_range(self, start: int, end: int, step: int) -> range:
        """Get appropriate range based on start and end positions"""
        if start < end:
            return range(start, end + step, step)
        return range(start, end - step, -step)

    def _handle_automated_z_scan(self) -> None:
        """Handle the automated Z-axis scanning process"""
        config = self.automation_config
        self.automation_config.normalize_bounds()
        z_dir = 1
        last_image_black = False
        
        current_z_step = config.z_step
        z_position = config.z_start if z_dir > 0 else config.z_end
        no_focus_count = focus_count = 0
        
        print("Starting X/Y Step at", self.get_position().to_gcode())
        
        while self._should_continue_scanning(z_position, z_dir) and not self.paused:
            if not self._process_z_position(z_position):
                image = self.camera.get_last_image()
                
                if ImageAnalyzer.is_black(image):
                    last_image_black = True
                    break
                
                focus_result = ImageAnalyzer.analyze_focus(image)
                focus_score = focus_result.focus_score
                focus_quality = self._evaluate_focus(focus_score)
                
                print("Focus Score", focus_score)
                z_position, current_z_step, no_focus_count, focus_count = self._handle_focus(
                    focus_quality, focus_score, z_position, z_dir, current_z_step, no_focus_count, focus_count
                )
                
                if self._should_break_scan(no_focus_count, focus_count):
                    break
                
            z_position += current_z_step * z_dir
        
        if not last_image_black:
            z_dir *= -1

    def _should_continue_scanning(self, z_position: int, z_dir: int) -> bool:
        """Determine if scanning should continue based on Z position and direction"""
        config = self.automation_config
        if z_dir > 0:
            return z_position <= config.z_end
        return z_position >= config.z_start

    def _process_z_position(self, z_position: int) -> bool:
        """Process movement to new Z position and capture image"""
        self.position.z = z_position
        self._send_and_wait(f"G0 Z{z_position / 100}")
        self._send_and_wait("M400")
        time.sleep(0.1)
        
        self.camera.capture_image()
        # Wait for image capture to complete
        while self.camera.is_taking_image:
            time.sleep(0.01)
        
        return False

    def _handle_focus(
        self, 
        focus_quality: FocusScore,
        focus_score: int,
        z_position: int,
        z_dir: int,
        current_z_step: int,
        no_focus_count: int,
        focus_count: int
    ) -> Tuple[int, int, int, int]:
        """Handle focus quality and update scanning parameters"""
        config = self.automation_config
        
        if focus_quality == FocusScore.GOOD:
            self.camera.save_image(True, folder=f'X{self.position.x} Y{self.position.y}', filename=f'Z{self.position.z} F{int(focus_score)}')
            focus_count += 1
            no_focus_count = 0
            return z_position, current_z_step, no_focus_count, focus_count
            
        elif focus_quality == FocusScore.MODERATE and current_z_step > config.z_step:
            z_position -= 1 * config.z_step * z_dir
            return z_position, config.z_step, no_focus_count, focus_count
            
        else:
            if current_z_step == config.z_step * 1:
                no_focus_count += 1
            return z_position, current_z_step, no_focus_count, focus_count

    def _should_break_scan(self, no_focus_count: int, focus_count: int) -> bool:
        """Determine if scanning should break based on focus counts"""
        return (no_focus_count >= 10) or (focus_count > 0 and no_focus_count > 6)