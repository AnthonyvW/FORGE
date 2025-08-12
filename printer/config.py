class PrinterConfig:
    """Base configuration for printer hardware settings"""
    def __init__(
        self,
        serial_port: str = 'COM7',
        baud_rate: int = 115200,
        max_x: int = 22000,  # Maximum X dimension in steps
        max_y: int = 22000,  # Maximum Y dimension in steps
        max_z: int = 6000,   # Maximum Z dimension in steps
    ):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

class AutomationConfig:
    """Configuration for automated scanning"""
    def __init__(
        self,
        x_start: int = 4990,#7098,
        y_start: int = 8218,#8556,
        x_end: int = 5170,#7388,
        y_end: int = 16306,#17236,
        xy_step: int = 160,
        z_start: int = 4280,#4250,
        z_end: int = 4330,#4430,
        z_step: int = 8,
        initial_z: int = 4330,#3000,
        good_focus_threshold: int = 400,
        moderate_focus_threshold: int = 500
    ):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.xy_step = xy_step
        self.z_start = z_start
        self.z_end = z_end
        self.z_step = z_step
        self.initial_z = initial_z
        self.good_focus_threshold = good_focus_threshold
        self.moderate_focus_threshold = moderate_focus_threshold
    
    def normalize_bounds(self) -> None:
        # Swap values if start is greater than end for x-axis
        if self.x_start > self.x_end:
            self.x_start, self.x_end = self.x_end, self.x_start
        
        # Swap values if start is greater than end for y-axis
        if self.y_start > self.y_end:
            self.y_start, self.y_end = self.y_end, self.y_start
        
        # Swap values if start is greater than end for z-axis
        if self.z_start > self.z_end:
            self.z_start, self.z_end = self.z_end, self.z_start
