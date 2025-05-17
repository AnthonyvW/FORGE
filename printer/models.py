from dataclasses import dataclass
from enum import Enum

@dataclass
class Position:
    x: int
    y: int
    z: int

    def to_gcode(self) -> str:
        """Convert position to G-code coordinates"""
        return f"X{self.x/100} Y{self.y/100} Z{self.z/100}"

class FocusScore(Enum):
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    POOR = "POOR"