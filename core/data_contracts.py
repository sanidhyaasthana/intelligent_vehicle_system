from dataclasses import dataclass
from typing import Tuple


@dataclass
class SystemEvent:
    image_path: str
    user_id: int
    vehicle_location: Tuple[float, float]
    user_location: Tuple[float, float]
    label: int

    def validate(self):
        assert isinstance(self.image_path, str) and len(self.image_path) > 0, 'image_path missing'
        assert isinstance(self.user_id, int), 'user_id must be int'
        assert isinstance(self.vehicle_location, tuple) and len(self.vehicle_location) == 2, 'vehicle_location invalid'
        assert isinstance(self.user_location, tuple) and len(self.user_location) == 2, 'user_location invalid'
        assert self.label in (0, 1), 'label must be 0 or 1'
