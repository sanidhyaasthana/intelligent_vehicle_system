"""
Synthetic geofence data generation.

Generates a realistic dataset of location-based authentication events with:

Legitimate Samples:
    - Inside defined geofences (home, office, parking)
    - Small GPS noise
    - During common times (morning 6-10, evening 17-22)
    - Reasonable speeds (0-30 m/s)

Attack Samples:
    - Outside all geofences
    - Random distances (100m to 50km away)
    - Unusual times (1-5 AM typically)
    - Boundary attacks (just outside geofence)
    - High speeds (suspicious movement)

Each sample includes:
    lat, lon, time_of_day (hour), gps_accuracy, speed, label
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import logging

from ..utils.geo_utils import haversine_distance

logger = logging.getLogger(__name__)


class GeoDataGenerator:
    """
    Generate synthetic geofence dataset for testing.
    
    Args:
        num_legit: Number of legitimate samples
        num_attack: Number of attack samples
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        num_legit: int = 1000,
        num_attack: int = 500,
        seed: int = 42,
    ):
        self.num_legit = num_legit
        self.num_attack = num_attack
        self.seed = seed
        
        np.random.seed(seed)
        
        # Define geofences (home, office, parking)
        self.geofences = [
            {'name': 'home', 'lat': 40.7128, 'lon': -74.0060, 'radius': 50.0},
            {'name': 'office', 'lat': 40.7580, 'lon': -73.9855, 'radius': 80.0},
            {'name': 'parking', 'lat': 40.7489, 'lon': -73.9680, 'radius': 40.0},
        ]
        
        # Home location for distance features
        self.home_lat = 40.7128
        self.home_lon = -74.0060
    
    def _generate_legitimate_sample(self) -> Dict:
        """Generate a legitimate location sample."""
        # Pick a random geofence
        geofence = self.geofences[np.random.randint(len(self.geofences))]
        
        # Sample point inside geofence with small noise
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, geofence['radius'] * 0.8)
        
        # Convert to lat/lon (approximate)
        lat_offset = radius / 111000.0
        lon_offset = radius / (111000.0 * np.cos(np.radians(geofence['lat'])))
        
        lat = geofence['lat'] + lat_offset * np.cos(angle)
        lon = geofence['lon'] + lon_offset * np.sin(angle)
        
        # Common access times: morning (6-10) or evening (17-22)
        if np.random.rand() > 0.5:
            time_of_day = np.random.randint(6, 11)  # Morning
        else:
            time_of_day = np.random.randint(17, 23)  # Evening
        
        # Add some night-time access
        if np.random.rand() < 0.1:
            time_of_day = np.random.randint(22, 24)
        
        # Realistic GPS accuracy
        gps_accuracy = np.random.uniform(5.0, 20.0)
        
        # Reasonable speed (pedestrian or vehicle)
        speed = np.random.uniform(0.0, 15.0)
        
        return {
            'lat': lat,
            'lon': lon,
            'time_of_day': time_of_day,
            'gps_accuracy': gps_accuracy,
            'speed': speed,
            'label': 1,  # Legitimate / genuine  (global convention: 1=genuine, 0=impostor)
        }
    
    def _generate_attack_sample(self) -> Dict:
        """Generate an attack (spoofing) sample."""
        attack_type = np.random.choice(['outside', 'boundary', 'unusual_time', 'high_speed'])
        
        if attack_type == 'outside':
            # Random location far from geofences
            lat = np.random.uniform(40.5, 41.0)
            lon = np.random.uniform(-74.3, -73.7)
            time_of_day = np.random.randint(6, 23)
            gps_accuracy = np.random.uniform(10.0, 50.0)
            speed = np.random.uniform(0.0, 20.0)
        
        elif attack_type == 'boundary':
            # Just outside a geofence
            geofence = self.geofences[np.random.randint(len(self.geofences))]
            angle = np.random.uniform(0, 2 * np.pi)
            radius = geofence['radius'] + np.random.uniform(10.0, 100.0)
            
            lat_offset = radius / 111000.0
            lon_offset = radius / (111000.0 * np.cos(np.radians(geofence['lat'])))
            
            lat = geofence['lat'] + lat_offset * np.cos(angle)
            lon = geofence['lon'] + lon_offset * np.sin(angle)
            
            time_of_day = np.random.randint(6, 23)
            gps_accuracy = np.random.uniform(20.0, 50.0)
            speed = np.random.uniform(5.0, 25.0)
        
        elif attack_type == 'unusual_time':
            # Deep in the night (1-5 AM)
            geofence = self.geofences[np.random.randint(len(self.geofences))]
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, geofence['radius'] * 1.2)
            
            lat_offset = radius / 111000.0
            lon_offset = radius / (111000.0 * np.cos(np.radians(geofence['lat'])))
            
            lat = geofence['lat'] + lat_offset * np.cos(angle)
            lon = geofence['lon'] + lon_offset * np.sin(angle)
            
            time_of_day = np.random.randint(1, 6)  # 1-5 AM
            gps_accuracy = np.random.uniform(15.0, 40.0)
            speed = np.random.uniform(10.0, 30.0)
        
        else:  # high_speed
            # Suspicious movement speed
            lat = np.random.uniform(40.5, 41.0)
            lon = np.random.uniform(-74.3, -73.7)
            time_of_day = np.random.randint(0, 24)
            gps_accuracy = np.random.uniform(20.0, 50.0)
            speed = np.random.uniform(25.0, 60.0)  # >25 m/s is suspicious
        
        return {
            'lat': lat,
            'lon': lon,
            'time_of_day': time_of_day,
            'gps_accuracy': gps_accuracy,
            'speed': speed,
            'label': 0,  # Attack / impostor  (global convention: 1=genuine, 0=impostor)
        }
    
    def generate(self) -> pd.DataFrame:
        """Generate complete dataset."""
        samples = []
        
        # Generate legitimate samples
        for _ in range(self.num_legit):
            samples.append(self._generate_legitimate_sample())
        
        # Generate attack samples
        for _ in range(self.num_attack):
            samples.append(self._generate_attack_sample())
        
        df = pd.DataFrame(samples)
        
        logger.info(
            f"Generated {len(df)} samples: "
            f"{self.num_legit} legitimate, {self.num_attack} attacks"
        )
        
        return df
    
    def save(self, output_path: str) -> None:
        """Generate and save dataset to CSV."""
        df = self.generate()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved geofence dataset to {output_path}")


def generate_geo_dataset(
    output_path: str = 'data/geo/location_data.csv',
    num_legit: int = 1000,
    num_attack: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Convenience function to generate and save geofence dataset.
    
    Args:
        output_path: Where to save the CSV
        num_legit: Number of legitimate samples
        num_attack: Number of attack samples
        seed: Random seed
        
    Returns:
        Generated DataFrame
    """
    generator = GeoDataGenerator(num_legit=num_legit, num_attack=num_attack, seed=seed)
    generator.save(output_path)
    return generator.generate()
