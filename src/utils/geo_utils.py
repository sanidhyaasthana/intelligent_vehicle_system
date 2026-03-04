"""
Geospatial Utilities
====================

Provides functions for GPS coordinate operations, distance calculations,
and geofence boundary checking.

Key Functions:
- Haversine distance between GPS coordinates
- GPS noise simulation
- Point-in-polygon/circle checking
- Geofence boundary detection

Design Choices:
- All distances in meters for consistency
- Coordinates in (latitude, longitude) format
- Supports both individual points and batch operations
"""

import numpy as np
from typing import Tuple, List, Union, Optional
import math


# Earth's radius in meters (mean radius)
EARTH_RADIUS_M = 6_371_000


def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two GPS coordinates.

    Uses the Haversine formula for accurate distance on a sphere.

    Args:
        lat1: Latitude of first point in degrees.
        lon1: Longitude of first point in degrees.
        lat2: Latitude of second point in degrees.
        lon2: Longitude of second point in degrees.

    Returns:
        Distance in meters.

    Example:
        >>> # Distance from NYC to LA (approximate)
        >>> dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        >>> print(f"{dist / 1000:.0f} km")  # ~3940 km

    Reference:
        https://en.wikipedia.org/wiki/Haversine_formula
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def haversine_distance_batch(coords1: np.ndarray,
                             coords2: np.ndarray) -> np.ndarray:
    """
    Compute haversine distances for batches of coordinates.

    Args:
        coords1: Array of shape (N, 2) with [lat, lon] pairs.
        coords2: Array of shape (N, 2) or (2,) for single point.

    Returns:
        Array of distances in meters, shape (N,).
    """
    coords1 = np.atleast_2d(coords1)
    coords2 = np.atleast_2d(coords2)

    # Broadcast if necessary
    if coords2.shape[0] == 1 and coords1.shape[0] > 1:
        coords2 = np.broadcast_to(coords2, coords1.shape)

    lat1 = np.radians(coords1[:, 0])
    lon1 = np.radians(coords1[:, 1])
    lat2 = np.radians(coords2[:, 0])
    lon2 = np.radians(coords2[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def sample_gps_noise(lat: float, lon: float,
                     std_meters: float,
                     n_samples: int = 1,
                     seed: Optional[int] = None) -> Union[Tuple[float, float], np.ndarray]:
    """
    Sample GPS coordinates with Gaussian noise.

    Simulates GPS measurement uncertainty.

    Args:
        lat: Base latitude in degrees.
        lon: Base longitude in degrees.
        std_meters: Standard deviation of noise in meters.
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        If n_samples=1: Tuple (lat, lon).
        If n_samples>1: Array of shape (n_samples, 2).

    Note:
        Uses local flat-Earth approximation for small distances.
        Accurate for std_meters << 10km.
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert std from meters to degrees (approximate)
    # 1 degree latitude ≈ 111,000 meters
    # 1 degree longitude ≈ 111,000 * cos(lat) meters
    lat_rad = math.radians(lat)
    meters_per_deg_lat = 111_000
    meters_per_deg_lon = 111_000 * math.cos(lat_rad)

    std_lat = std_meters / meters_per_deg_lat
    std_lon = std_meters / meters_per_deg_lon

    # Sample noise
    noise_lat = np.random.normal(0, std_lat, n_samples)
    noise_lon = np.random.normal(0, std_lon, n_samples)

    new_lat = lat + noise_lat
    new_lon = lon + noise_lon

    if n_samples == 1:
        return float(new_lat[0]), float(new_lon[0])
    else:
        return np.column_stack([new_lat, new_lon])


def offset_coordinates(lat: float, lon: float,
                       distance_m: float,
                       bearing_deg: float) -> Tuple[float, float]:
    """
    Calculate new coordinates given distance and bearing from a point.

    Args:
        lat: Starting latitude in degrees.
        lon: Starting longitude in degrees.
        distance_m: Distance to travel in meters.
        bearing_deg: Bearing in degrees (0 = North, 90 = East).

    Returns:
        Tuple (new_lat, new_lon) in degrees.
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)

    # Angular distance
    angular_dist = distance_m / EARTH_RADIUS_M

    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(angular_dist) +
        math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing_rad)
    )

    new_lon = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_dist) * math.cos(lat_rad),
        math.cos(angular_dist) - math.sin(lat_rad) * math.sin(new_lat)
    )

    return math.degrees(new_lat), math.degrees(new_lon)


def point_in_circle(lat: float, lon: float,
                    center_lat: float, center_lon: float,
                    radius_m: float) -> bool:
    """
    Check if a point is inside a circular geofence.

    Args:
        lat: Point latitude.
        lon: Point longitude.
        center_lat: Circle center latitude.
        center_lon: Circle center longitude.
        radius_m: Circle radius in meters.

    Returns:
        True if point is inside the circle.
    """
    distance = haversine_distance(lat, lon, center_lat, center_lon)
    return distance <= radius_m


def point_in_polygon(lat: float, lon: float,
                     vertices: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon geofence.

    Uses the ray casting algorithm.

    Args:
        lat: Point latitude.
        lon: Point longitude.
        vertices: List of (lat, lon) tuples defining polygon vertices.
                  Should be in order (clockwise or counter-clockwise).

    Returns:
        True if point is inside the polygon.

    Note:
        Works best for small polygons where Earth's curvature is negligible.
    """
    n = len(vertices)
    inside = False

    j = n - 1
    for i in range(n):
        lat_i, lon_i = vertices[i]
        lat_j, lon_j = vertices[j]

        if ((lon_i > lon) != (lon_j > lon)) and \
           (lat < (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i) + lat_i):
            inside = not inside

        j = i

    return inside


def distance_to_polygon_boundary(lat: float, lon: float,
                                 vertices: List[Tuple[float, float]]) -> float:
    """
    Compute distance from a point to the nearest polygon edge.

    Args:
        lat: Point latitude.
        lon: Point longitude.
        vertices: List of (lat, lon) tuples defining polygon vertices.

    Returns:
        Distance to nearest edge in meters.
    """
    min_dist = float('inf')
    n = len(vertices)

    for i in range(n):
        j = (i + 1) % n
        lat_i, lon_i = vertices[i]
        lat_j, lon_j = vertices[j]

        # Distance to line segment
        dist = _point_to_segment_distance(lat, lon, lat_i, lon_i, lat_j, lon_j)
        min_dist = min(min_dist, dist)

    return min_dist


def _point_to_segment_distance(px: float, py: float,
                               x1: float, y1: float,
                               x2: float, y2: float) -> float:
    """
    Compute distance from point to line segment (in coordinate space).

    Uses haversine for actual distance calculation.
    """
    # Vector from p1 to p2
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        # Segment is a point
        return haversine_distance(px, py, x1, y1)

    # Project point onto line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

    # Nearest point on segment
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    return haversine_distance(px, py, nearest_x, nearest_y)


class Geofence:
    """
    Geofence definition with support for circles and polygons.

    Attributes:
        name: Human-readable geofence name.
        fence_type: 'circle' or 'polygon'.
        center: Center coordinates for circles.
        radius: Radius in meters for circles.
        vertices: List of vertices for polygons.

    Example:
        >>> home = Geofence('home', 'circle', center=(40.7128, -74.0060), radius=50)
        >>> print(home.contains(40.7128, -74.0060))
        True
    """

    def __init__(self,
                 name: str,
                 fence_type: str,
                 center: Optional[Tuple[float, float]] = None,
                 radius: Optional[float] = None,
                 vertices: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize a geofence.

        Args:
            name: Geofence identifier.
            fence_type: 'circle' or 'polygon'.
            center: (lat, lon) center for circles.
            radius: Radius in meters for circles.
            vertices: List of (lat, lon) for polygons.
        """
        self.name = name
        self.fence_type = fence_type.lower()

        if self.fence_type == 'circle':
            if center is None or radius is None:
                raise ValueError("Circle geofence requires center and radius")
            self.center = center
            self.radius = radius
            self.vertices = None
        elif self.fence_type == 'polygon':
            if vertices is None or len(vertices) < 3:
                raise ValueError("Polygon geofence requires at least 3 vertices")
            self.center = None
            self.radius = None
            self.vertices = vertices
        else:
            raise ValueError(f"Unknown geofence type: {fence_type}")

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is inside this geofence."""
        if self.fence_type == 'circle':
            return point_in_circle(lat, lon, self.center[0], self.center[1], self.radius)
        else:
            return point_in_polygon(lat, lon, self.vertices)

    def distance_to_center(self, lat: float, lon: float) -> float:
        """Compute distance from point to geofence center."""
        if self.fence_type == 'circle':
            return haversine_distance(lat, lon, self.center[0], self.center[1])
        else:
            # For polygon, use centroid
            centroid = self.get_centroid()
            return haversine_distance(lat, lon, centroid[0], centroid[1])

    def distance_to_boundary(self, lat: float, lon: float) -> float:
        """Compute distance from point to geofence boundary."""
        if self.fence_type == 'circle':
            dist_to_center = haversine_distance(lat, lon, self.center[0], self.center[1])
            return abs(dist_to_center - self.radius)
        else:
            return distance_to_polygon_boundary(lat, lon, self.vertices)

    def get_centroid(self) -> Tuple[float, float]:
        """Get the centroid of the geofence."""
        if self.fence_type == 'circle':
            return self.center
        else:
            lats = [v[0] for v in self.vertices]
            lons = [v[1] for v in self.vertices]
            return (sum(lats) / len(lats), sum(lons) / len(lons))

    def is_boundary_region(self, lat: float, lon: float,
                           margin_m: float = 20.0) -> bool:
        """Check if point is near the boundary (within margin)."""
        return self.distance_to_boundary(lat, lon) <= margin_m

    @classmethod
    def from_config(cls, config: dict) -> 'Geofence':
        """Create geofence from configuration dictionary."""
        return cls(
            name=config['name'],
            fence_type=config['type'],
            center=tuple(config.get('center', [])) or None,
            radius=config.get('radius'),
            vertices=[tuple(v) for v in config.get('vertices', [])] or None
        )


class GeofenceManager:
    """
    Manager for multiple geofences.

    Provides convenient methods for checking against all geofences.

    Example:
        >>> manager = GeofenceManager()
        >>> manager.add_geofence(Geofence('home', 'circle', (40.7, -74.0), 50))
        >>> manager.add_geofence(Geofence('office', 'circle', (40.8, -73.9), 100))
        >>> print(manager.is_inside_any(40.7, -74.0))
        True
    """

    def __init__(self, geofences: Optional[List[Geofence]] = None):
        """Initialize with optional list of geofences."""
        self.geofences = geofences or []

    def add_geofence(self, geofence: Geofence) -> None:
        """Add a geofence to the manager."""
        self.geofences.append(geofence)

    def is_inside_any(self, lat: float, lon: float) -> bool:
        """Check if point is inside any geofence."""
        return any(gf.contains(lat, lon) for gf in self.geofences)

    def get_containing_geofences(self, lat: float, lon: float) -> List[str]:
        """Get names of all geofences containing the point."""
        return [gf.name for gf in self.geofences if gf.contains(lat, lon)]

    def distance_to_nearest(self, lat: float, lon: float) -> Tuple[float, Optional[str]]:
        """
        Get distance to nearest geofence center.

        Returns:
            Tuple of (distance_m, geofence_name).
        """
        if not self.geofences:
            return float('inf'), None

        min_dist = float('inf')
        nearest_name = None

        for gf in self.geofences:
            dist = gf.distance_to_center(lat, lon)
            if dist < min_dist:
                min_dist = dist
                nearest_name = gf.name

        return min_dist, nearest_name

    def is_in_boundary_region(self, lat: float, lon: float,
                              margin_m: float = 20.0) -> bool:
        """Check if point is in boundary region of any geofence."""
        for gf in self.geofences:
            if gf.is_boundary_region(lat, lon, margin_m):
                return True
        return False

    def get_home_geofence(self) -> Optional[Geofence]:
        """Get the 'home' geofence if it exists."""
        for gf in self.geofences:
            if gf.name.lower() == 'home':
                return gf
        return self.geofences[0] if self.geofences else None

    @classmethod
    def from_config(cls, config_list: List[dict]) -> 'GeofenceManager':
        """Create manager from list of geofence configurations."""
        geofences = [Geofence.from_config(cfg) for cfg in config_list]
        return cls(geofences)


def time_to_sincos(hour: float) -> Tuple[float, float]:
    """
    Convert time of day to sine/cosine encoding.

    This encoding preserves the circular nature of time
    (23:59 is close to 00:00).

    Args:
        hour: Hour of day (0-24).

    Returns:
        Tuple (sin_time, cos_time).
    """
    # Normalize to [0, 2π]
    angle = 2 * math.pi * hour / 24.0
    return math.sin(angle), math.cos(angle)


def sincos_to_time(sin_t: float, cos_t: float) -> float:
    """
    Convert sine/cosine encoding back to hour.

    Args:
        sin_t: Sine component.
        cos_t: Cosine component.

    Returns:
        Hour of day (0-24).
    """
    angle = math.atan2(sin_t, cos_t)
    if angle < 0:
        angle += 2 * math.pi
    return 24.0 * angle / (2 * math.pi)
