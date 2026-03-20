"""
FTTX Network Design Automation Script
QGIS Production Script for FTTX Network Planning
Handles 100,000+ homepass records with deterministic behavior

Extended: supports intermediate Dome Closures feeding groups of FDTs with
full fiber traceability and optional pickup-point placement.
"""

import time
import math
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsSpatialIndex, QgsField, QgsFields, QgsWkbTypes, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform, QgsDistanceArea, QgsProcessing, QgsFeatureSink,
    QgsProcessingUtils, QgsFeatureRequest, QgsUnitTypes, QgsVectorFileWriter,
    QgsVectorLayerUtils, QgsGeometryUtils
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtWidgets import (
    QApplication, QDialog, QComboBox, QLabel, QPushButton, QCheckBox,
    QWidget,
    QLineEdit, QDoubleSpinBox, QProgressBar, QFormLayout, QVBoxLayout,
    QHBoxLayout, QGroupBox, QTabWidget, QTextEdit, QMessageBox)
from qgis.PyQt.QtCore import pyqtSignal, QObject, QThread
from qgis.PyQt.QtGui import QFont

from qgis.analysis import (
    QgsNetworkDistanceStrategy, QgsGraphBuilder, QgsGraph, QgsGraphAnalyzer,
    QgsVectorLayerDirector
)

import processing
import numpy as np

# =============================================================================
# CUSTOM LOGGING HANDLER FOR GUI
# =============================================================================

class GuiLogHandler(logging.Handler, QObject):
    """Custom logging handler that emits signals for GUI updates."""
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

try:
    from sklearn.cluster import KMeans
except Exception:
    # Minimal, deterministic KMeans fallback when scikit-learn is not installed.
    class KMeans:
        def __init__(self, n_clusters=8, random_state=None, n_init=10, max_iter=300):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.n_init = n_init
            self.max_iter = max_iter
            self.cluster_centers_ = None

        def fit_predict(self, X):
            X = np.asarray(X, dtype=float)
            n_samples = X.shape[0]
            k = int(self.n_clusters)

            rng = np.random.RandomState(self.random_state)
            best_inertia = None
            best_labels = None
            best_centers = None

            n_init = max(1, int(self.n_init))
            for _ in range(n_init):
                if n_samples >= k:
                    indices = rng.choice(n_samples, k, replace=False)
                else:
                    indices = rng.choice(n_samples, k, replace=True)
                centers = X[indices].astype(float).copy()

                for _ in range(int(self.max_iter)):
                    # assign labels
                    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                    labels = np.argmin(dists, axis=1)

                    # recompute centers
                    new_centers = np.array([
                        X[labels == ci].mean(axis=0) if np.any(labels == ci) else centers[ci]
                        for ci in range(k)
                    ])

                    if np.allclose(new_centers, centers):
                        break
                    centers = new_centers

                inertia = np.sum((X - centers[labels]) ** 2)
                if best_inertia is None or inertia < best_inertia:
                    best_inertia = inertia
                    best_labels = labels
                    best_centers = centers

            self.cluster_centers_ = best_centers
            return best_labels

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

MIN_FAT_CAPACITY = 7
MAX_FAT_CAPACITY = 8
MAX_FDT_CAPACITY = 64
MAX_BOX_TO_BOX_DISTANCE = 250  # meters
MAX_BOX_TO_CUSTOMER_DISTANCE = 150  # meters
DROP_LENGTH_TOLERANCE = 1.0  # meters; tolerance for snap offsets and geometry precision
SNAP_OFFSET_DISTANCE = 1  # meters
MAX_FAT_PER_FDT = 8
MIN_FAT_PER_FDT = 8  # minimum FATs per FDT; underfilled FDTs borrow from connected roads
MAX_CUSTOMERS_PER_FAT = 8
MAX_FAT_TO_FAT_DISTANCE = 150
MAX_FAT_PER_WING = 4
SPLITTER_PATTERN = ["1x9", "1x9", "1x9", "1x8"]
SCRIPT_VERSION = "2026-03-06-pathfix-v3"

# closure / dome configuration
MAX_FDT_PER_8CORE = 4              # maximum FDTs served by one 8-core distribution cable
CLOSURE_SMALL_CAPACITY = 24        # cores for small-capacity feed cable
CLOSURE_LARGE_CAPACITY = 48        # cores for large-capacity feed cable
# A 24-core closure has 24/8=3 cables, each serving 4 FDTs max = 12 FDTs total.
# A 48-core closure has 48/8=6 cables, each serving 4 FDTs max = 24 FDTs total.
# Threshold: if FDT count > 12, use 48-core; otherwise 24-core.
CLOSURE_24_CORE_THRESHOLD = (CLOSURE_SMALL_CAPACITY // 8) * MAX_FDT_PER_8CORE

# Pickup point generation / clustering
# Approximate number of closures each generated pickup should serve.  Larger
# values yield fewer pickup points but longer feeder runs.
CLOSURES_PER_PICKUP = 8

# name of attribute field containing homepass/customer count
HOMEPASS_FIELD_NAME = "Name"  # change as needed

# radius used when multiple FATs share the same pole (meters)
FAT_OFFSET_RADIUS = 0.5

# Layer names (exact match required)
LAYER_NAMES = {
    'road': 'roads',
    'homepass': 'homepass',
    'pole': 'poles'
}
OPTIONAL_LAYER_CANDIDATES = {
    'building': ['buildings', 'building', 'Buildings', 'Building'],
    'pickup': ['pickup', 'pickup_point', 'Pickup', 'PickupPoint']  # optional layer for pickup point cabinets
}
FAST_MODE_DEFAULT = True  # enable performance-friendly clustering by default

@dataclass
class FTTXConfig:
    """User-configurable runtime settings."""
    max_box_to_box_distance: float = MAX_BOX_TO_BOX_DISTANCE
    max_box_to_customer_distance: float = MAX_BOX_TO_CUSTOMER_DISTANCE
    drop_length_tolerance: float = DROP_LENGTH_TOLERANCE
    snap_offset_distance: float = SNAP_OFFSET_DISTANCE
    fat_offset_radius: float = FAT_OFFSET_RADIUS
    splitter_pattern: List[str] = field(default_factory=lambda: SPLITTER_PATTERN.copy())
    homepass_field_name: str = HOMEPASS_FIELD_NAME
    enable_closure_generation: bool = True
    fast_mode: bool = FAST_MODE_DEFAULT

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RoadSegment:
    """Represents a road segment with its properties"""
    id: int
    geometry: QgsGeometry
    fid: int
    length: float = 0.0
    homepasses: List[QgsFeature] = field(default_factory=list)
    total_customers: int = 0
    fat_count: int = 0
    fats: List['FAT'] = field(default_factory=list)

@dataclass
class Homepass:
    """Represents a homepass point with customer count"""
    fid: int
    geometry: QgsPointXY
    customer_count: int
    road_id: Optional[int] = None
    distance_to_road: float = 0.0
    fat_id: Optional[str] = None
    
@dataclass
class FAT:
    """Fiber Access Terminal"""
    name: str
    geometry: QgsPointXY
    road_id: int
    total_customers: int
    cascade_order: int
    from_device: str
    to_device: str
    wing: str  # 'A' or 'B'
    wing_id: str = ""
    parent_fdt: str = ""
    fdt_serial: str = ""
    fat_sequence: str = ""
    fat_full_code: str = ""
    splitter_type: str = ""
    customers: List[int] = field(default_factory=list)
    customer_allocations: Dict[int, int] = field(default_factory=dict)  # fid -> count (for split points)
    snapped_geometry: Optional[QgsPointXY] = None
    offset_geometry: Optional[QgsPointXY] = None

@dataclass
class DomeClosure:
    """Intermediate dome closure feeding a cluster of FDTs

    Attributes:
        pickup_geometry: location of the pickup point feeding this closure (if any)
        zone_geometry: polygon representing this closure's service area.  Used
            for strict zone-based assignment of FDTs.  This is computed after
            closures are generated and may be a Voronoi cell or a buffered
            convex hull depending on configuration.
    """
    name: str
    geometry: QgsPointXY
    fdts: List['FDT'] = field(default_factory=list)
    total_fdt: int = 0
    feed_capacity: int = 0  # 24 or 48 cores
    snapped_geometry: Optional[QgsPointXY] = None
    pickup_geometry: Optional[QgsPointXY] = None
    pickup_name: Optional[str] = None
    zone_geometry: Optional[QgsGeometry] = None

@dataclass
class CableSegment:
    """Represents a cable segment between devices"""
    name: str
    start_device: str
    end_device: str
    geometry: QgsGeometry
    length: float
    fdt_id: str
    core_info: Dict = field(default_factory=dict)  # holds fiber tracing details
    
@dataclass
class FDT:
    """Fiber Distribution Terminal"""
    name: str
    geometry: QgsPointXY
    total_customers: int
    total_fat: int
    wing_a_count: int
    wing_b_count: int
    fdt_code: str = ""
    fdt_serial: str = ""
    fats: List[FAT] = field(default_factory=list)
    zone_geometry: Optional[QgsGeometry] = None
    snapped_geometry: Optional[QgsPointXY] = None

class WingType(Enum):
    A = "A"  # Right side
    B = "B"  # Left side

# =============================================================================
# LOGGING SETUP
# =============================================================================

class PerformanceLogger:
    """Performance logging utility"""
    def __init__(self):
        self.start_time = time.time()
        self.module_times = {}
        self.logger = logging.getLogger('FTTX_Design')
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def start_module(self, module_name: str):
        """Start timing a module"""
        self.module_times[module_name] = time.time()
        self.logger.info(f"Starting module: {module_name}")
    
    def end_module(self, module_name: str):
        """End timing a module and log duration"""
        if module_name in self.module_times:
            duration = time.time() - self.module_times[module_name]
            self.logger.info(f"Completed module: {module_name} - Duration: {duration:.2f} seconds")
            return duration
        return 0
    
    def log(self, message: str):
        """Log an info message"""
        self.logger.info(message)
    
    def error(self, message: str):
        """Log an error message"""
        self.logger.error(message)

# =============================================================================
# ERROR HANDLING
# =============================================================================

class FTTXError(Exception):
    """Base exception for FTTX design errors"""
    pass

class LayerNotFoundError(FTTXError):
    """Raised when required layer is not found"""
    pass

class CRSError(FTTXError):
    """Raised when CRS is invalid"""
    pass

class ValidationError(FTTXError):
    """Raised when validation fails"""
    pass

class ErrorCollector:
    """Deprecated no-op collector retained for compatibility."""
    def __init__(self):
        self.errors = []
        
    def add_error(self, error_type: str, reference_id: str, description: str):
        """No-op: error collection has been removed."""
        return
    
    def create_error_layer(self, crs: QgsCoordinateReferenceSystem) -> QgsVectorLayer:
        """No-op: returns None because error layer is removed."""
        return None
    
    def write_errors(self):
        """No-op: error writing has been removed."""
        return

# =============================================================================
# CORE UTILITIES
# =============================================================================

class SpatialIndexManager:
    """Manages spatial indices for all layers"""
    def __init__(self):
        self.indices = {}
        self.geometries = {}
        
    def build_index(self, layer: QgsVectorLayer, cache_geoms: bool = True) -> QgsSpatialIndex:
        """Build spatial index for a layer with optional geometry caching"""
        index = QgsSpatialIndex()
        geoms = {}
        
        features = layer.getFeatures()
        for feat in features:
            geom = feat.geometry()
            if geom and not geom.isEmpty():
                index.addFeature(feat)
                if cache_geoms:
                    geoms[feat.id()] = geom
        
        self.indices[layer.id()] = index
        if cache_geoms:
            self.geometries[layer.id()] = geoms
        
        return index
    
    def get_index(self, layer: QgsVectorLayer) -> QgsSpatialIndex:
        """Get or create spatial index for layer"""
        if layer.id() not in self.indices:
            return self.build_index(layer)
        return self.indices[layer.id()]
    
    def get_geometry(self, layer: QgsVectorLayer, fid: int) -> Optional[QgsGeometry]:
        """Get cached geometry for feature"""
        if layer.id() in self.geometries:
            return self.geometries[layer.id()].get(fid)
        return None
    
    def nearest_neighbor(self, layer: QgsVectorLayer, point: QgsPointXY, 
                        max_distance: float = None) -> Optional[QgsFeature]:
        """Find nearest feature in layer to point"""
        index = self.get_index(layer)
        nearest_ids = index.nearestNeighbor(point, 1, max_distance)
        
        if nearest_ids:
            fid = nearest_ids[0]
            request = QgsFeatureRequest().setFilterFid(fid)
            feat = next(layer.getFeatures(request))
            return feat
        return None

class DistanceCalculator:
    """Handles distance calculations with CRS awareness"""
    def __init__(self, crs: QgsCoordinateReferenceSystem):
        self.crs = crs
        self.distance_area = QgsDistanceArea()
        self.distance_area.setSourceCrs(crs, QgsProject.instance().transformContext())
        self.distance_area.setEllipsoid(crs.ellipsoidAcronym())
        
    def distance(self, geom1: QgsGeometry, geom2: QgsGeometry) -> float:
        """Calculate distance between geometries in meters"""
        if not geom1 or not geom2:
            return float('inf')
        
        # If CRS is geographic, transform to meters for distance
        if self.crs.isGeographic():
            return geom1.distance(geom2) * 111320  # Rough conversion
        else:
            return geom1.distance(geom2)
    
    def distance_points(self, point1: QgsPointXY, point2: QgsPointXY) -> float:
        """Calculate distance between points. Fast path for projected CRS."""
        if not self.crs.isGeographic():
            dx = point2.x() - point1.x()
            dy = point2.y() - point1.y()
            return math.sqrt(dx * dx + dy * dy)
        return self.distance(
            QgsGeometry.fromPointXY(point1),
            QgsGeometry.fromPointXY(point2)
        )
    
    def distance_along_road(self, road_geom: QgsGeometry, 
                           point1: QgsPointXY, point2: QgsPointXY) -> float:
        """Calculate distance along road geometry"""
        # Project points to road
        point1_proj = road_geom.nearestPoint(QgsGeometry.fromPointXY(point1))
        point2_proj = road_geom.nearestPoint(QgsGeometry.fromPointXY(point2))
        
        # Get distances along line
        dist1 = road_geom.lineLocatePoint(point1_proj)
        dist2 = road_geom.lineLocatePoint(point2_proj)
        
        # Return difference
        return abs(dist2 - dist1) * road_geom.length()

# =============================================================================
# LAYER VALIDATION AND LOADING
# =============================================================================

def validate_and_load_layers(project: QgsProject, logger: PerformanceLogger) -> Dict[str, QgsVectorLayer]:
    """Validate and load required layers from project"""
    layers = {}
    
    for layer_type, layer_name in LAYER_NAMES.items():
        layer = project.mapLayersByName(layer_name)
        if not layer:
            raise LayerNotFoundError(f"Required layer '{layer_name}' not found in project")
        
        layers[layer_type] = layer[0]
        logger.log(f"Loaded layer: {layer_name}")
        
        # Validate CRS
        crs = layer[0].crs()
        if crs.isGeographic():
            raise CRSError(f"Layer '{layer_name}' has geographic CRS {crs.authid()}. Projected CRS required.")
        
        logger.log(f"  - CRS: {crs.authid()}")
        logger.log(f"  - Feature count: {layer[0].featureCount()}")
    
    # check homepass field
    home_layer = layers.get('homepass')
    if home_layer:
        fields = [f.name() for f in home_layer.fields()]
        if HOMEPASS_FIELD_NAME not in fields:
            raise ValidationError(f"Homepass field '{HOMEPASS_FIELD_NAME}' not found in layer {home_layer.name()}")

    # optional layers
    for key, names in OPTIONAL_LAYER_CANDIDATES.items():
        for nm in names:
            found = project.mapLayersByName(nm)
            if found:
                layers[key] = found[0]
                logger.log(f"Loaded optional layer: {nm}")
                break
    
    return layers

# =============================================================================
# HELPERS
# =============================================================================

def _parse_homepass_value(value) -> int:
    """Convert homepass field value to integer. Handles text/mixed types safely.
    
    Rule: SUM(homepass) per FAT, not COUNT(points). Values must be numeric.
    If conversion fails, treat as 0.
    """
    try:
        hp = float(value)
    except (ValueError, TypeError):
        return 0
    return int(hp)


def safe_homepass_sum(features, field_name: str) -> int:
    """Sum homepass values from a list of features safely.
    
    IMPORTANT: Uses SUM(homepass values), NOT COUNT(points).
    Null or non-numeric values contribute 0.
    """
    total = 0
    for f in features:
        try:
            value = f[field_name]
        except Exception:
            value = None
        hp = _parse_homepass_value(value)
        total += hp
    return total

def _fat_sort_key(fat: FAT) -> Tuple[int, int, float, float, int]:
    """Deterministic FAT ordering key used for naming and sequencing."""
    return (
        int(fat.cascade_order),
        int(fat.road_id),
        round(float(fat.geometry.x()), 6),
        round(float(fat.geometry.y()), 6),
        int(fat.total_customers)
    )


def _cluster_fdts(fdts: List[FDT],
                  max_per_cluster: int,
                  dist_calc: DistanceCalculator) -> List[List[FDT]]:
    """Divide FDTs into equal, geographically compact regions for closure assignment.

    Design principle (matches reference images)
    ───────────────────────────────────────────
    Each closure controls one contiguous geographic region of streets.
    FDTs close to each other (on the same or nearby roads) stay together.
    All regions should contain approximately the same number of FDTs.

    Algorithm
    ---------
    Step 1 — Determine k (number of closures):
        k = ceil(N / target) where target = round(max_per_cluster * 0.75).
        This produces fuller clusters — matching the images where each of
        the 3 closures serves ~15-20 FDTs rather than 5 half-empty ones.

    Step 2 — K-Means on FDT coordinates:
        Standard K-Means finds k spatially compact centroids.
        Each FDT is assigned to its nearest centroid.

    Step 3 — Balanced reallocation:
        Move border FDTs between clusters until sizes are within ±1 of N/k.
        "Border FDT" = the FDT in the larger cluster that is closest to the
        smaller cluster's centroid AND within a geographic distance limit
        (prevents cross-city moves that destroy zone compactness).
        The distance limit is the diameter of a cluster of max_per_cluster
        FDTs at typical 250m road spacing = max_per_cluster * 250 / 2 m.

    Step 4 — Hard-cap overflow split:
        Any cluster still over max_per_cluster is chunked into extra clusters.

    Returns list of FDT lists (one per cluster/closure).
    """
    n = len(fdts)
    if n == 0:
        return []
    if n <= max_per_cluster:
        return [fdts[:]]

    # ── Step 1: number of clusters ────────────────────────────────────────────
    target_per = max(1, round(max_per_cluster * 0.75))   # ~18 for cap=24
    k = max(2, math.ceil(n / target_per))
    k = min(k, n)

    # ── Step 2: K-Means ───────────────────────────────────────────────────────
    coords = np.array([[f.geometry.x(), f.geometry.y()] for f in fdts],
                      dtype=float)
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500)
    labels = km.fit_predict(coords)

    clusters: List[List[FDT]] = [[] for _ in range(k)]
    for fdt, label in zip(fdts, labels):
        clusters[label].append(fdt)
    clusters = [c for c in clusters if c]
    k = len(clusters)
    if k == 0:
        return [fdts[:]]

    # ── Step 3: balanced reallocation ─────────────────────────────────────────
    # Distance limit: half the expected diameter of a full cluster.
    geo_limit = (max_per_cluster * MAX_BOX_TO_BOX_DISTANCE) / 2.0

    def _ctr(cl: List[FDT]) -> QgsPointXY:
        cx = sum(f.geometry.x() for f in cl) / len(cl)
        cy = sum(f.geometry.y() for f in cl) / len(cl)
        return QgsPointXY(cx, cy)

    for _ in range(n * 4):
        sizes   = [len(c) for c in clusters]
        max_sz  = max(sizes)
        min_sz  = min(sizes)
        if max_sz - min_sz <= 1 and max_sz <= max_per_cluster:
            break

        big_idx   = sizes.index(max_sz)
        small_idx = sizes.index(min_sz)
        if big_idx == small_idx:
            break

        big_cl   = clusters[big_idx]
        small_cl = clusters[small_idx]
        small_c  = _ctr(small_cl)

        # Find closest border FDT in the big cluster within distance limit
        best, best_d = None, float('inf')
        for fdt in big_cl:
            d = dist_calc.distance_points(fdt.geometry, small_c)
            if d < best_d and d <= geo_limit:
                best_d, best = d, fdt

        if best is None:
            break  # no valid candidate — accept imbalance

        big_cl.remove(best)
        small_cl.append(best)

    # ── Step 4: enforce hard cap ──────────────────────────────────────────────
    final: List[List[FDT]] = []
    for cl in clusters:
        if not cl:
            continue
        cl.sort(key=lambda f: (round(f.geometry.x(), 6), round(f.geometry.y(), 6)))
        for i in range(0, len(cl), max_per_cluster):
            chunk = cl[i:i + max_per_cluster]
            if chunk:
                final.append(chunk)

    return final


def _cluster_closures(closures: List[DomeClosure],
                      max_per_cluster: int,
                      dist_calc: DistanceCalculator) -> List[List[DomeClosure]]:
    """Greedy geographic clustering of dome closures.

    Same algorithm used for FDTs but operating on closures.
    """
    clusters: List[List[DomeClosure]] = []
    remaining = closures.copy()
    remaining.sort(key=lambda c: (round(c.geometry.x(), 6), round(c.geometry.y(), 6)))

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        remaining.sort(key=lambda c: dist_calc.distance_points(seed.geometry, c.geometry))
        while remaining and len(cluster) < max_per_cluster:
            cluster.append(remaining.pop(0))
        clusters.append(cluster)
    return clusters


def _generate_pickup_points(closures: List[DomeClosure],
                            roads: Dict[int, RoadSegment],
                            road_index: QgsSpatialIndex,
                            dist_calc: DistanceCalculator,
                            logger: PerformanceLogger) -> List[QgsPointXY]:
    """Automatically generate one or more pickup points for the closures.

    - Closures are clustered spatially using a simple greedy algorithm;
    - Each cluster yields a centroid which is then snapped to the nearest road
      to ensure a road-accessible pickup location.
    - The resulting points are assigned back to each closure's ``pickup_geometry``
      (and ``pickup_name``) so that feeder routing can use them.
    """
    if not closures:
        return []
    # decide number of clusters based on desired closures per pickup
    num_clusters = max(1, math.ceil(len(closures) / CLOSURES_PER_PICKUP))
    clusters = _cluster_closures(closures, math.ceil(len(closures) / num_clusters), dist_calc)

    generated: List[QgsPointXY] = []
    for ci, cluster in enumerate(clusters, start=1):
        if not cluster:
            continue
        # compute centroid of cluster
        cx = sum(c.geometry.x() for c in cluster) / len(cluster)
        cy = sum(c.geometry.y() for c in cluster) / len(cluster)
        centroid = QgsPointXY(cx, cy)
        # snap centroid to nearest road if possible
        if road_index:
            nearest = road_index.nearestNeighbor(centroid, 1)
            if nearest:
                road = roads.get(nearest[0])
                if road and road.geometry and not road.geometry.isEmpty():
                    # closestSegmentWithContext now returns four values (distance, point, vertex_index, left_of)
                    _d, snap_pt, _idx, _left = road.geometry.closestSegmentWithContext(centroid)
                    centroid = QgsPointXY(snap_pt.x(), snap_pt.y())
        pickup_name = f"PICKUP_{ci}"
        generated.append(centroid)
        for c in cluster:
            c.pickup_geometry = centroid
            c.pickup_name = pickup_name
    logger.log(f"Generated {len(generated)} automatic pickup point(s)")
    return generated

# =============================================================================
# HELPERS
# =============================================================================

def distribute_points_around_pole(pole_point: QgsPointXY, fat_count: int, radius: float = FAT_OFFSET_RADIUS) -> List[QgsPointXY]:
    """Return a list of points evenly spaced in a circle around pole_point.

    fat_count: number of offsets required.
    radius: distance from pole center in meters.
    """
    pts: List[QgsPointXY] = []
    if fat_count <= 0:
        return pts
    for i in range(fat_count):
        angle = 2 * math.pi * i / fat_count
        x = pole_point.x() + radius * math.cos(angle)
        y = pole_point.y() + radius * math.sin(angle)
        pts.append(QgsPointXY(x, y))
    return pts

def _line_intersects_other_roads(line_geom: QgsGeometry,
                                 candidate_road_id: int,
                                 roads: Dict[int, RoadSegment],
                                 road_index: Optional[QgsSpatialIndex]) -> bool:
    """Return True if line truly crosses any road except candidate_road_id.

    Uses crosses() instead of intersects() to exclude roads that merely
    touch at a shared endpoint (e.g. the target road itself meeting an
    adjacent road at a junction).  Bounding-box pre-filter is still used
    for performance, but only geometries that actually cross the line
    (not just overlap in bbox) are counted.
    """
    if not line_geom or line_geom.isEmpty():
        return False
    bbox = line_geom.boundingBox()
    if road_index:
        candidate_ids = road_index.intersects(bbox)
    else:
        candidate_ids = list(roads.keys())
    for rid in candidate_ids:
        if rid == candidate_road_id:
            continue
        road = roads.get(rid)
        if not road or not road.geometry or road.geometry.isEmpty():
            continue
        # crosses() is True only when the interiors intersect but neither
        # geometry is wholly contained — exactly what a genuine road crossing
        # looks like.  touching at a single endpoint (junction) returns False.
        if line_geom.crosses(road.geometry):
            return True
    return False

def _line_intersects_buildings(line_geom: QgsGeometry,
                               building_index: Optional[QgsSpatialIndex],
                               building_geoms: Dict[int, QgsGeometry]) -> bool:
    """Return True if line intersects any building geometry."""
    if not line_geom or line_geom.isEmpty() or not building_index:
        return False
    for bid in building_index.intersects(line_geom.boundingBox()):
        bgeom = building_geoms.get(bid)
        if bgeom and not bgeom.isEmpty() and line_geom.intersects(bgeom):
            return True
    return False

def _road_vertices(road_geom: QgsGeometry) -> List[QgsPointXY]:
    """Extract vertices from line/multiline road geometry."""
    if not road_geom or road_geom.isEmpty():
        return []
    if road_geom.isMultipart():
        pts: List[QgsPointXY] = []
        for part in road_geom.asMultiPolyline():
            pts.extend(part)
        return pts
    return list(road_geom.asPolyline())

def _point_side_of_road(point: QgsPointXY, road_geom: QgsGeometry) -> int:
    """Return +1 (left), -1 (right), 0 (undetermined) relative to road direction."""
    if not road_geom or road_geom.isEmpty():
        return 0
    try:
        _d, _pt, _idx, left_of = road_geom.closestSegmentWithContext(point)
        if left_of > 0:
            return 1
        if left_of < 0:
            return -1
    except Exception:
        pass
    return 0

def _is_near_road_junction(point: QgsPointXY,
                           road_geom: QgsGeometry,
                           dist_calc: DistanceCalculator,
                           threshold_m: float = 10.0) -> bool:
    """Corner-house check: point is near segment start/end nodes."""
    vertices = _road_vertices(road_geom)
    if len(vertices) < 2:
        return False
    d_start = dist_calc.distance_points(point, vertices[0])
    d_end = dist_calc.distance_points(point, vertices[-1])
    return min(d_start, d_end) <= threshold_m

def _offset_point_by_side(point: QgsPointXY,
                          road_geom: QgsGeometry,
                          side_sign: int,
                          offset_distance: float = SNAP_OFFSET_DISTANCE) -> QgsPointXY:
    """Offset point perpendicular to nearest road segment toward side_sign."""
    if side_sign == 0:
        return point
    vertices = _road_vertices(road_geom)
    if len(vertices) < 2:
        return point

    road_point = road_geom.nearestPoint(QgsGeometry.fromPointXY(point))
    best_seg = None
    best_dist = float('inf')
    for i in range(len(vertices) - 1):
        seg = QgsGeometry.fromPolylineXY([vertices[i], vertices[i + 1]])
        d = seg.distance(road_point)
        if d < best_dist:
            best_dist = d
            best_seg = (vertices[i], vertices[i + 1])
    if not best_seg:
        return point

    pt1, pt2 = best_seg
    dx = pt2.x() - pt1.x()
    dy = pt2.y() - pt1.y()
    length = math.sqrt(dx * dx + dy * dy)
    if length <= 0:
        return point
    dx /= length
    dy /= length

    # Left side relative to segment direction.
    if side_sign > 0:
        perp_x, perp_y = -dy, dx
    else:
        perp_x, perp_y = dy, -dx

    return QgsPointXY(
        point.x() + perp_x * offset_distance,
        point.y() + perp_y * offset_distance
    )

# =============================================================================
# ROAD PROCESSING
# =============================================================================

def process_roads(road_layer: QgsVectorLayer, 
                  index_mgr: SpatialIndexManager,
                  logger: PerformanceLogger) -> Dict[int, RoadSegment]:
    """Process road layer and build road segments dictionary"""
    logger.start_module("Road Processing")
    
    roads = {}
    
    # Build spatial index for roads
    index_mgr.build_index(road_layer, cache_geoms=True)
    
    for feat in road_layer.getFeatures():
        geom = feat.geometry()
        if geom and not geom.isEmpty():
            road = RoadSegment(
                id=feat.id(),
                geometry=geom,
                fid=feat.id(),
                length=geom.length()
            )
            roads[feat.id()] = road
    
    logger.end_module("Road Processing")
    logger.log(f"Processed {len(roads)} road segments")
    
    return roads

# =============================================================================
# HOMEPASS PROCESSING AND ROAD ASSIGNMENT
# =============================================================================

def assign_homepasses_to_roads(homepass_layer: QgsVectorLayer,
                               road_layer: QgsVectorLayer,
                               roads: Dict[int, RoadSegment],
                               index_mgr: SpatialIndexManager,
                               dist_calc: DistanceCalculator,
                               logger: PerformanceLogger,
                               error_collector: ErrorCollector) -> List[Homepass]:
    """Assign each homepass to road frontage using top-3 nearest + crossing filter."""
    logger.start_module("Homepass Assignment")
    
    homepasses = []
    # build or retrieve index for the road layer instead of using layer.parent()
    road_index = index_mgr.get_index(road_layer)
    
    features = list(homepass_layer.getFeatures())
    original_total = safe_homepass_sum(features, HOMEPASS_FIELD_NAME)

    hp_field_idx = homepass_layer.fields().indexOf(HOMEPASS_FIELD_NAME)

    for feat in features:
        geom = feat.geometry()
        if not geom or geom.isEmpty():
            continue

        point = geom.asPoint()
        point_geom = QgsGeometry.fromPointXY(point)

        if hp_field_idx >= 0:
            hp_val = _parse_homepass_value(feat.attribute(hp_field_idx))
        else:
            hp_val = 0

        candidate_ids = road_index.nearestNeighbor(point, 5)
        best_road_id = None
        best_dist = float('inf')

        for rid in candidate_ids:
            if rid not in roads:
                continue
            road = roads[rid]
            road_geom = road.geometry
            near_pt = road_geom.nearestPoint(point_geom).asPoint()
            d = dist_calc.distance_points(point, near_pt)
            # Only consider roads within a reasonable service distance
            if d > MAX_BOX_TO_CUSTOMER_DISTANCE:
                continue
            access_line = QgsGeometry.fromPolylineXY([point, near_pt])
            if _line_intersects_other_roads(access_line, rid, roads, road_index):
                continue
            if d < best_dist:
                best_dist = d
                best_road_id = rid

        # Fallback: among all candidates pick the geometrically nearest road
        # that is reachable (no road crossing), ignoring distance cap.
        # This ensures we never silently assign to the wrong street.
        if best_road_id is None and candidate_ids:
            fallback_best_id = None
            fallback_best_dist = float('inf')
            for rid in candidate_ids:
                if rid not in roads:
                    continue
                road = roads[rid]
                near_pt = road.geometry.nearestPoint(point_geom).asPoint()
                d = dist_calc.distance_points(point, near_pt)
                access_line = QgsGeometry.fromPolylineXY([point, near_pt])
                if _line_intersects_other_roads(access_line, rid, roads, road_index):
                    continue
                if d < fallback_best_dist:
                    fallback_best_dist = d
                    fallback_best_id = rid
            # If still nothing passes crossing check, take the absolute nearest
            if fallback_best_id is None:
                for rid in candidate_ids:
                    if rid not in roads:
                        continue
                    near_pt = roads[rid].geometry.nearestPoint(point_geom).asPoint()
                    d = dist_calc.distance_points(point, near_pt)
                    if d < fallback_best_dist:
                        fallback_best_dist = d
                        fallback_best_id = rid
            if fallback_best_id is not None:
                best_road_id = fallback_best_id
                best_dist = fallback_best_dist

        if best_road_id is not None and best_road_id in roads:
            road = roads[best_road_id]
            hp = Homepass(
                fid=feat.id(),
                geometry=point,
                customer_count=hp_val,
                road_id=best_road_id,
                distance_to_road=best_dist
            )
            homepasses.append(hp)
            road.homepasses.append(hp)
            road.total_customers += hp_val
    
    # verify sum consistency
    assigned_total = sum(hp.customer_count for hp in homepasses)
    if assigned_total != original_total:
        error_collector.add_error(
            "HOMEPASS_MISMATCH",
            "all",
            f"Assigned total {assigned_total} differs from original {original_total}"
        )
    
    logger.end_module("Homepass Assignment")
    logger.log(f"Processed {len(homepasses)} homepasses, total customers: {assigned_total}")
    
    return homepasses

# =============================================================================
# FAT GENERATION
# =============================================================================

def generate_fats(roads: Dict[int, RoadSegment],
                  road_layer: QgsVectorLayer,
                  index_mgr: SpatialIndexManager,
                  building_layer: Optional[QgsVectorLayer],
                  dist_calc: DistanceCalculator,
                  error_collector: ErrorCollector,
                  logger: PerformanceLogger) -> List[FAT]:
    """Generate FATs for each road segment using demand-driven allocation with advanced spatial constraints."""
    logger.start_module("FAT Generation")

    road_index = index_mgr.get_index(road_layer)
    building_index = None
    building_geoms: Dict[int, QgsGeometry] = {}
    if building_layer:
        building_index = index_mgr.build_index(building_layer, cache_geoms=True)
        building_geoms = index_mgr.geometries.get(building_layer.id(), {})

    fats: List[FAT] = []
    fat_counter = 1

    for road_id in sorted(roads.keys()):
        road = roads[road_id]
        if not road.homepasses or road.total_customers == 0:
            continue

        road_geom = road.geometry
        
        # Strict demand-driven FAT count: exactly how many FATs are needed.
        required_fat = math.ceil(road.total_customers / MAX_FAT_CAPACITY)
        road.fat_count = required_fat

        # Sort homepasses along road for deterministic allocation
        homepasses = sorted(
            road.homepasses,
            key=lambda hp: road_geom.lineLocatePoint(QgsGeometry.fromPointXY(hp.geometry))
        )

        # Build exactly `required_fat` capacity-safe groups using demand-driven allocation.
        # Homepass weights are respected and may be split across FATs.
        clusters: List[List[Homepass]] = []
        remaining_customers = int(road.total_customers)
        hp_idx = 0
        hp_remaining = _parse_homepass_value(homepasses[0].customer_count) if homepasses else 0
        if hp_remaining < 0:
            hp_remaining = 0

        for fat_slot in range(required_fat):
            remaining_fats = required_fat - fat_slot
            target_load = min(MAX_FAT_CAPACITY, remaining_customers - (remaining_fats - 1))
            if target_load <= 0:
                raise ValidationError(
                    f"Road {road_id}: invalid FAT target load {target_load} in slot {fat_slot + 1}"
                )

            cluster: List[Homepass] = []
            cluster_load = 0

            while cluster_load < target_load:
                if hp_idx >= len(homepasses):
                    raise ValidationError(
                        f"Road {road_id}: ran out of homepasses during strict FAT allocation"
                    )

                hp = homepasses[hp_idx]
                if hp_remaining <= 0:
                    hp_idx += 1
                    if hp_idx < len(homepasses):
                        hp_remaining = _parse_homepass_value(homepasses[hp_idx].customer_count)
                        if hp_remaining < 0:
                            hp_remaining = 0
                    continue

                take = min(hp_remaining, target_load - cluster_load)
                cluster.append(
                    Homepass(
                        fid=hp.fid,
                        geometry=hp.geometry,
                        customer_count=take,
                        road_id=hp.road_id,
                        distance_to_road=hp.distance_to_road,
                        fat_id=hp.fat_id
                    )
                )
                cluster_load += take
                hp_remaining -= take
                remaining_customers -= take

            clusters.append(cluster)

        if remaining_customers != 0:
            raise ValidationError(
                f"Road {road_id}: unallocated customers after FAT grouping: {remaining_customers}"
            )
        
        # Convert clusters to FATs with advanced spatial validation
        local_fats: List[FAT] = []
        for c in clusters:
            fat = create_fat_from_homepasses(
                c, road_id, fat_counter, road_geom, dist_calc,
                roads=roads, road_index=road_index,
                building_index=building_index, building_geoms=building_geoms
            )
            local_fats.append(fat)
            fat_counter += 1

        # Strict validation: created FAT count must match demand-derived count
        if len(local_fats) != required_fat:
            raise ValidationError(
                f"Road {road_id}: created {len(local_fats)} FATs but required {required_fat}"
            )
        
        # Validate all FATs meet constraints
        for fat, cluster in zip(local_fats, clusters):
            if fat.total_customers > MAX_FAT_CAPACITY:
                raise ValidationError(
                    f"FAT {fat.name} has {fat.total_customers} customers, exceeds {MAX_FAT_CAPACITY}"
                )
            if fat.total_customers <= 0:
                raise ValidationError(f"FAT {fat.name} is empty")
            
            # Validate drop distance and crossing constraints (advanced features)
            # Allow outlier customers that exceed drop distance (report as warnings)
            outliers = []
            for hp in cluster:
                if hp.road_id != fat.road_id:
                    raise ValidationError(f"FAT {fat.name} has cross-street assignment for customer {hp.fid}")
                d = dist_calc.distance_points(hp.geometry, fat.geometry)
                if d > MAX_BOX_TO_CUSTOMER_DISTANCE:
                    outliers.append((hp.fid, d))
                    error_collector.add_error(
                        "DROP_DISTANCE_EXCEEDED",
                        f"FAT {fat.name}-Customer {hp.fid}",
                        f"Distance {d:.2f}m exceeds {MAX_BOX_TO_CUSTOMER_DISTANCE}m (data quality issue)"
                    )
                    continue
                
                # Check for road/building crossing (with corner exception)
                is_corner = _is_near_road_junction(hp.geometry, road_geom, dist_calc, 10.0)
                link = QgsGeometry.fromPolylineXY([fat.geometry, hp.geometry])
                crosses_road = _line_intersects_other_roads(link, road_id, roads, road_index)
                crosses_building = _line_intersects_buildings(link, building_index, building_geoms)
                
                if is_corner:
                    if crosses_building:
                        error_collector.add_error(
                            "GEOMETRY_CROSSING",
                            f"FAT {fat.name}-Customer {hp.fid}",
                            "Path crosses building (corner location)"
                        )
                else:
                    if crosses_road or crosses_building:
                        error_collector.add_error(
                            "GEOMETRY_CROSSING",
                            f"FAT {fat.name}-Customer {hp.fid}",
                            f"Path crosses {'road' if crosses_road else 'building'}"
                        )
            
            if outliers:
                logger.log(
                    f"FAT {fat.name} has {len(outliers)} outlier customer(s) exceeding drop distance: "
                    f"{[fid for fid, _ in outliers]}"
                )

        logger.log(
            f"Road {road_id} → {road.total_customers} customers → {required_fat} FAT required → {len(local_fats)} FAT created"
        )
        fats.extend(local_fats)

    logger.end_module("FAT Generation")
    logger.log(f"Generated {len(fats)} FATs")

    # Global consistency check
    total_fat_customers = sum(int(f.total_customers) for f in fats)
    total_road_customers = sum(int(r.total_customers) for r in roads.values())
    if total_fat_customers != total_road_customers:
        raise ValidationError(
            f"FAT/customer mismatch: FAT total={total_fat_customers}, "
            f"road total={total_road_customers}"
        )

    return fats
def split_cluster_by_capacity(homepasses: List[Homepass], max_capacity: int) -> List[List[Homepass]]:
    """Split a cluster of homepasses into multiple groups respecting capacity.
    
    Uses SUM(homepass values), not COUNT(points). max_capacity is the max
    sum per group.
    """
    groups = []
    current_group = []
    current_customers = 0  # SUM of homepass values in current_group
    
    for hp in homepasses:
        remaining = _parse_homepass_value(hp.customer_count)
        if remaining <= 0:
            continue

        # Split one homepass into chunks when it exceeds FAT capacity.
        while remaining > 0:
            chunk = min(remaining, max_capacity)
            hp_part = Homepass(
                fid=hp.fid,
                geometry=hp.geometry,
                customer_count=chunk,
                road_id=hp.road_id,
                distance_to_road=hp.distance_to_road,
                fat_id=hp.fat_id
            )

            if current_customers + chunk <= max_capacity:
                current_group.append(hp_part)
                current_customers += chunk
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [hp_part]
                current_customers = chunk

            remaining -= chunk
    
    if current_group:
        groups.append(current_group)
    
    return groups

def create_fat_from_homepasses(homepasses: List[Homepass], road_id: int, 
                               counter: int, road_geom: QgsGeometry,
                               dist_calc: Optional[DistanceCalculator] = None,
                               roads: Optional[Dict[int, RoadSegment]] = None,
                               road_index: Optional[QgsSpatialIndex] = None,
                               building_index: Optional[QgsSpatialIndex] = None,
                               building_geoms: Optional[Dict[int, QgsGeometry]] = None) -> FAT:
    """Create a FAT from a list of homepasses"""
    # Calculate centroid of homepasses
    points = [hp.geometry for hp in homepasses]
    centroid_x = sum(p.x() for p in points) / len(points)
    centroid_y = sum(p.y() for p in points) / len(points)
    centroid = QgsPointXY(centroid_x, centroid_y)
    
    # Default placement is road-projected centroid.
    projected = road_geom.nearestPoint(QgsGeometry.fromPointXY(centroid))
    proj_point = projected.asPoint()

    # Determine dominant frontage side and place FAT on that side.
    side_votes: List[int] = []
    for hp in homepasses:
        s = _point_side_of_road(hp.geometry, road_geom)
        if s != 0:
            side_votes.append(1 if s > 0 else -1)
    side_sign = 1 if not side_votes or sum(side_votes) >= 0 else -1
    side_point = _offset_point_by_side(proj_point, road_geom, side_sign, SNAP_OFFSET_DISTANCE)

    def _candidate_passes_crossing(cand: QgsPointXY) -> bool:
        """Check that links from candidate to all customers avoid road/building crossing (with corner exception)."""
        if roads is None or road_index is None:
            return True
        for hp in homepasses:
            is_corner = _is_near_road_junction(hp.geometry, road_geom, dist_calc, 10.0)
            link = QgsGeometry.fromPolylineXY([cand, hp.geometry])
            crosses_road = _line_intersects_other_roads(link, road_id, roads, road_index)
            crosses_building = _line_intersects_buildings(
                link, building_index or None, building_geoms or {}
            )
            if is_corner:
                if crosses_building:
                    return False
            else:
                if crosses_road or crosses_building:
                    return False
        return True

    fat_point = side_point
    if dist_calc:
        # Prefer side-point, then projected, then centroid.
        # Must satisfy both drop limit AND crossing constraints (no road/building crossing).
        candidates = [side_point, proj_point, centroid]
        best_pt = side_point
        best_max = float('inf')
        for cand in candidates:
            max_drop = 0.0
            for hp in homepasses:
                d = dist_calc.distance_points(hp.geometry, cand)
                if d > max_drop:
                    max_drop = d
            if max_drop > MAX_BOX_TO_CUSTOMER_DISTANCE:
                continue
            if not _candidate_passes_crossing(cand):
                continue
            if max_drop < best_max:
                best_max = max_drop
                best_pt = cand
                break
        else:
            # No candidate passed both constraints; use projected (matches _cluster_fat_point)
            best_pt = proj_point
            if not _candidate_passes_crossing(best_pt):
                best_pt = centroid
        fat_point = best_pt
    
    # Get cascade order (based on position along road)
    cascade_order = int(road_geom.lineLocatePoint(projected) * 1000)
    
    # SUM of homepass values (not count of points)
    total_customers = sum(_parse_homepass_value(hp.customer_count) for hp in homepasses)
    
    # customer_allocations: fid -> count for this FAT (supports split points across FATs)
    customer_allocations = {hp.fid: _parse_homepass_value(hp.customer_count) for hp in homepasses}
    
    return FAT(
        # Temporary deterministic placeholder, replaced by strict
        # hierarchical FAT code after FDT/wing assignment.
        name=f"TEMP-FAT-{counter:06d}",
        geometry=fat_point,
        road_id=road_id,
        total_customers=total_customers,
        cascade_order=cascade_order,
        from_device="",
        to_device="",
        wing="",
        customers=[hp.fid for hp in homepasses],
        customer_allocations=customer_allocations
    )

# =============================================================================
# FAT CASCADING
# =============================================================================

def create_fat_cascade(fats: List[FAT],
                       roads: Dict[int, RoadSegment],
                       dist_calc: DistanceCalculator,
                       error_collector: ErrorCollector,
                       logger: PerformanceLogger) -> List[FAT]:
    """Create linear cascade of FATs along roads"""
    logger.start_module("FAT Cascading")
    
    # Group FATs by road
    fats_by_road = defaultdict(list)
    for fat in fats:
        fats_by_road[fat.road_id].append(fat)
    
    cascaded_fats = []
    
    for road_id, road_fats in fats_by_road.items():
        # Sort by cascade order (position along road)
        road_fats.sort(key=lambda f: f.cascade_order)
        
        # Determine road direction
        road_geom = roads[road_id].geometry
        
        # Create cascade chain
        for i, fat in enumerate(road_fats):
            if i == 0:
                fat.from_device = "FDT"  # First FAT connects to FDT
            else:
                fat.from_device = road_fats[i-1].name
            
            if i < len(road_fats) - 1:
                fat.to_device = road_fats[i+1].name
            else:
                fat.to_device = ""  # Last FAT has no downstream connection
            
            # Check distance between consecutive FATs
            if i > 0:
                prev_fat = road_fats[i-1]
                distance = dist_calc.distance_along_road(road_geom, prev_fat.geometry, fat.geometry)
                
                if distance > MAX_FAT_TO_FAT_DISTANCE:
                    # Need to insert intermediate FATs
                    error_collector.add_error(
                        "FAT_DISTANCE_EXCEEDED",
                        f"{prev_fat.name}-{fat.name}",
                        f"Distance {distance:.2f}m exceeds maximum {MAX_FAT_TO_FAT_DISTANCE}m"
                    )
                    
                    # Insert intermediate FATs
                    intermediate_fats = insert_intermediate_fats(
                        prev_fat, fat, road_geom, dist_calc, len(cascaded_fats)
                    )
                    cascaded_fats.extend(intermediate_fats)
        
        cascaded_fats.extend(road_fats)
    
    logger.end_module("FAT Cascading")
    logger.log(f"Cascaded {len(cascaded_fats)} FATs")
    
    return cascaded_fats

def insert_intermediate_fats(start_fat: FAT, end_fat: FAT, 
                            road_geom: QgsGeometry,
                            dist_calc: DistanceCalculator,
                            base_counter: int) -> List[FAT]:
    """Insert intermediate FATs when distance exceeds maximum"""
    distance = dist_calc.distance_along_road(road_geom, start_fat.geometry, end_fat.geometry)
    num_intermediate = math.ceil(distance / MAX_FAT_TO_FAT_DISTANCE) - 1
    
    if num_intermediate <= 0:
        return []
    
    # Calculate positions along road
    start_pos = road_geom.lineLocatePoint(QgsGeometry.fromPointXY(start_fat.geometry))
    end_pos = road_geom.lineLocatePoint(QgsGeometry.fromPointXY(end_fat.geometry))
    
    intermediate_fats = []
    for i in range(1, num_intermediate + 1):
        # Interpolate position
        t = i / (num_intermediate + 1)
        pos = start_pos + t * (end_pos - start_pos)
        
        # Get point at position
        point_geom = road_geom.interpolate(pos * road_geom.length())
        if point_geom:
            point = point_geom.asPoint()
            
            # Create intermediate FAT
            fat = FAT(
                name=f"FAT_{base_counter + i:06d}_INT",
                geometry=point,
                road_id=start_fat.road_id,
                total_customers=0,  # No customers
                cascade_order=int(pos * 1000),
                from_device="",
                to_device="",
                wing=""
            )
            
            # Set connections
            if i == 1:
                fat.from_device = start_fat.name
            else:
                fat.from_device = intermediate_fats[-1].name
            
            if i == num_intermediate:
                fat.to_device = end_fat.name
            else:
                fat.to_device = ""  # Will be set by next FAT
            
            intermediate_fats.append(fat)
    
    # Update connections
    for i in range(len(intermediate_fats)):
        if i < len(intermediate_fats) - 1:
            intermediate_fats[i].to_device = intermediate_fats[i+1].name
    
    return intermediate_fats

# =============================================================================
# WING DETERMINATION
# =============================================================================

def assign_wings_per_fdt(fdts: List[FDT], logger: PerformanceLogger):
    """Assign FAT wings per FDT with strictly non-overlapping contiguous zones.

    Wing layout
    -----------
    FATs in each FDT are sorted by their road position (cascade_order).
    They are then split exactly at the midpoint of the sorted list:

      Wing A  –  the LOWER half (road positions before the midpoint)
                 sorted DESCENDING so A-01 is closest to the FDT boundary
                 and A-04 is the far end of the chain.

      Wing B  –  the UPPER half (road positions at/after the midpoint)
                 sorted ASCENDING so B-01 is closest to the FDT boundary
                 and B-04 is the far end.

    The FDT is placed at the BOUNDARY between Wing A and Wing B.

    Borrowed FATs (from a connected road)
    --------------------------------------
    A borrowed FAT belongs to a DIFFERENT road than the FDT's primary road.
    It must always be placed at index 0 of its wing (= A-01 or B-01), because:
      - It physically lives at the junction next to the FDT
      - The cable must run FDT → borrowed_FAT (junction) → primary-road FATs outward
      - Placing it at the far end (A-04/B-04) would force the cable to travel
        all the way along the primary road and then jump to another street entirely

    Cascade chain:
        FDT → A-01 (1×9) → A-02 (1×9) → A-03 (1×9) → A-04 (1×8)
        FDT → B-01 (1×9) → B-02 (1×9) → B-03 (1×9) → B-04 (1×8)
    """
    logger.start_module("Wing Distribution")

    for fdt_index, fdt in enumerate(fdts, start=1):
        fdt_serial = f"{fdt_index:04d}"
        fdt_code   = f"FDT-{fdt_serial}"
        fdt.fdt_serial = fdt_serial
        fdt.fdt_code   = fdt_code
        fdt.name       = fdt_code

        all_fats   = list(fdt.fats)
        total_fats = len(all_fats)

        if total_fats > MAX_FAT_PER_FDT:
            raise ValidationError(
                f"{fdt.name} has {total_fats} FATs; exceeds {MAX_FAT_PER_FDT}"
            )
        if total_fats < MIN_FAT_PER_FDT:
            logger.log(
                f"WARNING: {fdt.name} has only {total_fats} FATs "
                f"(minimum {MIN_FAT_PER_FDT}); isolated road."
            )

        # ----------------------------------------------------------------
        # Step 1: separate primary-road FATs from borrowed FATs.
        # Primary road = the road_id shared by the majority of FATs.
        # ----------------------------------------------------------------
        from collections import Counter as _Counter
        primary_road_id = _Counter(f.road_id for f in all_fats).most_common(1)[0][0]

        primary_fats = sorted(
            [f for f in all_fats if f.road_id == primary_road_id],
            key=lambda f: f.cascade_order   # ascending road position
        )
        borrowed_fats = sorted(
            [f for f in all_fats if f.road_id != primary_road_id],
            key=lambda f: f.cascade_order
        )

        # ----------------------------------------------------------------
        # Step 2: split primary FATs exactly at the midpoint.
        # Lower half → Wing A (reversed: closest to FDT boundary at [0])
        # Upper half → Wing B (ascending: closest to FDT boundary at [0])
        # ----------------------------------------------------------------
        split = len(primary_fats) // 2
        wing_a = list(reversed(primary_fats[:split]))   # [0]=closest, [-1]=farthest
        wing_b = list(primary_fats[split:])             # [0]=closest, [-1]=farthest

        # ----------------------------------------------------------------
        # Step 3: insert borrowed FATs at the FRONT of their wing.
        #
        # A borrowed FAT is physically at the junction between roads, so it
        # is always the FIRST hop from the FDT.  Inserting at index 0 means:
        #   FDT → borrowed_FAT (A-01) → primary_road_FAT_closest (A-02) → ...
        #
        # Distribute borrowed FATs one at a time to the shorter wing,
        # inserting each at index 0 (shifting primary FATs to higher indices).
        # ----------------------------------------------------------------
        if borrowed_fats:
            # Sort borrowed FATs by distance to the FDT boundary midpoint
            if wing_a and wing_b:
                bx = (wing_a[0].geometry.x() + wing_b[0].geometry.x()) / 2
                by = (wing_a[0].geometry.y() + wing_b[0].geometry.y()) / 2
            elif wing_a:
                bx, by = wing_a[0].geometry.x(), wing_a[0].geometry.y()
            elif wing_b:
                bx, by = wing_b[0].geometry.x(), wing_b[0].geometry.y()
            else:
                bx = sum(f.geometry.x() for f in borrowed_fats) / len(borrowed_fats)
                by = sum(f.geometry.y() for f in borrowed_fats) / len(borrowed_fats)

            borrowed_fats.sort(key=lambda f:
                (f.geometry.x() - bx) ** 2 + (f.geometry.y() - by) ** 2
            )
            for fat in borrowed_fats:
                # Always insert at index 0 so this FAT is the first hop (A-01/B-01)
                if len(wing_a) <= len(wing_b):
                    wing_a.insert(0, fat)
                else:
                    wing_b.insert(0, fat)

        # ----------------------------------------------------------------
        # Step 4: enforce MAX_FAT_PER_WING.
        # If a wing is still too long, move its LAST element (farthest primary
        # road FAT) to the end of the other wing — never move index 0 which
        # could be a borrowed FAT placed at the junction.
        # ----------------------------------------------------------------
        while len(wing_a) > MAX_FAT_PER_WING and len(wing_b) < MAX_FAT_PER_WING:
            wing_b.append(wing_a.pop())    # pop() = last element = farthest primary FAT
        while len(wing_b) > MAX_FAT_PER_WING and len(wing_a) < MAX_FAT_PER_WING:
            wing_a.append(wing_b.pop())

        if len(wing_a) > MAX_FAT_PER_WING or len(wing_b) > MAX_FAT_PER_WING:
            raise ValidationError(
                f"{fdt_code}: wing overflow after rebalance "
                f"A={len(wing_a)}, B={len(wing_b)}"
            )

        # ----------------------------------------------------------------
        # Step 5: assign names, sequence numbers, wing labels.
        # wing_a[0] and wing_b[0] are ALWAYS closest to the FDT.
        # ----------------------------------------------------------------
        for wing_label, wing_fats in [(WingType.A.value, wing_a),
                                       (WingType.B.value, wing_b)]:
            for seq, fat in enumerate(wing_fats, start=1):
                fat.wing          = wing_label
                fat.wing_id       = wing_label
                fat.parent_fdt    = fdt_code
                fat.fdt_serial    = fdt_serial
                fat.fat_sequence  = f"{seq:02d}"
                fat.fat_full_code = f"FAT-{fdt_serial}{wing_label}-{seq:02d}"
                fat.name          = fat.fat_full_code

        fdt.wing_a_count = len(wing_a)
        fdt.wing_b_count = len(wing_b)
        fdt.fats         = wing_a + wing_b

        # Place FDT geometry at the midpoint between the two closest FATs
        if wing_a and wing_b:
            fdt.geometry = QgsPointXY(
                (wing_a[0].geometry.x() + wing_b[0].geometry.x()) / 2,
                (wing_a[0].geometry.y() + wing_b[0].geometry.y()) / 2
            )
        elif wing_a:
            fdt.geometry = wing_a[0].geometry
        elif wing_b:
            fdt.geometry = wing_b[0].geometry

        logger.log(
            f"{fdt.name} -> {total_fats} FAT -> A:{len(wing_a)}, B:{len(wing_b)} "
            f"| A-01={wing_a[0].name if wing_a else 'none'} "
            f"B-01={wing_b[0].name if wing_b else 'none'}"
        )

    logger.end_module("Wing Distribution")


def create_fdt_wing_cascades(fdts: List[FDT],
                             error_collector: ErrorCollector,
                             logger: PerformanceLogger):
    """Wire cascade from_device/to_device and assign splitter types.

    Each wing is already ordered closest-to-FDT first by assign_wings_per_fdt.
    The cascade chain is:
        FDT → wing[0] (1×9) → wing[1] (1×9) → wing[2] (1×9) → wing[3] (1×8)

    SPLITTER_PATTERN = ["1x9", "1x9", "1x9", "1x8"]
    Position 0 is closest to FDT, position 3 (1×8) is the far end.
    """
    logger.start_module("FDT Wing Cascading")

    for fdt in fdts:
        total_fats = len(fdt.fats)
        if total_fats > MAX_FAT_PER_FDT:
            raise ValidationError(
                f"{fdt.name} has {total_fats} FATs, exceeds {MAX_FAT_PER_FDT}"
            )

        wing_a = [f for f in fdt.fats if f.wing == WingType.A.value]
        wing_b = [f for f in fdt.fats if f.wing == WingType.B.value]

        # wing_a and wing_b are already in correct order (closest→farthest)
        # from assign_wings_per_fdt — do NOT re-sort here.

        if len(wing_a) > MAX_FAT_PER_WING or len(wing_b) > MAX_FAT_PER_WING:
            raise ValidationError(
                f"{fdt.name} wing FAT count exceeded: "
                f"A={len(wing_a)}, B={len(wing_b)}, max={MAX_FAT_PER_WING}"
            )

        fdt_outputs = 0
        for wing_fats in [wing_a, wing_b]:
            if not wing_fats:
                continue
            for i, fat in enumerate(wing_fats):
                # SPLITTER_PATTERN index 0 = 1×9 (closest to FDT)
                # SPLITTER_PATTERN index 3 = 1×8 (farthest, closes the chain)
                fat.splitter_type = SPLITTER_PATTERN[i]
                fat.from_device   = fdt.name if i == 0 else wing_fats[i - 1].name
                fat.to_device     = wing_fats[i + 1].name if i < len(wing_fats) - 1 else ""
            fdt_outputs += 1

        if fdt_outputs > 2:
            raise ValidationError(
                f"{fdt.name} has {fdt_outputs} outputs; maximum is 2"
            )

        logger.log(
            f"{fdt.name} -> total FAT {total_fats}, outputs {fdt_outputs}, "
            f"WingA {len(wing_a)}, WingB {len(wing_b)}"
        )

    logger.end_module("FDT Wing Cascading")

# =============================================================================
# FDT GENERATION
# =============================================================================

def _roads_touch(road_a, road_b) -> bool:
    """True if two road segments share an endpoint or meet at a junction.

    Uses touches() + intersects() combo:
    - touches() catches clean T/X junctions where geometries share a boundary point
    - intersects() fallback catches roads that slightly overlap due to digitising error
      but we exclude crossing roads using NOT crosses() to avoid false positives
    """
    if not road_a.geometry or not road_b.geometry:
        return False
    ga, gb = road_a.geometry, road_b.geometry
    # touches() is True when boundaries meet but interiors don't overlap
    if ga.touches(gb):
        return True
    # Some OSM exports have a tiny gap (<0.5m); use distance threshold
    return ga.distance(gb) < 0.5


def _get_touching_road_ids(road_id: int, roads: dict, road_index) -> list:
    """Return IDs of roads that physically touch (share a junction with) road_id.

    Only direct neighbours — no multi-hop expansion — so the search stays
    local and doesn't pull in FATs from across the city.
    """
    target = roads.get(road_id)
    if not target or target.geometry.isEmpty():
        return []
    bbox = target.geometry.boundingBox()
    bbox.grow(1.0)          # tiny buffer just to catch floating-point edge gaps
    result = []
    for rid in road_index.intersects(bbox):
        if rid == road_id:
            continue
        nb = roads.get(rid)
        if nb and _roads_touch(target, nb):
            result.append(rid)
    return result


def generate_fdts(fats: List[FAT],
                 error_collector: ErrorCollector,
                 logger: PerformanceLogger,
                 roads=None,
                 road_index=None,
                 dist_calc=None) -> List[FDT]:
    """Generate FDTs by clustering FATs, enforcing MIN_FAT_PER_FDT (6) and MAX_FAT_PER_FDT (8).

    Algorithm
    ---------
    Phase 1 – per-road grouping
        Each road's FATs are sorted by cascade_order and sliced into groups of
        up to MAX_FAT_PER_FDT.  This gives the initial FDT candidates.

    Phase 2 – merge undersized groups
        Any group with fewer than MIN_FAT_PER_FDT FATs is "deficient".
        Strategy (in priority order):

        A. Same-road merge: if this road has another group (the last chunk
           when a road has e.g. 9 FATs → [8, 1]), merge the small tail into
           the nearest same-road group that is below MAX_FAT_PER_FDT.

        B. Cross-road fill from touching neighbours: for each directly
           touching road (junction neighbour), look for groups that have
           MORE than MIN_FAT_PER_FDT FATs (i.e. ≥ 7) and can spare one FAT
           without going below MIN_FAT_PER_FDT.  Borrow the closest FAT.

        C. Last-resort cross-road merge: if still deficient, find the
           nearest group on any touching road that, when merged with the
           deficient group, would not exceed MAX_FAT_PER_FDT.  Merge them
           into a single group (absorbed group is removed).

        If none of the above fills the group it is logged as truly isolated.

    Phase 3 – build FDT objects from settled groups.
    """
    logger.start_module("FDT Generation")

    if not fats:
        return []

    # ------------------------------------------------------------------
    # Phase 1: initial per-road grouping
    # ------------------------------------------------------------------
    fats_by_road = defaultdict(list)
    for fat in fats:
        fats_by_road[fat.road_id].append(fat)

    # groups_by_road[road_id] = list of mutable FAT lists
    groups_by_road: Dict[int, List[List[FAT]]] = {}
    for road_id in sorted(fats_by_road.keys()):
        road_fats = sorted(fats_by_road[road_id], key=_fat_sort_key)
        groups = []
        for i in range(0, len(road_fats), MAX_FAT_PER_FDT):
            chunk = road_fats[i:i + MAX_FAT_PER_FDT]
            if chunk:
                groups.append(chunk)
        groups_by_road[road_id] = groups

    # ------------------------------------------------------------------
    # Phase 2: enforce minimum with targeted, local operations only
    # ------------------------------------------------------------------
    if roads and road_index:

        def _centroid(group):
            if not group:
                return None
            return QgsPointXY(
                sum(f.geometry.x() for f in group) / len(group),
                sum(f.geometry.y() for f in group) / len(group)
            )

        def _dist(group_a, group_b):
            ca = _centroid(group_a)
            cb = _centroid(group_b)
            if ca is None or cb is None:
                return float('inf')
            if dist_calc:
                return dist_calc.distance_points(ca, cb)
            return ((ca.x() - cb.x()) ** 2 + (ca.y() - cb.y()) ** 2) ** 0.5

        def _pt_dist(centroid_pt, fat):
            if centroid_pt is None:
                return float('inf')
            if dist_calc:
                return dist_calc.distance_points(centroid_pt, fat.geometry)
            return ((centroid_pt.x() - fat.geometry.x()) ** 2 +
                    (centroid_pt.y() - fat.geometry.y()) ** 2) ** 0.5

        for _ in range(20):
            any_changed = False

            # Purge emptied groups before each pass so they are never
            # accidentally considered as merge or borrow candidates
            for rid in list(groups_by_road.keys()):
                groups_by_road[rid] = [g for g in groups_by_road[rid] if g]

            for road_id in sorted(groups_by_road.keys()):
                for group_idx in range(len(groups_by_road[road_id])):
                    group = groups_by_road[road_id][group_idx]

                    # Skip empty (already consumed) or already meeting minimum
                    if not group or len(group) >= MIN_FAT_PER_FDT:
                        continue

                    logger.log(
                        f"FDT on road {road_id} group {group_idx}: "
                        f"{len(group)} FATs, needs {MIN_FAT_PER_FDT - len(group)} more."
                    )

                    # -- Strategy A: same-road merge --
                    # Combine with another non-empty group on this road if it fits
                    same_road_others = [
                        (gi, g) for gi, g in enumerate(groups_by_road[road_id])
                        if gi != group_idx
                        and g                                    # must be non-empty
                        and len(g) + len(group) <= MAX_FAT_PER_FDT
                    ]
                    if same_road_others:
                        best_gi, best_g = min(same_road_others,
                                              key=lambda x: _dist(group, x[1]))
                        group.extend(best_g)
                        best_g.clear()
                        any_changed = True
                        logger.log(
                            f"  Strategy A: merged same-road group {best_gi} -> "
                            f"group {group_idx} now has {len(group)} FATs."
                        )
                        continue

                    # -- Strategy B: borrow one FAT at a time from touching roads --
                    touching_ids = _get_touching_road_ids(road_id, roads, road_index)
                    grp_centroid = _centroid(group)

                    borrow_pool = []
                    for nb_id in touching_ids:
                        for nb_gi, nb_g in enumerate(groups_by_road.get(nb_id, [])):
                            # Only borrow from groups with surplus (> MIN means >= MIN+1 = 7+)
                            if not nb_g or len(nb_g) <= MIN_FAT_PER_FDT:
                                continue
                            for fat in nb_g:
                                borrow_pool.append(
                                    (_pt_dist(grp_centroid, fat), nb_id, nb_gi, fat)
                                )

                    borrow_pool.sort(key=lambda x: x[0])

                    for d, nb_id, nb_gi, fat in borrow_pool:
                        if len(group) >= MIN_FAT_PER_FDT:
                            break
                        nb_g = groups_by_road[nb_id][nb_gi]
                        if not nb_g or len(nb_g) <= MIN_FAT_PER_FDT:
                            continue
                        if fat not in nb_g:
                            continue
                        nb_g.remove(fat)
                        group.append(fat)
                        any_changed = True
                        logger.log(
                            f"  Strategy B: borrowed FAT {fat.name} from "
                            f"road {nb_id} (dist={d:.1f}m). "
                            f"Group now {len(group)} FATs."
                        )

                    if len(group) >= MIN_FAT_PER_FDT:
                        continue

                    # -- Strategy C: merge entire neighbour group --
                    merge_pool = []
                    for nb_id in touching_ids:
                        for nb_gi, nb_g in enumerate(groups_by_road.get(nb_id, [])):
                            if not nb_g:
                                continue
                            if len(group) + len(nb_g) <= MAX_FAT_PER_FDT:
                                merge_pool.append((_dist(group, nb_g), nb_id, nb_gi, nb_g))

                    if merge_pool:
                        merge_pool.sort(key=lambda x: x[0])
                        _, nb_id, nb_gi, nb_g = merge_pool[0]
                        group.extend(nb_g)
                        nb_g.clear()
                        any_changed = True
                        logger.log(
                            f"  Strategy C: merged neighbour group from road {nb_id} "
                            f"into road {road_id} group {group_idx}. "
                            f"Group now {len(group)} FATs."
                        )
                    elif len(group) < MIN_FAT_PER_FDT:
                        logger.log(
                            f"  Road {road_id} group {group_idx}: truly isolated, "
                            f"only {len(group)} FATs."
                        )
                        error_collector.add_error(
                            "FDT_UNDERFILLED",
                            f"Road {road_id} group {group_idx}",
                            f"FDT has {len(group)} FATs, minimum is {MIN_FAT_PER_FDT}."
                        )

            if not any_changed:
                break

        # Final purge of emptied groups
        for road_id in list(groups_by_road.keys()):
            groups_by_road[road_id] = [g for g in groups_by_road[road_id] if g]

    # ------------------------------------------------------------------
    # Phase 3: build FDT objects from settled groups
    # ------------------------------------------------------------------
    fdts = []
    fdt_counter = 1

    for road_id in sorted(groups_by_road.keys()):
        for group in groups_by_road[road_id]:
            if not group:
                continue
            if len(group) > MAX_FAT_PER_FDT:
                raise ValidationError(
                    f"Road {road_id}: group has {len(group)} FATs, "
                    f"exceeds MAX_FAT_PER_FDT ({MAX_FAT_PER_FDT})"
                )
            fdt = _make_fdt(group, fdt_counter)
            if fdt.total_customers > MAX_FDT_CAPACITY:
                raise ValidationError(
                    f"{fdt.name} has {fdt.total_customers} customers, "
                    f"exceeds {MAX_FDT_CAPACITY}"
                )
            fdts.append(fdt)
            logger.log(
                f"Road {road_id} -> {fdt.name} "
                f"({fdt.total_fat} FAT, {fdt.total_customers} customers)"
            )
            fdt_counter += 1

    logger.end_module("FDT Generation")
    logger.log(f"Generated {len(fdts)} FDTs")
    return fdts
def _make_fdt(group_fats: List[FAT], counter: int) -> FDT:
    """Create a single FDT object from a FAT group.

    Ensures downstream modules always receive:
    - fdt.fats populated with the member FAT objects
    - fdt.zone_geometry as a polygon covering the FDT service cluster
    """
    if not group_fats:
        raise ValidationError("Cannot create FDT from an empty FAT group")

    # FDT point: centroid of member FAT points (deterministic, no randomness)
    cx = sum(f.geometry.x() for f in group_fats) / len(group_fats)
    cy = sum(f.geometry.y() for f in group_fats) / len(group_fats)
    fdt_point = QgsPointXY(cx, cy)

    total_customers = sum(f.total_customers for f in group_fats)
    # Wing assignment is applied later per FDT by assign_wings_per_fdt.
    wing_a_count = 0
    wing_b_count = 0

    # Build a robust polygon zone around the FAT cluster.
    # We use convex hull + generous buffer so pole snapping and route validation
    # can work with sparse groups (1-2 FATs) and elongated alignments.
    pts = [QgsPointXY(f.geometry.x(), f.geometry.y()) for f in group_fats]
    pts.append(fdt_point)
    zone_geom = QgsGeometry.fromMultiPointXY(pts).convexHull().buffer(MAX_BOX_TO_BOX_DISTANCE, 12)
    if not zone_geom or zone_geom.isEmpty():
        zone_geom = QgsGeometry.fromPointXY(fdt_point).buffer(MAX_BOX_TO_BOX_DISTANCE, 12)

    fdt_serial = f"{counter:04d}"
    fdt_code = f"FDT-{fdt_serial}"
    return FDT(
        name=fdt_code,
        geometry=fdt_point,
        total_customers=total_customers,
        total_fat=len(group_fats),
        wing_a_count=wing_a_count,
        wing_b_count=wing_b_count,
        fdt_code=fdt_code,
        fdt_serial=fdt_serial,
        fats=list(group_fats),
        zone_geometry=zone_geom
    )

# =============================================================================
# POLE SNAPPING WITH OFFSET
# =============================================================================

def snap_to_poles(fdts: List[FDT],
                 fats: List[FAT],
                 homepasses: List[Homepass],
                 pole_layer: QgsVectorLayer,
                 roads: Dict[int, RoadSegment],
                 index_mgr: SpatialIndexManager,
                 dist_calc: DistanceCalculator,
                 error_collector: ErrorCollector,
                 logger: PerformanceLogger) -> Tuple[List[FDT], List[FAT]]:
    """Snap FDT and FAT points to nearest poles with offset"""
    logger.start_module("Pole Snapping")
    
    # Build spatial index for poles
    pole_index = index_mgr.build_index(pole_layer)
    pole_geoms = index_mgr.geometries.get(pole_layer.id(), {})
    
    # Group FDTs by zone for pole assignment
    for fdt in fdts:
        # Find nearest pole within FDT zone
        candidate_poles = pole_index.nearestNeighbor(fdt.geometry, 5)
        
        best_pole = None
        best_dist = float('inf')
        
        for pole_id in candidate_poles:
            pole_geom = pole_geoms.get(pole_id)
            if not pole_geom:
                continue
            
            pole_point = pole_geom.asPoint()
            
            # Check if pole is within FDT zone
            if fdt.zone_geometry and fdt.zone_geometry.contains(pole_geom):
                dist = dist_calc.distance_points(fdt.geometry, pole_point)
                if dist < best_dist:
                    best_dist = dist
                    best_pole = pole_point
                    fdt.snapped_geometry = pole_point
        
        if not best_pole:
            error_collector.add_error(
                "NO_POLE_IN_ZONE",
                fdt.name,
                "No suitable pole found in FDT zone"
            )
    
    # Build homepass lookup for customer-aware FAT snapping.
    hp_by_fid: Dict[int, Homepass] = {}
    for hp in homepasses:
        if hp.fid not in hp_by_fid:
            hp_by_fid[hp.fid] = hp

    # first pass: assign nearest pole coordinates to each FAT but do not
    # calculate offset_geometry yet; we'll adjust positions if multiple FATs
    # share the same pole.
    for fat in fats:
        parent_fdt = next((fdt for fdt in fdts if fat in fdt.fats), None)
        if not parent_fdt:
            continue

        assigned_points = [hp_by_fid[hp_id].geometry for hp_id in fat.customers if hp_id in hp_by_fid]
        road = roads.get(fat.road_id)

        candidate_poles = pole_index.nearestNeighbor(fat.geometry, 15)
        best_pole = None
        best_dist = float('inf')
        for pole_id in candidate_poles:
            pole_geom = pole_geoms.get(pole_id)
            if not pole_geom:
                continue
            pole_point = pole_geom.asPoint()
            if parent_fdt.zone_geometry and parent_fdt.zone_geometry.contains(pole_geom):
                candidate_start = pole_point
                if road:
                    candidate_start = calculate_offset_point(
                        pole_point, road.geometry, fat.wing, dist_calc
                    )
                if assigned_points:
                    max_drop = max(
                        dist_calc.distance_points(candidate_start, hp_point)
                        for hp_point in assigned_points
                    )
                    if max_drop > (MAX_BOX_TO_CUSTOMER_DISTANCE + DROP_LENGTH_TOLERANCE):
                        continue
                dist = dist_calc.distance_points(fat.geometry, pole_point)
                if dist < best_dist:
                    best_dist = dist
                    best_pole = pole_point
        if best_pole:
            fat.snapped_geometry = best_pole
            fat._pole_key = (best_pole.x(), best_pole.y())
        else:
            error_collector.add_error(
                "NO_POLE_IN_ZONE",
                fat.name,
                "No suitable pole found in FDT zone"
            )
            fat.snapped_geometry = fat.geometry

    # second pass: handle overlapping FATs sharing the same pole point
    pole_groups = {}
    for fat in fats:
        key = getattr(fat, '_pole_key', None)
        if key:
            pole_groups.setdefault(key, []).append(fat)
    for key, group in pole_groups.items():
        if len(group) > 1:
            pole_pt = QgsPointXY(key[0], key[1])
            offsets = distribute_points_around_pole(pole_pt, len(group))
            # sort by wing so all A's get first offsets, then B's
            sorted_group = sorted(group, key=lambda f: f.wing)
            for fat, new_pt in zip(sorted_group, offsets):
                fat.snapped_geometry = new_pt

    # final pass: compute offset_geometry now that snapped_geometry has been
    # finalized (with possible small displacements)
    for fat in fats:
        road = roads.get(fat.road_id)
        assigned_points = [hp_by_fid[hp_id].geometry for hp_id in fat.customers if hp_id in hp_by_fid]

        if fat.snapped_geometry and road:
            fat.offset_geometry = calculate_offset_point(
                fat.snapped_geometry, road.geometry, fat.wing, dist_calc
            )

        if assigned_points:
            start_point = fat.offset_geometry or fat.snapped_geometry or fat.geometry
            max_drop = max(dist_calc.distance_points(start_point, hp_point) for hp_point in assigned_points)
            if max_drop > (MAX_BOX_TO_CUSTOMER_DISTANCE + DROP_LENGTH_TOLERANCE):
                # Fall back to unsnapped FAT placement to preserve strict drop rule.
                fat.snapped_geometry = fat.geometry
                if road:
                    fat.offset_geometry = calculate_offset_point(
                        fat.geometry, road.geometry, fat.wing, dist_calc
                    )
                else:
                    fat.offset_geometry = None

                fallback_start = fat.offset_geometry or fat.snapped_geometry or fat.geometry
                fallback_max = max(
                    dist_calc.distance_points(fallback_start, hp_point)
                    for hp_point in assigned_points
                )
                if fallback_max > (MAX_BOX_TO_CUSTOMER_DISTANCE + DROP_LENGTH_TOLERANCE):
                    # Keep run alive; route validation will still enforce the same limit.
                    fat.offset_geometry = fat.geometry
                    error_collector.add_error(
                        "NO_VALID_FAT_LOCATION",
                        fat.name,
                        f"Best reachable drop is {fallback_max:.2f}m "
                        f"(limit {MAX_BOX_TO_CUSTOMER_DISTANCE + DROP_LENGTH_TOLERANCE:.2f}m)"
                    )
    
    logger.end_module("Pole Snapping")
    
    return fdts, fats

def calculate_offset_point(point: QgsPointXY, road_geom: QgsGeometry,
                          wing: str, dist_calc: DistanceCalculator) -> QgsPointXY:
    """Calculate frontage-side offset using side-of-line first, wing fallback."""
    side_sign = _point_side_of_road(point, road_geom)
    if side_sign == 0:
        side_sign = 1 if wing == WingType.A.value else -1
    return _offset_point_by_side(point, road_geom, side_sign, SNAP_OFFSET_DISTANCE)

def _dedupe_points(pts: List[QgsPointXY]) -> List[QgsPointXY]:
    """Remove consecutive duplicate points from a polyline vertex list.

    Two points are considered equal when their coordinates match within a
    sub-millimetre tolerance (1e-8 degrees / metres), which handles
    floating-point artefacts from road-snap projections.
    """
    if not pts:
        return pts
    result = [pts[0]]
    for p in pts[1:]:
        prev = result[-1]
        if abs(p.x() - prev.x()) > 1e-8 or abs(p.y() - prev.y()) > 1e-8:
            result.append(p)
    return result


# =============================================================================
# CABLE ROUTING
# =============================================================================

class CableRouter:
    """Handles cable routing along road network.

    Design principle
    ----------------
    FATs and FDTs are placed *slightly off* the road centre-line (frontage
    offset, pole-snap offset).  If we give their offset coordinates directly
    to the graph nearest-vertex lookup we often snap to a vertex on a
    *different* road whose vertex happens to be Euclidean-closer.  Dijkstra
    then routes through that wrong road and the subsequent
    ``_enforce_exact_cable_endpoints`` stub cuts across buildings.

    The fix is a two-step snap:
      1. Project the device point onto its *own* road geometry to get a
         road-edge anchor (``_road_snap``).  This is the point used for
         graph-vertex lookup — it is guaranteed to be on the correct road.
      2. Build the full cable polyline as:
             device_point → road_snap → [graph vertices] → road_snap → device_point
         so every segment of the cable lies either along a road or in the
         narrow frontage gap (< SNAP_OFFSET_DISTANCE).
    """

    def __init__(self, road_layer: QgsVectorLayer, dist_calc: DistanceCalculator,
                 roads: Optional[Dict[int, 'RoadSegment']] = None,
                 road_index: Optional[QgsSpatialIndex] = None):
        self.road_layer = road_layer
        self.dist_calc = dist_calc
        self.roads = roads or {}
        self.road_index = road_index
        self.graph_builder = None
        self.graph = None
        self._vertex_index: Optional[QgsSpatialIndex] = None
        
    def build_graph(self):
        """Build graph from road network using QgsVectorLayerDirector."""
        director = QgsVectorLayerDirector(
            self.road_layer, -1, '', '', '', QgsVectorLayerDirector.DirectionBoth
        )
        strategy = QgsNetworkDistanceStrategy()
        director.addStrategy(strategy)

        builder = QgsGraphBuilder(self.road_layer.crs())
        director.makeGraph(builder, [])

        self.graph_builder = builder
        self.graph = builder.graph()
        self._vertex_index = self._build_vertex_spatial_index()

    def _build_vertex_spatial_index(self) -> Optional[QgsSpatialIndex]:
        """Build spatial index of graph vertices for fast nearest-vertex lookup."""
        if not self.graph:
            return None
        idx = QgsSpatialIndex()
        for i in range(self.graph.vertexCount()):
            pt = self.graph.vertex(i).point()
            feat = QgsFeature(i)
            feat.setGeometry(QgsGeometry.fromPointXY(pt))
            idx.addFeature(feat)
        return idx

    def _road_snap(self, device_point: QgsPointXY,
                   road_id: Optional[int]) -> QgsPointXY:
        """Project a device point onto its own road centre-line.

        Returns the nearest point on the road geometry so that graph-vertex
        lookup is always relative to the correct road, not whichever road
        vertex happens to be Euclidean-closest to the offset device point.

        Falls back to the device point itself if the road is unavailable.
        """
        if road_id is None or road_id not in self.roads:
            return device_point
        road_geom = self.roads[road_id].geometry
        if not road_geom or road_geom.isEmpty():
            return device_point
        pt_geom = QgsGeometry.fromPointXY(device_point)
        snapped = road_geom.nearestPoint(pt_geom)
        if snapped and not snapped.isEmpty():
            return snapped.asPoint()
        return device_point

    def find_shortest_path(self, start: QgsPointXY, end: QgsPointXY,
                           constraints: Optional[Dict] = None,
                           start_road_id: Optional[int] = None,
                           end_road_id: Optional[int] = None) -> Tuple[QgsGeometry, float]:
        """Find shortest path along road network between two device points.

        The path is built as:
            start_device → start_road_snap → [Dijkstra graph path] → end_road_snap → end_device

        This guarantees:
        - Graph lookup uses the road-projected point (correct road, no cross-street snap)
        - The final cable geometry includes the short frontage stubs so
          ``_enforce_exact_cable_endpoints`` never needs to add a straight-line
          shortcut that crosses buildings.
        """
        if not self.graph:
            self.build_graph()

        # Step 1: project device points onto their own roads.
        start_road_pt = self._road_snap(start, start_road_id)
        end_road_pt   = self._road_snap(end,   end_road_id)

        # Step 2: find nearest graph vertices to the *road* projections.
        start_vertex = self.find_nearest_vertex(start_road_pt)
        end_vertex   = self.find_nearest_vertex(end_road_pt)

        # Helper: build straight-line fallback
        def _straight(a: QgsPointXY, b: QgsPointXY) -> Tuple[QgsGeometry, float]:
            g = QgsGeometry.fromPolylineXY([a, b])
            return g, self.dist_calc.distance_points(a, b)

        if start_vertex < 0 or end_vertex < 0:
            return _straight(start, end)

        if start_vertex == end_vertex:
            # Both snap to same vertex: build stub-only path
            pts = _dedupe_points([start, start_road_pt, end_road_pt, end])
            if len(pts) < 2:
                return _straight(start, end)
            return QgsGeometry.fromPolylineXY(pts), sum(
                self.dist_calc.distance_points(pts[i], pts[i+1])
                for i in range(len(pts)-1)
            )

        # Step 3: Dijkstra on graph
        tree, cost = QgsGraphAnalyzer.dijkstra(self.graph, start_vertex, 0)

        if end_vertex >= len(tree) or tree[end_vertex] == -1:
            return _straight(start, end)

        # Step 4: reconstruct vertex list from graph
        vertex_count = self.graph.vertexCount()
        edge_count   = self.graph.edgeCount()
        if end_vertex < 0 or end_vertex >= vertex_count:
            return _straight(start, end)

        try:
            graph_pts = [self.graph.vertex(end_vertex).point()]
        except Exception:
            return _straight(start, end)

        current   = end_vertex
        max_steps = vertex_count + edge_count + 10
        steps     = 0
        while current != start_vertex:
            if current < 0 or current >= len(tree):
                return _straight(start, end)
            edge_id = tree[current]
            if edge_id < 0 or edge_id >= edge_count:
                return _straight(start, end)
            edge    = self.graph.edge(edge_id)
            current = edge.fromVertex()
            if current < 0 or current >= vertex_count:
                return _straight(start, end)
            try:
                graph_pts.append(self.graph.vertex(current).point())
            except Exception:
                return _straight(start, end)
            steps += 1
            if steps > max_steps:
                return _straight(start, end)

        graph_pts.reverse()

        # Step 5: stitch full path:
        #   device_start → road_snap_start → graph_pts → road_snap_end → device_end
        # _dedupe_points removes consecutive duplicates so we don't get
        # zero-length segments when the device is already on the road.
        full_path = _dedupe_points(
            [start, start_road_pt] + graph_pts + [end_road_pt, end]
        )
        if len(full_path) < 2:
            return _straight(start, end)

        geom   = QgsGeometry.fromPolylineXY(full_path)
        length = geom.length()
        return geom, length
    
    def find_nearest_vertex(self, point: QgsPointXY) -> int:
        """Find nearest vertex index in graph. Uses spatial index when available."""
        if not self.graph:
            return -1
        if self._vertex_index:
            nearest = self._vertex_index.nearestNeighbor(point, 1)
            return nearest[0] if nearest else -1
        best_idx  = -1
        best_dist = float('inf')
        for i in range(self.graph.vertexCount()):
            vertex_point = self.graph.vertex(i).point()
            dist = self.dist_calc.distance_points(point, vertex_point)
            if dist < best_dist:
                best_dist = dist
                best_idx  = i
        return best_idx

def _anchor_point_for_device(geom: QgsGeometry) -> QgsPointXY:
    """Return exact anchor point for a device geometry."""
    if not geom or geom.isEmpty():
        raise ValidationError("Device geometry is empty")
    gtype = QgsWkbTypes.geometryType(geom.wkbType())
    if gtype == QgsWkbTypes.PointGeometry:
        return geom.asPoint()
    if gtype == QgsWkbTypes.PolygonGeometry:
        return geom.centroid().asPoint()
    if gtype == QgsWkbTypes.LineGeometry:
        return geom.centroid().asPoint()
    return geom.centroid().asPoint()

def _line_vertices(line_geom: QgsGeometry) -> List[QgsPointXY]:
    """Extract polyline vertices from single/multi line geometry."""
    if not line_geom or line_geom.isEmpty():
        return []
    if line_geom.isMultipart():
        parts = line_geom.asMultiPolyline()
        return list(parts[0]) if parts and parts[0] else []
    return list(line_geom.asPolyline())

def _enforce_exact_cable_endpoints(cable_geom: QgsGeometry,
                                   from_geom: QgsGeometry,
                                   to_geom: QgsGeometry) -> QgsGeometry:
    """Force cable first/last vertices to match device anchors exactly."""
    from_pt = _anchor_point_for_device(from_geom)
    to_pt = _anchor_point_for_device(to_geom)

    vertices = _line_vertices(cable_geom)
    if len(vertices) < 2:
        vertices = [from_pt, to_pt]
    else:
        vertices[0] = from_pt
        vertices[-1] = to_pt

    fixed = QgsGeometry.fromPolylineXY(vertices)

    # Strict zero-gap checks.
    fixed_vertices = _line_vertices(fixed)
    if len(fixed_vertices) < 2:
        raise ValidationError("Cable geometry has insufficient vertices after endpoint snapping")
    start_v = fixed_vertices[0]
    end_v = fixed_vertices[-1]

    distance_start = QgsGeometry.fromPointXY(start_v).distance(from_geom)
    distance_end = QgsGeometry.fromPointXY(end_v).distance(to_geom)
    if distance_start != 0 or distance_end != 0:
        raise ValidationError(
            f"Cable endpoint gap detected: start={distance_start}, end={distance_end}"
        )

    if not fixed.intersects(from_geom) or not fixed.intersects(to_geom):
        raise ValidationError("Cable does not intersect one or both device geometries")

    # touches() can be false for valid point-endpoint cases (and degenerate same-point links)
    # even when endpoint equality and intersects() are already satisfied.
    if (not fixed.touches(from_geom)) and distance_start != 0:
        raise ValidationError("Cable start does not touch source geometry")
    if (not fixed.touches(to_geom)) and distance_end != 0:
        raise ValidationError("Cable end does not touch destination geometry")

    return fixed

def route_fdt_to_fats(fdts: List[FDT],
                     cable_router: CableRouter,
                     error_collector: ErrorCollector,
                     logger: PerformanceLogger) -> List[CableSegment]:
    """Route cables from FDT to FATs along road network.

    Each cable runs:
        source_device → road_snap → Dijkstra path → road_snap → dest_FAT
    so the geometry always follows roads and never cuts across buildings.
    """
    logger.start_module("FDT to FAT Routing")
    
    cables = []
    
    fat_by_name = {}
    for fdt in fdts:
        for fat in fdt.fats:
            fat_by_name[fat.name] = fat

    for fdt in fdts:
        fdt_point = fdt.snapped_geometry or fdt.geometry
        fdt_geom  = QgsGeometry.fromPointXY(fdt_point)

        # Determine the FDT's road_id as the most common road_id among its FATs
        fdt_road_id: Optional[int] = None
        if fdt.fats:
            from collections import Counter
            fdt_road_id = Counter(f.road_id for f in fdt.fats).most_common(1)[0][0]

        # Route per wing so FDT outputs are at most two (A head, B head).
        for wing in [WingType.A.value, WingType.B.value]:
            wing_fats = sorted(
                [fat for fat in fdt.fats if fat.wing == wing],
                key=lambda f: f.cascade_order
            )
            if not wing_fats:
                continue

            for fat in wing_fats:
                source_name = fat.from_device or fdt.name

                if source_name == fdt.name:
                    start_point   = fdt_point
                    start_road_id = fdt_road_id
                else:
                    source_fat = fat_by_name.get(source_name)
                    if not source_fat:
                        raise ValidationError(f"Missing source FAT geometry for {source_name}")
                    start_point   = source_fat.snapped_geometry or source_fat.geometry
                    start_road_id = source_fat.road_id

                end_point   = fat.snapped_geometry or fat.geometry
                end_road_id = fat.road_id

                # Check if within FDT zone
                if fdt.zone_geometry and not fdt.zone_geometry.contains(
                        QgsGeometry.fromPointXY(end_point)):
                    error_collector.add_error(
                        "FAT_OUTSIDE_ZONE",
                        fat.name,
                        f"FAT {fat.name} is outside FDT zone"
                    )

                # Find road-following shortest path
                # road_ids are passed so each endpoint is first projected onto
                # its own road before graph-vertex lookup, preventing cross-road snapping.
                geom, length = cable_router.find_shortest_path(
                    start_point, end_point,
                    start_road_id=start_road_id,
                    end_road_id=end_road_id
                )
                length = geom.length()

                # Validate length
                if length > MAX_BOX_TO_BOX_DISTANCE:
                    error_collector.add_error(
                        "CABLE_LENGTH_EXCEEDED",
                        f"{fdt.name}-{fat.name}",
                        f"Cable length {length:.2f}m exceeds maximum {MAX_BOX_TO_BOX_DISTANCE}m"
                    )

                cable = CableSegment(
                    name=f"BOX_{source_name}_{fat.name}",
                    start_device=source_name,
                    end_device=fat.name,
                    geometry=geom,
                    length=length,
                    fdt_id=fdt.name
                )

                cables.append(cable)
    
    logger.end_module("FDT to FAT Routing")
    logger.log(f"Routed {len(cables)} FDT-FAT cables")
    
    return cables

def route_fat_to_homepasses(fats: List[FAT],
                           homepasses: List[Homepass],
                           dist_calc: DistanceCalculator,
                           error_collector: ErrorCollector,
                           logger: PerformanceLogger) -> List[CableSegment]:
    """Route cables from FATs to homepasses.
    
    When a point is split across FATs (borrowed), create multiple drop cables
    per (FAT, point) - one cable per customer allocated.
    """
    logger.start_module("FAT to Homepass Routing")
    
    cables = []
    
    # Build fid -> Homepass lookup
    hp_by_fid = {h.fid: h for h in homepasses}
    
    for fat in fats:
        start_point = fat.offset_geometry or fat.snapped_geometry or fat.geometry
        
        # Use customer_allocations when available (supports split points)
        if fat.customer_allocations:
            fid_count_pairs = list(fat.customer_allocations.items())
        else:
            fid_count_pairs = [(hp_id, 1) for hp_id in fat.customers]
        
        for hp_id, count in fid_count_pairs:
            hp = hp_by_fid.get(hp_id)
            if not hp:
                continue
            
            if hp.road_id != fat.road_id:
                raise ValidationError(
                    f"Invalid assignment: FAT {fat.name} road {fat.road_id} "
                    f"to customer {hp.fid} road {hp.road_id}"
                )

            end_point = hp.geometry
            geom = QgsGeometry.fromPolylineXY([start_point, end_point])
            length = dist_calc.distance_points(start_point, end_point)
            
            if length > (MAX_BOX_TO_CUSTOMER_DISTANCE + DROP_LENGTH_TOLERANCE):
                raise ValidationError(
                    f"Drop cable length exceeded: {fat.name} -> Customer_{hp_id} "
                    f"is {length:.2f}m (max {MAX_BOX_TO_CUSTOMER_DISTANCE + DROP_LENGTH_TOLERANCE:.2f}m)"
                )
            
            # Create one cable per customer (reflects split allocation)
            for i in range(count):
                cable = CableSegment(
                    name=f"DROP_{fat.name}_C{hp_id}_{i+1}" if count > 1 else f"DROP_{fat.name}_C{hp_id}",
                    start_device=fat.name,
                    end_device=f"Customer_{hp_id}",
                    geometry=geom,
                    length=length,
                    fdt_id=""
                )
                cables.append(cable)
    
    logger.end_module("FAT to Homepass Routing")
    logger.log(f"Routed {len(cables)} drop cables")
    
    return cables

def classify_length(length: float) -> int:
    """Standardize cable length to procurement bucket."""
    if length <= 9:
        return 10
    if 10 <= length <= 45:
        return 50
    if 46 <= length <= 96:
        return 100
    if 97 <= length <= 145:
        return 150
    return 150


def generate_core_list(capacity: int) -> List[Tuple[str, int]]:
    """Return ordered list of (buffer_color, core_number) pairs for a cable.

    The function respects the buffer/core examples given in the requirements.
    For a 48‑core feed there are four buffers (Blue, Orange, Green, Brown)
    with 12 cores each. For a 24‑core feed we use the first two buffers
    Blue/Orange with 12 cores each.
    """
    if capacity not in (CLOSURE_SMALL_CAPACITY, CLOSURE_LARGE_CAPACITY):
        raise ValidationError(f"Unsupported closure capacity {capacity}")
    if capacity == CLOSURE_SMALL_CAPACITY:
        buffers = ["Blue", "Orange"]
    else:
        buffers = ["Blue", "Orange", "Green", "Brown"]
    cores_per_buffer = capacity // len(buffers)
    cores = []
    for buf in buffers:
        for num in range(1, cores_per_buffer + 1):
            cores.append((buf, num))
    return cores


# -------------------------------------------------------------------------
# Closure zone helpers
# -------------------------------------------------------------------------

def compute_closure_service_zones(closures: List[DomeClosure]) -> None:
    """No-op: zone_geometry is now computed inside generate_closures from the
    K-Means cluster's convex hull, so no post-hoc Voronoi is needed.
    Retained for API compatibility."""
    pass


def enforce_closure_zone_membership(closures: List[DomeClosure],
                                    fdts: List[FDT],
                                    dist_calc: DistanceCalculator,
                                    logger: PerformanceLogger) -> None:
    """No-op: FDT cluster membership is definitive from _cluster_fdts().
    The K-Means clusters are the zones — no Voronoi reassignment is performed.
    Retained for API compatibility."""
    pass


def _equalise_closure_fdts(closures: List[DomeClosure],
                           dist_calc: DistanceCalculator,
                           logger: PerformanceLogger,
                           tolerance: int = 3) -> None:
    """Rebalance FDT counts across closures after Voronoi zone enforcement.

    Moves border FDTs from overloaded closures to underloaded ones so that
    all closures end up within `tolerance` FDTs of each other.

    Geographic guard: an FDT is only eligible to move if its distance to the
    receiving closure is <= MAX_BOX_TO_BOX_DISTANCE * 4.  This prevents
    cross-city moves that destroy zone compactness.

    The tolerance is set to 3 by default — tighter balancing is counter-
    productive when geographic constraints prevent valid moves.
    """
    if len(closures) < 2:
        return

    hard_cap  = CLOSURE_LARGE_CAPACITY // 2
    geo_limit = MAX_BOX_TO_BOX_DISTANCE * 4   # 1000 m geographic guard
    n_total   = sum(c.total_fdt for c in closures)
    max_iters = n_total + 1

    def _ctr(c: DomeClosure) -> QgsPointXY:
        return c.snapped_geometry or c.geometry

    for iteration in range(max_iters):
        counts  = [c.total_fdt for c in closures]
        max_cnt = max(counts)
        min_cnt = min(counts)

        if max_cnt - min_cnt <= tolerance:
            break

        big   = closures[counts.index(max_cnt)]
        small = closures[counts.index(min_cnt)]

        if small.total_fdt >= hard_cap:
            break

        small_ctr = _ctr(small)

        # Only consider FDTs within geographic reach of the receiving closure
        best_fdt  = None
        best_dist = float('inf')
        for fdt in big.fdts:
            fdt_pt = fdt.snapped_geometry or fdt.geometry
            d = dist_calc.distance_points(fdt_pt, small_ctr)
            if d < best_dist and d <= geo_limit:
                best_dist = d
                best_fdt  = fdt

        if best_fdt is None:
            # No geographically valid FDT to move — accept this imbalance
            logger.log(
                f"Equalise: cannot reduce imbalance "
                f"({big.name}={big.total_fdt} vs {small.name}={small.total_fdt}) "
                f"— no FDT within {geo_limit:.0f}m geographic limit"
            )
            break

        big.fdts.remove(best_fdt)
        big.total_fdt   -= 1
        small.fdts.append(best_fdt)
        small.total_fdt += 1

        logger.log(
            f"Equalise iter {iteration+1}: {best_fdt.name} "
            f"{big.name}({big.total_fdt+1}→{big.total_fdt}) → "
            f"{small.name}({small.total_fdt-1}→{small.total_fdt}), "
            f"dist={best_dist:.0f}m"
        )

    # Refresh feed capacities
    for c in closures:
        c.feed_capacity = (CLOSURE_LARGE_CAPACITY
                           if c.total_fdt > CLOSURE_24_CORE_THRESHOLD
                           else CLOSURE_SMALL_CAPACITY)


def generate_closures(fdts: List[FDT],
                      pickup_layer: Optional[QgsVectorLayer],
                      index_mgr: SpatialIndexManager,
                      dist_calc: DistanceCalculator,
                      roads: Dict[int, RoadSegment],
                      road_index: QgsSpatialIndex,
                      building_layer: Optional[QgsVectorLayer],
                      logger: PerformanceLogger,
                      pole_layer: Optional[QgsVectorLayer] = None) -> List[DomeClosure]:
    """Cluster FDTs into equal geographic regions and create one closure per region.

    Pipeline
    ────────
    1. _cluster_fdts()  — balanced K-Means → k equal geographic clusters.
       The cluster IS the definitive zone.  No post-hoc Voronoi reassignment.

    2. For each cluster, place the closure at the road-junction pole nearest
       to the cluster centroid.  A junction pole is preferred because it is
       where cable arms naturally branch outward (as shown in the reference).

    3. Build a convex-hull zone geometry from the cluster's FDT points so that
       correct_cable_zone_violations can detect genuine misassignments caused
       by snapping.  This zone is NOT used to reassign FDTs — it is only a
       diagnostic boundary.

    4. Assign pickup points and auto-generate them if no pickup layer exists.
    """
    closures: List[DomeClosure] = []
    if not fdts:
        return []

    max_per_closure = CLOSURE_LARGE_CAPACITY // 2   # hard cap = 24
    clusters = _cluster_fdts(fdts, max_per_closure, dist_calc)

    # Log initial balance
    if clusters:
        sizes = [len(c) for c in clusters]
        logger.log(
            f"Closure clustering: {len(clusters)} region(s), "
            f"FDTs per region — min={min(sizes)} max={max(sizes)} "
            f"avg={sum(sizes)/len(sizes):.1f}"
        )

    # ── Prepare spatial indices ───────────────────────────────────────────────
    building_index = None
    building_geoms: Dict[int, QgsGeometry] = {}
    if building_layer:
        building_index = index_mgr.build_index(building_layer, cache_geoms=True)
        building_geoms = index_mgr.geometries.get(building_layer.id(), {})

    pole_index = None
    pole_geoms: Dict[int, QgsGeometry] = {}
    if pole_layer and pole_layer.featureCount() > 0:
        pole_index = index_mgr.build_index(pole_layer, cache_geoms=True)
        pole_geoms = index_mgr.geometries.get(pole_layer.id(), {})

    pick_idx   = None
    pick_geoms = {}
    if pickup_layer and pickup_layer.featureCount() > 0:
        pick_idx   = index_mgr.build_index(pickup_layer, cache_geoms=True)
        pick_geoms = index_mgr.geometries.get(pickup_layer.id(), {})
        logger.log("Pickup layer available; closures will reference nearest pickup")

    # ── Build one closure per cluster ─────────────────────────────────────────
    for ci, cluster in enumerate(clusters, start=1):

        # Cluster centroid
        cx = sum(f.geometry.x() for f in cluster) / len(cluster)
        cy = sum(f.geometry.y() for f in cluster) / len(cluster)
        centroid = QgsPointXY(cx, cy)
        closure_pt = centroid
        snapped_pt = None
        road_geom  = None

        # ── Place closure at road-junction pole nearest to centroid ───────────
        # Prefer poles at road endpoints (junctions) for natural cable branching.
        if road_index and pole_index:
            candidate_ids = pole_index.nearestNeighbor(centroid, 25)
            best_pole   = None
            best_score  = -1
            best_dist   = float('inf')

            for pid in candidate_ids:
                pgeom = pole_geoms.get(pid)
                if not pgeom:
                    continue
                pole_pt   = pgeom.asPoint()
                pole_dist = dist_calc.distance_points(centroid, pole_pt)

                # Junction score: count road endpoints within 5 m of this pole
                nearby_rids    = road_index.nearestNeighbor(pole_pt, 8)
                junction_score = 0
                for rid in nearby_rids:
                    road = roads.get(rid)
                    if not road or not road.geometry or road.geometry.isEmpty():
                        continue
                    verts = _road_vertices(road.geometry)
                    if not verts:
                        continue
                    for ep in (verts[0], verts[-1]):
                        if dist_calc.distance_points(pole_pt, ep) <= 5.0:
                            junction_score += 1
                            break

                if (junction_score > best_score or
                        (junction_score == best_score and pole_dist < best_dist)):
                    best_score = junction_score
                    best_dist  = pole_dist
                    best_pole  = pole_pt

            if best_pole:
                snapped_pt = best_pole
                closure_pt = best_pole

        # Fallback: snap to nearest road then nearest pole
        if snapped_pt is None:
            if road_index:
                nr = road_index.nearestNeighbor(closure_pt, 1)
                if nr:
                    road = roads.get(nr[0])
                    if road and road.geometry and not road.geometry.isEmpty():
                        road_geom  = road.geometry
                        closure_pt = road_geom.nearestPoint(
                            QgsGeometry.fromPointXY(closure_pt)).asPoint()
            if pole_index:
                cids = pole_index.nearestNeighbor(closure_pt, 5)
                bp, bd = None, float('inf')
                for pid in cids:
                    pg = pole_geoms.get(pid)
                    if not pg:
                        continue
                    pp = pg.asPoint()
                    d  = dist_calc.distance_points(closure_pt, pp)
                    if d < bd:
                        bd, bp = d, pp
                if bp:
                    snapped_pt = bp
                    closure_pt = bp

        # Building avoidance
        if building_index:
            for bid in building_index.intersects(
                    QgsGeometry.fromPointXY(closure_pt).boundingBox()):
                bgeom = building_geoms.get(bid)
                if bgeom and bgeom.contains(QgsGeometry.fromPointXY(closure_pt)):
                    if road_geom:
                        closure_pt = road_geom.nearestPoint(
                            QgsGeometry.fromPointXY(closure_pt)).asPoint()
                    snapped_pt = None
                    break

        # ── Zone geometry: convex hull of cluster FDT points ─────────────────
        # Used only as a diagnostic boundary — NOT for FDT reassignment.
        fdt_pts = [QgsGeometry.fromPointXY(f.snapped_geometry or f.geometry)
                   for f in cluster]
        if len(fdt_pts) >= 3:
            merged   = QgsGeometry.unaryUnion(fdt_pts)
            zone_raw = merged.convexHull()
            # Buffer by MAX_BOX_TO_BOX_DISTANCE so FDTs on zone edge are included
            zone_geom = zone_raw.buffer(MAX_BOX_TO_BOX_DISTANCE, 8)
        elif fdt_pts:
            zone_geom = fdt_pts[0].buffer(MAX_BOX_TO_BOX_DISTANCE * 2, 8)
        else:
            zone_geom = QgsGeometry.fromPointXY(closure_pt).buffer(
                MAX_BOX_TO_BOX_DISTANCE * 2, 8)

        # ── Capacity ──────────────────────────────────────────────────────────
        cap = (CLOSURE_LARGE_CAPACITY if len(cluster) > CLOSURE_24_CORE_THRESHOLD
               else CLOSURE_SMALL_CAPACITY)

        # ── Pickup point ──────────────────────────────────────────────────────
        pickup_pt   = None
        pickup_name = None
        if pick_idx:
            nr = pick_idx.nearestNeighbor(closure_pt, 1)
            if nr:
                pid = nr[0]
                pg  = pick_geoms.get(pid)
                feat = None
                if pg and pg.wkbType() == QgsWkbTypes.PointGeometry:
                    pickup_pt = pg.asPoint()
                else:
                    req  = QgsFeatureRequest(pid)
                    feat = next(pickup_layer.getFeatures(req), None)
                    if feat and feat.geometry():
                        pickup_pt = feat.geometry().asPoint()
                pickup_name = f"PICKUP_{pid}"
                if feat and 'name' in feat.fields().names():
                    try:
                        pickup_name = str(feat['name'])
                    except Exception:
                        pass

        closure = DomeClosure(
            name=f"CLOSURE-{ci:04d}",
            geometry=closure_pt,
            fdts=list(cluster),
            total_fdt=len(cluster),
            feed_capacity=cap,
            pickup_geometry=pickup_pt,
            pickup_name=pickup_name,
            snapped_geometry=snapped_pt,
            zone_geometry=zone_geom,
        )
        closures.append(closure)
        logger.log(
            f"  CLOSURE-{ci:04d}: {len(cluster)} FDTs, "
            f"cap={cap} cores, junction_score={best_score if pole_index else 'n/a'}"
        )

    # ── Auto-generate pickup points if no pickup layer ────────────────────────
    if not pick_idx:
        _generate_pickup_points(closures, roads, road_index, dist_calc, logger)

    # ── Final balance report ──────────────────────────────────────────────────
    counts = [c.total_fdt for c in closures]
    logger.log(
        f"Final: {len(closures)} closure(s) — "
        f"min={min(counts)} max={max(counts)} avg={sum(counts)/len(counts):.1f}"
    )

    return closures


def correct_cable_zone_violations(closures: List[DomeClosure],
                                   cables: List[CableSegment],
                                   cable_router: CableRouter,
                                   logger: PerformanceLogger) -> List[CableSegment]:
    """Re-check FDT zone membership after routing and fix any misassignments.

    Since enforce_closure_zone_membership already ran during generate_closures,
    this pass is a lightweight safety net.  It checks each routed cable's FDT
    point against its closure's zone and, if the point falls outside, moves the
    FDT to the correct closure and re-routes the affected closures.
    """
    if not closures:
        return cables

    hard_cap    = CLOSURE_LARGE_CAPACITY // 2
    reassigned  = set()   # FDT names that were moved

    for closure in closures:
        for fdt in list(closure.fdts):
            test_pt  = fdt.snapped_geometry or fdt.geometry
            pt_geom  = QgsGeometry.fromPointXY(test_pt)
            in_zone  = (closure.zone_geometry and
                        closure.zone_geometry.contains(pt_geom))
            if in_zone:
                continue

            # FDT is outside this closure's zone — find the correct one
            correct = None
            for other in closures:
                if other is closure:
                    continue
                if (other.zone_geometry and
                        other.zone_geometry.contains(pt_geom) and
                        other.total_fdt < hard_cap):
                    correct = other
                    break

            # Fallback: nearest closure with capacity
            if correct is None:
                candidates = [(c, cable_router.dist_calc.distance_points(
                                  test_pt, c.snapped_geometry or c.geometry))
                              for c in closures
                              if c is not closure and c.total_fdt < hard_cap]
                if candidates:
                    correct = min(candidates, key=lambda x: x[1])[0]

            if correct is None:
                logger.log(f"No reassignment target for {fdt.name}; keeping in {closure.name}")
                continue

            logger.log(f"Reassigning {fdt.name}: {closure.name} → {correct.name}")
            closure.fdts.remove(fdt)
            closure.total_fdt -= 1
            correct.fdts.append(fdt)
            correct.total_fdt += 1
            reassigned.add(closure.name)
            reassigned.add(correct.name)

    if not reassigned:
        return cables

    # Update capacities
    for c in closures:
        c.feed_capacity = (CLOSURE_LARGE_CAPACITY
                           if c.total_fdt > CLOSURE_24_CORE_THRESHOLD
                           else CLOSURE_SMALL_CAPACITY)

    # Re-route only the affected closures; keep unaffected cables intact
    unaffected = [c for c in cables
                  if c.core_info.get("closure_name", "") not in reassigned]
    affected   = [c for c in closures if c.name in reassigned]
    re_routed  = route_closure_to_fdts(affected, cable_router, logger)

    logger.log(f"Zone correction: {len(reassigned)} closures re-routed "
               f"({len(re_routed)} cables)")
    return unaffected + re_routed


def route_closure_to_fdts(closures: List[DomeClosure],
                          cable_router: CableRouter,
                          logger: PerformanceLogger) -> List[CableSegment]:
    """Route Closure→FDT distribution cables via direct Dijkstra shortest path.

    Design principle (matches reference image exactly)
    ──────────────────────────────────────────────────
    Each FDT in a closure's cluster receives ONE direct cable routed along the
    road network from the closure point to the FDT point.  The "tree" that
    appears in the reference image is simply the visual result of multiple
    Dijkstra paths sharing common road segments — there is NO explicit cascade
    chaining between FDTs.  This is identical to how FAT→FAT box-to-box cables
    work, applied at the closure→FDT level.

    Key rules enforced here:
      1. Zone integrity  – only FDTs that belong to this closure are routed.
         FDT membership was already enforced by enforce_closure_zone_membership
         in generate_closures(), so closure.fdts is already clean.
      2. Correct road anchoring – both endpoints are road-snapped via their
         own road_id so Dijkstra never routes through the wrong street.
      3. FDTs sorted by distance from closure for deterministic core assignment.
      4. Core pairs (active + redundant) assigned sequentially; cascade groups
         of MAX_FDT_PER_8CORE (4) per 8-core sub-cable.
    """
    logger.start_module("Closure to FDT Routing")
    cables: List[CableSegment] = []

    road_index = cable_router.road_index

    for closure in closures:
        if not closure.fdts:
            continue

        # ── Closure point and its road anchor ────────────────────────────────
        closure_pt = closure.snapped_geometry or closure.geometry
        closure_road_id: Optional[int] = None
        if road_index:
            ids = road_index.nearestNeighbor(closure_pt, 1)
            if ids:
                closure_road_id = ids[0]

        # ── FDT primary road helper ───────────────────────────────────────────
        def _fdt_road(fdt: FDT) -> Optional[int]:
            if fdt.fats:
                from collections import Counter
                return Counter(f.road_id for f in fdt.fats).most_common(1)[0][0]
            return None

        # ── Sort FDTs by distance from closure (deterministic core order) ─────
        sorted_fdts = sorted(
            closure.fdts,
            key=lambda f: (
                round(cable_router.dist_calc.distance_points(
                    closure_pt, f.snapped_geometry or f.geometry), 3),
                f.name
            )
        )

        # ── Validate capacity ─────────────────────────────────────────────────
        cores = generate_core_list(closure.feed_capacity)
        max_fdt_for_capacity = (closure.feed_capacity // 8) * MAX_FDT_PER_8CORE
        if len(sorted_fdts) > max_fdt_for_capacity:
            logger.error(
                f"{closure.name} has {len(sorted_fdts)} FDTs but "
                f"{closure.feed_capacity} cores can only serve "
                f"{max_fdt_for_capacity}; skipping")
            continue

        # ── Route one direct cable per FDT ────────────────────────────────────
        for idx, fdt in enumerate(sorted_fdts):
            fdt_rid = _fdt_road(fdt)
            fdt_pt  = fdt.snapped_geometry or fdt.geometry

            group_idx    = idx // MAX_FDT_PER_8CORE
            pos_in_group = idx % MAX_FDT_PER_8CORE
            base_core    = group_idx * 8 + pos_in_group * 2

            if base_core + 1 >= len(cores):
                logger.error(
                    f"Not enough cores in {closure.name} for FDT {fdt.name} "
                    f"(idx {idx}); skipping")
                continue

            active    = cores[base_core]
            redundant = cores[base_core + 1]

            # Direct Dijkstra from closure to FDT along road network
            geom, length = cable_router.find_shortest_path(
                closure_pt, fdt_pt,
                start_road_id=closure_road_id,
                end_road_id=fdt_rid,
            )

            cables.append(CableSegment(
                name=f"{closure.name}-{fdt.name}",
                start_device=closure.name,
                end_device=fdt.name,
                geometry=geom,
                length=length,
                fdt_id=fdt.name,
                core_info={
                    "closure_name":           closure.name,
                    "buffer_color_active":    active[0],
                    "core_color_active":      active[1],
                    "buffer_color_redundant": redundant[0],
                    "core_color_redundant":   redundant[1],
                    "cable_capacity":         closure.feed_capacity,
                    "cascade_group":          group_idx,
                    "zone_violation":         False,
                }
            ))

    logger.end_module("Closure to FDT Routing")
    logger.log(f"Routed {len(cables)} closure-to-FDT cables")
    return cables


def route_pickup_to_closure(closures: List[DomeClosure],
                            pickup_layer: Optional[QgsVectorLayer],
                            index_mgr: SpatialIndexManager,
                            cable_router: CableRouter,
                            logger: PerformanceLogger) -> List[CableSegment]:
    """Route feeder cables from pickup points (or generated points) to closures.

    If a pickup layer is provided, the nearest pickup feature is used (or the
    closure's explicitly assigned pickup_geometry if set).  When no layer is
    supplied the script may have generated pickup points earlier; those are
    preferred.  In the absence of any pickup geometry the nearest graph vertex
    is used as a last-resort virtual pickup location.
    """
    cables: List[CableSegment] = []
    pick_idx = None
    pick_geoms = {}
    if pickup_layer and pickup_layer.featureCount() > 0:
        pick_idx = index_mgr.get_index(pickup_layer)
        pick_geoms = index_mgr.geometries.get(pickup_layer.id(), {})
    for closure in closures:
        # Derive closure's road_id for proper road-snapping in find_shortest_path
        closure_road_id: Optional[int] = None
        if cable_router.road_index:
            nearest_ids = cable_router.road_index.nearestNeighbor(closure.geometry, 1)
            if nearest_ids:
                closure_road_id = nearest_ids[0]

        # if closure has an explicit pickup_geometry (either from layer or auto-gen) use it
        if closure.pickup_geometry:
            start_pt = closure.pickup_geometry
        elif pick_idx:
            # choose nearest pickup to closure geometry from user-supplied layer
            nearest = pick_idx.nearestNeighbor(closure.geometry, 1)
            if nearest:
                pid = nearest[0]
                pg = pick_geoms.get(pid)
                if pg and pg.wkbType() == QgsWkbTypes.PointGeometry:
                    start_pt = pg.asPoint()
                else:
                    req = QgsFeatureRequest(pid)
                    feat = next(pickup_layer.getFeatures(req), None)
                    start_pt = feat.geometry().asPoint() if feat and feat.geometry() else closure.geometry
            else:
                start_pt = closure.geometry
        else:
            # no pickup information at all: use nearest graph vertex as virtual pickup
            v_idx = cable_router.find_nearest_vertex(closure.geometry)
            if v_idx >= 0 and cable_router.graph:
                start_pt = cable_router.graph.vertex(v_idx).point()
            else:
                start_pt = closure.geometry
        geom, length = cable_router.find_shortest_path(
            start_pt,
            closure.geometry,
            start_road_id=None,   # pickup point is already on road; no road_id needed
            end_road_id=closure_road_id,
        )
        pickup_name = closure.pickup_name or "PICKUP"
        cable_type = f"{closure.feed_capacity}C"
        cable = CableSegment(
            name=f"{pickup_name}-{closure.name}-{cable_type}",
            start_device=pickup_name,
            end_device=closure.name,
            geometry=geom,
            length=length,
            fdt_id="",
            core_info={"feed_capacity": closure.feed_capacity, "cable_type": cable_type}
        )
        cables.append(cable)
    logger.log(f"Routed {len(cables)} pickup-to-closure feeder cables")
    return cables

def _detect_street_field(road_layer: QgsVectorLayer) -> Optional[str]:
    """Pick best available street-name field from road layer."""
    names = [f.name() for f in road_layer.fields()]
    preferred = ["street_name", "street", "road_name", "name", "Street", "STREET", "ROAD_NAME", "Name"]
    lower_lookup = {n.lower(): n for n in names}
    for cand in preferred:
        if cand.lower() in lower_lookup:
            return lower_lookup[cand.lower()]
    for fld in road_layer.fields():
        if fld.type() == QVariant.String:
            return fld.name()
    return None

def _build_road_to_street_map(road_layer: QgsVectorLayer) -> Tuple[Dict[int, str], List[str]]:
    """Build road-id to street-name map and ordered unique street list."""
    street_field = _detect_street_field(road_layer)
    road_to_street: Dict[int, str] = {}
    street_set: Set[str] = set()

    for feat in road_layer.getFeatures():
        street = ""
        if street_field:
            try:
                street = str(feat[street_field] or "").strip()
            except Exception:
                street = ""
        if not street:
            street = f"ROAD_{feat.id()}"
        road_to_street[feat.id()] = street
        street_set.add(street)

    return road_to_street, sorted(street_set)

def _add_boq_row(layer: QgsVectorLayer,
                 level_type: str,
                 street_name: str,
                 total_homepass: int,
                 total_fdt: int,
                 total_fat: int,
                 total_poles: int,
                 total_closures: int,
                 total_box_to_box_cable: int,
                 cable_10m: int,
                 cable_50m: int,
                 cable_100m: int,
                 cable_150m: int,
                 total_standard_length: float,
                 feeder_cable_count: int,
                 feeder_cable_length_m: float,
                 closure_fdt_cable_count: int,
                 closure_fdt_cable_length_m: float):
    feat = QgsFeature(layer.fields())
    feat.setAttributes([
        level_type,
        street_name,
        total_homepass,
        total_fdt,
        total_fat,
        total_poles,
        total_closures,
        total_box_to_box_cable,
        cable_10m,
        cable_50m,
        cable_100m,
        cable_150m,
        total_standard_length,
        feeder_cable_count,
        feeder_cable_length_m,
        closure_fdt_cable_count,
        closure_fdt_cable_length_m,
    ])
    layer.dataProvider().addFeatures([feat])

def populate_boq_layer(boq_layer: QgsVectorLayer,
                       road_layer: QgsVectorLayer,
                       homepass_layer: QgsVectorLayer,
                       pole_layer: QgsVectorLayer,
                       homepasses: List[Homepass],
                       fdts: List[FDT],
                       fats: List[FAT],
                       b2b_cables: List[CableSegment],
                       closures: List[DomeClosure],
                       closure_cables: List[CableSegment],
                       pickup_cables: List[CableSegment]):
    """Populate BOQ with project-level and street-level procurement summaries.

    New columns added:
      total_poles                – distinct poles snapped to on this street
      total_closures             – dome closures whose centroid maps to this street
      feeder_cable_count         – pickup-to-closure cable segments on this street
      feeder_cable_length_m      – total feeder cable length (actual geometry, metres)
      closure_fdt_cable_count    – closure-to-FDT cable segments on this street
      closure_fdt_cable_length_m – total closure-to-FDT cable length (metres)
    """
    if road_layer.crs().isGeographic():
        raise CRSError(
            f"Road layer CRS {road_layer.crs().authid()} is geographic. Projected CRS in meters is required for BOQ."
        )
    homepass_fields = [f.name() for f in homepass_layer.fields()]
    if HOMEPASS_FIELD_NAME not in homepass_fields:
        raise ValidationError(
            f"Configured homepass field '{HOMEPASS_FIELD_NAME}' is missing from layer {homepass_layer.name()}."
        )

    road_to_street, all_streets = _build_road_to_street_map(road_layer)
    if not all_streets:
        all_streets = ["UNASSIGNED"]

    fat_by_name = {fat.name: fat for fat in fats}
    fdt_to_street: Dict[str, str] = {}
    for fdt in fdts:
        street = "UNASSIGNED"
        if fdt.fats:
            road_id = fdt.fats[0].road_id
            street = road_to_street.get(road_id, street)
        fdt_to_street[fdt.name] = street
        if street not in all_streets:
            all_streets.append(street)

    # ── Build spatial index for poles so we can assign them to streets ─────────
    pole_road_index = QgsSpatialIndex()
    pole_id_to_road: Dict[int, int] = {}  # pole fid -> nearest road fid
    if pole_layer and pole_layer.featureCount() > 0:
        road_si = QgsSpatialIndex()
        road_geom_cache: Dict[int, QgsGeometry] = {}
        for rf in road_layer.getFeatures():
            rg = rf.geometry()
            if rg and not rg.isEmpty():
                road_si.addFeature(rf)
                road_geom_cache[rf.id()] = rg
        for pf in pole_layer.getFeatures():
            pg = pf.geometry()
            if not pg or pg.isEmpty():
                continue
            pp = pg.asPoint()
            nearest_roads = road_si.nearestNeighbor(pp, 1)
            pole_id_to_road[pf.id()] = nearest_roads[0] if nearest_roads else -1

    # ── Collect poles actually used (snapped-to) by FATs or FDTs ───────────────
    # A pole is "used on a street" if the FAT/FDT that snapped to it belongs to
    # that street.  We track unique pole coordinates per street.
    street_pole_coords: Dict[str, set] = {}

    def _register_pole(geom: Optional[QgsPointXY], street: str):
        if not geom:
            return
        key = (round(geom.x(), 6), round(geom.y(), 6))
        street_pole_coords.setdefault(street, set()).add(key)

    for fat in fats:
        street = road_to_street.get(fat.road_id, "UNASSIGNED")
        _register_pole(fat.snapped_geometry, street)

    for fdt in fdts:
        street = fdt_to_street.get(fdt.name, "UNASSIGNED")
        _register_pole(fdt.snapped_geometry, street)

    # ── Map closures to streets via their constituent FDT streets ──────────────
    closure_to_street: Dict[str, str] = {}
    for closure in closures:
        if closure.fdts:
            # use the most common street among member FDTs
            from collections import Counter
            street_counts = Counter(fdt_to_street.get(f.name, "UNASSIGNED") for f in closure.fdts)
            closure_to_street[closure.name] = street_counts.most_common(1)[0][0]
        else:
            closure_to_street[closure.name] = "UNASSIGNED"

    # ── Map closure_cables (Closure→FDT) to streets via FDT ──────────────────
    fdt_name_to_street: Dict[str, str] = fdt_to_street  # alias for clarity

    # ── Map pickup_cables (Pickup→Closure) to streets via closure ─────────────
    closure_name_to_street: Dict[str, str] = closure_to_street

    # ── Initialize street buckets ──────────────────────────────────────────────
    _zero: Dict[str, object] = {
        "homepass": 0,
        "fdt": 0,
        "fat": 0,
        "cables": 0,
        "c10": 0, "c50": 0, "c100": 0, "c150": 0,
        "std_len": 0.0,
        "closures": 0,
        "feeder_cnt": 0,
        "feeder_len": 0.0,
        "cfdt_cnt": 0,
        "cfdt_len": 0.0,
    }

    street_rows: Dict[str, Dict] = {}
    for street in sorted(set(all_streets)):
        street_rows[street] = {k: (0.0 if isinstance(v, float) else type(v)()) for k, v in _zero.items()}

    def _ensure_street(s: str):
        if s not in street_rows:
            street_rows[s] = {k: (0.0 if isinstance(v, float) else type(v)()) for k, v in _zero.items()}

    # Homepass
    for hp in homepasses:
        street = road_to_street.get(hp.road_id, "UNASSIGNED")
        _ensure_street(street)
        street_rows[street]["homepass"] += _parse_homepass_value(hp.customer_count)

    # FAT count
    for fat in fats:
        street = road_to_street.get(fat.road_id, "UNASSIGNED")
        _ensure_street(street)
        street_rows[street]["fat"] += 1

    # FDT count
    for fdt in fdts:
        street = fdt_to_street.get(fdt.name, "UNASSIGNED")
        _ensure_street(street)
        street_rows[street]["fdt"] += 1

    # Box-to-Box cables (FDT→FAT distribution)
    for cable in b2b_cables:
        street = "UNASSIGNED"
        end_fat = fat_by_name.get(cable.end_device)
        if end_fat:
            street = road_to_street.get(end_fat.road_id, street)
        else:
            start_fat = fat_by_name.get(cable.start_device)
            if start_fat:
                street = road_to_street.get(start_fat.road_id, street)
            else:
                street = fdt_to_street.get(cable.fdt_id, street)
        _ensure_street(street)
        standardized = classify_length(float(cable.length))
        street_rows[street]["cables"] += 1
        street_rows[street]["std_len"] += standardized
        if standardized == 10:
            street_rows[street]["c10"] += 1
        elif standardized == 50:
            street_rows[street]["c50"] += 1
        elif standardized == 100:
            street_rows[street]["c100"] += 1
        else:
            street_rows[street]["c150"] += 1

    # Closures count per street
    for closure in closures:
        street = closure_to_street.get(closure.name, "UNASSIGNED")
        _ensure_street(street)
        street_rows[street]["closures"] += 1

    # Feeder cables (Pickup → Closure) per street
    for cable in pickup_cables:
        street = closure_name_to_street.get(cable.end_device, "UNASSIGNED")
        _ensure_street(street)
        street_rows[street]["feeder_cnt"] += 1
        street_rows[street]["feeder_len"] += float(cable.length)

    # Closure-to-FDT cables per street (keyed by FDT name = cable.fdt_id / end_device)
    for cable in closure_cables:
        fdt_name = cable.end_device  # end_device is the FDT name
        street = fdt_name_to_street.get(fdt_name, "UNASSIGNED")
        _ensure_street(street)
        street_rows[street]["cfdt_cnt"] += 1
        street_rows[street]["cfdt_len"] += float(cable.length)

    # ── Pole counts per street from collected coordinates ─────────────────────
    # ── Project-level totals ───────────────────────────────────────────────────
    total_c10   = sum(v["c10"]   for v in street_rows.values())
    total_c50   = sum(v["c50"]   for v in street_rows.values())
    total_c100  = sum(v["c100"]  for v in street_rows.values())
    total_c150  = sum(v["c150"]  for v in street_rows.values())
    total_std_len       = sum(v["std_len"]     for v in street_rows.values())
    total_street_cables = sum(v["cables"]      for v in street_rows.values())
    total_homepass      = sum(int(v["homepass"]) for v in street_rows.values())

    # All distinct poles snapped anywhere across the project
    all_pole_coords: set = set()
    for coords in street_pole_coords.values():
        all_pole_coords.update(coords)
    total_poles_project = len(all_pole_coords)

    configured_total_homepass = safe_homepass_sum(list(homepass_layer.getFeatures()), HOMEPASS_FIELD_NAME)
    if total_homepass != configured_total_homepass:
        raise ValidationError(
            f"Homepass aggregation mismatch: assigned={total_homepass}, configured_field_sum={configured_total_homepass}"
        )
    if total_street_cables != len(b2b_cables):
        raise ValidationError(
            f"Street cable aggregation mismatch: per-street={total_street_cables}, total={len(b2b_cables)}"
        )
    if total_homepass < len(fats):
        raise Warning("Homepass total appears incorrect. Check value aggregation.")

    # ── PROJECT summary row ────────────────────────────────────────────────────
    _add_boq_row(
        boq_layer,
        "PROJECT",
        "ALL",
        int(total_homepass),
        len(fdts),
        len(fats),
        total_poles_project,
        len(closures),
        len(b2b_cables),
        int(total_c10),
        int(total_c50),
        int(total_c100),
        int(total_c150),
        float(total_std_len),
        len(pickup_cables),
        float(sum(c.length for c in pickup_cables)),
        len(closure_cables),
        float(sum(c.length for c in closure_cables)),
    )

    # ── Per-STREET rows ────────────────────────────────────────────────────────
    for street in sorted(street_rows.keys()):
        row = street_rows[street]
        poles_on_street = len(street_pole_coords.get(street, set()))
        _add_boq_row(
            boq_layer,
            "STREET",
            street,
            int(row["homepass"]),
            int(row["fdt"]),
            int(row["fat"]),
            poles_on_street,
            int(row["closures"]),
            int(row["cables"]),
            int(row["c10"]),
            int(row["c50"]),
            int(row["c100"]),
            int(row["c150"]),
            float(row["std_len"]),
            int(row["feeder_cnt"]),
            float(row["feeder_len"]),
            int(row["cfdt_cnt"]),
            float(row["cfdt_len"]),
        )

# =============================================================================
# OUTPUT LAYER CREATION
# =============================================================================

def create_output_layers(crs: QgsCoordinateReferenceSystem,
                        project: QgsProject) -> Dict[str, QgsVectorLayer]:
    """Create memory layers for output data"""
    
    layers = {}

    # Remove stale output layers from previous runs.
    for old_name in ("FTTX_Errors", "BOQ", "FDT", "Closure"):
        for old_layer in project.mapLayersByName(old_name):
            project.removeMapLayer(old_layer.id())

    # FDT Layer
    fdt_layer = QgsVectorLayer(f"Point?crs={crs.authid()}", "FDT", "memory")
    fdt_provider = fdt_layer.dataProvider()
    fdt_provider.addAttributes([
        QgsField("Name", QVariant.String),
        QgsField("fdt_code", QVariant.String),
        QgsField("fdt_serial", QVariant.String),
        QgsField("Tag", QVariant.String),
        QgsField("total_customers", QVariant.Int),
        QgsField("total_FAT", QVariant.Int),
        QgsField("wing_A_count", QVariant.Int),
        QgsField("wing_B_count", QVariant.Int)
    ])
    fdt_layer.updateFields()
    layers['fdt'] = fdt_layer

    # FAT Layer
    fat_layer = QgsVectorLayer(f"Point?crs={crs.authid()}", "FAT", "memory")
    fat_provider = fat_layer.dataProvider()
    fat_provider.addAttributes([
        QgsField("Name", QVariant.String),
        QgsField("parent_fdt", QVariant.String),
        QgsField("fdt_serial", QVariant.String),
        QgsField("wing_id", QVariant.String),
        QgsField("fat_sequence", QVariant.String),
        QgsField("fat_full_code", QVariant.String),
        QgsField("from_device", QVariant.String),
        QgsField("to_device", QVariant.String),
        QgsField("road_id", QVariant.Int),
        QgsField("total_customers", QVariant.Int),
        QgsField("cascade_order", QVariant.Int),
        QgsField("wing", QVariant.String),
        QgsField("splitter_type", QVariant.String)
    ])
    fat_layer.updateFields()
    layers['fat'] = fat_layer
    
    # Box-to-Box Cable Layer
    b2b_layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", "Box_to_Box", "memory")
    b2b_provider = b2b_layer.dataProvider()
    b2b_provider.addAttributes([
        QgsField("name", QVariant.String),
        QgsField("start_device", QVariant.String),
        QgsField("end_device", QVariant.String),
        QgsField("cable_length", QVariant.Double),
        QgsField("fdt_id", QVariant.String)
    ])
    b2b_layer.updateFields()
    layers['b2b'] = b2b_layer
    
    # Box-to-Customer Cable Layer
    b2c_layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", "Box_to_Customer", "memory")
    b2c_provider = b2c_layer.dataProvider()
    b2c_provider.addAttributes([
        QgsField("name", QVariant.String),
        QgsField("fat_name", QVariant.String),
        QgsField("customer_id", QVariant.Int),
        QgsField("cable_length", QVariant.Double)
    ])
    b2c_layer.updateFields()
    layers['b2c'] = b2c_layer
    
    # BOQ Layer (non-spatial summary table)
    boq_layer = QgsVectorLayer("None", "BOQ", "memory")
    boq_provider = boq_layer.dataProvider()
    boq_provider.addAttributes([
        # Row identity
        QgsField("level_type", QVariant.String),
        QgsField("street_name", QVariant.String),
        # Homepass / devices
        QgsField("total_homepass", QVariant.Int),
        QgsField("total_fdt", QVariant.Int),
        QgsField("total_fat", QVariant.Int),
        QgsField("total_poles", QVariant.Int),
        QgsField("total_closures", QVariant.Int),
        # Box-to-box (FAT distribution) cables
        QgsField("total_box_to_box_cable", QVariant.Int),
        QgsField("cable_10m", QVariant.Int),
        QgsField("cable_50m", QVariant.Int),
        QgsField("cable_100m", QVariant.Int),
        QgsField("cable_150m", QVariant.Int),
        QgsField("total_standard_length", QVariant.Double),
        # Feeder cables (Pickup → Closure)
        QgsField("feeder_cable_count", QVariant.Int),
        QgsField("feeder_cable_length_m", QVariant.Double),
        # Closure-to-FDT distribution cables
        QgsField("closure_fdt_cable_count", QVariant.Int),
        QgsField("closure_fdt_cable_length_m", QVariant.Double),
    ])
    boq_layer.updateFields()
    layers['boq'] = boq_layer

    # Dome Closure Layer
    closure_layer = QgsVectorLayer(f"Point?crs={crs.authid()}", "Closure", "memory")
    closure_provider = closure_layer.dataProvider()
    closure_provider.addAttributes([
        QgsField("Name",          QVariant.String),
        QgsField("total_FDT",     QVariant.Int),
        QgsField("feed_capacity", QVariant.Int),
        QgsField("pickup_name",   QVariant.String),
    ])
    closure_layer.updateFields()
    layers['closure'] = closure_layer

    # Pickup Point Layer (generated or supplied)
    pickup_layer = QgsVectorLayer(f"Point?crs={crs.authid()}", "Pickup_Point", "memory")
    pickup_provider = pickup_layer.dataProvider()
    pickup_provider.addAttributes([
        QgsField("name", QVariant.String),
        QgsField("source", QVariant.String)
    ])
    pickup_layer.updateFields()
    layers['pickup_point'] = pickup_layer

    # Feeder Cable Layer (pickup to closure)
    feeder_layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", "Feeder_Cable", "memory")
    feeder_provider = feeder_layer.dataProvider()
    feeder_provider.addAttributes([
        QgsField("name", QVariant.String),
        QgsField("start_device", QVariant.String),
        QgsField("end_device", QVariant.String),
        QgsField("cable_type", QVariant.String),
        QgsField("length", QVariant.Double)
    ])
    feeder_layer.updateFields()
    layers['feeder'] = feeder_layer

    # Closure-to-FDT Cable Layer
    closure_cable_layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", "Closure_to_FDT", "memory")
    closure_cable_provider = closure_cable_layer.dataProvider()
    closure_cable_provider.addAttributes([
        QgsField("name",          QVariant.String),
        QgsField("closure_name",  QVariant.String),   # originating closure (always)
        QgsField("start_device",  QVariant.String),   # closure OR previous FDT in cascade
        QgsField("end_device",    QVariant.String),
        QgsField("cable_length",  QVariant.Double),
        QgsField("fdt_id",        QVariant.String),
        QgsField("buffer_act",    QVariant.String),
        QgsField("core_act",      QVariant.String),
        QgsField("buffer_red",    QVariant.String),
        QgsField("core_red",      QVariant.String),
        QgsField("cable_capacity",QVariant.Int),
        QgsField("cascade_group", QVariant.Int),
    ])
    closure_cable_layer.updateFields()
    layers['closure_cable'] = closure_cable_layer

    
    # Add all layers to project
    for layer in layers.values():
        project.addMapLayer(layer)
    
    return layers

def populate_output_layers(layers: Dict[str, QgsVectorLayer],
                          road_layer: QgsVectorLayer,
                          homepass_layer: QgsVectorLayer,
                          pole_layer: QgsVectorLayer,
                          homepasses: List[Homepass],
                          fdts: List[FDT],
                          fats: List[FAT],
                          b2b_cables: List[CableSegment],
                          b2c_cables: List[CableSegment],
                          closures: List[DomeClosure],
                          closure_cables: List[CableSegment],
                          pickup_cables: List[CableSegment],
                          pickup_layer: Optional[QgsVectorLayer],
                          logger: PerformanceLogger):
    """Populate output layers with features, including dome closures and cables"""
    logger.start_module("Output Layer Population")

    # FDT Layer
    fdt_features = []
    for fdt in fdts:
        feat = QgsFeature(layers['fdt'].fields())
        geom = fdt.snapped_geometry or fdt.geometry
        feat.setGeometry(QgsGeometry.fromPointXY(geom))
        feat.setAttributes([
            fdt.name,
            fdt.fdt_code or fdt.name,
            fdt.fdt_serial,
            "",  # Tag (can be filled later)
            fdt.total_customers,
            fdt.total_fat,
            fdt.wing_a_count,
            fdt.wing_b_count
        ])
        fdt_features.append(feat)
    layers['fdt'].dataProvider().addFeatures(fdt_features)
    
    # FAT Layer
    fat_features = []
    for fat in fats:
        feat = QgsFeature(layers['fat'].fields())
        geom = fat.offset_geometry or fat.snapped_geometry or fat.geometry
        feat.setGeometry(QgsGeometry.fromPointXY(geom))
        feat.setAttributes([
            fat.name,
            fat.parent_fdt,
            fat.fdt_serial,
            fat.wing_id or fat.wing,
            fat.fat_sequence,
            fat.fat_full_code or fat.name,
            fat.from_device,
            fat.to_device,
            fat.road_id,
            fat.total_customers,
            fat.cascade_order,
            fat.wing,
            fat.splitter_type
        ])
        fat_features.append(feat)
    
    layers['fat'].dataProvider().addFeatures(fat_features)
    
    # Box-to-Box Cable Layer
    b2b_features = []
    for cable in b2b_cables:
        feat = QgsFeature(layers['b2b'].fields())
        feat.setGeometry(cable.geometry)
        feat.setAttributes([
            cable.name,
            cable.start_device,
            cable.end_device,
            cable.length,
            cable.fdt_id
        ])
        b2b_features.append(feat)
    
    layers['b2b'].dataProvider().addFeatures(b2b_features)
    
    # Box-to-Customer Cable Layer
    b2c_features = []
    for cable in b2c_cables:
        feat = QgsFeature(layers['b2c'].fields())
        feat.setGeometry(cable.geometry)
        # Extract customer ID from name (handles DROP_FAT_C123 and DROP_FAT_C123_1)
        customer_id = 0
        if '_C' in cable.name:
            try:
                suffix = cable.name.split('_C')[-1]
                customer_id = int(suffix.split('_')[0])
            except (ValueError, IndexError):
                pass
        
        feat.setAttributes([
            cable.name,
            cable.start_device,
            customer_id,
            cable.length
        ])
        b2c_features.append(feat)
    
    layers['b2c'].dataProvider().addFeatures(b2c_features)
    
    # Dome Closure Layer features
    if closures and 'closure' in layers:
        closure_features = []
        for closure in closures:
            feat = QgsFeature(layers['closure'].fields())
            geom = closure.snapped_geometry or closure.geometry
            feat.setGeometry(QgsGeometry.fromPointXY(geom))
            feat.setAttributes([
                closure.name,
                closure.total_fdt,
                closure.feed_capacity,
                closure.pickup_name or "",
            ])
            closure_features.append(feat)
        layers['closure'].dataProvider().addFeatures(closure_features)

    # Pickup Point Features
    if 'pickup_point' in layers:
        pp_feats = []
        seen = set()
        # first, include any user-supplied pickup features
        if pickup_layer and pickup_layer.featureCount() > 0:
            for feat in pickup_layer.getFeatures():
                geom = feat.geometry()
                if not geom or geom.isEmpty():
                    continue
                if geom.wkbType() != QgsWkbTypes.PointGeometry:
                    continue
                pt = geom.asPoint()
                key = (pt.x(), pt.y())
                seen.add(key)
                new_feat = QgsFeature(layers['pickup_point'].fields())
                new_feat.setGeometry(QgsGeometry.fromPointXY(pt))
                name = ""
                if 'name' in feat.fields().names():
                    try:
                        name = str(feat['name'])
                    except Exception:
                        name = ""
                new_feat.setAttributes([name, "existing"])
                pp_feats.append(new_feat)
        # next add pickups assigned to closures (generated or nearest)
        if closures:
            for closure in closures:
                if closure.pickup_geometry:
                    key = (closure.pickup_geometry.x(), closure.pickup_geometry.y())
                    if key in seen:
                        continue
                    seen.add(key)
                    new_feat = QgsFeature(layers['pickup_point'].fields())
                    new_feat.setGeometry(QgsGeometry.fromPointXY(closure.pickup_geometry))
                    source = "existing" if pickup_layer and pickup_layer.featureCount() > 0 else "generated"
                    new_feat.setAttributes([closure.pickup_name or "PICKUP", source])
                    pp_feats.append(new_feat)
        layers['pickup_point'].dataProvider().addFeatures(pp_feats)

    # Feeder Cable Layer features
    if 'feeder' in layers:
        feeder_feats = []
        for cable in pickup_cables:
            feat = QgsFeature(layers['feeder'].fields())
            feat.setGeometry(cable.geometry)
            cable_type = cable.core_info.get('cable_type', '')
            feat.setAttributes([
                cable.name,
                cable.start_device,
                cable.end_device,
                cable_type,
                cable.length
            ])
            feeder_feats.append(feat)
        layers['feeder'].dataProvider().addFeatures(feeder_feats)

    # Closure-to-FDT Cable Layer features
    if closure_cables and 'closure_cable' in layers:
        cc_features = []
        for cable in closure_cables:
            feat = QgsFeature(layers['closure_cable'].fields())
            feat.setGeometry(cable.geometry)
            feat.setAttributes([
                cable.name,
                cable.core_info.get("closure_name", ""),
                cable.start_device,
                cable.end_device,
                cable.length,
                cable.fdt_id,
                cable.core_info.get("buffer_color_active", ""),
                cable.core_info.get("core_color_active", ""),
                cable.core_info.get("buffer_color_redundant", ""),
                cable.core_info.get("core_color_redundant", ""),
                cable.core_info.get("cable_capacity", ""),
                cable.core_info.get("cascade_group", "")
            ])
            cc_features.append(feat)
        layers['closure_cable'].dataProvider().addFeatures(cc_features)


    # BOQ table
    populate_boq_layer(
        layers['boq'],
        road_layer,
        homepass_layer,
        pole_layer,
        homepasses,
        fdts,
        fats,
        b2b_cables,
        closures,
        closure_cables,
        pickup_cables,
    )
    
    # Update extents
    for layer in layers.values():
        layer.updateExtents()
    
    logger.end_module("Output Layer Population")
    logger.log(f"Created: {len(fdt_features)} FDTs, {len(fat_features)} FATs, "
              f"{len(b2b_features)} box-to-box cables, {len(b2c_features)} drop cables")

# =============================================================================
# VALIDATION ENGINE
# =============================================================================

def validate_design(fdts: List[FDT],
                   fats: List[FAT],
                   b2b_cables: List[CableSegment],
                   b2c_cables: List[CableSegment],
                   error_collector: ErrorCollector,
                   logger: PerformanceLogger):
    """Deprecated: design validation was removed in BOQ-only workflow."""
    logger.start_module("Validation")
    logger.log("Validation module is disabled. BOQ generation is the active output workflow.")
    logger.end_module("Validation")

def validate_drop_cable_constraints(b2c_cables: List[CableSegment], logger: PerformanceLogger):
    """Hard validation: every FAT->Customer cable must be <= MAX_BOX_TO_CUSTOMER_DISTANCE."""
    logger.start_module("Drop Cable Constraint Validation")
    for cable in b2c_cables:
        if float(cable.length) > (MAX_BOX_TO_CUSTOMER_DISTANCE + DROP_LENGTH_TOLERANCE):
            raise ValidationError(
                f"Customer connection exceeds maximum distance: {cable.name} = "
                f"{float(cable.length):.2f}m > {MAX_BOX_TO_CUSTOMER_DISTANCE + DROP_LENGTH_TOLERANCE:.2f}m"
            )
    logger.log(
        f"Validated {len(b2c_cables)} drop cables: all <= {MAX_BOX_TO_CUSTOMER_DISTANCE + DROP_LENGTH_TOLERANCE:.2f}m"
    )
    logger.end_module("Drop Cable Constraint Validation")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def _apply_config_to_globals(config: FTTXConfig) -> Dict[str, object]:
    """Temporarily override script-level constants with GUI configuration values."""
    mapping = {
        "MAX_BOX_TO_BOX_DISTANCE": config.max_box_to_box_distance,
        "MAX_BOX_TO_CUSTOMER_DISTANCE": config.max_box_to_customer_distance,
        "DROP_LENGTH_TOLERANCE": config.drop_length_tolerance,
        "SNAP_OFFSET_DISTANCE": config.snap_offset_distance,
        "FAT_OFFSET_RADIUS": config.fat_offset_radius,
        "SPLITTER_PATTERN": config.splitter_pattern,
        "HOMEPASS_FIELD_NAME": config.homepass_field_name,
    }
    old_vals: Dict[str, object] = {}
    for key, new_value in mapping.items():
        old_vals[key] = globals().get(key)
        globals()[key] = new_value
    return old_vals


def _restore_globals(old_values: Dict[str, object]):
    for key, val in old_values.items():
        globals()[key] = val


def validate_selected_layers(layers: Dict[str, QgsVectorLayer],
                             config: FTTXConfig,
                             logger: PerformanceLogger):
    """Validate required and optional layers selected via the GUI."""
    required = ['road', 'homepass', 'pole']
    for key in required:
        layer = layers.get(key)
        if not layer:
            raise LayerNotFoundError(f"Required layer '{key}' was not selected.")
        logger.log(f"Using layer: {layer.name()}")
        crs = layer.crs()
        if crs.isGeographic():
            raise CRSError(
                f"Layer '{layer.name()}' has geographic CRS {crs.authid()}. Projected CRS required."
            )
        logger.log(f"  - CRS: {crs.authid()}")
        logger.log(f"  - Feature count: {layer.featureCount()}")

    for key in ['building', 'pickup']:
        layer = layers.get(key)
        if layer:
            logger.log(f"Using optional layer: {layer.name()}")
            crs = layer.crs()
            if crs.isGeographic():
                raise CRSError(
                    f"Layer '{layer.name()}' has geographic CRS {crs.authid()}. Projected CRS required."
                )
            logger.log(f"  - CRS: {crs.authid()}")
            logger.log(f"  - Feature count: {layer.featureCount()}")

    home_layer = layers['homepass']
    field_names = [f.name() for f in home_layer.fields()]
    if config.homepass_field_name not in field_names:
        raise ValidationError(
            f"Homepass field '{config.homepass_field_name}' not found in layer {home_layer.name()}"
        )


def run_fttx_design(config: Optional[FTTXConfig] = None,
                    layers: Optional[Dict[str, QgsVectorLayer]] = None,
                    progress_callback: Optional[callable] = None):
    """Main execution function with GUI-driven configuration and progress updates."""
    if config is None:
        config = FTTXConfig()
    if progress_callback is None:
        def progress_callback(value: int, message: str = ""):
            return None

    def _update_progress(value: int, message: str = ""):
        try:
            progress_callback(value, message)
        except Exception:
            pass
        try:
            if QApplication.instance():
                QApplication.processEvents()
        except Exception:
            pass

    old_globals = _apply_config_to_globals(config)
    try:
        # Initialize logger
        logger = PerformanceLogger()
        logger.log(f"Script version: {SCRIPT_VERSION}")
        logger.log("="*60)
        logger.log("Starting FTTX Network Design Automation")
        logger.log("="*60)

        # Validate layers
        _update_progress(5, "Layer Validation")
        project = QgsProject.instance()

        if layers is None:
            layers = validate_and_load_layers(project, logger)
        else:
            validate_selected_layers(layers, config, logger)

        # Initialize utilities
        crs = layers['road'].crs()
        index_mgr = SpatialIndexManager()
        dist_calc = DistanceCalculator(crs)
        error_collector = ErrorCollector()

        # Process roads
        _update_progress(10, "Road Processing")
        roads = process_roads(layers['road'], index_mgr, logger)

        # Assign homepasses to roads
        _update_progress(25, "Homepass Assignment")
        homepasses = assign_homepasses_to_roads(
            layers['homepass'], layers['road'], roads, index_mgr, dist_calc, logger,
            error_collector
        )

        # Generate FATs
        _update_progress(40, "FAT Generation")
        fats = generate_fats(
            roads,
            layers['road'],
            index_mgr,
            layers.get('building'),
            dist_calc,
            error_collector,
            logger
        )

        # FAT cascading
        _update_progress(50, "FAT Cascading")
        fats = create_fat_cascade(fats, roads, dist_calc, error_collector, logger)

        # Generate FDTs
        _update_progress(65, "FDT Generation")
        _fdt_road_index = index_mgr.get_index(layers['road'])
        fdts = generate_fdts(
            fats, error_collector, logger,
            roads=roads, road_index=_fdt_road_index, dist_calc=dist_calc
        )

        # Assign wings per FDT
        _update_progress(75, "Wing Assignment")
        assign_wings_per_fdt(fdts, logger)
        create_fdt_wing_cascades(fdts, error_collector, logger)

        # Snap to poles
        _update_progress(85, "Pole Snapping")
        fdts, fats = snap_to_poles(
            fdts, fats, homepasses, layers['pole'], roads, index_mgr, dist_calc, error_collector, logger
        )

        # Cable routing
        _update_progress(95, "Cable Routing")
        road_index = index_mgr.get_index(layers['road'])
        cable_router = CableRouter(layers['road'], dist_calc, roads=roads, road_index=road_index)
        cable_router.build_graph()

        b2b_cables = route_fdt_to_fats(fdts, cable_router, error_collector, logger)

        closures: List[DomeClosure] = []
        closure_cables: List[CableSegment] = []
        pickup_cables: List[CableSegment] = []

        if config.enable_closure_generation:
            closures = generate_closures(
                fdts,
                layers.get('pickup'),
                index_mgr,
                dist_calc,
                roads,
                index_mgr.get_index(layers['road']),
                layers.get('building'),
                logger,
                pole_layer=layers['pole'],
            )

            if closures:
                closure_cables = route_closure_to_fdts(closures, cable_router, logger)
                closure_cables = correct_cable_zone_violations(closures, closure_cables, cable_router, logger)
                pickup_cables = route_pickup_to_closure(closures, layers.get('pickup'), index_mgr, cable_router, logger)

            _update_progress(100, "Closure Generation")

        # Route FAT to homepass cables
        b2c_cables = route_fat_to_homepasses(fats, homepasses, dist_calc, error_collector, logger)
        validate_drop_cable_constraints(b2c_cables, logger)

        # Create output layers
        output_layers = create_output_layers(crs, project)

        # Populate output layers
        populate_output_layers(
            output_layers,
            layers['road'],
            layers['homepass'],
            layers['pole'],
            homepasses,
            fdts,
            fats,
            b2b_cables,
            b2c_cables,
            closures,
            closure_cables,
            pickup_cables,
            layers.get('pickup'),
            logger
        )

        # Final progress
        _update_progress(100, "Completed")

        # Final logging
        logger.log("="*60)
        logger.log("FTTX Network Design Completed Successfully")
        logger.log(f"Total execution time: {time.time() - logger.start_time:.2f} seconds")
        logger.log("="*60)

        # Summary
        print(f"\nFTTX Design Summary:")
        print(f"  - Roads processed: {len(roads)}")
        print(f"  - Homepasses processed: {len(homepasses)}")
        print(f"  - FATs created: {len(fats)}")
        print(f"  - FDTs created: {len(fdts)}")
        print(f"  - Dome closures created: {len(closures)}")
        print(f"  - Box-to-box cables: {len(b2b_cables)}")
        print(f"  - Closure-to-FDT cables: {len(closure_cables)}")
        print(f"  - Pickup-to-Closure cables: {len(pickup_cables)}")
        print(f"  - Drop cables: {len(b2c_cables)}")
        print(f"  - BOQ rows: {output_layers['boq'].featureCount()}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {str(e)}")
        raise
    finally:
        _restore_globals(old_globals)


class FTTXDesignerDialog(QDialog):
    """Dialog for configuring and running the FTTX design pipeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FTTX Designer")
        self.resize(600, 800)
        self._build_ui()
        self._populate_layers()
        self._setup_logging()

    def _build_ui(self):
        # Apply modern stylesheet
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cfcfcf;
                border-radius: 6px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 2px 5px;
            }
            QPushButton {
                background-color: #2d89ef;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1b5fad;
            }
            QPushButton#run_button {
                background-color: #107c10;
            }
            QPushButton#run_button:hover {
                background-color: #0b5a0b;
            }
            QProgressBar {
                border: 1px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2d89ef;
            }
            QTabWidget::pane {
                border: 1px solid #cfcfcf;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #f0f0f0;
                border: 1px solid #cfcfcf;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: none;
            }
        """)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Tab 1: Network Layers
        self._create_network_layers_tab()

        # Tab 2: Parameters
        self._create_parameters_tab()

        # Tab 3: Execution
        self._create_execution_tab()

        # Status label and progress bar
        self.status_label = QLabel("Status: Ready to start")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # Buttons
        self.run_button = QPushButton("▶ Run Network Design")
        self.run_button.setObjectName("run_button")
        self.cancel_button = QPushButton("Cancel")
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.run_button)
        btn_layout.addWidget(self.cancel_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        # Connections
        self.homepass_layer_dropdown.currentIndexChanged.connect(self._on_homepass_layer_changed)
        self.run_button.clicked.connect(self._on_run)
        self.cancel_button.clicked.connect(self.reject)

    def _create_network_layers_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        group = QGroupBox("Network Input Layers")
        form = QFormLayout()

        self.road_layer_dropdown = QComboBox()
        self.homepass_layer_dropdown = QComboBox()
        self.homepass_field_dropdown = QComboBox()
        self.pole_layer_dropdown = QComboBox()
        self.building_layer_dropdown = QComboBox()
        self.pickup_layer_dropdown = QComboBox()

        form.addRow(QLabel("Road Layer:"), self.road_layer_dropdown)
        form.addRow(QLabel("Homepass Layer:"), self.homepass_layer_dropdown)
        form.addRow(QLabel("Homepass Field:"), self.homepass_field_dropdown)
        form.addRow(QLabel("Pole Layer:"), self.pole_layer_dropdown)
        form.addRow(QLabel("Building Layer (optional):"), self.building_layer_dropdown)
        form.addRow(QLabel("Pickup Layer (optional):"), self.pickup_layer_dropdown)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Network Layers")

    def _create_parameters_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Distance Constraints
        dist_group = QGroupBox("Distance Constraints")
        dist_form = QFormLayout()

        self.max_box_box_spin = QDoubleSpinBox()
        self.max_box_box_spin.setRange(0, 10000)
        self.max_box_box_spin.setValue(MAX_BOX_TO_BOX_DISTANCE)
        self.max_box_box_spin.setSuffix(" m")

        self.max_box_customer_spin = QDoubleSpinBox()
        self.max_box_customer_spin.setRange(0, 10000)
        self.max_box_customer_spin.setValue(MAX_BOX_TO_CUSTOMER_DISTANCE)
        self.max_box_customer_spin.setSuffix(" m")

        self.drop_tolerance_spin = QDoubleSpinBox()
        self.drop_tolerance_spin.setRange(0, 100)
        self.drop_tolerance_spin.setValue(DROP_LENGTH_TOLERANCE)
        self.drop_tolerance_spin.setSuffix(" m")

        self.snap_offset_spin = QDoubleSpinBox()
        self.snap_offset_spin.setRange(0, 100)
        self.snap_offset_spin.setValue(SNAP_OFFSET_DISTANCE)
        self.snap_offset_spin.setSuffix(" m")

        self.fat_offset_spin = QDoubleSpinBox()
        self.fat_offset_spin.setRange(0, 10)
        self.fat_offset_spin.setValue(FAT_OFFSET_RADIUS)
        self.fat_offset_spin.setSuffix(" m")

        dist_form.addRow(QLabel("Max Box-to-Box Distance:"), self.max_box_box_spin)
        dist_form.addRow(QLabel("Max Box-to-Customer Distance:"), self.max_box_customer_spin)
        dist_form.addRow(QLabel("Drop Length Tolerance:"), self.drop_tolerance_spin)
        dist_form.addRow(QLabel("Snap Offset Distance:"), self.snap_offset_spin)
        dist_form.addRow(QLabel("FAT Offset Radius:"), self.fat_offset_spin)

        dist_group.setLayout(dist_form)
        layout.addWidget(dist_group)

        # Splitter Configuration
        splitter_group = QGroupBox("Splitter Configuration")
        splitter_form = QFormLayout()

        self.splitter_pattern_input = QLineEdit(",".join(SPLITTER_PATTERN))
        self.splitter_pattern_input.setMinimumWidth(300)

        splitter_form.addRow(QLabel("Splitter Pattern:"), self.splitter_pattern_input)

        splitter_group.setLayout(splitter_form)
        layout.addWidget(splitter_group)

        # Closure Generation
        closure_group = QGroupBox("Closure Generation")
        closure_layout = QVBoxLayout()

        self.closure_checkbox = QCheckBox("Generate Dome Closures and Pickup Points")
        self.closure_checkbox.setChecked(True)
        self.fast_mode_checkbox = QCheckBox("Fast Mode (use Euclidean for clustering)")
        self.fast_mode_checkbox.setChecked(FAST_MODE_DEFAULT)

        closure_layout.addWidget(self.closure_checkbox)
        closure_layout.addWidget(self.fast_mode_checkbox)
        closure_group.setLayout(closure_layout)
        layout.addWidget(closure_group)

        layout.addStretch()
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Parameters")

    def _create_execution_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Script Version
        version_label = QLabel(f"Script Version: {SCRIPT_VERSION}")
        version_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(version_label)

        # Pipeline Steps
        steps_group = QGroupBox("Design Pipeline")
        steps_layout = QVBoxLayout()
        steps = [
            "• Road Processing",
            "• Homepass Assignment",
            "• FAT Generation",
            "• FAT Cascading",
            "• FDT Generation",
            "• Wing Assignment",
            "• Pole Snapping",
            "• Cable Routing",
            "• Closure Generation (optional)",
            "• BOQ: poles / closures / feeder / closure-to-FDT cables per street",
        ]
        for step in steps:
            steps_layout.addWidget(QLabel(step))
        steps_group.setLayout(steps_layout)
        layout.addWidget(steps_group)

        # Execution Log
        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Execution")

    def _setup_logging(self):
        self.gui_handler = GuiLogHandler()
        self.gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.gui_handler.log_signal.connect(self._append_log)
        # Will be added to logger during run

    def _append_log(self, message):
        self.log_text.append(message)
        # Scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

    def _populate_layers(self):
        project = QgsProject.instance()
        vector_layers = [l for l in project.mapLayers().values() if isinstance(l, QgsVectorLayer)]
        layer_names = sorted([layer.name() for layer in vector_layers])

        def add_items(combo: QComboBox, allow_none: bool = False):
            combo.clear()
            if allow_none:
                combo.addItem("<None>")
            combo.addItems(layer_names)

        add_items(self.road_layer_dropdown)
        add_items(self.homepass_layer_dropdown)
        add_items(self.pole_layer_dropdown)
        add_items(self.building_layer_dropdown, allow_none=True)
        add_items(self.pickup_layer_dropdown, allow_none=True)

        # Trigger initial homepass field populate
        self._on_homepass_layer_changed()

    def _get_selected_layer(self, combo: QComboBox) -> Optional[QgsVectorLayer]:
        name = combo.currentText()
        if not name or name == "<None>":
            return None
        layers = QgsProject.instance().mapLayersByName(name)
        if not layers:
            return None
        layer = layers[0]
        if not isinstance(layer, QgsVectorLayer):
            return None
        return layer

    def _on_homepass_layer_changed(self):
        layer = self._get_selected_layer(self.homepass_layer_dropdown)
        self.homepass_field_dropdown.clear()
        if not layer:
            return
        fields = [f.name() for f in layer.fields()]
        self.homepass_field_dropdown.addItems(fields)

    def _update_progress(self, value: int, message: str = ""):
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(f"Status: {message}")
        QApplication.processEvents()

    def _on_run(self):
        road_layer = self._get_selected_layer(self.road_layer_dropdown)
        homepass_layer = self._get_selected_layer(self.homepass_layer_dropdown)
        pole_layer = self._get_selected_layer(self.pole_layer_dropdown)
        building_layer = self._get_selected_layer(self.building_layer_dropdown)
        pickup_layer = self._get_selected_layer(self.pickup_layer_dropdown)

        if not road_layer or not homepass_layer or not pole_layer:
            QMessageBox.warning(self, "Missing Layers", "Please select Road, Homepass, and Pole layers.")
            return

        config = FTTXConfig(
            max_box_to_box_distance=self.max_box_box_spin.value(),
            max_box_to_customer_distance=self.max_box_customer_spin.value(),
            drop_length_tolerance=self.drop_tolerance_spin.value(),
            snap_offset_distance=self.snap_offset_spin.value(),
            fat_offset_radius=self.fat_offset_spin.value(),
            splitter_pattern=[s.strip() for s in self.splitter_pattern_input.text().split(",") if s.strip()],
            homepass_field_name=self.homepass_field_dropdown.currentText(),
            enable_closure_generation=self.closure_checkbox.isChecked(),
            fast_mode=self.fast_mode_checkbox.isChecked()
        )

        layers = {
            'road': road_layer,
            'homepass': homepass_layer,
            'pole': pole_layer,
            'building': building_layer,
            'pickup': pickup_layer
        }

        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

        # Add GUI handler to logger
        logger = logging.getLogger('FTTX_Design')
        logger.addHandler(self.gui_handler)

        try:
            run_fttx_design(config, layers, self._update_progress)
            QMessageBox.information(self, "FTTX Designer", "FTTX design completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "FTTX Designer", f"Error: {str(e)}")
        finally:
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(True)
            logger.removeHandler(self.gui_handler)


def launch_fttx_designer():
    dialog = FTTXDesignerDialog()
    dialog.exec_()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__console__' or __name__ == '__main__':
    # Run when executed in QGIS Python Console
    launch_fttx_designer()

