"""
================================================================================
Road Name Matcher — PyQGIS GUI Tool
================================================================================
Author  : GIS Expert
Version : 1.0.0
Purpose : Match point locations to road names via spatial search + reverse
          geocoding fallback, with a full PyQt5 GUI for QGIS 3.x.

HOW TO RUN
----------
Option A — QGIS Python Console:
    exec(open('/path/to/road_name_matcher_qgis.py').read())

Option B — Run as script with QGIS Python interpreter:
    qgis_process run script:/path/to/road_name_matcher_qgis.py

Option C — Plugins > Python Console > Open Editor, paste script, Run.

REQUIREMENTS
------------
- QGIS 3.x
- Internet access for Nominatim fallback (free/OSM-based)
- No paid APIs required
================================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import re
import csv
import json
import time
import logging
import threading
import traceback
from datetime import datetime
from functools import lru_cache

# ── QGIS / PyQt ──────────────────────────────────────────────────────────────
from qgis.core import (
    QgsApplication,
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsSpatialIndex, QgsCoordinateTransform, QgsCoordinateReferenceSystem,
    QgsProject, QgsField, QgsVectorFileWriter, QgsWkbTypes,
    QgsCoordinateTransformContext, QgsFields, QgsFeatureSink,
    QgsExpression, QgsFeatureRequest, Qgis
)
from qgis.PyQt.QtCore import (
    Qt, QThread, pyqtSignal, QVariant, QTimer, QSize
)
from qgis.PyQt.QtGui import (
    QColor, QFont, QIcon, QPalette, QTextCursor, QPixmap
)
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QWidget, QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit, QFileDialog,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QScrollArea, QApplication, QMessageBox,
    QListWidget, QListWidgetItem, QSlider, QToolButton, QSizePolicy,
    QAbstractItemView
)

# ── Optional requests ─────────────────────────────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('RoadMatcher')

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'search_radius_m'   : 50.0,
    'metric_crs'        : 'EPSG:32632',
    'geocoder_timeout'  : 10,
    'retry_count'       : 3,
    'retry_backoff'     : 2.0,
    'rate_limit_sec'    : 1.1,
    'cache_path'        : os.path.join(os.path.expanduser('~'), 'road_geocache.json'),
    'user_agent'        : 'RoadNameMatcher/1.0 (QGIS PyQGIS Tool)',
    'nominatim_url'     : 'https://nominatim.openstreetmap.org/reverse',
    'dry_run_count'     : 10,
}

# Road name field candidates (ordered by preference)
ROAD_NAME_FIELDS = [
    'name', 'street', 'road_name', 'rd_name', 'fullname', 'str_name',
    'streetname', 'road', 'st_name', 'highway', 'ref', 'label', 'route',
    'osm_name', 'nam', 'ROAD_NAME', 'NAME', 'STREET', 'FULLNAME'
]

# Abbreviation expansions
ABBREV_MAP = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave?\b': 'avenue',
    r'\bdr\b': 'drive', r'\bct\b': 'court', r'\bblvd\b': 'boulevard',
    r'\bln\b': 'lane',  r'\bpl\b': 'place', r'\bcres?\b': 'crescent',
    r'\bcl\b': 'close', r'\bhwy\b': 'highway', r'\bexp\b': 'expressway',
}

# Estate / court keywords
ESTATE_KEYWORDS = ['estate', 'layout', 'phase', 'residential area', 'housing estate',
                   'scheme', 'gra', 'government reserved']
COURT_KEYWORDS  = ['court', 'courts']
WEAK_KEYWORDS   = ['close', 'crescent', 'gardens', 'grove', 'mews', 'rise', 'walk']

# Confidence thresholds
CONF_INTERSECT  = 95
CONF_NEAREST    = 75
CONF_GEOCODER   = 55
CONF_INFERRED   = 35


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Lowercase, strip, remove punctuation, expand abbreviations."""
    if not name:
        return ''
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', ' ', name)
    for pattern, replacement in ABBREV_MAP.items():
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', name).strip()


def detect_road_name_field(layer: QgsVectorLayer) -> list:
    """Return ordered list of (fieldname, score) for candidate road name fields."""
    fields = [f.name() for f in layer.fields()]
    candidates = []
    for f in fields:
        fl = f.lower()
        score = 0
        for pref in ROAD_NAME_FIELDS:
            if fl == pref.lower():
                score = 100
                break
            elif pref.lower() in fl:
                score = max(score, 60)
            elif fl in pref.lower():
                score = max(score, 40)
        if score > 0:
            candidates.append((f, score))
    candidates.sort(key=lambda x: -x[1])
    return candidates


def detect_keyword(text: str, keywords: list) -> (bool, str):
    """Check if any keyword appears in text. Returns (found, matched_text)."""
    if not text:
        return False, ''
    tl = text.lower()
    for kw in keywords:
        if kw in tl:
            return True, kw
    return False, ''


def get_metric_transform(src_crs: QgsCoordinateReferenceSystem,
                         metric_epsg: str) -> QgsCoordinateTransform:
    dst_crs = QgsCoordinateReferenceSystem(metric_epsg)
    return QgsCoordinateTransform(src_crs, dst_crs,
                                  QgsCoordinateTransformContext())


# ─────────────────────────────────────────────────────────────────────────────
#  GEOCODER
# ─────────────────────────────────────────────────────────────────────────────

class NominatimGeocoder:
    """Rate-limited, cached, retry-aware Nominatim reverse geocoder."""

    def __init__(self, config: dict):
        self.url        = config['nominatim_url']
        self.timeout    = config['geocoder_timeout']
        self.retries    = config['retry_count']
        self.backoff    = config['retry_backoff']
        self.rate_limit = config['rate_limit_sec']
        self.ua         = config['user_agent']
        self.cache_path = config['cache_path']
        self._cache     = self._load_cache()
        self._last_call = 0.0
        self.healthy    = HAS_REQUESTS

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            log.warning(f'Cache save failed: {e}')

    def _wait_rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_call = time.time()

    def reverse(self, lat: float, lon: float) -> dict:
        """
        Reverse geocode a lat/lon.
        Returns dict with keys: road, suburb, city, country, display_name, raw
        """
        key = f'{lat:.6f},{lon:.6f}'
        if key in self._cache:
            return self._cache[key]

        if not HAS_REQUESTS:
            return {'road': '', 'error': 'requests not available'}

        params = {'lat': lat, 'lon': lon, 'format': 'jsonv2',
                  'addressdetails': 1, 'zoom': 18}
        headers = {'User-Agent': self.ua}

        for attempt in range(self.retries):
            try:
                self._wait_rate_limit()
                resp = requests.get(self.url, params=params, headers=headers,
                                    timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                addr = data.get('address', {})
                result = {
                    'road'        : addr.get('road', addr.get('pedestrian',
                                             addr.get('path', ''))),
                    'suburb'      : addr.get('suburb', ''),
                    'neighbourhood': addr.get('neighbourhood', ''),
                    'city'        : addr.get('city', addr.get('town',
                                             addr.get('village', ''))),
                    'state'       : addr.get('state', ''),
                    'country'     : addr.get('country', ''),
                    'display_name': data.get('display_name', ''),
                    'raw'         : data,
                }
                self._cache[key] = result
                self._save_cache()
                return result
            except Exception as e:
                log.warning(f'Geocoder attempt {attempt+1} failed: {e}')
                if attempt < self.retries - 1:
                    time.sleep(self.backoff ** attempt)

        return {'road': '', 'error': 'all retries failed'}

    def ping(self) -> bool:
        """Check if Nominatim is reachable."""
        try:
            r = requests.get(self.url, params={'lat': 51.5, 'lon': -0.1,
                             'format': 'json'}, timeout=5,
                             headers={'User-Agent': self.ua})
            return r.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
#  GOOGLE MAPS GEOCODER  (optional last-resort fallback)
# ─────────────────────────────────────────────────────────────────────────────

class GoogleMapsGeocoder:
    """
    Last-resort reverse geocoder using the Google Maps Geocoding API.
    Only called when:
      (a) the user has enabled it via the UI toggle, AND
      (b) every free source (shapefile + Nominatim) failed to return a road name.
    Results are stored in the same shared JSON cache to avoid re-calling.
    """

    ENDPOINT = 'https://maps.googleapis.com/maps/api/geocode/json'

    def __init__(self, api_key: str, config: dict):
        self.api_key    = api_key.strip()
        self.timeout    = config.get('geocoder_timeout', 10)
        self.retries    = config.get('retry_count', 3)
        self.backoff    = config.get('retry_backoff', 2.0)
        self.rate_limit = config.get('rate_limit_sec', 0.05)  # Google allows 50 rps
        self.cache_path = config.get('cache_path', '')
        self._cache     = self._load_cache()
        self._last_call = 0.0

    def _load_cache(self) -> dict:
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            log.warning(f'Google cache save failed: {e}')

    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_call = time.time()

    def reverse(self, lat: float, lon: float) -> dict:
        """
        Returns dict with keys: road, suburb, city, display_name, raw
        Uses cache key prefixed with 'gmaps:' to distinguish from Nominatim entries.
        """
        key = f'gmaps:{lat:.6f},{lon:.6f}'
        if key in self._cache:
            return self._cache[key]

        if not HAS_REQUESTS or not self.api_key:
            return {'road': '', 'error': 'Google Maps API key missing or requests unavailable'}

        params = {'latlng': f'{lat},{lon}', 'key': self.api_key, 'result_type': 'route'}

        for attempt in range(self.retries):
            try:
                self._wait()
                resp = requests.get(self.ENDPOINT, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                if data.get('status') not in ('OK', 'ZERO_RESULTS'):
                    err_msg = data.get('error_message', data.get('status', 'unknown error'))
                    return {'road': '', 'error': f'Google API: {err_msg}'}

                road      = ''
                suburb    = ''
                city      = ''
                disp_name = ''

                results = data.get('results', [])
                if results:
                    best = results[0]
                    disp_name = best.get('formatted_address', '')
                    for comp in best.get('address_components', []):
                        types = comp.get('types', [])
                        if 'route' in types:
                            road = comp.get('long_name', '')
                        elif 'sublocality' in types or 'neighborhood' in types:
                            suburb = comp.get('long_name', '')
                        elif 'locality' in types:
                            city = comp.get('long_name', '')

                result = {
                    'road'        : road,
                    'suburb'      : suburb,
                    'city'        : city,
                    'display_name': disp_name,
                    'raw'         : data,
                }
                self._cache[key] = result
                self._save_cache()
                return result

            except Exception as e:
                log.warning(f'Google geocoder attempt {attempt+1} failed: {e}')
                if attempt < self.retries - 1:
                    time.sleep(self.backoff ** attempt)

        return {'road': '', 'error': 'Google geocoder: all retries failed'}

    def ping(self) -> bool:
        """Quick validity check — uses a known Lagos coordinate."""
        if not self.api_key or not HAS_REQUESTS:
            return False
        try:
            r = requests.get(self.ENDPOINT,
                             params={'latlng': '6.5244,3.3792', 'key': self.api_key},
                             timeout=6)
            data = r.json()
            return data.get('status') in ('OK', 'ZERO_RESULTS')
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
#  ROAD-SEGMENT STREET-NAME CACHE
#  Prevents duplicate geocoder calls when many points share the same nearest road
# ─────────────────────────────────────────────────────────────────────────────

class RoadSegmentCache:
    """
    Caches the reverse-geocoded street name for each road segment (by fid).
    When two or more points snap to the same road segment fid and that segment
    has no name in the shapefile, only the FIRST point triggers a geocoder call.
    All subsequent points on the same segment reuse the cached name instantly.
    """

    def __init__(self):
        self._store: dict = {}   # fid (int) → result dict from geocoder

    def get(self, fid: int):
        """Return cached result or None."""
        return self._store.get(fid)

    def set(self, fid: int, result: dict):
        self._store[fid] = result

    def size(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE PROCESSING WORKER  (runs in QThread)
# ─────────────────────────────────────────────────────────────────────────────

class MatchWorker(QThread):
    """Background thread that runs the road-matching pipeline."""

    progress      = pyqtSignal(int, int)          # current, total
    log_message   = pyqtSignal(str, str)          # message, level
    feature_done  = pyqtSignal(dict)              # per-feature result dict
    finished      = pyqtSignal(dict)              # summary dict
    error         = pyqtSignal(str)

    def __init__(self, config: dict, params: dict):
        super().__init__()
        self.config   = config
        self.params   = params
        self._cancel  = False

    def cancel(self):
        self._cancel = True

    def emit_log(self, msg: str, level: str = 'info'):
        self.log_message.emit(msg, level)
        getattr(log, level, log.info)(msg)

    # ── Main entry ────────────────────────────────────────────────────────────
    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            self.error.emit(f'Pipeline error: {e}\n{traceback.format_exc()}')

    def _run_pipeline(self):
        cfg    = self.config
        params = self.params

        # ── 1. Load road layer ────────────────────────────────────────────
        self.emit_log('Loading road layer…')
        road_layer = self._load_layer(params['road_path'], 'roads')
        if not road_layer or not road_layer.isValid():
            self.error.emit('Road layer failed to load or is invalid.')
            return

        # ── 2. Load point layer (file OR active QGIS layer) ───────────────
        self.emit_log('Loading point layer…')
        if params.get('use_active_layer') and params.get('active_layer_ref'):
            # Active layer is passed as a QgsVectorLayer reference directly
            pt_layer = params['active_layer_ref']
            if not pt_layer or not pt_layer.isValid():
                self.error.emit('Active layer is invalid.')
                return
            self.emit_log(f'Using active layer: "{pt_layer.name()}"')
        else:
            pt_layer = self._load_point_layer(params)
            if not pt_layer or not pt_layer.isValid():
                self.error.emit('Point layer failed to load or is invalid.')
                return

        # ── 3. Detect road name field ──────────────────────────────────────
        road_field_candidates = detect_road_name_field(road_layer)
        chosen_road_field = params.get('road_field') or (
            road_field_candidates[0][0] if road_field_candidates else None)
        if not chosen_road_field:
            self.error.emit('No road name field detected in road shapefile.')
            return
        self.emit_log(f'Using road name field: "{chosen_road_field}"')

        # ── 4. Coordinate transforms ───────────────────────────────────────
        metric_epsg = cfg['metric_crs']
        road_xfm = get_metric_transform(road_layer.crs(), metric_epsg)
        pt_xfm   = get_metric_transform(pt_layer.crs(),  metric_epsg)

        # ── 5. Build spatial index on reprojected road segments ────────────
        self.emit_log('Building spatial index…')
        road_index  = QgsSpatialIndex()
        road_geoms  = {}    # fid → reprojected geometry
        road_names  = {}    # fid → road name string (may be blank)

        road_field_names = [f.name() for f in road_layer.fields()]
        for feat in road_layer.getFeatures():
            geom = feat.geometry()
            if geom.isNull() or geom.isEmpty():
                continue
            geom_m = QgsGeometry(geom)
            geom_m.transform(road_xfm)
            road_index.insertFeature(feat)
            road_geoms[feat.id()] = geom_m
            road_names[feat.id()] = (
                str(feat[chosen_road_field]).strip()
                if chosen_road_field in road_field_names
                   and feat[chosen_road_field] is not None
                   and str(feat[chosen_road_field]).strip() not in ('', 'NULL', 'None')
                else ''
            )

        self.emit_log(f'Indexed {len(road_geoms)} road segments.')

        # ── 6. Prepare output ─────────────────────────────────────────────
        out_fields  = self._build_output_fields(pt_layer)
        output_path = params['output_path']
        out_format  = params.get('output_format', 'GPKG')
        sink, _     = self._create_sink(output_path, out_format,
                                        out_fields, pt_layer.crs())
        if sink is None:
            self.error.emit(f'Could not create output at: {output_path}')
            return

        # ── 7. Build geocoders ────────────────────────────────────────────
        nominatim       = NominatimGeocoder(cfg)
        google_enabled  = params.get('google_enabled', False)
        google_geocoder = None
        if google_enabled:
            gkey = params.get('google_api_key', '').strip()
            if gkey:
                google_geocoder = GoogleMapsGeocoder(gkey, cfg)
                self.emit_log('Google Maps geocoder enabled as last-resort fallback.')
            else:
                self.emit_log('Google Maps enabled but no API key provided — skipping.', 'warning')

        # Feature flags
        estate_enabled = params.get('estate_enabled', True)

        # ── 8. Road-segment geocode cache (avoids duplicate API calls) ────
        # When two points share the same nearest road segment fid AND that
        # segment has no name in the shapefile, only the first point calls
        # the geocoder.  All subsequent points reuse the cached result.
        seg_cache = RoadSegmentCache()

        # ── 9. Process points ─────────────────────────────────────────────
        summary = {
            'total': 0, 'by_shapefile': 0, 'by_geocoder': 0,
            'by_google': 0, 'estate': 0, 'court': 0,
            'unresolved': 0, 'errors': 0, 'seg_cache_hits': 0
        }
        dry_run   = params.get('dry_run', False)
        dry_limit = cfg['dry_run_count']
        features  = list(pt_layer.getFeatures())
        total     = min(len(features), dry_limit) if dry_run else len(features)

        self.emit_log(f'Processing {total} point{"s" if total!=1 else ""}…')

        for idx, feat in enumerate(features[:total]):
            if self._cancel:
                self.emit_log('Processing cancelled by user.', 'warning')
                break

            self.progress.emit(idx + 1, total)
            summary['total'] += 1

            try:
                result = self._process_feature(
                    feat, pt_layer, pt_xfm, road_index, road_geoms,
                    road_names, road_layer, chosen_road_field,
                    nominatim, google_geocoder, seg_cache,
                    cfg, params, estate_enabled)

                # Tally
                src = result['road_src']
                if src == 'shapefile':
                    summary['by_shapefile'] += 1
                elif src == 'google':
                    summary['by_google'] += 1
                elif src in ('geocoder', 'seg_cache'):
                    summary['by_geocoder'] += 1
                else:
                    summary['unresolved'] += 1

                if result.get('_seg_cache_hit'):
                    summary['seg_cache_hits'] += 1
                if result['is_estate']:
                    summary['estate'] += 1
                if result['is_court']:
                    summary['court'] += 1

                # Write feature
                out_feat = QgsFeature(out_fields)
                for i, v in enumerate(feat.attributes()):
                    if i < len(pt_layer.fields()):
                        out_feat[pt_layer.fields()[i].name()] = v
                for k, v in result.items():
                    if k.startswith('_'):
                        continue   # skip internal flags
                    try:
                        out_feat[k] = v
                    except Exception:
                        pass
                out_feat.setGeometry(feat.geometry())
                sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
                self.feature_done.emit({**result, '_idx': idx})

            except Exception as e:
                summary['errors'] += 1
                self.emit_log(f'Feature {feat.id()} error: {e}', 'warning')

        del sink
        if summary['seg_cache_hits']:
            self.emit_log(
                f'Segment cache saved {summary["seg_cache_hits"]} redundant API calls.')
        summary['output_path'] = output_path
        self.emit_log('✓ Processing complete.', 'info')
        self.finished.emit(summary)

    # ── Feature processor ─────────────────────────────────────────────────────
    def _process_feature(self, feat, pt_layer, pt_xfm, road_index,
                         road_geoms, road_names, road_layer,
                         chosen_road_field, nominatim, google_geocoder,
                         seg_cache, cfg, params, estate_enabled):
        """
        Match one point feature to a road name.

        Matching chain (stops at first success):
          A. Named road in shapefile (intersect or nearest within radius)
          B. Segment cache  — reuse Nominatim result for this road-segment fid
          C. Nominatim reverse geocode
          D. Google Maps reverse geocode  (only if enabled and key present)

        Estate/court detection is applied only when estate_enabled=True.
        """

        result = {
            'road_name'     : '',
            'road_src'      : 'unresolved',
            'road_field'    : chosen_road_field,
            'road_dist_m'   : -1.0,
            'is_estate'     : False,
            'is_court'      : False,
            'estate_txt'    : '',
            'court_txt'     : '',
            'confidence'    : 0,
            'status'        : 'unresolved',
            'notes'         : '',
            '_seg_cache_hit': False,   # internal flag, stripped before writing
        }

        geom = feat.geometry()
        if geom.isNull() or geom.isEmpty():
            result['status'] = 'null_geometry'
            result['notes']  = 'Point has null/empty geometry'
            return result

        # Reproject point to metric CRS
        geom_m = QgsGeometry(geom)
        geom_m.transform(pt_xfm)

        radius  = cfg['search_radius_m']
        pt_rect = geom_m.boundingBox()
        pt_rect.grow(radius)

        # ── A. Shapefile: find nearest road segment within radius ──────────
        candidates = road_index.intersects(pt_rect)
        best_dist  = float('inf')
        best_fid   = None

        for fid in candidates:
            rgeom = road_geoms.get(fid)
            if rgeom is None:
                continue
            dist = geom_m.distance(rgeom)
            if dist < best_dist:
                best_dist = dist
                best_fid  = fid

        if best_fid is not None:
            rname = road_names.get(best_fid, '')
            if rname:
                # Shapefile has a valid name — use it directly
                result['road_name']   = rname
                result['road_src']    = 'shapefile'
                result['road_dist_m'] = round(best_dist, 2)
                if best_dist <= 0.1:
                    result['confidence'] = CONF_INTERSECT
                    result['status']     = 'intersect'
                    result['notes']      = f'Intersecting road: {rname}'
                else:
                    result['confidence'] = CONF_NEAREST
                    result['status']     = 'nearest'
                    result['notes']      = f'Nearest road ({best_dist:.1f} m): {rname}'

        # ── B/C/D. Geocoder fallbacks (only needed when shapefile gave no name) ──
        if not result['road_name']:
            result['road_dist_m'] = round(best_dist, 2) if best_fid is not None else -1.0

            # Build WGS-84 point once (needed by all geocoders)
            geo_pt  = geom.asPoint()
            wgs84   = QgsCoordinateReferenceSystem('EPSG:4326')
            if pt_layer.crs() != wgs84:
                to_wgs = QgsCoordinateTransform(
                    pt_layer.crs(), wgs84, QgsCoordinateTransformContext())
                geo_pt = to_wgs.transform(geo_pt)
            lat, lon = geo_pt.y(), geo_pt.x()

            geo_result = None

            # B. Segment cache: if another point already queried this road fid
            if best_fid is not None:
                cached = seg_cache.get(best_fid)
                if cached is not None:
                    geo_result = cached
                    result['_seg_cache_hit'] = True
                    result['road_src']       = 'seg_cache'

            # C. Nominatim (first real API call for this road segment)
            if geo_result is None:
                geo_result = nominatim.reverse(lat, lon)
                if best_fid is not None:
                    # Store so future points on this segment skip the call
                    seg_cache.set(best_fid, geo_result)

            geo_road = geo_result.get('road', '')

            if geo_road:
                result['road_name']  = geo_road
                result['road_src']   = result['road_src'] if result['_seg_cache_hit'] else 'geocoder'
                result['confidence'] = CONF_GEOCODER
                result['status']     = 'seg_cached' if result['_seg_cache_hit'] else 'geocoded'
                result['notes']      = (
                    f'[seg-cache] {geo_result.get("display_name","")[:80]}'
                    if result['_seg_cache_hit']
                    else f'Nominatim: {geo_result.get("display_name","")[:80]}'
                )

            # D. Google Maps last-resort fallback
            elif google_geocoder is not None:
                g_result = google_geocoder.reverse(lat, lon)
                g_road   = g_result.get('road', '')
                if g_road:
                    result['road_name']  = g_road
                    result['road_src']   = 'google'
                    result['confidence'] = CONF_GEOCODER - 5   # slightly lower than Nominatim
                    result['status']     = 'google_geocoded'
                    result['notes']      = f'Google Maps: {g_result.get("display_name","")[:80]}'
                    # Also cache into seg_cache so parallel points skip Google too
                    if best_fid is not None and not seg_cache.get(best_fid):
                        seg_cache.set(best_fid, g_result)
                else:
                    result['notes'] = (
                        geo_result.get('error', 'Nominatim: no road') +
                        ' | ' + g_result.get('error', 'Google: no road')
                    )
            else:
                result['notes'] = geo_result.get('error', 'No road found by any geocoder')

        # ── Estate / court detection (only when toggle is ON) ──────────────
        if estate_enabled:
            probe_texts = []
            pt_field_names = [f.name() for f in pt_layer.fields()]
            for field_key in ['address_field', 'estate_field', 'court_field']:
                fname = params.get(field_key, '')
                if fname and fname in pt_field_names:
                    val = feat[fname]
                    if val and str(val).strip() not in ('', 'NULL', 'None'):
                        probe_texts.append(str(val))
            probe_texts.append(result['road_name'])
            probe_texts.append(result.get('notes', ''))

            combined  = ' '.join(probe_texts)
            is_estate, etxt = detect_keyword(combined, ESTATE_KEYWORDS)
            is_court,  ctxt = detect_keyword(combined, COURT_KEYWORDS)

            result['is_estate']  = is_estate
            result['is_court']   = is_court
            result['estate_txt'] = etxt
            result['court_txt']  = ctxt

            if result['status'] == 'unresolved' and (is_estate or is_court):
                result['confidence'] = CONF_INFERRED
                result['status']     = 'inferred'

        return result

    # ── Layer loaders ─────────────────────────────────────────────────────────
    def _load_layer(self, path: str, name: str) -> QgsVectorLayer:
        layer = QgsVectorLayer(path, name, 'ogr')
        if not layer.isValid():
            self.emit_log(f'Invalid layer: {path}', 'error')
            return None
        return layer

    def _load_point_layer(self, params: dict) -> QgsVectorLayer:
        path = params['point_path']
        if path.lower().endswith('.csv'):
            lat_f = params.get('lat_field', 'latitude')
            lon_f = params.get('lon_field', 'longitude')
            uri   = (f'file:///{path}?type=csv&xField={lon_f}'
                     f'&yField={lat_f}&crs=EPSG:4326&delimiter=,')
            layer = QgsVectorLayer(uri, 'points', 'delimitedtext')
        else:
            layer = QgsVectorLayer(path, 'points', 'ogr')
        return layer if layer.isValid() else None

    # ── Output helpers ────────────────────────────────────────────────────────
    def _build_output_fields(self, pt_layer: QgsVectorLayer) -> QgsFields:
        fields = QgsFields()
        # Original fields
        for f in pt_layer.fields():
            fields.append(f)
        # Append result fields
        added = {
            'road_name'  : QVariant.String,
            'road_src'   : QVariant.String,
            'road_field' : QVariant.String,
            'road_dist_m': QVariant.Double,
            'is_estate'  : QVariant.Bool,
            'is_court'   : QVariant.Bool,
            'estate_txt' : QVariant.String,
            'court_txt'  : QVariant.String,
            'confidence' : QVariant.Int,
            'status'     : QVariant.String,
            'notes'      : QVariant.String,
        }
        for fname, ftype in added.items():
            fields.append(QgsField(fname, ftype))
        return fields

    def _create_sink(self, path, fmt, fields, crs):
        try:
            if fmt == 'GPKG':
                writer = QgsVectorFileWriter(
                    path, 'UTF-8', fields, QgsWkbTypes.Point, crs,
                    'GPKG')
            else:
                writer = QgsVectorFileWriter(
                    path, 'UTF-8', fields, QgsWkbTypes.Point, crs,
                    'ESRI Shapefile')
            return writer, path
        except Exception as e:
            self.emit_log(f'Sink creation error: {e}', 'error')
            return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (professional dark GIS theme)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    'bg'         : '#1C1F26',
    'bg_card'    : '#252932',
    'bg_input'   : '#1A1D23',
    'accent'     : '#4ECDC4',
    'accent2'    : '#FF6B6B',
    'accent3'    : '#FFE66D',
    'text'       : '#E8EAF0',
    'text_dim'   : '#8B90A0',
    'border'     : '#343842',
    'success'    : '#6BCB77',
    'warning'    : '#FFD166',
    'error'      : '#EF476F',
    'tab_active' : '#4ECDC4',
    'tab_bg'     : '#1C1F26',
}

STYLE_MAIN = f"""
    QDialog {{
        background-color: {PALETTE['bg']};
    }}
    QWidget#rmw_root, QWidget#rmw_root QWidget {{
        background-color: {PALETTE['bg']};
        color: {PALETTE['text']};
        font-family: 'Segoe UI', 'Ubuntu', sans-serif;
        font-size: 13px;
    }}
    QWidget#rmw_root QTabWidget::pane {{
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        background: {PALETTE['bg_card']};
    }}
    QWidget#rmw_root QTabBar::tab {{
        background: {PALETTE['tab_bg']};
        color: {PALETTE['text_dim']};
        padding: 8px 20px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        min-width: 90px;
        font-weight: 500;
    }}
    QWidget#rmw_root QTabBar::tab:selected {{
        background: {PALETTE['bg_card']};
        color: {PALETTE['accent']};
        border-bottom: 2px solid {PALETTE['accent']};
    }}
    QWidget#rmw_root QTabBar::tab:hover:!selected {{
        background: {PALETTE['border']};
        color: {PALETTE['text']};
    }}
    QWidget#rmw_root QGroupBox {{
        border: 1px solid {PALETTE['border']};
        border-radius: 6px;
        margin-top: 14px;
        padding: 12px 10px 10px 10px;
        color: {PALETTE['text_dim']};
        font-size: 11px;
        font-weight: 600;
    }}
    QWidget#rmw_root QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {PALETTE['accent']};
    }}
    QWidget#rmw_root QLineEdit,
    QWidget#rmw_root QComboBox,
    QWidget#rmw_root QSpinBox,
    QWidget#rmw_root QDoubleSpinBox {{
        background: {PALETTE['bg_input']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 5px 8px;
        color: {PALETTE['text']};
        selection-background-color: {PALETTE['accent']};
    }}
    QWidget#rmw_root QLineEdit:focus,
    QWidget#rmw_root QComboBox:focus,
    QWidget#rmw_root QSpinBox:focus,
    QWidget#rmw_root QDoubleSpinBox:focus {{
        border-color: {PALETTE['accent']};
    }}
    QWidget#rmw_root QComboBox::drop-down {{
        border: none;
        padding-right: 8px;
    }}
    QWidget#rmw_root QComboBox QAbstractItemView {{
        background: {PALETTE['bg_input']};
        selection-background-color: {PALETTE['accent']};
        color: {PALETTE['text']};
        border: 1px solid {PALETTE['border']};
    }}
    QWidget#rmw_root QPushButton {{
        background: {PALETTE['bg_card']};
        border: 1px solid {PALETTE['border']};
        border-radius: 5px;
        padding: 7px 18px;
        color: {PALETTE['text']};
        font-weight: 500;
    }}
    QWidget#rmw_root QPushButton:hover {{
        border-color: {PALETTE['accent']};
        color: {PALETTE['accent']};
    }}
    QWidget#rmw_root QPushButton:pressed {{
        background: {PALETTE['accent']};
        color: {PALETTE['bg']};
    }}
    QWidget#rmw_root QPushButton#run_btn {{
        background: {PALETTE['accent']};
        color: {PALETTE['bg']};
        border: none;
        font-weight: 700;
        font-size: 14px;
        padding: 10px 32px;
        border-radius: 6px;
    }}
    QWidget#rmw_root QPushButton#run_btn:hover {{ background: #3dbdb5; }}
    QWidget#rmw_root QPushButton#run_btn:disabled {{
        background: {PALETTE['border']};
        color: {PALETTE['text_dim']};
    }}
    QWidget#rmw_root QPushButton#cancel_btn {{
        background: transparent;
        border: 1px solid {PALETTE['accent2']};
        color: {PALETTE['accent2']};
        padding: 10px 24px;
    }}
    QWidget#rmw_root QPushButton#cancel_btn:hover {{
        background: {PALETTE['accent2']};
        color: white;
    }}
    QWidget#rmw_root QProgressBar {{
        background: {PALETTE['bg_input']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        height: 10px;
        text-align: center;
        color: transparent;
    }}
    QWidget#rmw_root QProgressBar::chunk {{
        background: {PALETTE['accent']};
        border-radius: 4px;
    }}
    QWidget#rmw_root QTextEdit {{
        background: {PALETTE['bg_input']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        color: {PALETTE['text']};
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        padding: 6px;
    }}
    QWidget#rmw_root QLabel {{
        color: {PALETTE['text']};
        background: transparent;
    }}
    QWidget#rmw_root QCheckBox {{
        color: {PALETTE['text']};
        spacing: 6px;
        background: transparent;
    }}
    QWidget#rmw_root QCheckBox::indicator {{
        width: 16px; height: 16px;
        border: 1px solid {PALETTE['border']};
        border-radius: 3px;
        background: {PALETTE['bg_input']};
    }}
    QWidget#rmw_root QCheckBox::indicator:checked {{
        background: {PALETTE['accent']};
        border-color: {PALETTE['accent']};
    }}
    QWidget#rmw_root QTableWidget {{
        background: {PALETTE['bg_input']};
        border: 1px solid {PALETTE['border']};
        gridline-color: {PALETTE['border']};
        color: {PALETTE['text']};
        selection-background-color: {PALETTE['accent']};
    }}
    QWidget#rmw_root QHeaderView::section {{
        background: {PALETTE['bg_card']};
        color: {PALETTE['text_dim']};
        border: none;
        border-bottom: 1px solid {PALETTE['border']};
        padding: 6px 8px;
        font-size: 11px;
        font-weight: 600;
    }}
    QWidget#rmw_root QScrollBar:vertical {{
        background: {PALETTE['bg']};
        width: 8px;
        border-radius: 4px;
    }}
    QWidget#rmw_root QScrollBar::handle:vertical {{
        background: {PALETTE['border']};
        border-radius: 4px;
        min-height: 20px;
    }}
    QWidget#rmw_root QScrollBar::handle:vertical:hover {{
        background: {PALETTE['accent']};
    }}
    QWidget#rmw_root QScrollBar::add-line:vertical,
    QWidget#rmw_root QScrollBar::sub-line:vertical {{ height: 0; }}
    QWidget#rmw_root QListWidget {{
        background: {PALETTE['bg_input']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        color: {PALETTE['text']};
    }}
    QWidget#rmw_root QListWidget::item:selected {{
        background: {PALETTE['accent']};
        color: {PALETTE['bg']};
    }}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER WIDGETS
# ─────────────────────────────────────────────────────────────────────────────

def make_label(text, color=None, bold=False, size=None):
    lbl = QLabel(text)
    style = ''
    if color:
        style += f'color: {color};'
    if bold:
        style += 'font-weight: 700;'
    if size:
        style += f'font-size: {size}px;'
    if style:
        lbl.setStyleSheet(style)
    return lbl


def make_divider():
    f = QFrame()
    f.setObjectName('divider')
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f'background: {PALETTE["border"]}; max-height: 1px;')
    return f


def file_row(parent, label_text, placeholder, browse_cb, btn_text='Browse…'):
    """Returns (row_layout, line_edit)."""
    row  = QHBoxLayout()
    lbl  = make_label(label_text, color=PALETTE['text_dim'])
    lbl.setFixedWidth(140)
    le   = QLineEdit()
    le.setPlaceholderText(placeholder)
    btn  = QPushButton(btn_text)
    btn.setFixedWidth(90)
    btn.clicked.connect(browse_cb(le))
    row.addWidget(lbl)
    row.addWidget(le, 1)
    row.addWidget(btn)
    return row, le


class StatusBadge(QLabel):
    """Coloured round status pill."""
    def __init__(self, text='—', color='gray'):
        super().__init__(f'  {text}  ')
        self.setStyleSheet(
            f'background: {color}; color: white; border-radius: 10px;'
            f'font-size: 11px; font-weight: 700; padding: 2px 8px;')
        self.setAlignment(Qt.AlignCenter)


class SummaryCard(QFrame):
    """A metric card with number + label."""
    def __init__(self, value='0', label='', color=None):
        super().__init__()
        color = color or PALETTE['accent']
        self.setStyleSheet(
            f'background: {PALETTE["bg_card"]}; border: 1px solid {PALETTE["border"]};'
            f'border-radius: 8px; padding: 4px;')
        lay = QVBoxLayout(self)
        lay.setSpacing(2)
        self.val_lbl = QLabel(str(value))
        self.val_lbl.setAlignment(Qt.AlignCenter)
        self.val_lbl.setStyleSheet(
            f'color: {color}; font-size: 28px; font-weight: 700;')
        self.lbl_lbl = QLabel(label)
        self.lbl_lbl.setAlignment(Qt.AlignCenter)
        self.lbl_lbl.setStyleSheet(
            f'color: {PALETTE["text_dim"]}; font-size: 11px; font-weight: 600;'
            f'letter-spacing: 0.5px; text-transform: uppercase;')
        lay.addWidget(self.val_lbl)
        lay.addWidget(self.lbl_lbl)

    def update_value(self, v):
        self.val_lbl.setText(str(v))


# ─────────────────────────────────────────────────────────────────────────────
#  KEYWORD EDITOR  (mini widget for estate/court keywords)
# ─────────────────────────────────────────────────────────────────────────────

class KeywordEditor(QWidget):
    def __init__(self, title, keywords: list):
        super().__init__()
        self._keywords = list(keywords)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        hdr = QHBoxLayout()
        hdr.addWidget(make_label(title, color=PALETTE['accent'], bold=True))
        hdr.addStretch()
        self.list_w = QListWidget()
        self.list_w.setMaximumHeight(100)
        for kw in self._keywords:
            self.list_w.addItem(kw)
        self.inp = QLineEdit()
        self.inp.setPlaceholderText('Add keyword…')
        add_btn = QPushButton('+')
        add_btn.setFixedWidth(32)
        add_btn.clicked.connect(self._add)
        rm_btn  = QPushButton('−')
        rm_btn.setFixedWidth(32)
        rm_btn.clicked.connect(self._remove)
        row = QHBoxLayout()
        row.addWidget(self.inp, 1)
        row.addWidget(add_btn)
        row.addWidget(rm_btn)
        lay.addLayout(hdr)
        lay.addWidget(self.list_w)
        lay.addLayout(row)

    def _add(self):
        kw = self.inp.text().strip().lower()
        if kw and kw not in self._keywords:
            self._keywords.append(kw)
            self.list_w.addItem(kw)
            self.inp.clear()

    def _remove(self):
        row = self.list_w.currentRow()
        if row >= 0:
            kw = self.list_w.takeItem(row).text()
            if kw in self._keywords:
                self._keywords.remove(kw)

    def get_keywords(self) -> list:
        return list(self._keywords)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN GUI DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class RoadMatcherDialog(QDialog):
    """Main GUI dialog for Road Name Matcher."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Road Name Matcher  —  PyQGIS Tool v1.0')
        self.setMinimumSize(860, 680)
        self.resize(960, 740)
        # NOTE: stylesheet is applied inside _build_ui on the named root widget,
        # NOT on the QDialog itself — this avoids QGIS theme overrides.
        self._worker          = None
        self._config          = dict(DEFAULT_CONFIG)
        self._results         = []
        self._active_layer_ref = None

        self._build_ui()
        self._geocoder_ping()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI BUILD
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Root container widget — stylesheet anchored here, not on QDialog ──
        # This prevents QGIS's own QDialog stylesheet from overriding our theme.
        self._root_w = QWidget(self)
        self._root_w.setObjectName('rmw_root')
        self._root_w.setStyleSheet(STYLE_MAIN)

        dialog_lay = QVBoxLayout(self)
        dialog_lay.setContentsMargins(0, 0, 0, 0)
        dialog_lay.setSpacing(0)
        dialog_lay.addWidget(self._root_w)

        root = QVBoxLayout(self._root_w)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────────────
        header = self._make_header()
        root.addWidget(header)

        # ── Tab widget ────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        tab_input    = self._tab_input()
        tab_fields   = self._tab_fields()
        tab_settings = self._tab_settings()
        tab_output   = self._tab_output()
        tab_run      = self._tab_run()

        self.tabs.addTab(tab_input,    '① Inputs')
        self.tabs.addTab(tab_fields,   '② Field Map')
        self.tabs.addTab(tab_settings, '③ Settings')
        self.tabs.addTab(tab_output,   '④ Output')
        self.tabs.addTab(tab_run,      '⑤ Run / Log')

        root.addWidget(self.tabs, 1)

        # ── Status bar ────────────────────────────────────────────────────
        status_bar = self._make_status_bar()
        root.addWidget(status_bar)

    def _make_header(self):
        frame = QFrame()
        frame.setFixedHeight(64)
        frame.setStyleSheet(
            f'background: {PALETTE["bg_card"]}; border-bottom: 2px solid {PALETTE["accent"]};')
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(20, 0, 20, 0)

        title = QLabel('Road Name Matcher')
        title.setStyleSheet(
            f'color: {PALETTE["accent"]}; font-size: 20px; font-weight: 800;'
            f'letter-spacing: 1px;')
        sub = QLabel(' PyQGIS Spatial Tool')
        sub.setStyleSheet(f'color: {PALETTE["text_dim"]}; font-size: 13px;')

        self.geocoder_badge = StatusBadge('Geocoder: checking…', '#555')
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addStretch()
        lay.addWidget(self.geocoder_badge)
        return frame

    def _make_status_bar(self):
        frame = QFrame()
        frame.setFixedHeight(30)
        frame.setStyleSheet(
            f'background: {PALETTE["bg_card"]}; border-top: 1px solid {PALETTE["border"]};')
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 0, 16, 0)
        self.status_lbl = QLabel('Ready.')
        self.status_lbl.setStyleSheet(f'color: {PALETTE["text_dim"]}; font-size: 11px;')
        self.mini_prog  = QProgressBar()
        self.mini_prog.setFixedWidth(140)
        self.mini_prog.setFixedHeight(6)
        self.mini_prog.setTextVisible(False)
        self.mini_prog.setValue(0)
        lay.addWidget(self.status_lbl, 1)
        lay.addWidget(self.mini_prog)
        return frame

    # ── Tab 1: Inputs ─────────────────────────────────────────────────────────
    def _tab_input(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 20, 20, 10)
        lay.setSpacing(14)

        # Road shapefile
        road_grp = QGroupBox('Road Layer')
        rg_lay = QVBoxLayout(road_grp)
        rg_lay.setSpacing(8)

        road_row = QHBoxLayout()
        self.road_path_le = QLineEdit()
        self.road_path_le.setPlaceholderText('/path/to/roads.shp  or  roads.gpkg')
        road_browse = QPushButton('Browse…')
        road_browse.setFixedWidth(90)
        road_browse.clicked.connect(self._browse_road)
        self.road_status = StatusBadge('Not loaded', PALETTE['error'])
        road_row.addWidget(make_label('Road file:', color=PALETTE['text_dim']))
        road_row.addWidget(self.road_path_le, 1)
        road_row.addWidget(road_browse)
        road_row.addWidget(self.road_status)
        rg_lay.addLayout(road_row)

        self.road_crs_lbl = make_label('CRS: —', PALETTE['text_dim'])
        self.road_crs_lbl.setStyleSheet(f'color: {PALETTE["text_dim"]}; font-size: 11px;')
        rg_lay.addWidget(self.road_crs_lbl)
        lay.addWidget(road_grp)

        # ── Point layer ────────────────────────────────────────────────────
        pt_grp = QGroupBox('Point Layer')
        pg_lay = QVBoxLayout(pt_grp)
        pg_lay.setSpacing(8)

        # ── Active layer toggle row ────────────────────────────────────────
        active_row = QHBoxLayout()
        self.use_active_btn = QPushButton('⚡  Use Active QGIS Layer')
        self.use_active_btn.setCheckable(True)
        self.use_active_btn.setChecked(False)
        self.use_active_btn.setToolTip(
            'When ON, the tool reads whichever layer is currently selected\n'
            'in the QGIS Layers panel — no file browse needed.')
        self.use_active_btn.setStyleSheet(
            f'QPushButton {{ background: {PALETTE["bg_card"]}; border: 1px solid {PALETTE["border"]};'
            f'border-radius: 5px; padding: 6px 14px; color: {PALETTE["text"]}; font-weight: 600; }}'
            f'QPushButton:checked {{ background: {PALETTE["accent"]}; color: {PALETTE["bg"]}; '
            f'border: none; }}'
            f'QPushButton:hover {{ border-color: {PALETTE["accent"]}; }}')
        self.use_active_btn.toggled.connect(self._on_active_layer_toggled)

        self.active_layer_lbl = make_label('No active layer detected', PALETTE['text_dim'])
        self.active_layer_lbl.setStyleSheet(
            f'color: {PALETTE["text_dim"]}; font-size: 11px; font-style: italic;')

        active_row.addWidget(self.use_active_btn)
        active_row.addSpacing(10)
        active_row.addWidget(self.active_layer_lbl, 1)
        pg_lay.addLayout(active_row)

        # Divider
        pg_lay.addWidget(make_divider())

        # File path row (disabled when active layer is ON)
        pt_row = QHBoxLayout()
        self.pt_path_le = QLineEdit()
        self.pt_path_le.setPlaceholderText('/path/to/points.shp  or  .gpkg  or  .csv')
        pt_browse = QPushButton('Browse…')
        pt_browse.setFixedWidth(90)
        pt_browse.clicked.connect(self._browse_points)
        self.pt_status = StatusBadge('Not loaded', PALETTE['error'])
        pt_row.addWidget(make_label('Point file:', color=PALETTE['text_dim']))
        pt_row.addWidget(self.pt_path_le, 1)
        pt_row.addWidget(pt_browse)
        pt_row.addWidget(self.pt_status)
        pg_lay.addLayout(pt_row)
        self._pt_file_widgets = [self.pt_path_le, pt_browse]   # toggled on/off

        # CSV lat/lon
        csv_row = QHBoxLayout()
        csv_row.addWidget(make_label('Lat field:', color=PALETTE['text_dim']))
        self.lat_field_le = QLineEdit('latitude')
        self.lat_field_le.setFixedWidth(120)
        csv_row.addWidget(self.lat_field_le)
        csv_row.addSpacing(16)
        csv_row.addWidget(make_label('Lon field:', color=PALETTE['text_dim']))
        self.lon_field_le = QLineEdit('longitude')
        self.lon_field_le.setFixedWidth(120)
        csv_row.addWidget(self.lon_field_le)
        csv_row.addStretch()
        self.csv_hint = make_label('(Only used for CSV input)', PALETTE['text_dim'])
        self.csv_hint.setStyleSheet(
            f'color: {PALETTE["text_dim"]}; font-size: 11px; font-style: italic;')
        csv_row.addWidget(self.csv_hint)
        pg_lay.addLayout(csv_row)

        self.pt_crs_lbl = make_label('CRS: —', PALETTE['text_dim'])
        self.pt_crs_lbl.setStyleSheet(f'color: {PALETTE["text_dim"]}; font-size: 11px;')
        pg_lay.addWidget(self.pt_crs_lbl)
        lay.addWidget(pt_grp)

        # ── Estate / Court geocoding toggle ────────────────────────────────
        estate_grp = QGroupBox('Estate & Court Detection')
        eg_lay = QHBoxLayout(estate_grp)
        eg_lay.setSpacing(12)

        self.estate_toggle_btn = QPushButton('🏘  Estate / Court Detection: ON')
        self.estate_toggle_btn.setCheckable(True)
        self.estate_toggle_btn.setChecked(True)
        self.estate_toggle_btn.setToolTip(
            'When ON: the tool scans text fields and geocoder responses\n'
            'for estate and court keywords and flags matching points.\n'
            'Turn OFF to skip this step entirely (faster processing).')
        self.estate_toggle_btn.setStyleSheet(
            f'QPushButton {{ background: {PALETTE["success"]}; color: {PALETTE["bg"]}; '
            f'border: none; border-radius: 5px; padding: 8px 18px; font-weight: 700; }}'
            f'QPushButton:!checked {{ background: {PALETTE["bg_card"]}; '
            f'color: {PALETTE["text_dim"]}; border: 1px solid {PALETTE["border"]}; }}'
            f'QPushButton:hover {{ opacity: 0.85; }}')
        self.estate_toggle_btn.toggled.connect(self._on_estate_toggled)

        self.estate_status_lbl = make_label(
            'Enabled — points will be scanned for estate/court keywords.',
            PALETTE['success'])
        self.estate_status_lbl.setStyleSheet(
            f'color: {PALETTE["success"]}; font-size: 11px;')
        self.estate_status_lbl.setWordWrap(True)

        eg_lay.addWidget(self.estate_toggle_btn)
        eg_lay.addWidget(self.estate_status_lbl, 1)
        lay.addWidget(estate_grp)

        # Connect path changes
        self.road_path_le.editingFinished.connect(self._on_road_changed)
        self.pt_path_le.editingFinished.connect(self._on_point_changed)

        lay.addStretch()
        return w

    def _on_active_layer_toggled(self, checked: bool):
        """Enable/disable file-browse controls and read the active QGIS layer."""
        for w in self._pt_file_widgets:
            w.setEnabled(not checked)
        self.lat_field_le.setEnabled(not checked)
        self.lon_field_le.setEnabled(not checked)

        if checked:
            try:
                active = iface.activeLayer()  # iface is available inside QGIS
            except Exception:
                active = None

            if active and active.isValid() and active.type() == active.VectorLayer:
                self.active_layer_lbl.setText(
                    f'✓  "{active.name()}"  —  {active.featureCount()} features  '
                    f'[{active.crs().authid()}]')
                self.active_layer_lbl.setStyleSheet(
                    f'color: {PALETTE["success"]}; font-size: 11px;')
                self._active_layer_ref = active
                self._populate_point_fields(active)
                self.pt_crs_lbl.setText(
                    f'CRS: {active.crs().authid()} — {active.crs().description()[:60]}')
            else:
                self.active_layer_lbl.setText(
                    '✗  No valid vector layer is selected in the Layers panel.')
                self.active_layer_lbl.setStyleSheet(
                    f'color: {PALETTE["error"]}; font-size: 11px;')
                self._active_layer_ref = None
                self.use_active_btn.setChecked(False)
        else:
            self.active_layer_lbl.setText('No active layer detected')
            self.active_layer_lbl.setStyleSheet(
                f'color: {PALETTE["text_dim"]}; font-size: 11px; font-style: italic;')
            self._active_layer_ref = None

    def _on_estate_toggled(self, checked: bool):
        if checked:
            self.estate_toggle_btn.setText('🏘  Estate / Court Detection: ON')
            self.estate_status_lbl.setText(
                'Enabled — points will be scanned for estate/court keywords.')
            self.estate_status_lbl.setStyleSheet(f'color: {PALETTE["success"]}; font-size: 11px;')
        else:
            self.estate_toggle_btn.setText('🚫  Estate / Court Detection: OFF')
            self.estate_status_lbl.setText(
                'Disabled — estate/court fields will be left blank (faster).')
            self.estate_status_lbl.setStyleSheet(
                f'color: {PALETTE["text_dim"]}; font-size: 11px;')

    # ── Tab 2: Field Mapping ──────────────────────────────────────────────────
    def _tab_fields(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 20, 20, 10)
        lay.setSpacing(14)

        # Road name field
        road_f_grp = QGroupBox('Road Name Field (from road layer)')
        rf_lay = QVBoxLayout(road_f_grp)
        self.road_field_combo = QComboBox()
        self.road_field_combo.setPlaceholderText('— auto-detected —')
        self.road_field_candidates_tbl = QTableWidget(0, 2)
        self.road_field_candidates_tbl.setHorizontalHeaderLabels(['Field Name', 'Confidence'])
        self.road_field_candidates_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.road_field_candidates_tbl.setMaximumHeight(130)
        self.road_field_candidates_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.road_field_candidates_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.road_field_candidates_tbl.itemClicked.connect(self._on_candidate_click)
        rf_lay.addWidget(make_label('Override road name field:', color=PALETTE['text_dim']))
        rf_lay.addWidget(self.road_field_combo)
        rf_lay.addWidget(make_label('Detected candidates:', color=PALETTE['text_dim']))
        rf_lay.addWidget(self.road_field_candidates_tbl)
        lay.addWidget(road_f_grp)

        # Point optional fields
        pt_f_grp = QGroupBox('Point Layer — Optional Field Mapping')
        pf_lay  = QGridLayout(pt_f_grp)
        pf_lay.setSpacing(8)

        def combo_row(label, row):
            lbl = make_label(label, color=PALETTE['text_dim'])
            cb  = QComboBox()
            cb.addItem('— none —')
            pf_lay.addWidget(lbl, row, 0)
            pf_lay.addWidget(cb,  row, 1)
            return cb

        self.id_combo      = combo_row('Point ID field:',      0)
        self.addr_combo    = combo_row('Address text field:',  1)
        self.estate_combo  = combo_row('Estate name field:',   2)
        self.court_combo   = combo_row('Court name field:',    3)
        lay.addWidget(pt_f_grp)

        # Keyword editors
        kw_lay = QHBoxLayout()
        self.estate_kw_editor = KeywordEditor('Estate Keywords', ESTATE_KEYWORDS)
        self.court_kw_editor  = KeywordEditor('Court Keywords',  COURT_KEYWORDS)
        kw_lay.addWidget(self.estate_kw_editor)
        kw_lay.addSpacing(12)
        kw_lay.addWidget(self.court_kw_editor)
        lay.addLayout(kw_lay)
        lay.addStretch()
        return w

    # ── Tab 3: Settings ───────────────────────────────────────────────────────
    def _tab_settings(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 20, 20, 10)
        lay.setSpacing(14)

        # Spatial settings
        sp_grp = QGroupBox('Spatial Matching')
        sg_lay = QGridLayout(sp_grp)
        sg_lay.setSpacing(10)

        sg_lay.addWidget(make_label('Search radius (m):', PALETTE['text_dim']), 0, 0)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(1, 5000)
        self.radius_spin.setValue(DEFAULT_CONFIG['search_radius_m'])
        self.radius_spin.setSuffix(' m')
        sg_lay.addWidget(self.radius_spin, 0, 1)

        sg_lay.addWidget(make_label('Metric CRS (EPSG):', PALETTE['text_dim']), 1, 0)
        self.crs_le = QLineEdit(DEFAULT_CONFIG['metric_crs'])
        self.crs_le.setPlaceholderText('e.g. EPSG:32632 or EPSG:27700')
        sg_lay.addWidget(self.crs_le, 1, 1)

        crs_hint = make_label(
            'Tip: Use UTM zone for your area (Nigeria → EPSG:32632, UK → EPSG:27700)',
            PALETTE['text_dim'])
        crs_hint.setStyleSheet(f'color: {PALETTE["text_dim"]}; font-size: 11px; font-style: italic;')
        sg_lay.addWidget(crs_hint, 2, 0, 1, 2)
        lay.addWidget(sp_grp)

        # Geocoder settings
        gc_grp = QGroupBox('Reverse Geocoder (Nominatim / OSM)')
        gg_lay = QGridLayout(gc_grp)
        gg_lay.setSpacing(10)

        gg_lay.addWidget(make_label('Timeout (s):', PALETTE['text_dim']), 0, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(2, 60)
        self.timeout_spin.setValue(DEFAULT_CONFIG['geocoder_timeout'])
        self.timeout_spin.setSuffix(' s')
        gg_lay.addWidget(self.timeout_spin, 0, 1)

        gg_lay.addWidget(make_label('Max retries:', PALETTE['text_dim']), 1, 0)
        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(0, 10)
        self.retry_spin.setValue(DEFAULT_CONFIG['retry_count'])
        gg_lay.addWidget(self.retry_spin, 1, 1)

        gg_lay.addWidget(make_label('Rate limit (s):', PALETTE['text_dim']), 2, 0)
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(0.5, 10)
        self.rate_spin.setValue(DEFAULT_CONFIG['rate_limit_sec'])
        self.rate_spin.setSuffix(' s')
        gg_lay.addWidget(self.rate_spin, 2, 1)

        gg_lay.addWidget(make_label('Geocoder URL:', PALETTE['text_dim']), 3, 0)
        self.geocoder_url_le = QLineEdit(DEFAULT_CONFIG['nominatim_url'])
        gg_lay.addWidget(self.geocoder_url_le, 3, 1)

        gg_lay.addWidget(make_label('Cache file:', PALETTE['text_dim']), 4, 0)
        self.cache_le = QLineEdit(DEFAULT_CONFIG['cache_path'])
        cache_browse = QPushButton('…')
        cache_browse.setFixedWidth(32)
        cache_browse.clicked.connect(self._browse_cache)
        cache_row = QHBoxLayout()
        cache_row.addWidget(self.cache_le, 1)
        cache_row.addWidget(cache_browse)
        cache_w = QWidget()
        cache_w.setLayout(cache_row)
        gg_lay.addWidget(cache_w, 4, 1)

        gg_lay.addWidget(make_label('User-Agent:', PALETTE['text_dim']), 5, 0)
        self.ua_le = QLineEdit(DEFAULT_CONFIG['user_agent'])
        gg_lay.addWidget(self.ua_le, 5, 1)

        lay.addWidget(gc_grp)

        # ── Google Maps — last-resort fallback ─────────────────────────────
        gm_grp = QGroupBox('Google Maps API  (Last-Resort Fallback)')
        gm_lay = QGridLayout(gm_grp)
        gm_lay.setSpacing(10)

        # Toggle button
        self.google_toggle_btn = QPushButton('🔴  Google Maps Fallback: OFF')
        self.google_toggle_btn.setCheckable(True)
        self.google_toggle_btn.setChecked(False)
        self.google_toggle_btn.setToolTip(
            'When ON: if both the road shapefile AND Nominatim return no road name,\n'
            'the tool makes one final call to the Google Maps Geocoding API.\n'
            'Requires a valid API key. Google charges per request beyond free tier.')
        self.google_toggle_btn.setStyleSheet(
            f'QPushButton {{ background: {PALETTE["bg_card"]}; border: 1px solid {PALETTE["border"]};'
            f'border-radius: 5px; padding: 8px 14px; color: {PALETTE["text_dim"]}; font-weight: 700; }}'
            f'QPushButton:checked {{ background: #34A853; color: white; border: none; }}'
            f'QPushButton:hover {{ border-color: {PALETTE["accent"]}; }}')
        self.google_toggle_btn.toggled.connect(self._on_google_toggled)
        gm_lay.addWidget(self.google_toggle_btn, 0, 0, 1, 2)

        # API key input
        gm_lay.addWidget(make_label('API Key:', PALETTE['text_dim']), 1, 0)
        self.google_key_le = QLineEdit()
        self.google_key_le.setPlaceholderText('AIza…  (paste your Google Maps Geocoding API key)')
        self.google_key_le.setEchoMode(QLineEdit.Password)   # hide key by default
        gm_key_row = QHBoxLayout()
        gm_key_row.addWidget(self.google_key_le, 1)

        # Show/hide key toggle
        self.google_show_key_btn = QPushButton('👁')
        self.google_show_key_btn.setFixedWidth(36)
        self.google_show_key_btn.setToolTip('Show / hide API key')
        self.google_show_key_btn.setCheckable(True)
        self.google_show_key_btn.toggled.connect(
            lambda on: self.google_key_le.setEchoMode(
                QLineEdit.Normal if on else QLineEdit.Password))
        gm_key_row.addWidget(self.google_show_key_btn)

        # Test key button
        self.google_test_btn = QPushButton('Test Key')
        self.google_test_btn.setFixedWidth(80)
        self.google_test_btn.setEnabled(False)
        self.google_test_btn.clicked.connect(self._test_google_key)
        gm_key_row.addWidget(self.google_test_btn)

        gm_key_w = QWidget()
        gm_key_w.setLayout(gm_key_row)
        gm_lay.addWidget(gm_key_w, 1, 1)

        # Status label
        self.google_status_lbl = make_label(
            'Disabled. Enable to use Google Maps as a last-resort fallback.',
            PALETTE['text_dim'])
        self.google_status_lbl.setStyleSheet(
            f'color: {PALETTE["text_dim"]}; font-size: 11px;')
        self.google_status_lbl.setWordWrap(True)
        gm_lay.addWidget(self.google_status_lbl, 2, 0, 1, 2)

        # Disable key entry until toggle is on
        self.google_key_le.setEnabled(False)
        self.google_show_key_btn.setEnabled(False)

        disclaimer = make_label(
            '⚠  Google Maps API may incur costs beyond the free monthly quota. '
            'Only called after shapefile + Nominatim both fail.',
            PALETTE['warning'])
        disclaimer.setStyleSheet(
            f'color: {PALETTE["warning"]}; font-size: 10px;')
        disclaimer.setWordWrap(True)
        gm_lay.addWidget(disclaimer, 3, 0, 1, 2)
        lay.addWidget(gm_grp)
        adv_grp = QGroupBox('Advanced')
        av_lay  = QGridLayout(adv_grp)
        av_lay.setSpacing(10)

        self.dry_run_chk = QCheckBox('Dry run (process first N points only)')
        self.dry_run_chk.setChecked(False)
        av_lay.addWidget(self.dry_run_chk, 0, 0, 1, 2)

        av_lay.addWidget(make_label('Dry run count:', PALETTE['text_dim']), 1, 0)
        self.dry_run_spin = QSpinBox()
        self.dry_run_spin.setRange(1, 500)
        self.dry_run_spin.setValue(DEFAULT_CONFIG['dry_run_count'])
        av_lay.addWidget(self.dry_run_spin, 1, 1)

        lay.addWidget(adv_grp)
        lay.addStretch()
        return w

    # ── Tab 4: Output ─────────────────────────────────────────────────────────
    def _tab_output(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 20, 20, 10)
        lay.setSpacing(14)

        out_grp = QGroupBox('Output File')
        og_lay  = QGridLayout(out_grp)
        og_lay.setSpacing(10)

        og_lay.addWidget(make_label('Output path:', PALETTE['text_dim']), 0, 0)
        self.out_path_le = QLineEdit()
        self.out_path_le.setPlaceholderText('/path/to/output_matched.gpkg')
        out_browse = QPushButton('Browse…')
        out_browse.setFixedWidth(90)
        out_browse.clicked.connect(self._browse_output)
        orow = QHBoxLayout()
        orow.addWidget(self.out_path_le, 1)
        orow.addWidget(out_browse)
        og_lay.addLayout(orow, 0, 1)

        og_lay.addWidget(make_label('Format:', PALETTE['text_dim']), 1, 0)
        self.out_fmt_combo = QComboBox()
        self.out_fmt_combo.addItems(['GeoPackage (.gpkg)', 'Shapefile (.shp)'])
        og_lay.addWidget(self.out_fmt_combo, 1, 1)
        lay.addWidget(out_grp)

        # Output fields preview
        fld_grp = QGroupBox('Output Fields Preview')
        fl_lay  = QVBoxLayout(fld_grp)
        tbl = QTableWidget(11, 3)
        tbl.setHorizontalHeaderLabels(['Field Name', 'Type', 'Description'])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setSelectionMode(QAbstractItemView.NoSelection)
        rows = [
            ('road_name',   'String',  'Matched road/street name'),
            ('road_src',    'String',  'Source: shapefile / geocoder / unresolved'),
            ('road_field',  'String',  'Road layer field used'),
            ('road_dist_m', 'Double',  'Distance to matched road (metres)'),
            ('is_estate',   'Boolean', 'Point is in/near an estate'),
            ('is_court',    'Boolean', 'Point is in/near a court'),
            ('estate_txt',  'String',  'Matched estate keyword'),
            ('court_txt',   'String',  'Matched court keyword'),
            ('confidence',  'Integer', 'Confidence score 0–100'),
            ('status',      'String',  'intersect / nearest / geocoded / unresolved'),
            ('notes',       'String',  'Processing notes / details'),
        ]
        for r, (name, typ, desc) in enumerate(rows):
            tbl.setItem(r, 0, QTableWidgetItem(name))
            tbl.setItem(r, 1, QTableWidgetItem(typ))
            tbl.setItem(r, 2, QTableWidgetItem(desc))
            tbl.item(r, 0).setForeground(QColor(PALETTE['accent']))
        fl_lay.addWidget(tbl)
        lay.addWidget(fld_grp)
        lay.addStretch()
        return w

    # ── Tab 5: Run / Log ──────────────────────────────────────────────────────
    def _tab_run(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 20, 20, 16)
        lay.setSpacing(12)

        # Summary cards
        card_row = QHBoxLayout()
        self.card_total   = SummaryCard('—', 'Total',      PALETTE['text'])
        self.card_shp     = SummaryCard('—', 'Shapefile',  PALETTE['success'])
        self.card_geo     = SummaryCard('—', 'Geocoded',   PALETTE['accent'])
        self.card_estate  = SummaryCard('—', 'Estates',    PALETTE['warning'])
        self.card_court   = SummaryCard('—', 'Courts',     PALETTE['accent3'])
        self.card_unresl  = SummaryCard('—', 'Unresolved', PALETTE['error'])
        for card in [self.card_total, self.card_shp, self.card_geo,
                     self.card_estate, self.card_court, self.card_unresl]:
            card_row.addWidget(card)
        lay.addLayout(card_row)

        # Progress
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_lbl = make_label('0 / 0', PALETTE['text_dim'])
        self.progress_lbl.setFixedWidth(80)
        prog_row.addWidget(self.progress_bar, 1)
        prog_row.addWidget(self.progress_lbl)
        lay.addLayout(prog_row)

        # Log area
        lay.addWidget(make_label('Processing Log', PALETTE['text_dim']))
        self.log_te = QTextEdit()
        self.log_te.setReadOnly(True)
        self.log_te.setMinimumHeight(240)
        lay.addWidget(self.log_te, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.dry_badge = StatusBadge('DRY RUN', '#cc8800')
        self.dry_badge.setVisible(False)
        self.run_btn    = QPushButton('▶  RUN MATCHER')
        self.run_btn.setObjectName('run_btn')
        self.cancel_btn = QPushButton('✕  Cancel')
        self.cancel_btn.setObjectName('cancel_btn')
        self.cancel_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.dry_badge)
        btn_row.addSpacing(10)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.run_btn)
        lay.addLayout(btn_row)
        return w

    # ─────────────────────────────────────────────────────────────────────────
    #  BROWSE CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────
    def _browse_road(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select Road Layer', '',
            'Vector Files (*.shp *.gpkg *.geojson);;All files (*)')
        if path:
            self.road_path_le.setText(path)
            self._on_road_changed()

    def _browse_points(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select Point Layer', '',
            'Vector / CSV Files (*.shp *.gpkg *.csv *.geojson);;All files (*)')
        if path:
            self.pt_path_le.setText(path)
            self._on_point_changed()

    def _browse_output(self):
        fmt  = 'gpkg' if self.out_fmt_combo.currentIndex() == 0 else 'shp'
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Output Layer', '',
            f'{"GeoPackage" if fmt=="gpkg" else "Shapefile"} (*.{fmt})')
        if path:
            if not path.endswith(f'.{fmt}'):
                path += f'.{fmt}'
            self.out_path_le.setText(path)

    def _browse_cache(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Geocoder Cache File', '', 'JSON (*.json)')
        if path:
            self.cache_le.setText(path)

    # ─────────────────────────────────────────────────────────────────────────
    #  LAYER VALIDATION CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────
    def _on_road_changed(self):
        path = self.road_path_le.text().strip()
        if not path or not os.path.exists(path):
            self.road_status.setText('  Not found  ')
            self.road_status.setStyleSheet(
                f'background: {PALETTE["error"]}; color: white; border-radius: 10px;'
                f'font-size: 11px; font-weight: 700; padding: 2px 8px;')
            return
        layer = QgsVectorLayer(path, 'road_test', 'ogr')
        if layer.isValid():
            self.road_status.setText('  ✓ Valid  ')
            self.road_status.setStyleSheet(
                f'background: {PALETTE["success"]}; color: {PALETTE["bg"]}; border-radius: 10px;'
                f'font-size: 11px; font-weight: 700; padding: 2px 8px;')
            self.road_crs_lbl.setText(f'CRS: {layer.crs().authid()} — {layer.crs().description()[:60]}')
            self._populate_road_fields(layer)
        else:
            self.road_status.setText('  ✗ Invalid  ')
            self.road_status.setStyleSheet(
                f'background: {PALETTE["error"]}; color: white; border-radius: 10px;'
                f'font-size: 11px; font-weight: 700; padding: 2px 8px;')

    def _on_point_changed(self):
        path = self.pt_path_le.text().strip()
        if not path or not os.path.exists(path):
            self.pt_status.setText('  Not found  ')
            self.pt_status.setStyleSheet(
                f'background: {PALETTE["error"]}; color: white; border-radius: 10px;'
                f'font-size: 11px; font-weight: 700; padding: 2px 8px;')
            return
        if path.lower().endswith('.csv'):
            layer = QgsVectorLayer(
                f'file:///{path}?type=csv&xField={self.lon_field_le.text()}'
                f'&yField={self.lat_field_le.text()}&crs=EPSG:4326&delimiter=,',
                'pt_test', 'delimitedtext')
        else:
            layer = QgsVectorLayer(path, 'pt_test', 'ogr')

        if layer.isValid():
            self.pt_status.setText('  ✓ Valid  ')
            self.pt_status.setStyleSheet(
                f'background: {PALETTE["success"]}; color: {PALETTE["bg"]}; border-radius: 10px;'
                f'font-size: 11px; font-weight: 700; padding: 2px 8px;')
            self.pt_crs_lbl.setText(f'CRS: {layer.crs().authid()} — {layer.crs().description()[:60]}')
            self._populate_point_fields(layer)
        else:
            self.pt_status.setText('  ✗ Invalid  ')
            self.pt_status.setStyleSheet(
                f'background: {PALETTE["error"]}; color: white; border-radius: 10px;'
                f'font-size: 11px; font-weight: 700; padding: 2px 8px;')

    def _populate_road_fields(self, layer: QgsVectorLayer):
        self.road_field_combo.clear()
        candidates = detect_road_name_field(layer)
        all_fields = [f.name() for f in layer.fields()]
        for f in all_fields:
            self.road_field_combo.addItem(f)
        if candidates:
            self.road_field_combo.setCurrentText(candidates[0][0])

        # Populate candidate table
        tbl = self.road_field_candidates_tbl
        tbl.setRowCount(len(candidates))
        for r, (fname, score) in enumerate(candidates):
            tbl.setItem(r, 0, QTableWidgetItem(fname))
            score_item = QTableWidgetItem(f'{score}%')
            color = (PALETTE['success'] if score >= 80
                     else PALETTE['warning'] if score >= 50
                     else PALETTE['error'])
            score_item.setForeground(QColor(color))
            tbl.setItem(r, 1, score_item)

    def _populate_point_fields(self, layer: QgsVectorLayer):
        fields = ['— none —'] + [f.name() for f in layer.fields()]
        for cb in [self.id_combo, self.addr_combo, self.estate_combo, self.court_combo]:
            cb.clear()
            cb.addItems(fields)

    def _on_candidate_click(self, item):
        row = item.row()
        name = self.road_field_candidates_tbl.item(row, 0).text()
        self.road_field_combo.setCurrentText(name)

    # ─────────────────────────────────────────────────────────────────────────
    #  GOOGLE MAPS TOGGLE & TEST
    # ─────────────────────────────────────────────────────────────────────────
    def _on_google_toggled(self, checked: bool):
        self.google_key_le.setEnabled(checked)
        self.google_show_key_btn.setEnabled(checked)
        self.google_test_btn.setEnabled(checked)
        if checked:
            self.google_toggle_btn.setText('🟢  Google Maps Fallback: ON')
            self.google_status_lbl.setText(
                'Enabled. Google Maps will be called ONLY when shapefile and '
                'Nominatim both return no road name.')
            self.google_status_lbl.setStyleSheet(f'color: {PALETTE["success"]}; font-size: 11px;')
        else:
            self.google_toggle_btn.setText('🔴  Google Maps Fallback: OFF')
            self.google_status_lbl.setText(
                'Disabled. Enable to use Google Maps as a last-resort fallback.')
            self.google_status_lbl.setStyleSheet(
                f'color: {PALETTE["text_dim"]}; font-size: 11px;')

    def _test_google_key(self):
        key = self.google_key_le.text().strip()
        if not key:
            QMessageBox.warning(self, 'No Key', 'Please enter a Google Maps API key first.')
            return
        self.google_status_lbl.setText('Testing key…')

        def _do_test():
            gc = GoogleMapsGeocoder(key, self._config)
            ok = gc.ping()
            QTimer.singleShot(0, lambda: self._on_google_test_result(ok))

        threading.Thread(target=_do_test, daemon=True).start()

    def _on_google_test_result(self, ok: bool):
        if ok:
            self.google_status_lbl.setText('✓  API key is valid and reachable.')
            self.google_status_lbl.setStyleSheet(f'color: {PALETTE["success"]}; font-size: 11px;')
        else:
            self.google_status_lbl.setText(
                '✗  Key test failed. Check the key, billing, and Geocoding API activation.')
            self.google_status_lbl.setStyleSheet(f'color: {PALETTE["error"]}; font-size: 11px;')

    # ─────────────────────────────────────────────────────────────────────────
    #  GEOCODER PING
    # ─────────────────────────────────────────────────────────────────────────
    def _geocoder_ping(self):
        if not HAS_REQUESTS:
            self.geocoder_badge.setText('  Geocoder: no requests  ')
            self.geocoder_badge.setStyleSheet(
                f'background: {PALETTE["warning"]}; color: {PALETTE["bg"]}; border-radius: 10px;'
                f'font-size: 11px; font-weight: 700; padding: 2px 8px;')
            return

        def _ping():
            try:
                resp = requests.get(
                    DEFAULT_CONFIG['nominatim_url'],
                    params={'lat': 6.5, 'lon': 3.4, 'format': 'json'},
                    timeout=6, headers={'User-Agent': DEFAULT_CONFIG['user_agent']})
                ok = resp.status_code == 200
            except Exception:
                ok = False
            QTimer.singleShot(0, lambda: self._update_geocoder_badge(ok))

        threading.Thread(target=_ping, daemon=True).start()

    def _update_geocoder_badge(self, ok: bool):
        if ok:
            self.geocoder_badge.setText('  Geocoder: Online ✓  ')
            self.geocoder_badge.setStyleSheet(
                f'background: {PALETTE["success"]}; color: {PALETTE["bg"]}; border-radius: 10px;'
                f'font-size: 11px; font-weight: 700; padding: 2px 8px;')
        else:
            self.geocoder_badge.setText('  Geocoder: Offline ✗  ')
            self.geocoder_badge.setStyleSheet(
                f'background: {PALETTE["error"]}; color: white; border-radius: 10px;'
                f'font-size: 11px; font-weight: 700; padding: 2px 8px;')

    # ─────────────────────────────────────────────────────────────────────────
    #  RUN / CANCEL
    # ─────────────────────────────────────────────────────────────────────────
    def _collect_config(self) -> dict:
        cfg = dict(self._config)
        cfg['search_radius_m']   = self.radius_spin.value()
        cfg['metric_crs']        = self.crs_le.text().strip()
        cfg['geocoder_timeout']  = self.timeout_spin.value()
        cfg['retry_count']       = self.retry_spin.value()
        cfg['rate_limit_sec']    = self.rate_spin.value()
        cfg['cache_path']        = self.cache_le.text().strip()
        cfg['user_agent']        = self.ua_le.text().strip()
        cfg['nominatim_url']     = self.geocoder_url_le.text().strip()
        cfg['dry_run_count']     = self.dry_run_spin.value()
        return cfg

    def _collect_params(self) -> dict:
        fmt_idx          = self.out_fmt_combo.currentIndex()
        use_active       = getattr(self, 'use_active_btn', None) and self.use_active_btn.isChecked()
        active_layer_ref = getattr(self, '_active_layer_ref', None) if use_active else None
        return {
            'road_path'       : self.road_path_le.text().strip(),
            'point_path'      : self.pt_path_le.text().strip() if not use_active else '',
            'lat_field'       : self.lat_field_le.text().strip(),
            'lon_field'       : self.lon_field_le.text().strip(),
            'road_field'      : self.road_field_combo.currentText(),
            'id_field'        : self.id_combo.currentText() if self.id_combo.currentIndex() > 0 else '',
            'address_field'   : self.addr_combo.currentText() if self.addr_combo.currentIndex() > 0 else '',
            'estate_field'    : self.estate_combo.currentText() if self.estate_combo.currentIndex() > 0 else '',
            'court_field'     : self.court_combo.currentText() if self.court_combo.currentIndex() > 0 else '',
            'output_path'     : self.out_path_le.text().strip(),
            'output_format'   : 'GPKG' if fmt_idx == 0 else 'SHP',
            'dry_run'         : self.dry_run_chk.isChecked(),
            # ── New flags ──────────────────────────────────────────────────
            'use_active_layer': use_active,
            'active_layer_ref': active_layer_ref,
            'estate_enabled'  : self.estate_toggle_btn.isChecked(),
            'google_enabled'  : self.google_toggle_btn.isChecked(),
            'google_api_key'  : self.google_key_le.text().strip(),
        }

    def _validate(self, params: dict) -> str:
        if not params['road_path'] or not os.path.exists(params['road_path']):
            return 'Road layer path is missing or does not exist.'
        if params.get('use_active_layer'):
            if not params.get('active_layer_ref'):
                return ('Active Layer mode is ON but no valid layer is selected.\n'
                        'Select a vector layer in the Layers panel and try again.')
        else:
            if not params['point_path'] or not os.path.exists(params['point_path']):
                return 'Point layer path is missing or does not exist.'
        if not params['output_path']:
            return 'Please specify an output path.'
        if params.get('google_enabled') and not params.get('google_api_key'):
            return ('Google Maps fallback is enabled but no API key was entered.\n'
                    'Either paste a key in Settings → Google Maps API, or disable the toggle.')
        return ''

    def _on_run(self):
        cfg    = self._collect_config()
        params = self._collect_params()

        err = self._validate(params)
        if err:
            QMessageBox.warning(self, 'Validation Error', err)
            return

        # Update global keywords from editors
        global ESTATE_KEYWORDS, COURT_KEYWORDS
        ESTATE_KEYWORDS = self.estate_kw_editor.get_keywords()
        COURT_KEYWORDS  = self.court_kw_editor.get_keywords()

        # Reset UI
        self.log_te.clear()
        self.progress_bar.setValue(0)
        self.progress_lbl.setText('0 / 0')
        for c in [self.card_total, self.card_shp, self.card_geo,
                  self.card_estate, self.card_court, self.card_unresl]:
            c.update_value('—')
        self._results.clear()

        self.dry_badge.setVisible(params['dry_run'])
        self.tabs.setCurrentIndex(4)   # switch to Run tab

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        # Start worker
        self._worker = MatchWorker(cfg, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._on_log)
        self._worker.feature_done.connect(self._on_feature)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self._append_log('⚠ Cancellation requested…', PALETTE['warning'])

    # ─────────────────────────────────────────────────────────────────────────
    #  WORKER SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    def _on_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.mini_prog.setMaximum(total)
        self.mini_prog.setValue(current)
        self.progress_lbl.setText(f'{current} / {total}')
        self.status_lbl.setText(f'Processing {current} of {total}…')

    def _on_log(self, msg: str, level: str):
        colors = {'info': PALETTE['text'], 'warning': PALETTE['warning'],
                  'error': PALETTE['error'], 'debug': PALETTE['text_dim']}
        self._append_log(msg, colors.get(level, PALETTE['text']))

    def _append_log(self, msg: str, color: str = None):
        color = color or PALETTE['text']
        ts    = datetime.now().strftime('%H:%M:%S')
        html  = (f'<span style="color:{PALETTE["text_dim"]};font-size:10px;">[{ts}]</span> '
                 f'<span style="color:{color};">{msg}</span><br>')
        self.log_te.moveCursor(QTextCursor.End)
        self.log_te.insertHtml(html)
        self.log_te.moveCursor(QTextCursor.End)

    def _on_feature(self, result: dict):
        self._results.append(result)

    def _on_finished(self, summary: dict):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_lbl.setText('✓ Complete.')

        self.card_total.update_value(summary.get('total', 0))
        self.card_shp.update_value(summary.get('by_shapefile', 0))
        self.card_geo.update_value(summary.get('by_geocoder', 0))
        self.card_estate.update_value(summary.get('estate', 0))
        self.card_court.update_value(summary.get('court', 0))
        self.card_unresl.update_value(summary.get('unresolved', 0))

        out = summary.get('output_path', '')
        self._append_log('─' * 60, PALETTE['border'])
        self._append_log('✓ PROCESSING COMPLETE', PALETTE['success'])
        self._append_log(f'  Total processed      : {summary.get("total", 0)}', PALETTE['text'])
        self._append_log(f'  Matched (shapefile)  : {summary.get("by_shapefile", 0)}', PALETTE['success'])
        self._append_log(f'  Matched (Nominatim)  : {summary.get("by_geocoder", 0)}', PALETTE['accent'])
        self._append_log(f'  Matched (Google Maps): {summary.get("by_google", 0)}', '#34A853')
        self._append_log(f'  Seg-cache hits saved : {summary.get("seg_cache_hits", 0)}', PALETTE['accent3'])
        self._append_log(f'  Estates detected     : {summary.get("estate", 0)}', PALETTE['warning'])
        self._append_log(f'  Courts detected      : {summary.get("court", 0)}', PALETTE['accent3'])
        self._append_log(f'  Unresolved           : {summary.get("unresolved", 0)}', PALETTE['error'])
        if summary.get('errors', 0):
            self._append_log(f'  Errors (skipped)     : {summary["errors"]}', PALETTE['error'])
        self._append_log(f'  Output saved to: {out}', PALETTE['accent'])

        QMessageBox.information(self, 'Complete',
            f'Processing complete!\n\n'
            f'Total: {summary.get("total", 0)}\n'
            f'Shapefile matches: {summary.get("by_shapefile", 0)}\n'
            f'Geocoder matches: {summary.get("by_geocoder", 0)}\n'
            f'Unresolved: {summary.get("unresolved", 0)}\n\n'
            f'Output: {out}')

    def _on_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_lbl.setText('✗ Error.')
        self._append_log(f'ERROR: {msg}', PALETTE['error'])
        QMessageBox.critical(self, 'Processing Error', msg)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


# ── Global reference — prevents garbage collection when run from console ──────
_ROAD_MATCHER_DLG = None


def main():
    """
    Launch the Road Name Matcher GUI.

    HOW TO RUN
    ----------
    Option A  — QGIS Python Console (recommended):
        exec(open(r'/full/path/to/road_name_matcher_qgis.py').read())

    Option B  — QGIS Python Console Editor tab:
        Paste the whole script, then click the green Run button.

    Option C  — standalone (outside QGIS, for testing layout only):
        python road_name_matcher_qgis.py

    IMPORTANT: The global variable _ROAD_MATCHER_DLG keeps the dialog alive.
    Do not assign the return value to a local variable inside exec() — Python
    will garbage-collect it before it ever renders.  The global handles this.
    """
    global _ROAD_MATCHER_DLG

    # ── Determine execution context ───────────────────────────────────────────
    # Inside QGIS the QApplication event loop is already running.
    # We must NOT call app.exec_() in that case — it would block QGIS entirely.
    # Instead we call dlg.exec_() which opens a modal sub-loop, or just show()
    # when the caller is the QGIS console (non-blocking is better UX there).
    _inside_qgis = QgsApplication.instance() is not None

    if not _inside_qgis:
        # Standalone: create a minimal QApplication
        import sys
        _app = QApplication.instance() or QApplication(sys.argv)
    else:
        _app = None  # already running inside QGIS

    # ── Create dialog and store in global so GC cannot destroy it ─────────────
    _ROAD_MATCHER_DLG = RoadMatcherDialog()

    if _inside_qgis:
        # Non-blocking show — the QGIS event loop drives painting & events.
        # raise_() + activateWindow() bring it to the front reliably.
        _ROAD_MATCHER_DLG.setWindowFlags(
            _ROAD_MATCHER_DLG.windowFlags() | Qt.Window
        )
        _ROAD_MATCHER_DLG.show()
        _ROAD_MATCHER_DLG.raise_()
        _ROAD_MATCHER_DLG.activateWindow()
    else:
        # Standalone: blocking exec loop
        _ROAD_MATCHER_DLG.show()
        _app.exec_()

    return _ROAD_MATCHER_DLG


# ── Auto-launch when exec()'d from the QGIS Python Console ───────────────────
# exec() runs the file in the console's namespace, so __name__ == '__main__'
# is NOT set.  We use a try/except on iface to detect the QGIS console and
# launch automatically either way.
if __name__ == '__main__':
    # Launched as a normal Python script (outside QGIS)
    main()
else:
    # exec()'d from QGIS Python Console — launch immediately
    main()

