# =============================================================================
# PyQGIS: FTTx Road Topology Fixer — OSM Edition
# =============================================================================
# Targets the specific FTTx design error:
#   "FAT(s) are on isolated roads (roads with no junction connections)"
#
# Run in QGIS Python Console:
#   exec(open('/path/to/fttx_road_topology_fixer.py').read())
#
# Pipeline:
#   1. Fix geometries
#   2. Remove duplicates / slivers
#   3. Snap (tight tolerance for near-misses)
#   4. Extend dangling endpoints → nearest line (within extend_max)
#   5. ISOLATED SEGMENT PASS — detect fully-isolated segments and
#      bridge them to the nearest reachable road (within bridge_max)
#   6. Split at all intersections (creates topological nodes)
#   7. Topology report — list any remaining isolated segments
#   8. Optionally save to file
# =============================================================================

import traceback
import math

from qgis.core import (
    QgsProject, QgsGeometry, QgsPointXY, QgsWkbTypes,
    QgsFeature, QgsSpatialIndex, QgsVectorLayer,
    QgsVectorFileWriter, QgsMessageLog, Qgis,
    QgsCoordinateReferenceSystem
)
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QLineEdit, QPushButton,
    QFileDialog, QComboBox, QCheckBox, QTextEdit,
    QGroupBox, QProgressBar, QApplication, QMessageBox,
    QSpinBox
)
from qgis.PyQt.QtCore import Qt, QObject, pyqtSignal
from qgis.PyQt.QtGui import QFont
import processing


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(msg, "FTTxTopologyFixer", level=level)


def nearest_point_on_line(line_geom: QgsGeometry, point_xy: QgsPointXY):
    """Returns (QgsPointXY nearest, float distance). Handles degenerate cases."""
    try:
        result = line_geom.closestSegmentWithContext(point_xy)
        nearest_pt = result[1]
        dist = math.sqrt(abs(result[0]))
        return nearest_pt, dist
    except Exception:
        return None, float('inf')


def pt_dist(a: QgsPointXY, b: QgsPointXY) -> float:
    return math.hypot(a.x() - b.x(), a.y() - b.y())


def pt_eq(a: QgsPointXY, b: QgsPointXY, tol: float = 1e-6) -> bool:
    return pt_dist(a, b) < tol


def is_line_layer(layer) -> bool:
    from qgis.core import QgsMapLayer, QgsVectorLayer
    if layer is None:
        return False
    if not isinstance(layer, QgsVectorLayer):
        return False
    return layer.geometryType() == QgsWkbTypes.LineGeometry


def get_endpoints(feature: QgsFeature):
    """Returns list of (part_idx_or_None, which_0_or_neg1, QgsPointXY)."""
    geom = feature.geometry()
    if geom is None or geom.isEmpty():
        return []
    results = []
    if geom.isMultipart():
        for pi, part in enumerate(geom.asMultiPolyline()):
            if len(part) >= 2:
                results.append((pi, 0, part[0]))
                results.append((pi, -1, part[-1]))
    else:
        part = geom.asPolyline()
        if len(part) >= 2:
            results.append((None, 0, part[0]))
            results.append((None, -1, part[-1]))
    return results


def move_endpoint(feature: QgsFeature, part_idx, which: int, new_pt: QgsPointXY):
    """Returns a new QgsGeometry with the given endpoint moved to new_pt."""
    geom = feature.geometry()
    if geom.isMultipart():
        parts = [list(p) for p in geom.asMultiPolyline()]
        coords = parts[part_idx]
        if len(coords) < 2:
            return None
        if which == 0:
            coords[0] = new_pt
        else:
            coords[-1] = new_pt
        parts[part_idx] = coords
        return QgsGeometry.fromMultiPolylineXY(parts)
    else:
        coords = list(geom.asPolyline())
        if len(coords) < 2:
            return None
        if which == 0:
            coords[0] = new_pt
        else:
            coords[-1] = new_pt
        return QgsGeometry.fromPolylineXY(coords)


def all_vertices(feature: QgsFeature):
    """Yield every QgsPointXY vertex of a (multi)polyline feature."""
    geom = feature.geometry()
    if geom is None or geom.isEmpty():
        return
    if geom.isMultipart():
        for part in geom.asMultiPolyline():
            for v in part:
                yield v
    else:
        for v in geom.asPolyline():
            yield v


def endpoint_has_vertex_neighbour(ep_pt: QgsPointXY, own_fid: int,
                                   layer, index: QgsSpatialIndex,
                                   feats: dict, tol: float) -> bool:
    """
    Strict FTTx-style check: returns True only if ep_pt coincides with
    a VERTEX (not just any interior point) of another feature.
    FTTx tools build graphs by matching endpoints to vertices only —
    a dangling end touching the middle of another line does not form a junction.
    """
    ep_geom    = QgsGeometry.fromPointXY(ep_pt)
    search_box = ep_geom.buffer(tol, 4).boundingBox()
    for cid in index.intersects(search_box):
        if cid == own_fid:
            continue
        other = feats.get(cid)
        if other is None:
            continue
        for v in all_vertices(other):
            if pt_dist(ep_pt, v) < tol:
                return True
    return False


def is_segment_isolated(fid: int, layer, index: QgsSpatialIndex,
                         feats: dict, tol: float = 0.01) -> bool:
    """
    FTTx-strict isolation check.
    A segment is isolated when EVERY endpoint has no vertex-coincident neighbour.
    Uses vertex-to-vertex matching to mirror the FTTx graph builder.
    Note: a segment whose dangling end only touches the interior of another line
    is still considered isolated — split-at-intersections must have run first.
    """
    f = feats.get(fid)
    if f is None:
        return False
    endpoints = get_endpoints(f)
    if not endpoints:
        return True
    for _, _, pt in endpoints:
        if endpoint_has_vertex_neighbour(pt, fid, layer, index, feats, tol):
            return False
    return True


def count_touching_neighbours(fid: int, layer, index: QgsSpatialIndex,
                               feats: dict, tol: float = 0.01) -> int:
    """Count distinct neighbours sharing a vertex with fid (FTTx-strict)."""
    f = feats.get(fid)
    if f is None:
        return 0
    touching = set()
    for _, _, pt in get_endpoints(f):
        ep_geom    = QgsGeometry.fromPointXY(pt)
        search_box = ep_geom.buffer(tol, 4).boundingBox()
        for cid in index.intersects(search_box):
            if cid == fid:
                continue
            other = feats.get(cid)
            if other is None:
                continue
            for v in all_vertices(other):
                if pt_dist(pt, v) < tol:
                    touching.add(cid)
                    break
    return len(touching)


def highlight_isolated_segments(layer, isolated_fids: list):
    """Select + zoom to isolated segments so user can fix with Vertex tool."""
    if not isolated_fids:
        return
    layer.selectByIds(isolated_fids)
    iface.mapCanvas().zoomToSelected(layer)
    iface.mapCanvas().refresh()
    iface.messageBar().pushMessage(
        "FTTx Topology",
        f"{len(isolated_fids)} isolated segment(s) selected — fix with Vertex tool (V)",
        level=Qgis.Warning, duration=20
    )


def apply_geom_edit(layer, fid, new_geom, index, feats):
    """Apply a geometry edit and keep index + feats dict in sync."""
    old_feat = feats[fid]
    index.deleteFeature(old_feat)
    ok = layer.changeGeometry(fid, new_geom)
    updated = layer.getFeature(fid)
    index.addFeature(updated)
    feats[fid] = updated
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  CORE WORKER
# ─────────────────────────────────────────────────────────────────────────────

class FTTxTopoWorker(QObject):

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    log_msg  = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _emit(self, pct, msg):
        self.progress.emit(pct, msg)
        self.log_msg.emit(msg)
        QApplication.processEvents()

    def run(self):
        try:
            self._execute()
        except Exception:
            self.finished.emit(False, "Unexpected error:\n" + traceback.format_exc())

    # ── Main pipeline ──────────────────────────────────────────────────
    def _execute(self):
        p = self.params
        snap_tol    = p["snap_tolerance"]
        extend_max  = p["extend_max"]
        bridge_max  = p["bridge_max"]
        output_path = p.get("output_save_path") or None
        layer       = p["input_layer"]

        if not is_line_layer(layer):
            self.finished.emit(False, "Selected layer is not a line layer.")
            return

        feat_count = layer.featureCount()
        self._emit(2, f"Input: {layer.name()} ({feat_count} features)")

        # ── Step 1: Fix geometries ─────────────────────────────────────
        self._emit(4, "Step 1/7 — Fixing geometries …")
        try:
            fixed = processing.run("native:fixgeometries", {
                'INPUT': layer, 'OUTPUT': 'memory:fixed'
            })['OUTPUT']
        except Exception:
            self.finished.emit(False, "fixgeometries failed:\n" + traceback.format_exc())
            return

        # ── Step 2: Remove duplicate geometries ────────────────────────
        self._emit(10, "Step 2/7 — Removing duplicate geometries …")
        try:
            dedup = processing.run("native:deleteduplicategeometries", {
                'INPUT': fixed, 'OUTPUT': 'memory:dedup'
            })['OUTPUT']
            removed = fixed.featureCount() - dedup.featureCount()
            self._emit(14, f"  Removed {removed} duplicate(s)")
        except Exception:
            dedup = fixed
            self._emit(14, "  (dedup not available, skipping)")

        if self._cancelled:
            self.finished.emit(False, "Cancelled."); return

        # ── Step 3: Snap ───────────────────────────────────────────────
        self._emit(16, f"Step 3/7 — Snapping to self (tolerance {snap_tol} m) …")
        try:
            snapped = processing.run("native:snapgeometries", {
                'INPUT': dedup, 'REFERENCE_LAYER': dedup,
                'TOLERANCE': snap_tol, 'BEHAVIOR': 0,
                'OUTPUT': 'memory:snapped'
            })['OUTPUT']
        except Exception:
            self.finished.emit(False, "snapgeometries failed:\n" + traceback.format_exc())
            return

        # ── Build working structures ───────────────────────────────────
        wlayer = snapped
        feats  = {f.id(): f for f in wlayer.getFeatures()}
        idx    = QgsSpatialIndex()
        for f in feats.values():
            idx.addFeature(f)

        self._emit(22, f"  Working layer: {len(feats)} features")
        if not wlayer.isEditable():
            wlayer.startEditing()

        # ── Step 4: Extend dangling endpoints ──────────────────────────
        self._emit(24, f"Step 4/7 — Extending dangling endpoints (max {extend_max} m) …")
        extended, skipped_far = 0, 0

        try:
            ep_layer = processing.run("native:extractspecificvertices", {
                'INPUT': wlayer, 'VERTICES': '0,-1',
                'OUTPUT': 'memory:endpoints'
            })['OUTPUT']
        except Exception:
            self.finished.emit(False, "Endpoint extraction failed:\n" + traceback.format_exc())
            return

        total_ep = ep_layer.featureCount()

        for ep_idx, ep_feat in enumerate(ep_layer.getFeatures()):
            if self._cancelled:
                wlayer.rollBack()
                self.finished.emit(False, "Cancelled."); return

            ep_geom  = ep_feat.geometry()
            if ep_geom is None or ep_geom.isEmpty():
                continue
            ep_point = ep_geom.asPoint()

            search_rect = ep_geom.buffer(extend_max + 1.0, 4).boundingBox()
            cands = idx.intersects(search_rect)
            if not cands:
                continue

            # Already touching?
            if any(ep_geom.distance(wlayer.getFeature(c).geometry() or
                   QgsGeometry()) < 1e-4 for c in cands):
                continue

            # Find nearest
            best_dist, best_cid, best_pt = float('inf'), None, None
            for cid in cands:
                cg = wlayer.getFeature(cid).geometry()
                if cg is None:
                    continue
                np_, d = nearest_point_on_line(cg, ep_point)
                if d < best_dist:
                    best_dist, best_cid, best_pt = d, cid, np_

            if best_cid is None or best_dist > extend_max:
                skipped_far += 1
                continue

            # Find owner
            owner_fid, owner_pi, owner_which = None, None, None
            for cid in cands:
                f = feats.get(cid)
                if f is None:
                    continue
                for pi, which, v in get_endpoints(f):
                    if pt_dist(v, ep_point) < 1.0:
                        owner_fid, owner_pi, owner_which = cid, pi, which
                        break
                if owner_fid is not None:
                    break

            # Fallback: closest endpoint globally
            if owner_fid is None:
                min_d = None
                for fid, f in feats.items():
                    for pi, which, v in get_endpoints(f):
                        d = pt_dist(v, ep_point)
                        if min_d is None or d < min_d:
                            min_d = d
                            owner_fid, owner_pi, owner_which = fid, pi, which

            if owner_fid is None:
                continue

            new_geom = move_endpoint(feats[owner_fid], owner_pi, owner_which, best_pt)
            if new_geom and not new_geom.isEmpty():
                apply_geom_edit(wlayer, owner_fid, new_geom, idx, feats)
                extended += 1

            if ep_idx % 500 == 0:
                pct = 24 + int(20 * ep_idx / max(total_ep, 1))
                self._emit(pct, f"  Endpoints: {ep_idx}/{total_ep}")

        self._emit(44, f"  Extended: {extended}  |  Too far (skipped): {skipped_far}")

        if self._cancelled:
            wlayer.rollBack(); self.finished.emit(False, "Cancelled."); return

        # ── Step 5: Isolated segment bridging ──────────────────────────
        self._emit(46, f"Step 5/7 — Detecting & bridging isolated segments (max {bridge_max} m) …")

        # Refresh feats after edits
        feats = {f.id(): f for f in wlayer.getFeatures()}
        idx   = QgsSpatialIndex()
        for f in feats.values():
            idx.addFeature(f)

        isolated_fids = [
            fid for fid in feats
            if is_segment_isolated(fid, wlayer, idx, feats, 1e-4)
        ]

        self._emit(48, f"  Found {len(isolated_fids)} isolated segment(s)")
        bridged = 0
        still_isolated = []

        for fid in isolated_fids:
            if self._cancelled:
                wlayer.rollBack(); self.finished.emit(False, "Cancelled."); return

            f = feats.get(fid)
            if f is None:
                continue
            eps = get_endpoints(f)
            if not eps:
                continue

            # Try to connect EACH endpoint to nearest neighbour VERTEX.
            # Snapping to a vertex (not line interior) means no split pass
            # is needed to create the junction — the node already exists.
            any_connected = False
            for pi, which, ep_pt in eps:
                ep_geom  = QgsGeometry.fromPointXY(ep_pt)
                search_b = ep_geom.buffer(bridge_max + 1.0, 4).boundingBox()
                cands    = [c for c in idx.intersects(search_b) if c != fid]
                if not cands:
                    continue

                # Candidate 1: nearest vertex on any neighbour
                best_vd, best_vpt = float('inf'), None
                # Candidate 2: nearest point on line (fallback — needs split)
                best_ld, best_lpt = float('inf'), None

                for cid in cands:
                    other = feats.get(cid)
                    if other is None:
                        continue
                    # vertex search
                    for v in all_vertices(other):
                        d = pt_dist(ep_pt, v)
                        if d < best_vd:
                            best_vd, best_vpt = d, v
                    # line-interior search
                    cg = wlayer.getFeature(cid).geometry()
                    if cg:
                        lp, ld = nearest_point_on_line(cg, ep_pt)
                        if lp and ld < best_ld:
                            best_ld, best_lpt = ld, lp

                # Prefer vertex snap (creates junction without needing split)
                if best_vd <= bridge_max:
                    snap_pt = best_vpt
                elif best_ld <= bridge_max and best_lpt:
                    # Line-interior snap — junction is created by split pass
                    snap_pt = best_lpt
                else:
                    continue

                new_geom = move_endpoint(f, pi, which, snap_pt)
                if new_geom and not new_geom.isEmpty():
                    apply_geom_edit(wlayer, fid, new_geom, idx, feats)
                    f = feats[fid]
                    any_connected = True

            if any_connected:
                bridged += 1
            else:
                still_isolated.append(fid)

        self._emit(60, f"  Bridged: {bridged}  |  Still isolated: {len(still_isolated)}")

        # Commit
        if wlayer.isEditable():
            if not wlayer.commitChanges():
                errs = wlayer.commitErrors()
                self.finished.emit(False, "Commit failed:\n" + "\n".join(errs))
                return

        self._emit(62, "  Edits committed.")

        if self._cancelled:
            self.finished.emit(False, "Cancelled."); return

        # ── Step 6a: Re-fix geometries after edits ─────────────────────
        # Geometry editing (snap/bridge) can introduce self-intersections
        # or zero-length segments that crash splitwithlines. Fix again.
        self._emit(63, "Step 6/7 — Re-fixing geometries after edits …")
        try:
            refixed = processing.run("native:fixgeometries", {
                'INPUT': wlayer,
                'OUTPUT': 'memory:refixed'
            })['OUTPUT']
        except Exception:
            self.finished.emit(False, "Re-fix geometries failed:\n" + traceback.format_exc())
            return

        # Log any features that are still invalid after fix
        bad_fids = [f.id() for f in refixed.getFeatures()
                    if f.geometry() is None or not f.geometry().isGeosValid()]
        if bad_fids:
            self._emit(63, f"  ⚠ {len(bad_fids)} feature(s) still invalid after fix — "
                           f"will be skipped by split. IDs: {bad_fids[:10]}")
        else:
            self._emit(63, "  All geometries valid.")

        # ── Step 6b: Split at intersections ────────────────────────────
        self._emit(64, "  Splitting lines at intersections …")

        # INVALID_FEATURES=1 → skip invalid features instead of raising
        # This matches QGIS Processing setting "Skip (ignore) features with invalid geometries"
        split_params = {
            'INPUT':  refixed,
            'LINES':  refixed,
            'OUTPUT': 'memory:roads_fttx_ready',
        }
        context = processing.createContext()
        context.setInvalidGeometryCheck(
            # QgsFeatureRequest.GeometrySkipInvalid = 1
            getattr(__import__('qgis.core', fromlist=['QgsFeatureRequest'])
                    .QgsFeatureRequest, 'GeometrySkipInvalid', 1)
        )
        try:
            split = processing.run("native:splitwithlines", split_params,
                                   context=context)['OUTPUT']
        except Exception:
            # Last-resort fallback: run without strict geometry checking
            try:
                import processing as proc_mod
                feedback = proc_mod.createFeedback()
                split = processing.run(
                    "native:splitwithlines", split_params,
                    feedback=feedback
                )['OUTPUT']
            except Exception:
                self.finished.emit(False, "Split failed:\n" + traceback.format_exc())
                return

        self._emit(80, f"  Split complete → {split.featureCount()} features")

        # ── Step 7: Topology audit (FTTx-strict vertex check) ──────────
        self._emit(82, "Step 7/7 — Topology audit (FTTx-strict vertex check) …")

        split_feats = {f.id(): f for f in split.getFeatures()}
        split_idx   = QgsSpatialIndex()
        for f in split_feats.values():
            split_idx.addFeature(f)

        # Use tol=0.01 m — tight enough to catch snapping errors,
        # loose enough to survive floating-point projection rounding
        AUDIT_TOL = 0.01
        remaining_isolated = [
            fid for fid in split_feats
            if is_segment_isolated(fid, split, split_idx, split_feats, AUDIT_TOL)
        ]

        junctions = sum(
            1 for fid in split_feats
            if count_touching_neighbours(fid, split, split_idx, split_feats, AUDIT_TOL) >= 2
        )

        self._emit(95, f"  Remaining isolated: {len(remaining_isolated)}")
        self._emit(96, f"  Segments with ≥2 neighbours (junctions): {junctions}")

        # Highlight any remaining isolated segments in the QGIS canvas
        if remaining_isolated:
            try:
                highlight_isolated_segments(split, remaining_isolated)
                self._emit(97, f"  ⚠ Isolated segments selected & zoomed in canvas for manual review.")
            except Exception:
                pass

        # ── Save ───────────────────────────────────────────────────────
        if output_path:
            self._emit(97, f"Saving to {output_path} …")
            err_code, err_msg = QgsVectorFileWriter.writeAsVectorFormat(
                split, output_path, "UTF-8", split.crs()
            )
            if err_code != QgsVectorFileWriter.NoError:
                self._emit(97, f"  Save warning: {err_msg}")
            else:
                self._emit(97, "  Saved.")

        QgsProject.instance().addMapLayer(split)
        self._emit(100, "Added 'roads_fttx_ready' to project.")

        # ── Report ─────────────────────────────────────────────────────
        isolated_note = ""
        if remaining_isolated:
            isolated_note = (
                f"\n⚠ {len(remaining_isolated)} segment(s) still isolated after bridging.\n"
                f"  These roads are physically disconnected beyond {bridge_max} m.\n"
                f"  Options:\n"
                f"    a) Increase bridge_max and re-run\n"
                f"    b) Manually connect them in QGIS with the Vertex tool\n"
                f"    c) Delete them if they are mapping errors\n"
                f"  Feature IDs: {remaining_isolated[:20]}"
            )
        else:
            isolated_note = "\n✅ No isolated segments detected — FTTx topology check should pass."

        summary = (
            f"━━━ PIPELINE COMPLETE ━━━\n"
            f"  Input features          : {feat_count}\n"
            f"  Output features (split) : {split.featureCount()}\n"
            f"  Endpoints extended      : {extended}\n"
            f"  Isolated segments bridged: {bridged}\n"
            f"  Segments still isolated : {len(remaining_isolated)}\n"
            f"  Junction nodes          : {junctions}\n"
            f"{isolated_note}\n\n"
            f"💡 Load the layer in your FTTx tool and re-run the design check."
        )
        self.finished.emit(True, summary)


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────

STYLE = """
QDialog {
    background: #111827;
    color: #dde5f0;
    font-family: 'Consolas', 'Courier New', monospace;
}
QGroupBox {
    border: 1px solid #1e3a52;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 8px;
    color: #4a9fd4;
    font-weight: bold;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLabel { color: #8aa8c8; font-size: 12px; }
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {
    background: #0d1117;
    border: 1px solid #1e3a52;
    border-radius: 3px;
    color: #dde5f0;
    padding: 4px 8px;
    font-size: 12px;
    min-height: 28px;
}
QDoubleSpinBox:focus, QSpinBox:focus,
QLineEdit:focus, QComboBox:focus { border: 1px solid #3a7abf; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #0d1117; color: #dde5f0;
    selection-background-color: #1e3a52;
}
QPushButton {
    background: #0f2440;
    border: 1px solid #1e4a7a;
    border-radius: 3px;
    color: #5aabef;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: bold;
    min-height: 30px;
}
QPushButton:hover  { background: #162e52; border-color: #3a7abf; color: #fff; }
QPushButton:pressed{ background: #0a1a30; }
QPushButton#run_btn {
    background: #0a2e1a;
    border-color: #1a6a3a;
    color: #4dd888;
    min-height: 36px;
    font-size: 13px;
}
QPushButton#run_btn:hover { background: #0f3a22; color: #fff; }
QPushButton#cancel_btn {
    background: #2e0f0f;
    border-color: #6a1a1a;
    color: #d87070;
}
QPushButton#cancel_btn:hover { background: #3a1414; }
QTextEdit {
    background: #090e18;
    border: 1px solid #1e3a52;
    border-radius: 3px;
    color: #6dc897;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    padding: 4px;
}
QProgressBar {
    background: #0d1117;
    border: 1px solid #1e3a52;
    border-radius: 3px;
    text-align: center;
    color: #5aabef;
    font-size: 11px;
    min-height: 18px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #0a2e1a, stop:1 #1a6a3a);
    border-radius: 2px;
}
QCheckBox { color: #8aa8c8; spacing: 6px; }
QCheckBox::indicator {
    width:14px; height:14px;
    border:1px solid #1e3a52; border-radius:2px;
    background:#0d1117;
}
QCheckBox::indicator:checked { background:#1a6a3a; border-color:#3aaa6a; }
"""


class FTTxTopoDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📡  FTTx Road Topology Fixer — OSM Edition")
        self.setMinimumWidth(600)
        self.setStyleSheet(STYLE)
        self.worker = None
        self._build_ui()
        self._populate_layers()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # Header
        title = QLabel("FTTx ROAD TOPOLOGY FIXER")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Consolas", 13, QFont.Bold))
        title.setStyleSheet("color:#3a9fd4; letter-spacing:3px; padding:4px 0;")
        root.addWidget(title)

        sub = QLabel("Fixes isolated OSM road segments that block FAT → FDT assignment")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color:#3a607a; font-size:11px; margin-bottom:4px;")
        root.addWidget(sub)

        # Input
        grp_in = QGroupBox("Input Layer")
        fl_in  = QFormLayout(grp_in)
        fl_in.setSpacing(8)
        self.layer_combo = QComboBox()
        self.btn_refresh = QPushButton("↺")
        self.btn_refresh.setFixedWidth(32)
        self.btn_refresh.clicked.connect(self._populate_layers)
        row_l = QHBoxLayout()
        row_l.addWidget(self.layer_combo)
        row_l.addWidget(self.btn_refresh)
        fl_in.addRow("OSM road layer:", row_l)
        root.addWidget(grp_in)

        # Parameters
        grp_p = QGroupBox("Parameters")
        fl_p  = QFormLayout(grp_p)
        fl_p.setSpacing(8)

        self.snap_spin = QDoubleSpinBox()
        self.snap_spin.setRange(0.01, 50.0)
        self.snap_spin.setValue(0.5)
        self.snap_spin.setSuffix(" m")
        self.snap_spin.setDecimals(2)
        self.snap_spin.setToolTip(
            "Snap near-misses within this distance together.\n"
            "0.3–0.8 m is ideal for OSM urban road data."
        )
        fl_p.addRow("Snap tolerance:", self.snap_spin)

        self.extend_spin = QDoubleSpinBox()
        self.extend_spin.setRange(0.0, 500.0)
        self.extend_spin.setValue(10.0)
        self.extend_spin.setSuffix(" m")
        self.extend_spin.setDecimals(2)
        self.extend_spin.setToolTip(
            "Maximum gap to auto-extend a dangling endpoint.\n"
            "Endpoints further away are left alone (true cul-de-sacs)."
        )
        fl_p.addRow("Max endpoint extend:", self.extend_spin)

        self.bridge_spin = QDoubleSpinBox()
        self.bridge_spin.setRange(0.0, 2000.0)
        self.bridge_spin.setValue(50.0)
        self.bridge_spin.setSuffix(" m")
        self.bridge_spin.setDecimals(1)
        self.bridge_spin.setToolTip(
            "Maximum distance to bridge a FULLY ISOLATED segment to the nearest road.\n"
            "This targets the FAT-on-isolated-road error.\n"
            "Set higher for rural/sparse OSM data (100–500 m).\n"
            "Set lower for dense urban grids (20–50 m)."
        )
        fl_p.addRow("Max bridge distance\n(isolated fix):", self.bridge_spin)

        root.addWidget(grp_p)

        # Output
        grp_o = QGroupBox("Output")
        fl_o  = QFormLayout(grp_o)
        fl_o.setSpacing(8)
        self.save_check = QCheckBox("Save result to file")
        self.save_check.toggled.connect(self._toggle_save)
        fl_o.addRow("", self.save_check)
        self.path_edit  = QLineEdit()
        self.path_edit.setPlaceholderText("/path/to/roads_fttx_ready.gpkg")
        self.path_edit.setEnabled(False)
        self.btn_browse = QPushButton("Browse …")
        self.btn_browse.setEnabled(False)
        self.btn_browse.clicked.connect(self._browse)
        row_p = QHBoxLayout()
        row_p.addWidget(self.path_edit)
        row_p.addWidget(self.btn_browse)
        fl_o.addRow("Output path:", row_p)
        root.addWidget(grp_o)

        # Log
        grp_log = QGroupBox("Pipeline Log")
        ll = QVBoxLayout(grp_log)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(140)
        ll.addWidget(self.log_edit)
        root.addWidget(grp_log)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet("color:#3a607a; font-size:11px;")
        root.addWidget(self.status_lbl)

        # Buttons
        btn_row = QHBoxLayout()
        self.run_btn    = QPushButton("▶  Run Topology Fix")
        self.run_btn.setObjectName("run_btn")
        self.cancel_btn = QPushButton("■  Cancel")
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.setEnabled(False)
        self.close_btn  = QPushButton("Close")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn.clicked.connect(self._cancel)
        self.close_btn.clicked.connect(self.close)
        self.diag_btn = QPushButton("Diagnose Layer")
        self.diag_btn.setToolTip(
            "FTTx-strict audit on the selected layer without modifying it.\n"
            "Isolated segments are selected and zoomed in the QGIS canvas."
        )
        self.diag_btn.clicked.connect(self._diagnose)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.diag_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)
        root.addLayout(btn_row)

    def _populate_layers(self):
        self.layer_combo.clear()
        for lid, layer in QgsProject.instance().mapLayers().items():
            if is_line_layer(layer):
                self.layer_combo.addItem(layer.name(), lid)
        if self.layer_combo.count() == 0:
            self.layer_combo.addItem("(no line layers found)")

    def _toggle_save(self, checked):
        self.path_edit.setEnabled(checked)
        self.btn_browse.setEnabled(checked)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Cleaned Roads", "",
            "GeoPackage (*.gpkg);;Shapefile (*.shp);;All Files (*)"
        )
        if path:
            self.path_edit.setText(path)

    def _run(self):
        lid = self.layer_combo.currentData()
        if not lid:
            QMessageBox.warning(self, "No layer", "Select a line layer first.")
            return
        layer = QgsProject.instance().mapLayer(lid)
        if layer is None or not is_line_layer(layer):
            QMessageBox.warning(self, "Invalid layer", "Please select a polyline layer.")
            return

        out_path = None
        if self.save_check.isChecked():
            out_path = self.path_edit.text().strip() or None
            if not out_path:
                QMessageBox.warning(self, "No path", "Enter an output file path.")
                return

        params = {
            "input_layer"     : layer,
            "snap_tolerance"  : self.snap_spin.value(),
            "extend_max"      : self.extend_spin.value(),
            "bridge_max"      : self.bridge_spin.value(),
            "output_save_path": out_path,
        }

        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self._set_running(True)

        self.worker = FTTxTopoWorker(params)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_msg.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.run()

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
        self.cancel_btn.setEnabled(False)
        self._append("⚠ Cancellation requested …")

    def _diagnose(self):
        """Audit the selected layer for isolated segments without modifying it."""
        lid = self.layer_combo.currentData()
        if not lid:
            QMessageBox.warning(self, "No layer", "Select a line layer first.")
            return
        layer = QgsProject.instance().mapLayer(lid)
        if layer is None or not is_line_layer(layer):
            QMessageBox.warning(self, "Invalid layer", "Please select a polyline layer.")
            return

        self._append(f"\n── Diagnosing: {layer.name()} ({layer.featureCount()} features) ──")
        QApplication.processEvents()

        feats = {f.id(): f for f in layer.getFeatures()}
        idx   = QgsSpatialIndex()
        for f in feats.values():
            idx.addFeature(f)

        AUDIT_TOL = 0.01
        isolated = [
            fid for fid in feats
            if is_segment_isolated(fid, layer, idx, feats, AUDIT_TOL)
        ]
        junctions = sum(
            1 for fid in feats
            if count_touching_neighbours(fid, layer, idx, feats, AUDIT_TOL) >= 2
        )

        self._append(f"  Total segments   : {len(feats)}")
        self._append(f"  Isolated segments: {len(isolated)}")
        self._append(f"  Junction nodes   : {junctions}")

        if isolated:
            self._append(f"  Feature IDs      : {isolated[:30]}")
            try:
                highlight_isolated_segments(layer, isolated)
                self._append("  ⚠ Isolated segments selected & zoomed in canvas.")
                self._append("  Fix manually: enable layer editing → Vertex tool (V)")
                self._append("  → drag endpoint to snap onto nearest road vertex.")
            except Exception as e:
                self._append(f"  (highlight failed: {e})")
            self.status_lbl.setText(f"⚠ {len(isolated)} isolated segment(s) found")
            self.status_lbl.setStyleSheet("color:#f0a030; font-size:11px;")
        else:
            self._append("  ✅ No isolated segments — topology looks good for FTTx.")
            self.status_lbl.setText("✅ No isolated segments")
            self.status_lbl.setStyleSheet("color:#4dd888; font-size:11px;")

    def _set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.diag_btn.setEnabled(not running)
        self.layer_combo.setEnabled(not running)
        self.snap_spin.setEnabled(not running)
        self.extend_spin.setEnabled(not running)
        self.bridge_spin.setEnabled(not running)

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_lbl.setText(msg[:80])

    def _on_log(self, msg):
        self._append(msg)

    def _on_finished(self, ok, msg):
        self._set_running(False)
        self._append("\n" + msg)
        if ok:
            self.progress_bar.setValue(100)
            self.status_lbl.setText("✅ Complete!")
            self.status_lbl.setStyleSheet("color:#4dd888; font-size:11px;")
        else:
            self.status_lbl.setText("❌ Failed — see log.")
            self.status_lbl.setStyleSheet("color:#d87070; font-size:11px;")

    def _append(self, msg):
        self.log_edit.append(msg)
        self.log_edit.verticalScrollBar().setValue(
            self.log_edit.verticalScrollBar().maximum()
        )


# ─────────────────────────────────────────────────────────────────────────────
#  LAUNCH
# ─────────────────────────────────────────────────────────────────────────────

def open_fttx_fixer():
    dlg = FTTxTopoDialog()
    dlg.show()
    return dlg

_fttx_dlg = open_fttx_fixer()
