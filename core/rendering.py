"""
Rendering utilities for BlueSky Gym environments.

Provides reusable canvas management, coordinate projections, and drawing
functions for top-down and side-profile views. No BlueSky dependency —
environments pass world coordinates, this module handles screen output.
"""

import numpy as np
import pygame

NM2KM = 1.852
DEG2KM = 111.32

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
SKY_BLUE = (135, 206, 235)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (80, 80, 80)
LIGHT_GRAY = (155, 155, 155)
CRIMSON = (220, 20, 60)
OLIVE_GREEN = (154, 205, 50)
SLATE_GRAY = (119, 136, 153)
BRIGHT_GREEN = (3, 252, 11)
RED_ORANGE = (235, 52, 52)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
class PygameCanvas:
    """Manages a pygame window and per-frame surface lifecycle.

    In ``mode="human"`` (default) frames are blitted to a live pygame window.
    In ``mode="rgb_array"`` no window is opened; ``end_frame`` instead returns
    the frame as an ``(H, W, 3)`` uint8 array, so rendering works headless
    (e.g. with ``SDL_VIDEODRIVER=dummy``) for recording GIFs/videos.
    """

    def __init__(self, width, height, render_fps=120, bg_color=SKY_BLUE, mode="human"):
        self.width = width
        self.height = height
        self.window_size = (width, height)
        self.render_fps = render_fps
        self.bg_color = bg_color
        self.mode = mode
        self.window = None
        self.clock = None

    def begin_frame(self):
        if self.mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(self.window_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()
        elif not pygame.get_init():
            pygame.init()   # offscreen: no display window needed
        canvas = pygame.Surface(self.window_size)
        canvas.fill(self.bg_color)
        return canvas

    def end_frame(self, canvas):
        if self.mode != "human":
            # offscreen: return the frame as an (H, W, 3) uint8 array
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)
        return None


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------
class TopDownProjection:
    """North-up flat-earth projection from lat/lon to screen pixels.

    Uses a flat-earth approximation to convert geographic coordinates to
    screen positions relative to a reference point. This is a simplification
    that assumes a locally flat surface — it does not account for Earth
    curvature and breaks down near the poles or for large airspaces
    (> ~500 km). For the typical scenario sizes in BlueSky Gym this is
    accurate to within ~0.3%.

    Convention: North is up, East is right.

    Parameters
    ----------
    max_distance : float
        World width in km that maps to the viewport width.
    ref_lat : float
        Reference latitude (center of view) in degrees.
    ref_lon : float
        Reference longitude (center of view) in degrees.
    window_size : tuple of (width, height), optional
        Canvas size in pixels. Uses the full canvas as the drawing area.
        Provide this OR viewport, not both.
    viewport : tuple of (x_off, y_off, width, height), optional
        Sub-region of the canvas for split-level views. Overrides window_size.
    """

    def __init__(self, max_distance, ref_lat, ref_lon, window_size=None, viewport=None):
        if viewport is None:
            viewport = (0, 0, *window_size)
        self.max_distance = max_distance
        self.viewport = viewport
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self._cos_ref_lat = np.cos(np.deg2rad(ref_lat))

        x_off, y_off, w, h = viewport
        self._origin = (x_off + w / 2, y_off + h / 2)
        self._px_per_km = w / max_distance

    @property
    def center(self):
        return self._origin

    @property
    def px_per_km(self):
        return self._px_per_km

    def update_ref(self, ref_lat, ref_lon):
        """Update the reference point (e.g. to follow the ownship)."""
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self._cos_ref_lat = np.cos(np.deg2rad(ref_lat))

    def project(self, lat, lon):
        """Convert lat/lon to canvas (x, y)."""
        dx_km = (lon - self.ref_lon) * self._cos_ref_lat * DEG2KM
        dy_km = (lat - self.ref_lat) * DEG2KM
        return (self._origin[0] + dx_km * self._px_per_km,
                self._origin[1] - dy_km * self._px_per_km)

    def direction(self, heading_deg, length_km):
        """Convert a heading and length to a screen delta (dx, dy).

        Heading follows aviation convention: 0° = North, 90° = East, clockwise.
        """
        heading_rad = np.deg2rad(heading_deg)
        px = length_km * self._px_per_km
        return np.sin(heading_rad) * px, -np.cos(heading_rad) * px

    def scale(self, km):
        """Convert a world distance in km to pixels."""
        return km * self._px_per_km

    def clip(self, canvas):
        """Restrict drawing to this projection's viewport. Call unclip() when done."""
        canvas.set_clip(pygame.Rect(self.viewport))

    def unclip(self, canvas):
        """Remove viewport clipping and draw a border around the viewport."""
        canvas.set_clip(None)
        pygame.draw.rect(canvas, WHITE, pygame.Rect(self.viewport), width=1)


class SideProfileProjection:
    """Side-profile projection: horizontal distance + altitude to screen pixels.

    Parameters
    ----------
    max_distance : float
        World width in km mapped to viewport width.
    max_altitude : float
        Maximum altitude value mapped to the top of the drawable area.
    window_size : tuple of (width, height), optional
        Canvas size in pixels. Uses the full canvas as the drawing area.
        Provide this OR viewport, not both.
    viewport : tuple of (x_off, y_off, width, height), optional
        Sub-region of the canvas for split-level views. Overrides window_size.
    x_offset : float
        Horizontal pixel offset for the origin (ownship position).
    ground_height : int
        Height in pixels of the ground strip at the bottom of the viewport.
    """

    def __init__(self, max_distance, max_altitude,
                 window_size=None, viewport=None,
                 x_offset=25, ground_height=50):
        if viewport is None:
            viewport = (0, 0, *window_size)
        self.max_distance = max_distance
        self.max_altitude = max_altitude
        self.viewport = viewport
        self.x_offset = x_offset
        self.ground_height = ground_height
        self._x_off, self._y_off, self._w, self._h = viewport

    def project(self, distance_km, altitude):
        """Convert distance/altitude to canvas (x, y)."""
        x = ((distance_km + self.x_offset) / self.max_distance) * self._w + self._x_off
        y = (1.0 - altitude / self.max_altitude) * (self._h - self.ground_height) + self._y_off
        return x, y

    def altitude_to_y(self, altitude):
        """Convert altitude to canvas y-coordinate (for split-level use)."""
        return (1.0 - altitude / self.max_altitude) * (self._h - self.ground_height) + self._y_off

    def scale_horizontal(self, km):
        """Convert horizontal distance in km to pixels."""
        return (km / self.max_distance) * self._w

    def scale_vertical(self, altitude_units):
        """Convert altitude units to pixels."""
        return (altitude_units / self.max_altitude) * (self._h - self.ground_height)

    def clip(self, canvas):
        """Restrict drawing to this projection's viewport. Call unclip() when done."""
        canvas.set_clip(pygame.Rect(self.viewport))

    def unclip(self, canvas):
        """Remove viewport clipping and draw a border around the viewport."""
        canvas.set_clip(None)
        pygame.draw.rect(canvas, WHITE, pygame.Rect(self.viewport), width=1)


# ---------------------------------------------------------------------------
# Top-down drawing functions
# ---------------------------------------------------------------------------
def draw_aircraft(canvas, x, y, heading_deg, body_km, heading_km, projection,
                  *, color=BLACK, body_width=4, heading_width=1,
                  heading_color=None):
    """Draw an aircraft: body line centered on (x, y) with a heading indicator."""
    if heading_color is None:
        heading_color = color

    bdx, bdy = projection.direction(heading_deg, body_km)
    pygame.draw.line(canvas, color,
                     (x - bdx / 2, y - bdy / 2),
                     (x + bdx / 2, y + bdy / 2),
                     width=body_width)

    hdx, hdy = projection.direction(heading_deg, heading_km)
    pygame.draw.line(canvas, heading_color,
                     (x, y), (x + hdx, y + hdy),
                     width=heading_width)


def draw_intruder(canvas, x, y, heading_deg, projection, *,
                  body_km=3, heading_km=10, safety_radius_km=None,
                  in_intrusion=False,
                  color_safe=GRAY, color_intrusion=CRIMSON):
    """Draw an intruder aircraft with optional safety circle."""
    color = color_intrusion if in_intrusion else color_safe

    draw_aircraft(canvas, x, y, heading_deg,
                  body_km=body_km, heading_km=heading_km,
                  projection=projection, color=color)

    if safety_radius_km is not None:
        radius_px = projection.scale(safety_radius_km)
        pygame.draw.circle(canvas, color, (int(x), int(y)),
                           radius=int(radius_px), width=2)


def draw_waypoint(canvas, x, y, margin_km, projection, *,
                  reached=False,
                  color_active=WHITE, color_reached=LIGHT_GRAY):
    """Draw a waypoint marker: filled dot + margin ring."""
    color = color_reached if reached else color_active
    pygame.draw.circle(canvas, color, (int(x), int(y)),
                       radius=4, width=0)
    margin_px = projection.scale(margin_km)
    pygame.draw.circle(canvas, color, (int(x), int(y)),
                       radius=int(margin_px), width=2)


def draw_polygon(canvas, points, *, color=BLACK, filled=True, width=0):
    """Draw a polygon shape (airspace boundary, obstacle, etc.)."""
    if filled:
        pygame.draw.polygon(canvas, color, points)
    else:
        pygame.draw.polygon(canvas, color, points, width=max(width, 1))


def draw_radial_line(canvas, x, y, heading_deg, length_km, projection, *,
                     color=BLACK, width=1):
    """Draw a line from (x, y) in a given heading direction."""
    dx, dy = projection.direction(heading_deg, length_km)
    pygame.draw.line(canvas, color,
                     (x, y), (x + dx, y + dy),
                     width=width)


def draw_line(canvas, x1, y1, x2, y2, *, color=BLACK, width=1):
    """Draw a line between two screen positions."""
    pygame.draw.line(canvas, color,
                     (int(x1), int(y1)), (int(x2), int(y2)),
                     width=width)


# ---------------------------------------------------------------------------
# Side-profile drawing functions
# ---------------------------------------------------------------------------
def draw_side_aircraft(canvas, x, y, length_px, *, color=BLACK, width=5):
    """Draw an aircraft as a horizontal line in side-profile view."""
    pygame.draw.line(canvas, color,
                     (int(x), int(y)),
                     (int(x + length_px), int(y)),
                     width=width)


def draw_side_intruder(canvas, x, y, length_px, projection, *,
                       color=BLACK, width=5, in_intrusion=False,
                       hor_margin_km=None, ver_margin_alt=None):
    """Draw a side-profile intruder aircraft with optional collision box.

    Parameters
    ----------
    in_intrusion : bool
        If True, the collision box is drawn in red.
    hor_margin_km : float, optional
        Half-width of collision box in km (matches safety_radius_km in top view).
    ver_margin_alt : float, optional
        Half-height of collision box in altitude units.
    """
    draw_side_aircraft(canvas, x, y, length_px, color=color, width=width)

    if hor_margin_km is not None and ver_margin_alt is not None:
        hw = projection.scale_horizontal(hor_margin_km)
        hh = projection.scale_vertical(ver_margin_alt)
        box_color = CRIMSON if in_intrusion else BLACK
        draw_collision_box(canvas, x + length_px / 2, y, hw, hh, color=box_color)


def draw_collision_box(canvas, cx, cy, half_width_px, half_height_px, *,
                       color=BLACK, width=1):
    """Draw a rectangle outline centered on (cx, cy)."""
    left = cx - half_width_px
    right = cx + half_width_px
    top = cy - half_height_px
    bottom = cy + half_height_px
    pygame.draw.line(canvas, color, (left, top), (right, top), width=width)
    pygame.draw.line(canvas, color, (left, bottom), (right, bottom), width=width)
    pygame.draw.line(canvas, color, (left, top), (left, bottom), width=width)
    pygame.draw.line(canvas, color, (right, top), (right, bottom), width=width)


def draw_ground(canvas, projection, *, color=OLIVE_GREEN):
    """Draw a ground rectangle at the bottom of the projection's viewport."""
    x_off, y_off, w, h = projection.viewport
    pygame.draw.rect(canvas, color,
                     pygame.Rect((x_off, y_off + h - projection.ground_height),
                                 (w, projection.ground_height)))


def draw_runway(canvas, x_start, y, length_px, *, color=SLATE_GRAY, width=3):
    """Draw a runway line on the ground surface."""
    pygame.draw.line(canvas, color,
                     (int(x_start), int(y)),
                     (int(x_start + length_px), int(y)),
                     width=width)


def draw_target_altitude(canvas, y, projection, *, color=WHITE):
    """Draw a horizontal target altitude line spanning the projection's viewport width."""
    x_off, _, w, _ = projection.viewport
    pygame.draw.line(canvas, color,
                     (x_off, int(y)),
                     (x_off + w, int(y)))
