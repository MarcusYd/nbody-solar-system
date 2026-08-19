"""Interactive 2D N-body solar-system simulation.

Physical state uses SI units: metres, seconds, kilograms, and metres per
second. Display coordinates are logarithmically compressed and are not to
scale.
"""

import math
import time

import numpy as np
import pygame

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# -----------------------------
# Constants (SI)
# -----------------------------
G = 6.67430e-11
AU = 1.495978707e11
SOFTENING = 1.0e9  # m

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0

# -----------------------------
# Bodies: name, mass (kg), approximate semi-major axis (m)
# -----------------------------
MAJOR_BODIES = [
    ("Sun", 1.9885e30, 0.0),
    ("Mercury", 3.3011e23, 0.387098 * AU),
    ("Venus", 4.8675e24, 0.723332 * AU),
    ("Earth", 5.97237e24, 1.000000 * AU),
    ("Mars", 6.4171e23, 1.523679 * AU),
    ("Jupiter", 1.8982e27, 5.2044 * AU),
    ("Saturn", 5.6834e26, 9.5826 * AU),
    ("Uranus", 8.6810e25, 19.2184 * AU),
    ("Neptune", 1.02413e26, 30.1104 * AU),
]


def make_major_body_state():
    """Create approximate circular initial conditions in a barycentric frame.

    Returns names, masses (kg), positions (m), and velocities (m/s). The
    planet angles are illustrative rather than historical ephemeris data.
    """
    names = [body[0] for body in MAJOR_BODIES]
    masses = np.array([body[1] for body in MAJOR_BODIES], dtype=np.float64)
    semi_major_axes = np.array([body[2] for body in MAJOR_BODIES], dtype=np.float64)

    count = len(names)
    positions = np.zeros((count, 2), dtype=np.float64)
    velocities = np.zeros((count, 2), dtype=np.float64)

    angles = np.linspace(0.0, 2 * math.pi * 0.85, count)
    for i in range(1, count):
        theta = angles[i]
        radius = semi_major_axes[i]
        positions[i] = [radius * math.cos(theta), radius * math.sin(theta)]
        speed = math.sqrt(G * masses[0] / radius)
        velocities[i] = [-speed * math.sin(theta), speed * math.cos(theta)]

    # Translate the heliocentric initial conditions into a barycentric frame
    # without changing any body's velocity relative to the Sun.
    total_mass = masses.sum()
    positions -= (masses[:, None] * positions).sum(axis=0) / total_mass
    velocities -= (masses[:, None] * velocities).sum(axis=0) / total_mass

    return names, masses, positions, velocities

# -----------------------------
# Visual compression mapping (meters -> compressed units)
# -----------------------------
RMAX_PHYS = 32 * AU


def compress_positions(pos):
    """Map ``(N, 2)`` positions in metres to logarithmic display units."""
    rad = np.linalg.norm(pos, axis=1)
    out = np.zeros_like(pos, dtype=np.float64)
    mask = rad > 0
    compressed_radius = np.log1p(rad[mask] / AU) / math.log1p(RMAX_PHYS / AU)
    out[mask] = pos[mask] * (compressed_radius / rad[mask])[:, None]
    return out

# -----------------------------
# NumPy physics (vectorized)
# -----------------------------
def accel_majors_numpy(pos_major, m_major):
    """Return full N-body acceleration (m/s^2) for the major bodies."""
    d = pos_major[None, :, :] - pos_major[:, None, :]  # (M,M,2), j - i
    dist2 = d[..., 0]**2 + d[..., 1]**2 + SOFTENING**2
    inv_dist3 = dist2 ** (-1.5)
    np.fill_diagonal(inv_dist3, 0.0)
    acc = G * np.sum(d * inv_dist3[..., None] * m_major[None, :, None], axis=1)  # (M,2)
    return acc

def accel_asteroids_numpy(pos_ast, pos_major, m_major):
    """Return acceleration for massless asteroids due to major bodies only."""
    # d[a, j, :] = pos_major[j] - pos_ast[a]
    d = pos_major[None, :, :] - pos_ast[:, None, :]  # (A,M,2)
    dist2 = d[..., 0]**2 + d[..., 1]**2 + SOFTENING**2
    inv_dist3 = dist2 ** (-1.5)
    acc = G * np.sum(d * inv_dist3[..., None] * m_major[None, :, None], axis=1)  # (A,2)
    return acc

def step_numpy(pos_major, vel_major, pos_ast, vel_ast, m_major, dt):
    """Advance all bodies by ``dt`` seconds using a kick-drift-kick step."""
    a0m = accel_majors_numpy(pos_major, m_major)
    vhm = vel_major + 0.5 * dt * a0m
    pos_major2 = pos_major + dt * vhm
    a1m = accel_majors_numpy(pos_major2, m_major)
    vel_major2 = vhm + 0.5 * dt * a1m

    if pos_ast.shape[0] > 0:
        a0a = accel_asteroids_numpy(pos_ast, pos_major, m_major)
        vha = vel_ast + 0.5 * dt * a0a
        pos_ast2 = pos_ast + dt * vha
        # Use updated major-body positions for the second half kick.
        a1a = accel_asteroids_numpy(pos_ast2, pos_major2, m_major)
        vel_ast2 = vha + 0.5 * dt * a1a
    else:
        pos_ast2, vel_ast2 = pos_ast, vel_ast

    return pos_major2, vel_major2, pos_ast2, vel_ast2

# -----------------------------
# Numba physics
# -----------------------------
if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def accel_majors_numba(pos_major, m_major, Gc, softening):
        """Numba equivalent of ``accel_majors_numpy``."""
        M = pos_major.shape[0]
        acc = np.zeros((M, 2), dtype=np.float64)
        for i in range(M):
            xi = pos_major[i, 0]
            yi = pos_major[i, 1]
            ax = 0.0
            ay = 0.0
            for j in range(M):
                if j == i:
                    continue
                dx = pos_major[j, 0] - xi
                dy = pos_major[j, 1] - yi
                dist2 = dx*dx + dy*dy + softening*softening
                inv = 1.0 / math.sqrt(dist2)
                inv3 = inv*inv*inv
                s = Gc * m_major[j] * inv3
                ax += dx * s
                ay += dy * s
            acc[i, 0] = ax
            acc[i, 1] = ay
        return acc

    @njit(fastmath=True)
    def accel_asteroids_numba(pos_ast, pos_major, m_major, Gc, softening):
        """Numba equivalent of ``accel_asteroids_numpy``."""
        A = pos_ast.shape[0]
        M = pos_major.shape[0]
        acc = np.zeros((A, 2), dtype=np.float64)
        for a in range(A):
            xa = pos_ast[a, 0]
            ya = pos_ast[a, 1]
            ax = 0.0
            ay = 0.0
            for j in range(M):
                dx = pos_major[j, 0] - xa
                dy = pos_major[j, 1] - ya
                dist2 = dx*dx + dy*dy + softening*softening
                inv = 1.0 / math.sqrt(dist2)
                inv3 = inv*inv*inv
                s = Gc * m_major[j] * inv3
                ax += dx * s
                ay += dy * s
            acc[a, 0] = ax
            acc[a, 1] = ay
        return acc

    @njit(fastmath=True)
    def step_numba(pos_major, vel_major, pos_ast, vel_ast, m_major, dt, Gc, softening):
        """Numba equivalent of the NumPy kick-drift-kick update."""
        a0m = accel_majors_numba(pos_major, m_major, Gc, softening)
        M = pos_major.shape[0]
        vhm = np.empty_like(vel_major)
        pos_major2 = np.empty_like(pos_major)
        for i in range(M):
            vhm[i, 0] = vel_major[i, 0] + 0.5 * dt * a0m[i, 0]
            vhm[i, 1] = vel_major[i, 1] + 0.5 * dt * a0m[i, 1]
            pos_major2[i, 0] = pos_major[i, 0] + dt * vhm[i, 0]
            pos_major2[i, 1] = pos_major[i, 1] + dt * vhm[i, 1]
        a1m = accel_majors_numba(pos_major2, m_major, Gc, softening)
        vel_major2 = np.empty_like(vel_major)
        for i in range(M):
            vel_major2[i, 0] = vhm[i, 0] + 0.5 * dt * a1m[i, 0]
            vel_major2[i, 1] = vhm[i, 1] + 0.5 * dt * a1m[i, 1]

        A = pos_ast.shape[0]
        if A > 0:
            a0a = accel_asteroids_numba(pos_ast, pos_major, m_major, Gc, softening)
            vha = np.empty_like(vel_ast)
            pos_ast2 = np.empty_like(pos_ast)
            for a in range(A):
                vha[a, 0] = vel_ast[a, 0] + 0.5 * dt * a0a[a, 0]
                vha[a, 1] = vel_ast[a, 1] + 0.5 * dt * a0a[a, 1]
                pos_ast2[a, 0] = pos_ast[a, 0] + dt * vha[a, 0]
                pos_ast2[a, 1] = pos_ast[a, 1] + dt * vha[a, 1]
            a1a = accel_asteroids_numba(pos_ast2, pos_major2, m_major, Gc, softening)
            vel_ast2 = np.empty_like(vel_ast)
            for a in range(A):
                vel_ast2[a, 0] = vha[a, 0] + 0.5 * dt * a1a[a, 0]
                vel_ast2[a, 1] = vha[a, 1] + 0.5 * dt * a1a[a, 1]
        else:
            pos_ast2 = pos_ast
            vel_ast2 = vel_ast

        return pos_major2, vel_major2, pos_ast2, vel_ast2

# -----------------------------
# Asteroid factory
# -----------------------------
def make_asteroids(n, sun_position, sun_velocity, m_sun, seed=None):
    """Create massless asteroid test particles around the current Sun state.

    Positions are in metres and velocities are in metres per second. Passing
    an integer ``seed`` makes generation deterministic; the default preserves
    random generation for interactive use.
    """
    rng = np.random.default_rng(seed)

    if n <= 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    r_min = 2.1 * AU
    r_max = 3.3 * AU

    radii = rng.uniform(r_min, r_max, size=n)
    angles = rng.uniform(0, 2 * np.pi, size=n)

    pos = np.zeros((n, 2), dtype=np.float64)
    pos[:, 0] = radii * np.cos(angles)
    pos[:, 1] = radii * np.sin(angles)
    pos += np.asarray(sun_position, dtype=np.float64)

    # circular speed around Sun
    speeds = np.sqrt(G * m_sun / radii)
    vel = np.zeros((n, 2), dtype=np.float64)
    vel[:, 0] = -speeds * np.sin(angles)
    vel[:, 1] = speeds * np.cos(angles)

    # add small perturbations (a few %)
    vel *= rng.normal(1.0, 0.02, size=(n, 1))

    # Convert Sun-relative circular velocities to the simulation frame.
    vel += np.asarray(sun_velocity, dtype=np.float64)

    return pos, vel

# -----------------------------
# Simple UI slider
# -----------------------------
class Slider:
    def __init__(self, rect, min_val, max_val, initial):
        self.rect = pygame.Rect(rect)
        self.min = min_val
        self.max = max_val
        self.value = float(initial)
        self.dragging = False

    def handle_event(self, event):
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                changed = True
                self._set_from_mouse(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            changed = True
            self._set_from_mouse(event.pos[0])
        return changed

    def _set_from_mouse(self, mx):
        t = (mx - self.rect.left) / max(1, self.rect.width)
        t = max(0.0, min(1.0, t))
        self.value = self.min + t * (self.max - self.min)

    def draw(self, surf, font, label):
        # track
        pygame.draw.rect(surf, (40, 40, 55), self.rect, border_radius=6)
        # fill
        t = (self.value - self.min) / (self.max - self.min)
        fill = pygame.Rect(
            self.rect.left, self.rect.top, int(self.rect.width * t), self.rect.height
        )
        pygame.draw.rect(surf, (80, 120, 200), fill, border_radius=6)
        # knob
        knob_x = self.rect.left + int(self.rect.width * t)
        pygame.draw.circle(
            surf,
            (230, 230, 230),
            (knob_x, self.rect.centery),
            self.rect.height // 2,
        )
        # text
        txt = font.render(f"{label}: {int(self.value)}", True, (235, 235, 235))
        surf.blit(txt, (self.rect.left, self.rect.top - 22))

# -----------------------------
# Main
# -----------------------------
def main():
    pygame.init()
    W, H = 1300, 720
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Solar System + Asteroid Belt (NumPy vs Numba)")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 16)
    font_big = pygame.font.SysFont("consolas", 18, bold=True)

    # Layout
    panel_w = 420
    view_rect = pygame.Rect(0, 0, W - panel_w, H)
    panel_rect = pygame.Rect(W - panel_w, 0, panel_w, H)

    # Major body state
    names, m_major, pos_major, vel_major = make_major_body_state()
    M = len(names)

    # Asteroids
    asteroid_target = 2000
    pos_ast, vel_ast = make_asteroids(
        asteroid_target, pos_major[0], vel_major[0], m_major[0]
    )

    # Controls
    dt = 6 * 3600.0
    steps_per_frame = 40
    sim_time = 0.0
    selected = 3

    # Camera
    zoom = 300.0
    pan = np.array([0.0, 0.0], dtype=np.float64)
    dragging = False
    drag_start = (0, 0)
    pan_start = pan.copy()

    def world_to_screen(p_comp):
        x = view_rect.centerx + pan[0] + p_comp[0] * zoom
        y = view_rect.centery + pan[1] - p_comp[1] * zoom
        return int(x), int(y)

    # Colors/sizes
    BG = (12, 12, 16)
    PANEL = (18, 18, 26)
    GRID = (35, 35, 45)
    WHITE = (235, 235, 235)

    colors_major = [
        (255, 220, 120),
        (170, 170, 170),
        (235, 210, 160),
        (120, 170, 255),
        (255, 120, 90),
        (240, 190, 120),
        (235, 220, 190),
        (160, 220, 220),
        (120, 160, 255),
    ]
    sizes_major = [10, 3, 4, 4, 3, 7, 6, 5, 5]
    asteroid_color = (140, 140, 160)

    # Slider for asteroids
    slider = Slider((panel_rect.left + 24, panel_rect.top + 520, panel_w - 48, 18),
                    min_val=0, max_val=20000, initial=asteroid_target)

    # Backend toggle
    backend = "numpy"

    # Warm-up numba so first toggle isn’t “fake slow”
    if NUMBA_AVAILABLE:
        _ = step_numba(pos_major.copy(), vel_major.copy(),
                       pos_ast[:1].copy(), vel_ast[:1].copy(),
                       m_major, dt, G, SOFTENING)

    # Perf HUD trackers
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    phys_time_acc = 0.0
    frames_acc = 0
    steps_acc = 0
    hud_lines = ["(warming up...)"]

    # Metrics for "interactions"
    def interactions_per_step(A):
        # majors full N-body: M*(M-1)
        # asteroids: A*M (each asteroid interacts with each major)
        return M*(M-1) + A*M

    running = True
    while running:
        clock.tick(60)

        # -------- events --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_RIGHT, pygame.K_DOWN):
                    selected = (selected + 1) % M
                elif event.key in (pygame.K_LEFT, pygame.K_UP):
                    selected = (selected - 1) % M
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    steps_per_frame = min(steps_per_frame + 10, 1000)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    steps_per_frame = max(1, steps_per_frame - 10)
                elif event.key == pygame.K_r:
                    zoom = 300.0
                    pan[:] = 0.0
                elif event.key == pygame.K_t:
                    if NUMBA_AVAILABLE:
                        backend = "numba" if backend == "numpy" else "numpy"

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and view_rect.collidepoint(event.pos):
                    dragging = True
                    drag_start = event.pos
                    pan_start = pan.copy()
                elif event.button == 4:
                    zoom *= 1.12
                elif event.button == 5:
                    zoom = max(40.0, zoom / 1.12)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            if event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - drag_start[0]
                dy = event.pos[1] - drag_start[1]
                pan[:] = pan_start + np.array([dx, dy], dtype=np.float64)

            # slider events (in panel)
            slider_changed = slider.handle_event(event)
            if slider_changed:
                new_target = int(slider.value)
                cur = pos_ast.shape[0]
                if new_target != cur:
                    if new_target < cur:
                        pos_ast = pos_ast[:new_target].copy()
                        vel_ast = vel_ast[:new_target].copy()
                    else:
                        add_n = new_target - cur
                        add_pos, add_vel = make_asteroids(
                            add_n, pos_major[0], vel_major[0], m_major[0]
                        )
                        pos_ast = np.vstack([pos_ast, add_pos])
                        vel_ast = np.vstack([vel_ast, add_vel])

        # -------- physics --------
        phys_start = time.perf_counter()
        for _ in range(steps_per_frame):
            if backend == "numpy":
                pos_major, vel_major, pos_ast, vel_ast = step_numpy(
                    pos_major, vel_major, pos_ast, vel_ast, m_major, dt
                )
            else:
                pos_major, vel_major, pos_ast, vel_ast = step_numba(
                    pos_major, vel_major, pos_ast, vel_ast, m_major, dt, G, SOFTENING
                )
            sim_time += dt
        phys_time_acc += (time.perf_counter() - phys_start)

        # -------- render --------
        screen.fill(BG)
        pygame.draw.rect(screen, BG, view_rect)

        # grid
        cx, cy = view_rect.centerx + int(pan[0]), view_rect.centery + int(pan[1])
        pygame.draw.line(screen, GRID, (view_rect.left, cy), (view_rect.right, cy), 1)
        pygame.draw.line(screen, GRID, (cx, view_rect.top), (cx, view_rect.bottom), 1)
        for k in range(1, 6):
            pygame.draw.circle(screen, GRID, (cx, cy), int(zoom * (k / 5.0)), 1)

        # positions (compressed)
        comp_major = compress_positions(pos_major)
        if pos_ast.shape[0] > 0:
            comp_ast = compress_positions(pos_ast)
        else:
            comp_ast = np.zeros((0, 2), dtype=np.float64)

        # asteroids (draw as points)
        # Drawing can become the bottleneck at high asteroid counts.
        for i in range(comp_ast.shape[0]):
            sx, sy = world_to_screen(comp_ast[i])
            if view_rect.collidepoint((sx, sy)):
                screen.set_at((sx, sy), asteroid_color)

        # majors
        for i in range(M):
            sx, sy = world_to_screen(comp_major[i])
            pygame.draw.circle(screen, colors_major[i], (sx, sy), sizes_major[i])

        # highlight selected
        sx, sy = world_to_screen(comp_major[selected])
        pygame.draw.circle(screen, WHITE, (sx, sy), sizes_major[selected] + 4, 2)

        # panel
        pygame.draw.rect(screen, PANEL, panel_rect)

        # slider
        slider.draw(screen, font, "Asteroids")

        # HUD + perf
        frames_acc += 1
        steps_acc += steps_per_frame

        wall_now = time.perf_counter()
        cpu_now = time.process_time()
        wall_dt = wall_now - wall0
        cpu_dt = cpu_now - cpu0

        # Update HUD about 4 times/sec
        if wall_dt >= 0.25:
            fps = frames_acc / wall_dt
            steps_per_sec = steps_acc / wall_dt
            sim_days_per_sec = (steps_per_sec * dt) / 86400.0

            A = pos_ast.shape[0]
            inter_per_step = interactions_per_step(A)
            inter_per_sec = steps_per_sec * inter_per_step

            phys_ms_per_frame = (phys_time_acc / max(1, frames_acc)) * 1000.0
            # "CPU%" estimate: process time / wall time * 100 (single-process approximation)
            cpu_pct_est = (cpu_dt / wall_dt) * 100.0 if wall_dt > 0 else 0.0

            if NUMBA_AVAILABLE:
                backend_label = f"Backend: {backend.upper()}  (press T to toggle)"
            else:
                backend_label = "Backend: NUMPY (Numba not available)"

            hud_lines = [
                backend_label,
                f"Selected: {names[selected]}",
                f"Asteroids: {A}",
                (
                    f"dt: {dt/3600:.0f} h | steps/frame: {steps_per_frame} | "
                    f"warp: {steps_per_frame*dt/86400:.1f} days/frame"
                ),
                f"Sim time: {sim_time/SECONDS_PER_YEAR:.2f} years",
                "",
                f"FPS: {fps:.1f}",
                f"Physics steps/s: {steps_per_sec:.1f}",
                f"Sim days/s: {sim_days_per_sec:.2f}",
                f"Interactions/s: {inter_per_sec:,.0f}   (majors: {M*(M-1)}, ast: {A*M})",
                f"Physics ms/frame: {phys_ms_per_frame:.2f}",
                f"CPU% (est): {cpu_pct_est:.1f}",
                "",
                "Controls:",
                "  T: toggle NumPy/Numba",
                "  +/-: time warp",
                "  Mouse wheel: zoom",
                "  Drag: pan",
                "  Arrows: select major body",
            ]

            # reset accumulators
            wall0 = wall_now
            cpu0 = cpu_now
            frames_acc = 0
            steps_acc = 0
            phys_time_acc = 0.0

        # Draw the most recently computed HUD values.
        x0 = panel_rect.left + 18
        y0 = panel_rect.top + 18
        for idx, line in enumerate(hud_lines):
            surf = (
                font_big.render(line, True, WHITE)
                if idx == 0
                else font.render(line, True, WHITE)
            )
            screen.blit(surf, (x0, y0))
            y0 += 22

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
