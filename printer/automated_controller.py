import time
from .models import Position, FocusScore
from .config import AutomationConfig
from .base_controller import BasePrinterController
from image_processing.analyzers import ImageAnalyzer
from image_processing.machine_vision import MachineVision

from forgeConfig import (
    ForgeSettings,
)

from .base_controller import command


def _scan_bounds_plotter(proc_queue, y_min: float, y_max: float):
    """
    Messages accepted:
      ("data",  y, r, g, b, ylum)        # add color sample
      ("focus", y, hard_count, soft_count) # add focus counts
      ("break",)                          # insert NaN gap in both graphs
      ("title", text)
      ("done",)   # leave windows open (blocking show)
      ("close",)  # close immediately
    """
    import time, math, matplotlib

    # ---- pick a GUI backend that exists ----
    import tkinter  # noqa: F401
    backend = "TkAgg"

    if backend is None:
        # No GUI: drain queue until done/close and exit quietly
        t0 = time.time()
        while True:
            try:
                msg = proc_queue.get(timeout=0.2)
                if isinstance(msg, tuple) and msg and msg[0] in ("done", "close"):
                    break
            except Exception:
                if time.time() - t0 > 2.0:
                    break
        return

    try:
        matplotlib.use(backend, force=True)
    except Exception:
        return

    import matplotlib.pyplot as plt

    # ---- Figure 1: Color vs Y ----
    plt.ion()
    fig1 = plt.figure(figsize=(8, 5), dpi=120)
    ax1  = fig1.add_subplot(111)
    base_title_1 = "Average Color vs Y (live)"
    ax1.set_title(base_title_1)
    ax1.set_xlabel("Y position (mm)")
    ax1.set_ylabel("Value")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(y_min, y_max)
    ys, rs, gs, bs, yls = [], [], [], [], []
    (l_r,) = ax1.plot([], [], label="R")
    (l_g,) = ax1.plot([], [], label="G")
    (l_b,) = ax1.plot([], [], label="B")
    (l_y,) = ax1.plot([], [], label="Y (luminance)")
    ax1.legend(loc="best")
    fig1.canvas.draw_idle()
    try: plt.show(block=False)
    except Exception: pass

    # ---- Figure 2: Focus counts vs Y ----
    fig2 = plt.figure(figsize=(8, 5), dpi=120)
    ax2  = fig2.add_subplot(111)
    base_title_2 = "Focus Tiles vs Y (live)"
    ax2.set_title(base_title_2)
    ax2.set_xlabel("Y position (mm)")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(y_min, y_max)
    ys_f, hard_counts, soft_counts = [], [], []
    (l_hard,) = ax2.plot([], [], label="Hard (>= min_score)")
    (l_soft,) = ax2.plot([], [], label="Soft (>= soft_min_score)")
    ax2.legend(loc="best")
    fig2.canvas.draw_idle()
    try: plt.show(block=False)
    except Exception: pass

    last_elapsed = None  # seconds (float)

    running = True
    while running:
        try:
            msg = proc_queue.get(timeout=0.05)
        except Exception:
            msg = None

        if msg:
            tag = msg[0]
            if tag == "data":
                _, y, r, g, b, ylum = msg
                ys.append(float(y)); rs.append(float(r)); gs.append(float(g)); bs.append(float(b)); yls.append(float(ylum))
                l_r.set_data(ys, rs); l_g.set_data(ys, gs); l_b.set_data(ys, bs); l_y.set_data(ys, yls)
                ax1.relim(); ax1.autoscale_view(scalex=False, scaley=True)

            elif tag == "focus":
                _, y, h, s = msg
                ys_f.append(float(y)); hard_counts.append(int(h)); soft_counts.append(int(s))
                l_hard.set_data(ys_f, hard_counts); l_soft.set_data(ys_f, soft_counts)
                ax2.relim(); ax2.autoscale_view(scalex=False, scaley=True)

            elif tag == "break":
                # Insert NaNs to create a visual gap (no connecting line)
                nan = math.nan
                ys.append(nan); rs.append(nan); gs.append(nan); bs.append(nan); yls.append(nan)
                ys_f.append(nan); hard_counts.append(nan); soft_counts.append(nan)

            elif tag == "title":
                # allow setting a new base title text for fig1 if you want
                base_title_1 = str(msg[1]) or base_title_1
                # re-apply elapsed if we have one
                if last_elapsed is not None:
                    ax1.set_title(f"{base_title_1}   (t={last_elapsed:.1f}s)")
                else:
                    ax1.set_title(base_title_1)

            elif tag == "elapsed":
                # NEW: show elapsed seconds on both figure titles
                _, secs = msg
                last_elapsed = float(secs)
                ax1.set_title(f"{base_title_1}   (t={last_elapsed:.1f}s)")
                ax2.set_title(f"{base_title_2}   (t={last_elapsed:.1f}s)")
            elif tag == "close":
                running = False

            elif tag == "done":
                try:
                    plt.ioff()
                    plt.show()
                except Exception:
                    pass
                running = False

        # draw frames
        try:
            fig1.canvas.draw_idle()
            fig2.canvas.draw_idle()
            plt.pause(0.01)
        except Exception:
            break

    try:
        plt.close(fig1); plt.close(fig2)
    except Exception:
        pass


class AutomatedPrinter(BasePrinterController):
    """Extended printer controller with automation capabilities"""
    def __init__(self, forgeConfig: ForgeSettings, automation_config: AutomationConfig, camera):
        super().__init__(forgeConfig)
        self.automation_config = automation_config
        self.camera = camera
        self.machine_vision = MachineVision(camera, tile_size=48, stride=48, top_percent=0.15, min_score=50.0, soft_min_score=35.0)
        self.is_automated = False

        # Autofocus
        self.register_handler("AUTOFOCUS_DESCENT", self.autofocus_descent_macro)
        self.register_handler("AUTOFOCUS", self.autofocus_macro)
        self.register_handler("FINE_AUTOFOCUS", self.fine_autofocus)
        
        # Automation Routines
        self.register_handler("SCAN_SAMPLE_BOUNDS", self.scan_sample_bounds)


    # Autofocus
    def autofocus_descent_macro(self, cmd: command) -> None:
        """
        Autofocus (known descent): start above the sample and move *down only* to find focus.
        Strategy:
        - Coarse descent in 0.20 mm steps *until Z floor (0.00 mm)*.
        - Fine polish in 0.04 mm steps around the coarse peak (both directions, clamped).
        Units are ticks (0.01 mm). Printer min step = 4 ticks (0.04 mm).
        """

        # ---------------------- constants ---------------------------------------
        Z_MIN_MM     = 0.00
        TICKS_PER_MM = 100                 # 0.01 mm
        STEP_TICKS   = 4                   # 0.04 mm
        MAX_OFFSET_MM     = 5.60
        MAX_OFFSET_TICKS  = int(round(MAX_OFFSET_MM * TICKS_PER_MM))  # 560
        Z_MIN_TICKS  = int(round(Z_MIN_MM * TICKS_PER_MM))  # -> 0

        # Coarse descent
        COARSE_STEP_MM    = 0.20
        COARSE_STEP_TICKS = int(round(COARSE_STEP_MM * TICKS_PER_MM))  # 20

        # Optional early-stop guards (kept as-is unless you want them disabled)
        DROP_STOP_PEAK_THRESHOLD = 100   # stop if we fall ≥ this from the best-so-far while descending
        DROP_STOP_BASE_THRESHOLD = 100   # if baseline is best and we drop far below it, stop

        # Refine pass
        REFINE_FINE_STEP_TICKS   = STEP_TICKS    # 0.04 mm
        REFINE_NO_IMPROVE_LIMIT  = 2

        # ---------------------- helpers -----------------------------------------
        def status(msg: str, log: bool = True) -> None:
            self._handle_status(self.status_cmd(msg), log)

        def quantize_ticks(z_ticks: int) -> int:
            return int(round(z_ticks / STEP_TICKS) * STEP_TICKS)

        def move_to_ticks(z_ticks: int) -> None:
            z_ticks = max(z_ticks, Z_MIN_TICKS)  # clamp to floor
            z_mm = z_ticks / TICKS_PER_MM
            self._exec_gcode(f"G0 Z{z_mm:.2f}", wait=True)

        def capture_and_score() -> float:
            self._exec_gcode("M400", wait=True)
            time.sleep(0.1)  # vibration settle
            self.camera.capture_image()
            while self.camera.is_taking_image:
                time.sleep(0.01)
            image = self.camera.get_last_frame(prefer="still", wait_for_still=False)
            if image is None:
                return float("-inf")
            try:
                if ImageAnalyzer.is_black(image):
                    return float("-inf")
                res = ImageAnalyzer.analyze_focus(image)
                return float(getattr(res, "focus_score", float("-inf")))
            except Exception:
                return float("-inf")

        # envelope: only allow start down to start - MAX_OFFSET (bounded by Z_MIN)
        def within_envelope(zt: int) -> bool:
            return (start_ticks - MAX_OFFSET_TICKS) <= zt <= start_ticks and zt >= Z_MIN_TICKS

        # Score cache to avoid re-imaging the same Z
        scores = {}  # z_ticks -> score
        def score_at(zt: int) -> float:
            zt = quantize_ticks(zt)
            if zt < Z_MIN_TICKS or not within_envelope(zt):
                return float("-inf")
            if zt in scores:
                return scores[zt]
            move_to_ticks(zt)
            s = capture_and_score()
            scores[zt] = s
            return s

        # ---------------------- start / baseline --------------------------------
        status(cmd.message or "Autofocus (descent) starting…", cmd.log)

        if self.pause_point():
            return

        pos = self.get_position()
        start_ticks = quantize_ticks(int(round(getattr(pos, "z", 1600))))  # default 16.00 mm
        status(f"Start @ Z={start_ticks / TICKS_PER_MM:.2f} mm (descent expected)", cmd.log)

        move_to_ticks(start_ticks)
        baseline = capture_and_score()
        scores[start_ticks] = baseline
        best_ticks = start_ticks
        best_score = baseline
        status(f"[AF-Descent] Baseline Z={start_ticks / TICKS_PER_MM:.2f}  score={baseline:.3f}", False)

        # ---------------------- COARSE: down-only until floor --------------------
        if self.pause_point():
            status("Autofocus paused/stopped.", True); return

        max_k_down = min(MAX_OFFSET_TICKS // COARSE_STEP_TICKS,
                        (start_ticks - Z_MIN_TICKS) // COARSE_STEP_TICKS)

        peak_score = baseline
        peak_ticks = start_ticks

        for k in range(1, max_k_down + 1):
            if self.pause_point():
                status("Autofocus paused/stopped.", True); return

            target = quantize_ticks(start_ticks - k * COARSE_STEP_TICKS)

            # If the next step reaches or crosses the floor, take a final measurement at the floor then stop.
            if target <= Z_MIN_TICKS:
                target = Z_MIN_TICKS
                s = score_at(target)
                delta_base = s - baseline
                status(f"[AF-Descent] ↓0.20mm  Z={target / TICKS_PER_MM:.2f} (FLOOR)  score={s:.3f}  Δbase={delta_base:+.1f}", True)
                if s > best_score:
                    best_score, best_ticks = s, target
                if s > peak_score:
                    peak_score, peak_ticks = s, target
                break

            s = score_at(target)
            delta_base = s - baseline
            status(f"[AF-Descent] ↓0.20mm  Z={target / TICKS_PER_MM:.2f}  score={s:.3f}  Δbase={delta_base:+.1f}", False)

            # Track best/peak
            if s > best_score:
                best_score, best_ticks = s, target
            if s > peak_score:
                peak_score, peak_ticks = s, target

            # Optional guards (leave enabled/disable as you prefer)
            if best_ticks == start_ticks:
                base_drop = baseline - s
                if base_drop >= DROP_STOP_BASE_THRESHOLD:
                    status(f"[AF-Descent] Early stop (baseline-drop): {base_drop:.1f} below baseline "
                        f"({baseline:.1f} → {s:.1f})", True)
                    break

            peak_drop = peak_score - s
            if peak_drop >= DROP_STOP_PEAK_THRESHOLD:
                status(f"[AF-Descent] Early stop (peak-drop): {peak_drop:.1f} from peak "
                    f"({peak_score:.1f} → {s:.1f})", True)
                break

        # Use the best coarse point as center for fine search
        center = best_ticks

        # ---------------------- REFINE: fine polish around peak ------------------
        if self.pause_point():
            status("Autofocus paused/stopped.", True); return

        def climb_one_side(start: int, step_ticks: int) -> tuple[int, float]:
            """
            Move from start in +/-step_ticks increments until score worsens N times.
            Clamped to Z_MIN and descent envelope (allows small 'up' around peak).
            """
            zt = start
            best_local_z = start
            best_local_s = scores.get(start, score_at(start))
            no_imp = 0
            while True:
                if self.pause_point():
                    status("Autofocus paused/stopped.", True); return best_local_z, best_local_s
                nxt = quantize_ticks(zt + step_ticks)
                if nxt < Z_MIN_TICKS:
                    status("[AF-Descent] Z floor hit; stopping in this direction.", False)
                    break
                # allow slight 'up' from center if still ≤ start and within global envelope
                if not within_envelope(min(nxt, start)):
                    break
                s = score_at(nxt)
                status(f"[AF-Descent] 0.04mm {'up' if step_ticks>0 else 'down'}: Z={nxt / TICKS_PER_MM:.2f}  score={s:.3f}", False)
                if s > best_local_s + 1e-6:
                    best_local_z, best_local_s = nxt, s
                    zt = nxt
                    no_imp = 0
                else:
                    no_imp += 1
                    zt = nxt
                    if no_imp >= REFINE_NO_IMPROVE_LIMIT:
                        break
            return best_local_z, best_local_s

        # Explore finely both sides around center
        up_z, up_s     = climb_one_side(center,  REFINE_FINE_STEP_TICKS)
        down_z, down_s = climb_one_side(center, -REFINE_FINE_STEP_TICKS)

        local_best_z, local_best_s = (up_z, up_s) if up_s >= down_s else (down_z, down_s)
        if local_best_s > best_score:
            best_ticks, best_score = local_best_z, local_best_s

        # ---------------------- finalize ----------------------------------------
        if self.pause_point():
            return
        move_to_ticks(best_ticks)
        status(f"Autofocus (descent) complete: Best Z={best_ticks / TICKS_PER_MM:.2f} mm  Score={best_score:.3f}", True)

    def fine_autofocus(self, cmd: command) -> None:
        """
        Fine autofocus around current Z only (±0.16 mm) at 0.04 mm steps.
        Designed for quick re-focus when drift is small and starting point is near-best.
        """

        # ---- constants ----
        TICKS_PER_MM = 100     # 0.01 mm units
        STEP_TICKS   = 4       # 0.04 mm (printer min step)
        RANGE_MM     = 0.16
        RANGE_TICKS  = int(round(RANGE_MM * TICKS_PER_MM))    # 16 ticks
        MAX_STEPS    = RANGE_TICKS // STEP_TICKS              # 4 steps per side
        NO_IMPROVE_LIMIT = 1   # stop after this many non-improving steps per side

        # ---- helpers ----
        def status(msg: str, log: bool = True) -> None:
            self._handle_status(self.status_cmd(msg), log)

        def quantize_ticks(z_ticks: int) -> int:
            return int(round(z_ticks / STEP_TICKS) * STEP_TICKS)

        def within_window(zt: int, center: int) -> bool:
            return center - RANGE_TICKS <= zt <= center + RANGE_TICKS

        def move_to_ticks(z_ticks: int) -> None:
            z_mm = z_ticks / TICKS_PER_MM
            self._exec_gcode(f"G0 Z{z_mm:.2f}", wait=True)

        def capture_and_score() -> float:
            self._exec_gcode("M400", wait=True)
            time.sleep(0.1)  # vibration settle
            self.camera.capture_image()
            while self.camera.is_taking_image:
                time.sleep(0.01)
            image = self.camera.get_last_frame(prefer="still", wait_for_still=False)
            if image is None:
                return float("-inf")
            try:
                if ImageAnalyzer.is_black(image):
                    return float("-inf")
                res = ImageAnalyzer.analyze_focus(image)
                return float(getattr(res, "focus_score", float("-inf")))
            except Exception:
                return float("-inf")

        scores = {}  # cache: z_ticks -> score
        def score_at(zt: int) -> float:
            zt = quantize_ticks(zt)
            if zt in scores:
                return scores[zt]
            move_to_ticks(zt)
            s = capture_and_score()
            scores[zt] = s
            return s

        # ---- start: current Z as center ----
        status(cmd.message or "Fine autofocus…", cmd.log)

        pos = self.get_position()
        center = quantize_ticks(int(round(getattr(pos, "z", 1600))))  # fallback 16.00 mm
        status(f"[AF-Fine] Center Z={center / TICKS_PER_MM:.2f} mm", False)

        # Score center (baseline)
        best_z = center
        baseline_s = score_at(center)
        best_s = baseline_s
        status(f"[AF-Fine] Baseline score={baseline_s:.3f}  Δbase={0:+.1f}", False)

        # ---- climb one side helper ----
        def climb_one_side(start: int, step_sign: int) -> tuple[int, float]:
            """Hill-climb from start in ±0.04 mm steps up to MAX_STEPS within window."""
            zt = start
            best_local_z = start
            best_local_s = scores[zt]
            no_improve = 0
            for _ in range(MAX_STEPS):
                if self.pause_point():
                    status("Fine autofocus paused/stopped.", True)
                    return best_local_z, best_local_s

                nxt = quantize_ticks(zt + step_sign * STEP_TICKS)
                if not within_window(nxt, center):
                    break

                s = score_at(nxt)
                d = s - baseline_s
                status(
                    f"[AF-Fine] step {'+' if step_sign>0 else '-'}0.04: "
                    f"Z={nxt / TICKS_PER_MM:.2f}  score={s:.3f}  Δbase={d:+.1f}",
                    False
                )

                if s > best_local_s + 1e-6:
                    best_local_z, best_local_s = nxt, s
                    zt = nxt
                    no_improve = 0
                else:
                    no_improve += 1
                    zt = nxt
                    if no_improve > NO_IMPROVE_LIMIT:
                        break
            return best_local_z, best_local_s

        # ---- search both sides (down and up) ----
        down_z, down_s = climb_one_side(center, -1)
        up_z,   up_s   = climb_one_side(center,  1)

        if up_s >= down_s:
            best_z, best_s = up_z, up_s
        else:
            best_z, best_s = down_z, down_s

        # ---- finalize ----
        if self.pause_point():
            return
        move_to_ticks(best_z)
        status(
            f"[AF-Fine] Best Z={best_z / TICKS_PER_MM:.2f} mm  "
            f"Score={best_s:.3f}  Δbase={(best_s - baseline_s):.1f}",
            True
        )

    def autofocus_macro(self, cmd: command) -> None:
        """
        Autofocus with adaptive coarse→refine:
        - Coarse (0.40 mm): alternate outward; once a side hits ≥ +50 vs baseline, skip the other side and continue only on the biased side.
        - Refine: 0.20 mm hill-climb until improvement stalls, then 0.04 mm hill-climb to peak.
        Units are ticks (0.01 mm). Printer min step = 4 ticks (0.04 mm).
        """

        # ---------------------- constants ---------------------------------------
        Z_MIN_MM    = 0.00
        TICKS_PER_MM = 100               # 0.01 mm
        STEP_TICKS   = 4                 # 0.04 mm
        MAX_OFFSET_MM     = 5.60
        MAX_OFFSET_TICKS  = int(round(MAX_OFFSET_MM * TICKS_PER_MM))  # 560
        Z_MIN_TICKS = int(round(Z_MIN_MM * TICKS_PER_MM))  # -> 0

        # Coarse pass
        COARSE_STEP_MM    = 0.40
        COARSE_STEP_TICKS = int(round(COARSE_STEP_MM * TICKS_PER_MM)) # 40
        IMPROVEMENT_THRESHOLD = 50       # bias trigger
        DROP_STOP_PEAK_THRESHOLD = 100    # early-stop if drop ≥ this from the biased-side peak
        DROP_STOP_BASE_THRESHOLD = 100    # early-stop if drop ≥ this below baseline when baseline is still best

        # Refine pass (adaptive)
        REFINE_COARSE_STEP_MM    = 0.20
        REFINE_COARSE_STEP_TICKS = int(round(REFINE_COARSE_STEP_MM * TICKS_PER_MM))  # 20
        REFINE_FINE_STEP_TICKS   = STEP_TICKS  # 0.04 mm
        REFINE_NO_IMPROVE_LIMIT  = 2           # stop after N non-improving steps per fine side

        # ---------------------- helpers -----------------------------------------
        def status(msg: str, log: bool = True) -> None:
            self._handle_status(self.status_cmd(msg), log)

        def quantize_ticks(z_ticks: int) -> int:
            return int(round(z_ticks / STEP_TICKS) * STEP_TICKS)

        def move_to_ticks(z_ticks: int) -> None:
            z_ticks = max(z_ticks, Z_MIN_TICKS)  # clamp to floor
            z_mm = z_ticks / TICKS_PER_MM
            self._exec_gcode(f"G0 Z{z_mm:.2f}", wait=True)

        def capture_and_score() -> float:
            self._exec_gcode("M400", wait=True)
            time.sleep(0.1)  # vibration settle
            self.camera.capture_image()
            while self.camera.is_taking_image:
                time.sleep(0.01)
            image = self.camera.get_last_frame(prefer="still", wait_for_still=False)
            if image is None:
                return float("-inf")
            try:
                if ImageAnalyzer.is_black(image):
                    return float("-inf")
                res = ImageAnalyzer.analyze_focus(image)
                return float(res.focus_score)
            except Exception:
                return float("-inf")

        def within_envelope(zt: int) -> bool:
            # existing start±MAX_OFFSET bounds…
            return (start_ticks - MAX_OFFSET_TICKS) <= zt <= (start_ticks + MAX_OFFSET_TICKS) and zt >= Z_MIN_TICKS


        # Score cache to avoid re-imaging same Z
        scores = {}  # z_ticks -> score
        def score_at(zt: int) -> float:
            zt = quantize_ticks(zt)
            if zt < Z_MIN_TICKS or not within_envelope(zt):
                return float("-inf")
            if zt in scores:
                return scores[zt]
            move_to_ticks(zt)
            s = capture_and_score()
            scores[zt] = s
            return s

        # ---------------------- start / baseline --------------------------------
        status(cmd.message or "Autofocus starting…", cmd.log)

        if self.pause_point():
            return

        pos = self.get_position()
        start_ticks = quantize_ticks(int(round(getattr(pos, "z", 1600))))  # default 16.00 mm
        status(f"Start @ Z={start_ticks / TICKS_PER_MM:.2f} mm", cmd.log)

        move_to_ticks(start_ticks)
        baseline = capture_and_score()
        scores[start_ticks] = baseline
        best_ticks = start_ticks
        best_score = baseline
        status(f"[AF] Baseline Z={start_ticks / TICKS_PER_MM:.2f}  score={baseline:.3f}", False)

        # ---------------------- COARSE: alternating, then biased ----------------
        k_right = 1
        k_left  = 1
        max_k   = MAX_OFFSET_TICKS // COARSE_STEP_TICKS  # 14 steps per side (±5.6mm on 0.40mm grid)
        left_max_k_safe  = min(max_k, (start_ticks - Z_MIN_TICKS) // COARSE_STEP_TICKS)
        right_max_k_safe = max_k  # no known Z_MAX; keep your existing envelope for the + side
        bias_side = None               # 'right' or 'left'
        last_side = None
        peak_score_on_bias = baseline  # running peak after bias is set
        early_drop_armed = False       # becomes True once bias is set
        peak_score_on_bias = baseline   # running peak on the biased side once bias is set
        
        # Alternate until bias triggers; after that, probe only on bias side
        while True:
            if self.pause_point():
                status("Autofocus paused/stopped.", True); return

            right_has = k_right <= right_max_k_safe
            left_has  = k_left  <= left_max_k_safe
            if not right_has and not left_has:
                break

            # pick side (alternate until bias; once biased, only probe that side)
            if bias_side:
                side = bias_side if ((bias_side == 'right' and right_has) or (bias_side == 'left' and left_has)) \
                    else ('right' if right_has else 'left')
            else:
                if last_side == 'left' and right_has:
                    side = 'right'
                elif last_side == 'right' and left_has:
                    side = 'left'
                elif right_has:
                    side = 'right'
                elif left_has:
                    side = 'left'
                else:
                    break

            # target Z for this coarse step
            if side == 'right':
                target = quantize_ticks(start_ticks + k_right * COARSE_STEP_TICKS)
            else:
                target = quantize_ticks(start_ticks - k_left * COARSE_STEP_TICKS)
                if target < Z_MIN_TICKS:
                    status("[AF-Coarse] Reached Z floor (0.00 mm); stopping left exploration.", True)
                    # Exhaust the left side so we won't pick it again
                    k_left = left_max_k_safe + 1
                    last_side = side
                    continue

            # take a measurement
            s = score_at(target)
            if s > best_score:
                best_score, best_ticks = s, target

            improv = s - baseline
            status(f"[AF-Coarse] side={side:<5} Z={target / TICKS_PER_MM:.2f}  score={s:.3f}  Δbase={improv:+.1f}", False)

            # 1) Early-stop on baseline drop: if we started already in focus,
            #    Δbase will plunge; if baseline is still the best-so-far, stop now.
            if best_ticks == start_ticks:
                base_drop = baseline - s  # positive when current is worse than baseline
                if base_drop >= DROP_STOP_BASE_THRESHOLD:
                    status(f"[AF-Coarse] Early stop (baseline-drop): score fell {base_drop:.1f} below baseline "
                        f"({baseline:.1f} → {s:.1f})", True)
                    break

            # 2) Usual bias logic (unchanged)
            if not bias_side and improv >= IMPROVEMENT_THRESHOLD:
                bias_side = side
                peak_score_on_bias = s
                status(f"[AF-Coarse] Bias → {bias_side.upper()} (≥+{IMPROVEMENT_THRESHOLD})", True)

            # 3) Early-stop on peak drop (only on biased side)
            if bias_side and side == bias_side:
                if s > peak_score_on_bias:
                    peak_score_on_bias = s
                else:
                    peak_drop = peak_score_on_bias - s
                    if peak_drop >= DROP_STOP_PEAK_THRESHOLD:
                        status(f"[AF-Coarse] Early stop (peak-drop): score fell {peak_drop:.1f} from peak "
                            f"({peak_score_on_bias:.1f} → {s:.1f})", True)
                        break

            # advance counters
            if side == 'right':
                k_right += 1
            else:
                k_left += 1
            last_side = side

            # if biased and that side is exhausted, end coarse
            if bias_side and ((bias_side == 'right' and not (k_right <= max_k)) or
                            (bias_side == 'left'  and not (k_left  <= max_k))):
                break

        # ---------------------- REFINE: adaptive hill-climb ---------------------
        if self.pause_point():
            status("Autofocus paused/stopped.", True); return

        # 1) Decide initial direction: compare ±0.20 mm around the best coarse Z
        def try_dir_step(center: int, delta_ticks: int) -> tuple[str, int, float]:
            """Return (dir, zt, score) for a candidate step; dir in {'up','down'}."""
            if delta_ticks == 0:
                return ('none', center, scores.get(center, float('-inf')))
            up_zt = quantize_ticks(center + delta_ticks)
            down_zt = quantize_ticks(center - delta_ticks)
            up_s   = score_at(up_zt)
            down_s = score_at(down_zt)
            return ('up', up_zt, up_s) if up_s >= down_s else ('down', down_zt, down_s)

        # choose coarse refine direction
        dir1, z1, s1 = try_dir_step(best_ticks, REFINE_COARSE_STEP_TICKS)
        status(f"[AF-Refine] Probe 0.20mm {dir1}: Z={z1 / TICKS_PER_MM:.2f}  score={s1:.3f}", False)
        if s1 > best_score:
            best_score, best_ticks = s1, z1

        # 2) March in 0.20 mm steps while improving
        current_ticks = z1
        prev_score = s1
        while True:
            if self.pause_point():
                status("Autofocus paused/stopped.", True); return
            step = REFINE_COARSE_STEP_TICKS if dir1 == 'up' else -REFINE_COARSE_STEP_TICKS
            nxt = quantize_ticks(current_ticks + step)
            if nxt < Z_MIN_TICKS:
                status("[AF-Refine] Z floor hit; stopping in this direction.", False)
                break
            if not within_envelope(nxt):
                break
            s = score_at(nxt)
            status(f"[AF-Refine] 0.20mm step {dir1}: Z={nxt / TICKS_PER_MM:.2f}  score={s:.3f}", False)
            if s > best_score:
                best_score, best_ticks = s, nxt
            if s + 1e-6 >= prev_score:  # still improving or equal
                current_ticks, prev_score = nxt, s
                continue
            else:
                # improvement stalled; prepare to fine search around previous point
                break

        # 3) Fine hill-climb at 0.04 mm around the best-so-far
        center = best_ticks
        def climb_one_side(start: int, step_ticks: int) -> tuple[int, float]:
            """Move from start in +step_ticks increments until score worsens N times."""
            zt = start
            best_local_z = start
            best_local_s = scores.get(start, score_at(start))
            no_improve = 0
            while True:
                if self.pause_point():
                    status("Autofocus paused/stopped.", True); return best_local_z, best_local_s
                nxt = quantize_ticks(zt + step_ticks)
                if nxt < Z_MIN_TICKS:
                    status("[AF-Refine] Z floor hit; stopping in this direction.", False)
                    break
                if not within_envelope(nxt):
                    break
                s = score_at(nxt)
                status(f"[AF-Refine] 0.04mm step {'up' if step_ticks>0 else 'down'}: Z={nxt / TICKS_PER_MM:.2f}  score={s:.3f}", False)
                if s > best_local_s + 1e-6:
                    best_local_z, best_local_s = nxt, s
                    zt = nxt
                    no_improve = 0
                else:
                    no_improve += 1
                    zt = nxt
                    if no_improve >= REFINE_NO_IMPROVE_LIMIT:
                        break
            return best_local_z, best_local_s

        # Explore both sides finely from the center; keep the best
        up_z, up_s     = climb_one_side(center,  REFINE_FINE_STEP_TICKS)
        down_z, down_s = climb_one_side(center, -REFINE_FINE_STEP_TICKS)
        if up_s >= down_s:
            center, local_best_s = up_z, up_s
        else:
            center, local_best_s = down_z, down_s

        if local_best_s > best_score:
            best_score, best_ticks = local_best_s, center

        # ---------------------- finalize ----------------------------------------
        if self.pause_point():
            return
        move_to_ticks(best_ticks)
        status(f"Autofocus complete: Best Z={best_ticks / TICKS_PER_MM:.2f} mm  Score={best_score:.3f}", True)


    # Automation
    # --- 1) Handler --------------------------------------------------------------
    def scan_sample_bounds(self, cmd: command) -> None:
        STEP_MM  = 2.00
        Y_MAX_MM = 224.0
        Y_MIN_MM = 40.0

        def report(msg: str, log: bool = True) -> None:
            self._handle_status(self.status_cmd(msg), log)

        # --- start plotter process (spawn-safe) ---
        plot_ok    = [False]
        plot_queue = None
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            plot_queue = ctx.Queue()
            plot_proc  = ctx.Process(target=_scan_bounds_plotter, args=(plot_queue, Y_MIN_MM, Y_MAX_MM), daemon=True)
            plot_proc.start()
            plot_ok[0] = True
            plot_queue.put(("title", "Average Color vs Y (live)"))
        except Exception as e:
            report(f"[SCAN_SAMPLE_BOUNDS] Live plot process unavailable: {e}")

        def send_data(y_now: float, r: float, g: float, b: float, ylum: float, hard_ct: int, soft_ct: int) -> None:
            if plot_ok[0] and plot_queue is not None:
                try:
                    plot_queue.put(("data",  float(y_now), float(r), float(g), float(b), float(ylum)))
                    plot_queue.put(("focus", float(y_now), int(hard_ct), int(soft_ct)))
                except Exception:
                    plot_ok[0] = False


        def send_break():
            if plot_ok[0] and plot_queue is not None:
                try:
                    plot_queue.put(("break",))
                except Exception:
                    plot_ok[0] = False

        # --- capture start Y ---
        start_y = float(self.position.y) / 100
        start_time = time.time()

        report(f"[SCAN_SAMPLE_BOUNDS] Start @ Y={start_y:.3f} mm")
        self.pause_point()

        # 1) Autofocus from above
        report("[SCAN_SAMPLE_BOUNDS] Running autofocus_descent_macro…")
        self.autofocus_descent_macro(cmd)
        self.pause_point()

        def send_elapsed():
            if plot_ok[0] and plot_queue is not None:
                try:
                    plot_queue.put(("elapsed", time.time() - start_time))
                except Exception:
                    pass

        # --- measurement helper: color + focus counts ---
        def refine_and_measure(y_now: float) -> None:
            """
            Measure at current Y:
            - If focus is too weak (hard<10 and soft<15), skip fine_autofocus.
            - Otherwise run fine_autofocus, then measure.
            - Stream color + focus counts to the live plots.
            """
            try:
                # Pre-check focus to decide whether to run fine AF
                pre = self.machine_vision.compute_focused_tiles()
                pre_hard = len(pre.get("hard", []))
                pre_soft = len(pre.get("soft", []))

                run_fine = not (pre_hard < 10 and pre_soft < 15)
                if run_fine:
                    self.fine_autofocus(cmd)
                else:
                    time.sleep(0.2)  # wait when skipping fine autofocus

                # Read color
                r, g, b, ylum = self.machine_vision.get_average_color()

                # Compute focus tiles for reporting after optional fine AF
                all_tiles = self.machine_vision.compute_focused_tiles(filter_invalid=True)
                hard_tiles = len(all_tiles.get("hard", []))
                soft_tiles = len(all_tiles.get("soft", []))

                report(f"[SCAN_SAMPLE_BOUNDS] Y={y_now:.3f} "
                    f"→ Avg(R,G,B,Y)=({r:.1f},{g:.1f},{b:.1f},{ylum:.3f})  "
                    f"Focus: hard={hard_tiles} soft={soft_tiles}  "
                    f"{'(fine AF skipped)' if not run_fine else ''}")

                # Stream to both graphs
                send_data(y_now, r, g, b, ylum, hard_tiles, soft_tiles)
                send_elapsed()

            except Exception as e:
                report(f"[SCAN_SAMPLE_BOUNDS] Y={y_now:.3f} → measurement failed: {e}")

        # Make first measurement at start
        refine_and_measure(start_y)

        # 2) Sweep +Y
        y = start_y
        while y < Y_MAX_MM - 1e-9:
            y = min(y + STEP_MM, Y_MAX_MM)
            self._exec_gcode(f"G0 Y{y:.3f}")
            self.pause_point()
            refine_and_measure(y)

        # 3) Return to start **without drawing a connecting line**
        if abs(y - start_y) > 1e-9:
            send_break()  # prevents the line from connecting the last +Y point to start
            self._exec_gcode(f"G0 Y{start_y:.3f}")
            self.pause_point()
            report("[SCAN_SAMPLE_BOUNDS] Running autofocus_macro at start position…")
            self.autofocus_macro(cmd)  # <-- run the full autofocus here (not descent, not fine)
        # Measure at start after autofocus_macro (with skip logic inside refine_and_measure)
        refine_and_measure(start_y)

        # 4) Sweep -Y
        y = start_y
        while y > Y_MIN_MM + 1e-9:
            y = max(y - STEP_MM, Y_MIN_MM)
            self._exec_gcode(f"G0 Y{y:.3f}")
            self.pause_point()
            refine_and_measure(y)

        total_time = time.time() - start_time
        report(f"[SCAN_SAMPLE_BOUNDS] Scan complete. Total time: {total_time:.2f} seconds")
        send_elapsed()  # final elapsed push so titles show the final time

        if plot_ok[0] and plot_queue is not None:
            try:
                plot_queue.put(("done",))
            except Exception:
                pass


    # --- 3) Convenience starter --------------------------------------------------
    def start_scan_sample_bounds(self) -> None:
        """
        Enqueue SCAN_SAMPLE_BOUNDS as a single command (macro-style handler).
        """
        self.enqueue_cmd(command(kind="SCAN_SAMPLE_BOUNDS", value=0, message="Scan sample bounds", log=True))



    def start_autofocus(self) -> None:
        """Start the automation process"""

        self.reset_after_stop()
        
        # Enqueue the macro like any other command
        self.enqueue_cmd(command(
            kind="AUTOFOCUS",
            value="",
            message= "Beginning Autofocus Macro",
            log=True
        ))

    def start_fine_autofocus(self) -> None:
        """Start the automation process"""

        self.reset_after_stop()
        
        # Enqueue the macro like any other command
        self.enqueue_cmd(command(
            kind="FINE_AUTOFOCUS",
            value="",
            message= "Beginning Fine Autofocus Macro",
            log=True
        ))

    def start_automation(self) -> None:
        """Start the automation process"""

        self.reset_after_stop()
        
        # Enqueue the macro like any other command
        self.enqueue_cmd(command(
            kind="SCAN_SAMPLE_BOUNDS",
            value="",
            message= "Scan sample bounds",
            log=True
        ))

    def setPosition1(self) -> None:
        self.automation_config.x_start = self.position.x
        self.automation_config.y_start = self.position.y
        self.automation_config.z_start = self.position.z

    def setPosition2(self) -> None:
        self.automation_config.x_end = self.position.x
        self.automation_config.y_end = self.position.y
        self.automation_config.z_end = self.position.z

    def _get_range(self, start: int, end: int, step: int) -> range:
        """Get appropriate range based on start and end positions"""
        if start < end:
            return range(start, end + step, step)
        return range(start, end - step, -step)