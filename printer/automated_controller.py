import time
from typing import Callable, Optional

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


_AFTPM = 100          # ticks/mm (0.01 mm units)
_AFSTEP = 4           # 0.04 mm (printer min step)
_AF_ZFLOOR = 0        # 0.00 mm -> 0 ticks

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


    def _af_status(self, msg: str, log: bool = True) -> None:
        self._handle_status(self.status_cmd(msg), log)

    def _af_quantize(self, z_ticks: int) -> int:
        return int(round(z_ticks / _AFSTEP) * _AFSTEP)

    def _af_move_to_ticks(self, z_ticks: int) -> None:
        z_ticks = max(z_ticks, _AF_ZFLOOR)
        z_mm = z_ticks / _AFTPM
        self._exec_gcode(f"G0 Z{z_mm:.2f}", wait=True)

    def _af_capture_and_score(self) -> float:
        # Synchronize motion, settle, take still, score focus
        self._exec_gcode("M400", wait=True)
        time.sleep(0.1)  # vibration settle
        self.camera.capture_image()
        while self.camera.is_taking_image:
            time.sleep(0.01)
        img = self.camera.get_last_frame(prefer="still", wait_for_still=False)
        if img is None:
            return float("-inf")
        try:
            if ImageAnalyzer.is_black(img):
                return float("-inf")
            res = ImageAnalyzer.analyze_focus(img)
            return float(getattr(res, "focus_score", float("-inf")))
        except Exception:
            return float("-inf")

    def _af_score_at(self, zt: int, cache: dict[int, float], bounds_ok: Optional[Callable[[int], bool]] = None) -> float:
        """Quantize → bounds → cache → move → score."""
        zt = self._af_quantize(zt)
        if zt < _AF_ZFLOOR:
            return float("-inf")
        if bounds_ok and not bounds_ok(zt):
            return float("-inf")
        if zt in cache:
            return cache[zt]
        self._af_move_to_ticks(zt)
        s = self._af_capture_and_score()
        cache[zt] = s
        return s

    def _af_climb_fine(self, start: int, step_ticks: int, cache: dict[int, float],
                    bounds_ok: Optional[Callable[[int], bool]] = None, no_improve_limit: int = 2) -> tuple[int, float]:
        """
        0.04 mm hill-climb in +/- direction until 'no_improve_limit' consecutive
        non-improving steps or bounds violated. Returns (best_z, best_score).
        """
        zt = start
        best_z = start
        best_s = cache.get(start, self._af_score_at(start, cache, bounds_ok))
        no_imp = 0
        while True:
            nxt = self._af_quantize(zt + step_ticks)
            if nxt < _AF_ZFLOOR or (bounds_ok and not bounds_ok(nxt)):
                break
            s = self._af_score_at(nxt, cache, bounds_ok)
            self._af_status(
                f"[AF-Fine] 0.04mm step {'up' if step_ticks>0 else 'down'}: "
                f"Z={nxt / _AFTPM:.2f}  score={s:.3f}", False
            )
            if s > best_s + 1e-6:
                best_z, best_s = nxt, s
                zt = nxt
                no_imp = 0
            else:
                no_imp += 1
                zt = nxt
                if no_imp >= no_improve_limit:
                    break
        return best_z, best_s

    def _af_refine_around(self, center: int, cache: dict[int, float],
                        bounds_ok: Optional[Callable[[int], bool]] = None,
                        fine_step_ticks: int = _AFSTEP,
                        no_improve_limit: int = 2) -> tuple[int, float]:
        """
        Fine search both sides around 'center' using 0.04 mm steps.
        """
        up_z,   up_s   = self._af_climb_fine(center,  fine_step_ticks, cache, bounds_ok, no_improve_limit)
        down_z, down_s = self._af_climb_fine(center, -fine_step_ticks, cache, bounds_ok, no_improve_limit)
        return (up_z, up_s) if up_s >= down_s else (down_z, down_s)

    # Autofocus
    def autofocus_descent_macro(self, cmd: command) -> None:
        """
        Autofocus when starting above the sample and moving down only to find focus.
        Coarse: 0.20 mm descent steps to Z floor; Refine: 0.04 mm around best.
        """
        COARSE_STEP_MM = 0.20
        COARSE_STEP = int(round(COARSE_STEP_MM * _AFTPM))
        MAX_OFFSET_MM = 5.60
        MAX_OFFSET = int(round(MAX_OFFSET_MM * _AFTPM))
        DROP_STOP_PEAK = 100
        DROP_STOP_BASE = 100

        self._af_status(cmd.message or "Autofocus (descent) starting…", cmd.log)
        if self.pause_point(): return

        pos = self.get_position()
        start = self._af_quantize(int(round(getattr(pos, "z", 1600))))
        self._af_status(f"Start @ Z={start / _AFTPM:.2f} mm (descent expected)", cmd.log)

        # Envelope: only allow [start - MAX_OFFSET, start], clamped to floor
        def within_env(zt: int) -> bool:
            return (start - MAX_OFFSET) <= zt <= start and zt >= _AF_ZFLOOR

        scores: dict[int, float] = {}

        # Baseline
        self._af_move_to_ticks(start)
        baseline = self._af_capture_and_score()
        scores[start] = baseline
        best_z = start
        best_s = baseline
        self._af_status(f"[AF-Descent] Baseline Z={start / _AFTPM:.2f}  score={baseline:.3f}", False)

        # Coarse descent to floor
        peak_s = baseline
        peak_z = start
        steps = min(MAX_OFFSET // COARSE_STEP, (start - _AF_ZFLOOR) // COARSE_STEP)
        for k in range(1, steps + 1):
            if self.pause_point():
                self._af_status("Autofocus paused/stopped.", True); return

            target = self._af_quantize(start - k * COARSE_STEP)
            if target <= _AF_ZFLOOR:
                target = _AF_ZFLOOR

            s = self._af_score_at(target, scores, within_env)
            d_base = s - baseline
            self._af_status(f"[AF-Descent] ↓0.20mm  Z={target / _AFTPM:.2f}"
                            f"{' (FLOOR)' if target == _AF_ZFLOOR else ''}  score={s:.3f}  Δbase={d_base:+.1f}", False)

            if s > best_s: best_s, best_z = s, target
            if s > peak_s: peak_s, peak_z = s, target

            if best_z == start and (baseline - s) >= DROP_STOP_BASE:
                self._af_status(f"[AF-Descent] Early stop (baseline-drop)", True)
                break
            if (peak_s - s) >= DROP_STOP_PEAK:
                self._af_status(f"[AF-Descent] Early stop (peak-drop)", True)
                break
            if target == _AF_ZFLOOR:
                break

        # Refine around best
        if self.pause_point(): self._af_status("Autofocus paused/stopped.", True); return
        center = best_z
        local_z, local_s = self._af_refine_around(center, scores, within_env, _AFSTEP, no_improve_limit=2)
        if local_s > best_s:
            best_z, best_s = local_z, local_s

        if self.pause_point(): return
        self._af_move_to_ticks(best_z)
        self._af_status(f"Autofocus (descent) complete: Best Z={best_z / _AFTPM:.2f} mm  Score={best_s:.3f}", True)

    def fine_autofocus(self, cmd: command) -> None:
        """
        Fine autofocus around current Z (±0.16 mm) at 0.04 mm steps.
        """
        RANGE_MM = 0.16
        RANGE_TICKS = int(round(RANGE_MM * _AFTPM))  # ±16 ticks
        NO_IMPROVE_LIMIT = 1

        self._af_status(cmd.message or "Fine autofocus…", cmd.log)

        pos = self.get_position()
        center = self._af_quantize(int(round(getattr(pos, "z", 1600))))  # fallback 16.00 mm
        self._af_status(f"[AF-Fine] Center Z={center / _AFTPM:.2f} mm", False)

        # Window bounds: stay within center ± RANGE
        def within_window(zt: int) -> bool:
            return center - RANGE_TICKS <= zt <= center + RANGE_TICKS

        scores: dict[int, float] = {}
        baseline = self._af_score_at(center, scores, within_window)
        self._af_status(f"[AF-Fine] Baseline score={baseline:.3f}  Δbase={0:+.1f}", False)

        best_z, best_s = self._af_refine_around(
            center=center,
            cache=scores,
            bounds_ok=within_window,
            fine_step_ticks=_AFSTEP,
            no_improve_limit=NO_IMPROVE_LIMIT
        )

        if self.pause_point():  # graceful stop
            return

        self._af_move_to_ticks(best_z)
        self._af_status(
            f"[AF-Fine] Best Z={best_z / _AFTPM:.2f} mm  "
            f"Score={best_s:.3f}  Δbase={(best_s - baseline):.1f}",
            True
        )

    def autofocus_macro(self, cmd: command) -> None:
        """
        General autofocus: Coarse 0.40 mm alternating outward with biasing,
        then 0.20 mm refine march, then 0.04 mm fine polish around peak.
        """
        COARSE_STEP_MM = 0.40
        COARSE_STEP = int(round(COARSE_STEP_MM * _AFTPM))    # 40
        REFINE_COARSE_MM = 0.20
        REFINE_COARSE = int(round(REFINE_COARSE_MM * _AFTPM))  # 20
        MAX_OFFSET_MM = 5.60
        MAX_OFFSET = int(round(MAX_OFFSET_MM * _AFTPM))
        IMPROVE_THRESH = 50
        DROP_STOP_PEAK = 100
        DROP_STOP_BASE = 100

        self._af_status(cmd.message or "Autofocus starting…", cmd.log)
        if self.pause_point(): return

        pos = self.get_position()
        start = self._af_quantize(int(round(getattr(pos, "z", 1600))))
        self._af_status(f"Start @ Z={start / _AFTPM:.2f} mm", cmd.log)

        # Envelope: [start - MAX_OFFSET, start + MAX_OFFSET], floor clamped
        def within_env(zt: int) -> bool:
            return (start - MAX_OFFSET) <= zt <= (start + MAX_OFFSET) and zt >= _AF_ZFLOOR

        scores: dict[int, float] = {}

        # Baseline
        self._af_move_to_ticks(start)
        baseline = self._af_capture_and_score()
        scores[start] = baseline
        best_z = start
        best_s = baseline
        self._af_status(f"[AF] Baseline Z={start / _AFTPM:.2f}  score={baseline:.3f}", False)

        # Coarse alternating with bias
        k_right = 1; k_left = 1
        max_k = MAX_OFFSET // COARSE_STEP
        left_max_safe  = min(max_k, (start - _AF_ZFLOOR) // COARSE_STEP)
        right_max_safe = max_k  # no known Z max bound other than envelope
        bias_side = None   # 'right' or 'left'
        last_side = None
        peak_on_bias = baseline

        while True:
            if self.pause_point():
                self._af_status("Autofocus paused/stopped.", True); return

            right_has = k_right <= right_max_safe
            left_has  = k_left  <= left_max_safe
            if not right_has and not left_has:
                break

            # pick side
            if bias_side:
                side = bias_side if ((bias_side == 'right' and right_has) or (bias_side == 'left' and left_has)) \
                    else ('right' if right_has else 'left')
            else:
                if last_side == 'left' and right_has:   side = 'right'
                elif last_side == 'right' and left_has: side = 'left'
                elif right_has:                          side = 'right'
                else:                                    side = 'left'

            target = (start + k_right * COARSE_STEP) if side == 'right' else (start - k_left * COARSE_STEP)
            target = self._af_quantize(target)

            if side == 'left' and target < _AF_ZFLOOR:
                self._af_status("[AF-Coarse] Reached Z floor; stopping left exploration.", True)
                k_left = left_max_safe + 1
                last_side = side
                continue

            s = self._af_score_at(target, scores, within_env)
            if s > best_s: best_s, best_z = s, target

            improv = s - baseline
            self._af_status(f"[AF-Coarse] side={side:<5} Z={target / _AFTPM:.2f}  score={s:.3f}  Δbase={improv:+.1f}", False)

            # early-stop if baseline is still best and we plunged
            if best_z == start and (baseline - s) >= DROP_STOP_BASE:
                self._af_status("[AF-Coarse] Early stop (baseline-drop)", True)
                break

            # set bias when improvement threshold crossed
            if not bias_side and improv >= IMPROVE_THRESH:
                bias_side = side
                peak_on_bias = s
                self._af_status(f"[AF-Coarse] Bias → {bias_side.upper()} (≥+{IMPROVE_THRESH})", True)

            # early-stop on peak drop on bias side
            if bias_side and side == bias_side:
                if s > peak_on_bias:
                    peak_on_bias = s
                elif (peak_on_bias - s) >= DROP_STOP_PEAK:
                    self._af_status("[AF-Coarse] Early stop (peak-drop)", True)
                    break

            # advance counters
            if side == 'right': k_right += 1
            else:               k_left  += 1
            last_side = side

            # if biased and side exhausted, end coarse
            if bias_side and ((bias_side == 'right' and not (k_right <= max_k)) or
                            (bias_side == 'left'  and not (k_left  <= max_k))):
                break

        # Refine (0.20 mm marching in best direction), then fine polish
        if self.pause_point():
            self._af_status("Autofocus paused/stopped.", True); return

        # pick better of ±0.20 around best coarse
        up_zt   = self._af_quantize(best_z + REFINE_COARSE)
        down_zt = self._af_quantize(best_z - REFINE_COARSE)
        up_s    = self._af_score_at(up_zt, scores, within_env)
        down_s  = self._af_score_at(down_zt, scores, within_env)
        dir1, z1, s1 = (('up', up_zt, up_s) if up_s >= down_s else ('down', down_zt, down_s))
        self._af_status(f"[AF-Refine] Probe 0.20mm {dir1}: Z={z1 / _AFTPM:.2f}  score={s1:.3f}", False)
        if s1 > best_s: best_s, best_z = s1, z1

        # march in 0.20 until not improving
        current, prev = z1, s1
        while True:
            if self.pause_point():
                self._af_status("Autofocus paused/stopped.", True); return
            step = REFINE_COARSE if dir1 == 'up' else -REFINE_COARSE
            nxt = self._af_quantize(current + step)
            if nxt < _AF_ZFLOOR or not within_env(nxt):
                break
            s = self._af_score_at(nxt, scores, within_env)
            self._af_status(f"[AF-Refine] 0.20mm step {dir1}: Z={nxt / _AFTPM:.2f}  score={s:.3f}", False)
            if s > best_s: best_s, best_z = s, nxt
            if s + 1e-6 >= prev:
                current, prev = nxt, s
            else:
                break

        # fine hill-climb around best
        center = best_z
        local_z, local_s = self._af_refine_around(center, scores, within_env, _AFSTEP, no_improve_limit=2)
        if local_s > best_s:
            best_z, best_s = local_z, local_s

        if self.pause_point(): return
        self._af_move_to_ticks(best_z)
        self._af_status(f"Autofocus complete: Best Z={best_z / _AFTPM:.2f} mm  Score={best_s:.3f}", True)


    # Automation
    # --- Handler --------------------------------------------------------------
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


    # --- Convenience starter --------------------------------------------------
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