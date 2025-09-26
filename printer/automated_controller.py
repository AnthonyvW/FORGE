import time
from .models import Position, FocusScore
from .config import AutomationConfig
from .base_controller import BasePrinterController
from image_processing.analyzers import ImageAnalyzer

from forgeConfig import (
    ForgeSettings,
)

from .base_controller import command

class AutomatedPrinter(BasePrinterController):
    """Extended printer controller with automation capabilities"""
    def __init__(self, forgeConfig: ForgeSettings, automation_config: AutomationConfig, camera):
        super().__init__(forgeConfig)
        self.automation_config = automation_config
        self.camera = camera
        self.is_automated = False

        self.register_handler("AUTOFOCUS", self.autofocus_macro)
        self.register_handler("FINE_AUTOFOCUS", self.fine_autofocus)



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
            image = getattr(self.camera, "get_last_image", lambda: None)()
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

        # Score center
        best_z = center
        best_s = score_at(center)
        status(f"[AF-Fine] Baseline score={best_s:.3f}", False)

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
                status(f"[AF-Fine] step {'+' if step_sign>0 else '-'}0.04: Z={nxt / TICKS_PER_MM:.2f}  score={s:.3f}", False)
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
        # Try one step each way first to choose direction, but still explore both with early-stop
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
        status(f"[AF-Fine] Best Z={best_z / TICKS_PER_MM:.2f} mm  Score={best_s:.3f}", True)



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
            image = getattr(self.camera, "get_last_image", lambda: None)()
            if image is None:
                return float("-inf")
            try:
                if ImageAnalyzer.is_black(image):
                    return float("-inf")
                res = ImageAnalyzer.analyze_focus(image)
                return float(getattr(res, "focus_score", float("-inf")))
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
            kind="AUTOFOCUS",
            value="",
            message= "Beginning Autofocus Macro",
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