"""
Event generator: turns match tracking data into footballing event data.

This follows the methodology described in
"Generating Footballing Event Data from Match Tracking Data":

  Step 1 - Possession tracking
      A possession begins when the ball comes within the possession radius of a
      player (and that player is the closest player to the ball). The possession
      ends once the ball leaves that player. Noise from momentary ball movement
      and failed tackles is filtered out to leave a clean possession list.

  Step 2 - Event detection
      The possession list is walked with a sliding (previous, current, next)
      window. ``review_events`` inspects each triple and emits the appropriate
      event(s): passes, balls lost / interceptions, shots, goals, set pieces and
      aerial challenges.

All geometry is done in normalised [0, 1] pitch space; distances are evaluated
in metres on a standard 105 x 68 m pitch (see ``pitch.py``).
"""

from lib.possession import Possession
from lib.event import Event, EventLog
from lib import pitch

# --- Tuning constants -------------------------------------------------------
#
# The article describes a possession as the ball being within ".75 units" of a
# player. Tracking sources differ in scale, so we evaluate possession in metres
# on a standard pitch and expose the radius as a parameter. ~3 m is a robust
# default for tracking-derived possession; lower it for very clean data.
POSSESSION_RADIUS_M = 3.0

# A possession has to last at least this many frames to be considered real
# (filters single-frame proximity noise).
MIN_POSSESSION_FRAMES = 3

# A short opposition possession sandwiched between two possessions of the same
# TEAM, lasting no more than this many frames, is treated as a failed tackle /
# deflection and removed. (Same player: a failed tackle; same team: the ball
# grazed an opponent on its way between teammates.) Without this, every
# contested ball reads as two turnovers -- a phantom BALL LOST + RECOVERY pair
# in each direction.
FAILED_TACKLE_MAX_FRAMES = 9

# Same-player possessions separated by no more than this gap are merged (the
# ball briefly left the radius but the player kept the ball).
SAME_PLAYER_MERGE_GAP = 12

# Number of loose-ball frames after a possession before we go looking for the
# ball going out of play / a shot in the gap.
MIN_FLIGHT_FRAMES = 1

# A gap between possessions counts as a stoppage ("out of play") only if the
# ball is *continuously* undetected for at least this many seconds. A real dead
# ball (throw-in, goal kick, corner) stays gone for several seconds; brief,
# intermittent detection misses -- the norm with YOLO tracking, where the ball
# is only found in a fraction of frames -- must NOT be read as the ball leaving
# play, or nearly every possession change becomes a phantom SET PIECE. Lower it
# for clean tracking sources (e.g. Metrica) where the ball is blanked only when
# genuinely dead. Tune via ``generate_events(long_blank_seconds=...)``.
LONG_BLANK_SECONDS_RAW = 3.0
LONG_BLANK_SECONDS_CLEAN = 1.0

# A possession hand-off between teammates only counts as a PASS when the ball
# travelled at least this far (metres). Below it, the "pass" is the possession
# radius flapping between two players standing next to each other.
PASS_MIN_TRAVEL_M = 2.0

# A pass at least this long (metres) is flagged as a LONG BALL.
LONG_BALL_M = 32.0

# A pass that starts in a wide area of the final third and ends in the box is a
# CROSS. ``WIDE_BAND`` is the normalised width near each touchline.
WIDE_BAND = 0.22
FINAL_THIRD = 1.0 / 3.0

# Two opponents contesting a loose (airborne) ball within this distance, when
# the ball was lost in flight, are treated as an aerial challenge.
CHALLENGE_RADIUS_M = 4.0
AERIAL_MIN_FLIGHT_FRAMES = 8

# Region in front of the goal that a goalward ball must reach to count as a
# shot: slightly wider than the literal goal mouth (a saved shot dies at the
# keeper, a couple of metres off the line / off-centre), but deliberately NOT
# the whole box -- long clearances routinely drift into the penalty area and
# must not read as shots.
SHOT_AREA_DEPTH = 0.07
SHOT_AREA_HALF_WIDTH = 0.09

# A shot must also START within plausible shooting range of the goal. The
# false positives on real footage are long balls / clearances from midfield
# that drift goalward and are gathered by the keeper (or run out near the
# goal); those "shots" start 50+ m out. 40 m covers all but freak long-range
# attempts.
SHOT_MAX_START_DIST_M = 40.0

# A change of possession only counts as a turnover (BALL LOST + RECOVERY) when
# the recovering possession holds the ball at least this many frames. A
# shorter touch is the possession radius flapping between two nearby players
# (or a graze), not a real recovery -- emitting nothing is safer than a
# phantom turnover pair.
TURNOVER_MIN_HOLD_FRAMES = 6

# BALL LOST is subtyped INTERCEPTION only when the ball actually travelled
# loose for at least this many frames before the opponent won it (i.e. a pass
# or clearance was cut out mid-flight). Adjacent-possession steals are plain
# BALL LOST. The old threshold (1 frame) labelled nearly every noisy exchange
# an interception.
INTERCEPTION_MIN_FLIGHT_FRAMES = 4

# --- FIFA-style possession physics (Vidal-Codina et al., arXiv 2202.00804) ---
#
# A possession GAIN is only real if the player actually touched the ball, and
# a touch changes the ball's physics: direction or speed. A short "possession"
# during which the ball sails through the player's radius unaltered is the
# possession-flicker artifact that inflated BALL LOST to ~800/match. Measured
# rationale: three data-side fixes (ReID, ball cleanup, blank threshold) all
# failed to move event counts -- the flicker is structural to proximity-only
# possession, so the fix must be physical.
GAIN_CHECK_MAX_S = 1.0        # only vet possessions shorter than this
GAIN_DIR_MAX_COS = 0.85       # cos(in,out) >= this => direction unchanged (~32deg)
GAIN_MIN_SPEED_DELTA_MS = 1.5 # m/s in-vs-out difference that indicates a touch
GAIN_PHYS_WINDOW_S = 0.25     # seconds around the interval to measure in/out

# FIFA loss rule (b2): if the same player regains the ball before ANY other
# player takes control, no loss happened -- regardless of how long the ball
# was loose (a dribble that pushes the ball ahead). Bounded by this horizon
# and vetoed if the ball went dead in between.
REGAIN_MAX_GAP_S = 4.0

# --- IFAB set piece formation triggers (FIFA paper section 2.3.4) ----------
#
# A real restart produces a distinctive PLAYER CONFIGURATION (corner: someone
# at the corner mark; kick-off: everyone in their own half; throw-in: someone
# beyond the touchline; goal kick: someone inside their goal area). A tracking
# blank does NOT. So a long ball blank only becomes a SET PIECE if a formation
# confirms it; otherwise it is treated as a detection failure and suppressed.
TRIG_CENTRE_R = 0.06          # kick-off: player within this of the centre mark
TRIG_OWNHALF_TOL = 0.05       # kick-off: own-half tolerance (norm. length)
TRIG_MAX_OFFENDERS = 2        # kick-off: players allowed off-side of halfway
TRIG_CORNER_R = 0.035         # corner: player within this of a corner mark
TRIG_TOUCHLINE = 0.012        # throw-in: player at/beyond the touchline
TRIG_GOAL_AREA_HW = 0.15      # goal kick: goal-area half-width (normalised)
TRIG_LOOKBACK_S = 1.0         # dead-ball tail examined before the restart
PATTERN_TOL = 0.10            # executor start_loc must be near the trigger zone


class EventGenerator():
    def __init__(self, match, possession_radius=POSSESSION_RADIUS_M,
                 long_blank_seconds=None):
        self.match = match
        self.source = match.source
        self.possession_radius = possession_radius
        # Seconds of *continuous* ball-blank that count as the ball going out of
        # play. Defaults higher for noisy "raw" YOLO data than for clean sources.
        if long_blank_seconds is None:
            long_blank_seconds = (LONG_BLANK_SECONDS_RAW if self.source == "raw"
                                  else LONG_BLANK_SECONDS_CLEAN)
        self.long_blank_seconds = long_blank_seconds
        self.possessions = []
        # attacking goal x (0 or 1) per normalised team key
        self.attack_goal = {}
        # {frame: (x, y) normalised} for frames where the ball was seen.
        # Filled by detect_possessions; used by the gain physics, the regain
        # rule and the dead-gap checks without re-walking Match frames.
        self._ball_pos = {}

    def long_blank_frames(self):
        """Continuous ball-less frames needed to call the ball out of play."""
        return max(1, int(round(self.long_blank_seconds * self.match_fps())))

    # Every ``*_FRAMES`` constant in this module was tuned against 24 fps
    # tracking. Read literally at another rate they silently change meaning in
    # SECONDS -- at 15 fps (stride 2 on 30 fps footage) they become ~1.6x
    # stricter, which is what halved passes and duels. Scale them instead.
    TUNED_FPS = 24

    def _frames(self, frames_at_tuned_fps):
        """Convert a threshold tuned at 24 fps into this match's frame count."""
        return max(1, int(round(frames_at_tuned_fps
                                * self.match_fps() / self.TUNED_FPS)))

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #
    def generate(self):
        self.possessions = self.detect_possessions()
        self.possessions = self.filter_possessions(self.possessions)
        self._infer_attacking_directions()
        return self._build_events()

    # ------------------------------------------------------------------ #
    # Coordinate helpers                                                 #
    # ------------------------------------------------------------------ #
    def _norm_ball(self, moment):
        loc = moment.ball_loc()
        if loc is None:
            return None
        return pitch.to_normalised(self.source, loc[0], loc[1])

    def _nearest_player(self, moment, ball_norm):
        """Return (player, distance_m, player_norm) of the closest player."""
        nearest = None
        nearest_dist = None
        nearest_loc = None
        for entry in moment.players:
            frame = entry["frame"]
            if frame is None or frame.coordinates is None:
                continue
            p_norm = pitch.to_normalised(self.source, frame.x, frame.y)
            dist = pitch.distance_m(ball_norm, p_norm)
            if nearest_dist is None or dist < nearest_dist:
                nearest = entry["object"]
                nearest_dist = dist
                nearest_loc = p_norm
        return nearest, nearest_dist, nearest_loc

    # ------------------------------------------------------------------ #
    # Step 1: possession detection                                       #
    # ------------------------------------------------------------------ #
    def detect_possessions(self):
        possessions = []
        current = None
        for frame_number in range(1, self.match.frames + 1):
            moment = self.match.frame(frame_number)
            if moment is None:
                continue
            time = moment.time
            ball_norm = self._norm_ball(moment)
            if ball_norm is not None:
                self._ball_pos[frame_number] = ball_norm

            holder = None
            holder_loc = None
            if ball_norm is not None:
                player, dist, _ = self._nearest_player(moment, ball_norm)
                if player is not None and dist is not None and dist <= self.possession_radius:
                    holder = player
                    # holder_loc = la PELOTA (no los pies del poseedor). PROBADO
                    # 26-ago con los pies (idea de Sol: la homografia no vale para
                    # la pelota en el aire): el AUC EMPEORO 0,579->0,549. Los
                    # endpoints caen en los bordes de la posesion, donde la pelota
                    # esta cerca del piso, asi que la proyeccion aerea no los
                    # afecta; y usar el pie del "poseedor" mete ruido cuando la
                    # pelota mal-asigno al jugador. La pelota es mejor endpoint.
                    holder_loc = ball_norm

            if holder is None:
                # Ball is loose (in flight / out of play). Close any possession.
                if current is not None:
                    possessions.append(current)
                    current = None
                continue

            if current is not None and str(current.player.id) == str(holder.id):
                current.extend(frame_number, time, holder_loc)
            else:
                if current is not None:
                    possessions.append(current)
                current = Possession(holder, frame_number, time, holder_loc)

        if current is not None:
            possessions.append(current)
        return possessions

    # ------------------------------------------------------------------ #
    # Step 1b: possession noise filtering                                #
    # ------------------------------------------------------------------ #
    def filter_possessions(self, possessions):
        # Physics first: discard "possessions" where the ball sailed through a
        # player's radius without changing direction or speed (nobody touched
        # it). Removing them exposes A-[fake B]-A adjacencies that the merges
        # below then collapse -- this is what kills possession flicker.
        possessions = self._validate_gains(possessions)
        possessions = self._remove_failed_tackles(possessions)
        possessions = self._merge_same_player(possessions)
        possessions = self._drop_short_possessions(possessions)
        # Merging / dropping can expose new adjacencies, so settle once more.
        possessions = self._merge_same_player(possessions)
        return possessions

    # ------------------------------------------------------------------ #
    # FIFA-style gain physics (anti-flicker)                              #
    # ------------------------------------------------------------------ #
    def _ball_segment(self, f_from, f_to):
        """Ball displacement between the nearest detections inside a window.

        Returns ``(dx_m, dy_m, dt_s)`` in metres/seconds, or None when fewer
        than two detections exist in the window (blank ball -- unmeasurable).
        """
        first = last = None
        for f in range(f_from, f_to + 1):
            p = self._ball_pos.get(f)
            if p is not None:
                first = (f, p)
                break
        for f in range(f_to, f_from - 1, -1):
            p = self._ball_pos.get(f)
            if p is not None:
                last = (f, p)
                break
        if first is None or last is None or last[0] <= first[0]:
            return None
        dx = (last[1][0] - first[1][0]) * pitch.PITCH_LENGTH_M
        dy = (last[1][1] - first[1][1]) * pitch.PITCH_WIDTH_M
        dt = (last[0] - first[0]) / self.match_fps()
        return dx, dy, dt

    def _gain_is_real(self, poss):
        """FIFA rule: a touch changes the ball's direction or speed.

        Compares the ball's trajectory just before the possession starts with
        just after it ends. Unmeasurable (ball blanks) => keep the possession:
        we only discard when the physics POSITIVELY shows nobody touched it.
        """
        w = max(2, int(round(GAIN_PHYS_WINDOW_S * self.match_fps())))
        seg_in = self._ball_segment(poss.start_frame - w, poss.start_frame)
        seg_out = self._ball_segment(poss.end_frame, poss.end_frame + w)
        if seg_in is None or seg_out is None:
            return True

        len_in = (seg_in[0] ** 2 + seg_in[1] ** 2) ** 0.5
        len_out = (seg_out[0] ** 2 + seg_out[1] ** 2) ** 0.5
        v_in = len_in / seg_in[2]
        v_out = len_out / seg_out[2]
        speed_changed = abs(v_out - v_in) >= GAIN_MIN_SPEED_DELTA_MS

        # A near-static segment has no meaningful direction; judge by speed
        # alone (ball stopped dead or kicked from rest are both real touches).
        if len_in < 0.3 or len_out < 0.3:
            return speed_changed

        cos = (seg_in[0] * seg_out[0] + seg_in[1] * seg_out[1]) / (len_in * len_out)
        direction_changed = cos < GAIN_DIR_MAX_COS
        return direction_changed or speed_changed

    def _validate_gains(self, possessions):
        """Drop short possessions whose ball physics show no actual touch."""
        max_check = int(round(GAIN_CHECK_MAX_S * self.match_fps()))
        kept = []
        for poss in possessions:
            # Long possessions are self-evidently real (dribbles constantly
            # alter the ball); only brief ones can be flicker.
            if poss.duration >= max_check or self._gain_is_real(poss):
                kept.append(poss)
        return kept

    def _gap_went_dead(self, f_from, f_to):
        """True when the ball went out of play between two frames."""
        blank_run = 0
        for f in range(f_from, f_to + 1):
            p = self._ball_pos.get(f)
            if p is None:
                blank_run += 1
                if blank_run >= self.long_blank_frames():
                    return True
                continue
            blank_run = 0
            if pitch.is_out_of_bounds(p, margin=0.0):
                return True
        return False

    def _remove_failed_tackles(self, possessions):
        """Drop a brief opponent touch wedged between one player's possessions."""
        cleaned = []
        i = 0
        n = len(possessions)
        while i < n:
            if 0 < i < n - 1:
                prev = cleaned[-1] if cleaned else possessions[i - 1]
                curr = possessions[i]
                nxt = possessions[i + 1]
                is_failed_tackle = (
                    curr.duration <= self._frames(FAILED_TACKLE_MAX_FRAMES)
                    and prev.same_team(nxt)
                    and not curr.same_team(prev)
                )
                if is_failed_tackle:
                    # Skip the failed tackle entirely; the player kept the ball.
                    i += 1
                    continue
            cleaned.append(possessions[i])
            i += 1
        return cleaned

    def _merge_same_player(self, possessions):
        if not possessions:
            return possessions
        merged = [possessions[0]]
        regain_max = int(round(REGAIN_MAX_GAP_S * self.match_fps()))
        for poss in possessions[1:]:
            last = merged[-1]
            gap = poss.start_frame - last.end_frame
            mergeable = False
            if last.same_player(poss):
                if gap <= self._frames(SAME_PLAYER_MERGE_GAP):
                    mergeable = True
                elif gap <= regain_max:
                    # FIFA loss rule (b2): the same player regaining the ball
                    # with NO other control frame in between is not a loss --
                    # e.g. a dribble knocking the ball ahead. Possessions are
                    # consecutive control intervals, so reaching here already
                    # means nobody else had the ball; only veto the merge if
                    # the ball went dead (a set piece separates real events).
                    mergeable = not self._gap_went_dead(last.end_frame,
                                                        poss.start_frame)
            if mergeable:
                last.end_frame = poss.end_frame
                last.end_time = poss.end_time
                last.end_loc = poss.end_loc
                last.touches += poss.touches
            else:
                merged.append(poss)
        return merged

    def _drop_short_possessions(self, possessions):
        kept = []
        for poss in possessions:
            # A one-touch pass is legitimately short, so only drop a short
            # possession when it is not a clean hand-off between two others.
            min_poss = self._frames(MIN_POSSESSION_FRAMES)
            if poss.duration < min_poss and poss.touches < min_poss:
                continue
            kept.append(poss)
        return kept

    # ------------------------------------------------------------------ #
    # Attacking direction inference                                      #
    # ------------------------------------------------------------------ #
    def _infer_attacking_directions(self):
        """Each team attacks the goal furthest from its average x position."""
        sums = {}
        counts = {}
        for frame_number in range(1, self.match.frames + 1):
            moment = self.match.frame(frame_number)
            if moment is None:
                continue
            for entry in moment.players:
                frame = entry["frame"]
                if frame is None or frame.coordinates is None:
                    continue
                team = str(entry["object"].team)
                norm = pitch.to_normalised(self.source, frame.x, frame.y)
                sums[team] = sums.get(team, 0.0) + norm[0]
                counts[team] = counts.get(team, 0) + 1

        means = {t: sums[t] / counts[t] for t in sums if counts[t]}
        for team, mean_x in means.items():
            # A team whose players sit in the left half (mean_x < 0.5) defends
            # the left goal and therefore attacks the right goal (x = 1).
            self.attack_goal[team] = 1.0 if mean_x < 0.5 else 0.0

    def _attacking_goal(self, team):
        return self.attack_goal.get(str(team), 1.0)

    # ------------------------------------------------------------------ #
    # Step 2: event detection via a sliding triple window                #
    # ------------------------------------------------------------------ #
    def _build_events(self):
        log = EventLog()
        possessions = self.possessions
        n = len(possessions)

        for i in range(n):
            prev = possessions[i - 1] if i > 0 else None
            curr = possessions[i]
            nxt = possessions[i + 1] if i < n - 1 else None
            self.review_events(prev, curr, nxt, log)
        return log

    def review_events(self, prev, curr, nxt, log):
        """Inspect a (previous, current, next) possession triple and log events."""
        # --- Restart that *precedes* this possession --------------------- #
        # If there is no previous possession, or the ball clearly went out of
        # play between prev and curr, then curr begins from a set piece.
        if prev is None:
            self._emit_set_piece(curr, log, opening=True)
        else:
            flight = self._gap_analysis(prev, curr)
            if flight["out_of_play"]:
                if flight["out_of_bounds"]:
                    # The ball was SEEN leaving the pitch: a real restart.
                    self._emit_set_piece(curr, log, restart=flight)
                else:
                    # Only a detection blank. A real restart produces an IFAB
                    # player formation (corner/kick-off/throw-in/goal-kick);
                    # a YOLO miss does not. No formation => phantom, suppress.
                    subtype = self._formation_set_piece(curr)
                    if subtype is not None:
                        self._emit_set_piece(curr, log, restart=flight,
                                             forced_subtype=subtype)

        if nxt is None:
            # Nothing follows; we cannot classify the end of this possession.
            return

        flight = self._gap_analysis(curr, nxt)

        # --- Goal / shot in the gap before the restart ------------------- #
        if flight["goal"]:
            self._emit_shot(curr, nxt, flight, log, is_goal=True)
            return
        if flight["out_of_play"]:
            if flight["shot"]:
                self._emit_shot(curr, nxt, flight, log, is_goal=False)
            elif flight["out_of_bounds"] or \
                    self._formation_set_piece(nxt) is not None:
                # Ball seen out, or the restart formation confirms the
                # stoppage was real: the ball did leave play.
                self._emit_ball_out(curr, flight, log)
            # else: a bare detection blank with no restart formation --
            # phantom stoppage, no event (mirrors the set piece suppression).
            return

        # --- Continuous play -------------------------------------------- #
        if curr.same_team(nxt):
            if not curr.same_player(nxt) and self._ball_travelled(curr, nxt):
                self._emit_pass(curr, nxt, flight, log)
            # same player after merge, or a hand-off where the ball never
            # actually travelled (radius noise): no event.
        else:
            # Possession changed teams without the ball leaving play.
            if self._is_aerial_challenge(curr, nxt, flight):
                self._emit_challenge(curr, nxt, flight, log)
            elif flight["shot"]:
                # Goalward, gathered by the opposition keeper -> saved shot.
                self._emit_shot(curr, nxt, flight, log, is_goal=False, saved=True)
            elif nxt.duration >= self._frames(TURNOVER_MIN_HOLD_FRAMES):
                self._emit_turnover(curr, nxt, flight, log)
            # else: fleeting opposition touch -- not enough evidence of a
            # real turnover, so no event is emitted.

    # ------------------------------------------------------------------ #
    # Gap analysis between two possessions                               #
    # ------------------------------------------------------------------ #
    def _gap_analysis(self, a, b):
        """Inspect the ball's behaviour between possession ``a`` and ``b``."""
        start_frame = a.end_frame
        end_frame = b.start_frame
        attacking_goal = self._attacking_goal(a.team)

        flight_frames = max(end_frame - start_frame - 1, 0)
        went_none = False
        max_blank_run = 0     # longest run of *consecutive* ball-less frames
        cur_blank_run = 0
        out_norm = None  # last in/near-bounds ball position before going out
        ball_path = []
        for fn in range(start_frame, end_frame + 1):
            moment = self.match.frame(fn)
            if moment is None:
                continue
            ball = self._norm_ball(moment)
            if ball is None:
                went_none = True
                cur_blank_run += 1
                if cur_blank_run > max_blank_run:
                    max_blank_run = cur_blank_run
                continue
            cur_blank_run = 0
            ball_path.append((fn, ball))

        start_loc = a.end_loc
        end_loc = b.start_loc

        # Did the ball leave the field of play during the gap?
        out_of_bounds = False
        for _, ball in ball_path:
            if pitch.is_out_of_bounds(ball, margin=0.0):
                out_of_bounds = True
                out_norm = ball
        # A *continuous* stretch with no ball detection implies the ball went
        # out of play. We require a sustained blank (``long_blank_frames``) so
        # that the constant brief misses of YOLO tracking don't masquerade as
        # stoppages and inflate SET PIECE / BALL LOST OUT counts.
        long_blank = max_blank_run >= self.long_blank_frames()

        out_of_play = out_of_bounds or long_blank

        # The ball's closest approach to the attacking goal during the gap is a
        # better indicator of a shot than the next possession's start (which,
        # after a goal, is a kick-off back at the centre spot).
        goal_point = (attacking_goal, 0.5)
        effective_end = end_loc
        if ball_path:
            effective_end = min(
                (ball for _, ball in ball_path),
                key=lambda b: pitch.distance_m(b, goal_point),
            )
            if out_norm is not None and pitch.distance_m(out_norm, goal_point) < \
                    pitch.distance_m(effective_end, goal_point):
                effective_end = out_norm

        # Goalward? The ball headed meaningfully towards a's attacking goal,
        # from a position a shot could plausibly be taken from.
        goalward = self._is_goalward(start_loc, effective_end, attacking_goal)
        in_shot_range = (
            start_loc is not None and
            pitch.distance_m(start_loc, goal_point) <= SHOT_MAX_START_DIST_M
        )
        reached_goal_area = (
            pitch.near_goal(effective_end, attacking_goal,
                            depth=SHOT_AREA_DEPTH, half_width=SHOT_AREA_HALF_WIDTH)
            or (out_norm is not None and
                pitch.near_goal(out_norm, attacking_goal,
                                depth=SHOT_AREA_DEPTH, half_width=SHOT_AREA_HALF_WIDTH))
            or self._crossed_byline_at_goal(ball_path, attacking_goal)
        )
        shot = goalward and in_shot_range and reached_goal_area

        # A goal: ball reaches the goal mouth and the next restart is a kick-off
        # from the centre by the conceding team.
        goal = False
        if shot and out_of_play:
            if pitch.near_centre(b.start_loc) and not b.same_team(a):
                goal = True

        return {
            "flight_frames": flight_frames,
            "out_of_play": out_of_play,
            "out_of_bounds": out_of_bounds,
            "long_blank": long_blank,
            "out_norm": out_norm,
            "shot": shot,
            "goal": goal,
            "goalward": goalward,
            "attacking_goal": attacking_goal,
            "start_loc": start_loc,
            "end_loc": end_loc,
        }

    def match_fps(self):
        """Frames per second of the tracking actually being read.

        Never hardcode this: with ``--frame-stride`` the raw CSV ticks at
        fps/stride (e.g. 15 instead of 30), and since every threshold below is
        expressed in FRAMES, a wrong value silently rescales all of them in
        real time -- which suppressed passes and duels on the first strided run.
        ``Match`` resolves the true rate from the .meta.json sidecar.
        """
        if self.source == "metrica":
            return 25
        return getattr(self.match, "fps", None) or 24

    def _is_goalward(self, start_loc, end_loc, goal_x):
        if start_loc is None or end_loc is None:
            return False
        goal = (goal_x, 0.5)
        # Ball must end meaningfully closer to the goal than it started.
        d_start = pitch.distance_m(start_loc, goal)
        d_end = pitch.distance_m(end_loc, goal)
        return d_end < d_start - 3.0

    def _crossed_byline_at_goal(self, ball_path, goal_x):
        for _, ball in ball_path:
            x, y = ball
            if goal_x == 1 and x >= 1 - pitch.BYLINE_MARGIN and abs(y - 0.5) <= pitch.GOAL_HALF_WIDTH:
                return True
            if goal_x == 0 and x <= pitch.BYLINE_MARGIN and abs(y - 0.5) <= pitch.GOAL_HALF_WIDTH:
                return True
        return False

    def _is_aerial_challenge(self, curr, nxt, flight):
        """Best-effort aerial-duel detection (limited by 2D tracking)."""
        if flight["flight_frames"] < self._frames(AERIAL_MIN_FLIGHT_FRAMES):
            return False
        # The two players must have been close together when the ball dropped.
        moment = self.match.frame(nxt.start_frame)
        if moment is None:
            return False
        a_loc = moment.player_loc(curr.player.id)
        b_loc = moment.player_loc(nxt.player.id)
        if a_loc is None or b_loc is None:
            return False
        a_norm = pitch.to_normalised(self.source, a_loc[0], a_loc[1])
        b_norm = pitch.to_normalised(self.source, b_loc[0], b_loc[1])
        return pitch.distance_m(a_norm, b_norm) <= CHALLENGE_RADIUS_M

    # ------------------------------------------------------------------ #
    # Event emitters                                                     #
    # ------------------------------------------------------------------ #
    def _ball_travelled(self, curr, nxt, min_m=PASS_MIN_TRAVEL_M):
        """True when the ball moved at least ``min_m`` between possessions."""
        if curr.end_loc is None or nxt.start_loc is None:
            return True  # can't measure; don't silently drop the event
        return pitch.distance_m(curr.end_loc, nxt.start_loc) >= min_m

    def _emit_pass(self, curr, nxt, flight, log):
        subtype = self._pass_subtype(curr.end_loc, nxt.start_loc, curr.team)
        log.add(Event({
            "team": curr.team,
            "type": "PASS",
            "subtype": subtype,
            "start_frame": curr.end_frame,
            "start_time": curr.end_time,
            "end_frame": nxt.start_frame,
            "end_time": nxt.start_time,
            "from_player": curr.player,
            "to_player": nxt.player,
            "start_loc": curr.end_loc,
            "end_loc": nxt.start_loc,
        }))

    def _pass_subtype(self, start_loc, end_loc, team):
        if start_loc is None or end_loc is None:
            return ""
        length = pitch.distance_m(start_loc, end_loc)
        goal_x = self._attacking_goal(team)
        # CROSS: from a wide area of the final third into the box.
        ex, ey = end_loc
        sx, sy = start_loc
        in_final_third = (ex >= 1 - FINAL_THIRD) if goal_x == 1 else (ex <= FINAL_THIRD)
        from_wide = (sy <= WIDE_BAND or sy >= 1 - WIDE_BAND)
        into_box = pitch.near_goal(end_loc, goal_x, depth=0.17, half_width=0.20)
        if in_final_third and from_wide and into_box:
            return "CROSS"
        if length >= LONG_BALL_M:
            return "LONG BALL"
        return ""

    def _emit_turnover(self, curr, nxt, flight, log):
        """Ball lost to the opposition without leaving the field of play."""
        intercepted = flight["flight_frames"] >= self._frames(INTERCEPTION_MIN_FLIGHT_FRAMES)
        log.add(Event({
            "team": curr.team,
            "type": "BALL LOST",
            "subtype": "INTERCEPTION" if intercepted else "",
            "start_frame": curr.end_frame,
            "start_time": curr.end_time,
            "end_frame": nxt.start_frame,
            "end_time": nxt.start_time,
            "from_player": curr.player,
            "to_player": None,
            "start_loc": curr.end_loc,
            "end_loc": nxt.start_loc,
        }))
        log.add(Event({
            "team": nxt.team,
            "type": "RECOVERY",
            "subtype": "INTERCEPTION" if intercepted else "",
            "start_frame": nxt.start_frame,
            "start_time": nxt.start_time,
            "end_frame": nxt.start_frame,
            "end_time": nxt.start_time,
            "from_player": nxt.player,
            "to_player": None,
            "start_loc": nxt.start_loc,
            "end_loc": nxt.start_loc,
        }))

    def _emit_challenge(self, curr, nxt, flight, log):
        log.add(Event({
            "team": curr.team,
            "type": "CHALLENGE",
            "subtype": "AERIAL-LOST",
            "start_frame": curr.end_frame,
            "start_time": curr.end_time,
            "end_frame": nxt.start_frame,
            "end_time": nxt.start_time,
            "from_player": curr.player,
            "to_player": None,
            "start_loc": curr.end_loc,
            "end_loc": nxt.start_loc,
        }))
        log.add(Event({
            "team": nxt.team,
            "type": "CHALLENGE",
            "subtype": "AERIAL-WON",
            "start_frame": curr.end_frame,
            "start_time": curr.end_time,
            "end_frame": nxt.start_frame,
            "end_time": nxt.start_time,
            "from_player": nxt.player,
            "to_player": None,
            "start_loc": curr.end_loc,
            "end_loc": nxt.start_loc,
        }))

    def _emit_shot(self, curr, nxt, flight, log, is_goal=False, saved=False):
        if is_goal:
            subtype = "GOAL"
        elif saved:
            subtype = "SAVED"
        elif flight["out_of_play"]:
            subtype = "OFF TARGET"
        else:
            subtype = ""
        end_loc = flight["out_norm"] or nxt.start_loc or curr.end_loc
        log.add(Event({
            "team": curr.team,
            "type": "SHOT",
            "subtype": subtype,
            "start_frame": curr.end_frame,
            "start_time": curr.end_time,
            "end_frame": nxt.start_frame,
            "end_time": nxt.start_time,
            "from_player": curr.player,
            "to_player": None,
            "start_loc": curr.end_loc,
            "end_loc": (flight["attacking_goal"], 0.5) if is_goal else end_loc,
        }))
        if is_goal:
            log.add(Event({
                "team": curr.team,
                "type": "GOAL",
                "subtype": "",
                "start_frame": curr.end_frame,
                "start_time": curr.end_time,
                "end_frame": nxt.start_frame,
                "end_time": nxt.start_time,
                "from_player": curr.player,
                "to_player": None,
                "start_loc": curr.end_loc,
                "end_loc": (flight["attacking_goal"], 0.5),
            }))

    def _emit_ball_out(self, curr, flight, log):
        log.add(Event({
            "team": curr.team,
            "type": "BALL LOST",
            "subtype": "OUT",
            "start_frame": curr.end_frame,
            "start_time": curr.end_time,
            "end_frame": curr.end_frame,
            "end_time": curr.end_time,
            "from_player": curr.player,
            "to_player": None,
            "start_loc": curr.end_loc,
            "end_loc": flight["out_norm"] or curr.end_loc,
        }))

    # ------------------------------------------------------------------ #
    # IFAB formation triggers (FIFA paper 2.3.4)                          #
    # ------------------------------------------------------------------ #
    def _players_normalised(self, frame_number):
        """[(team, (x, y) normalised)] for every player seen in a frame."""
        moment = self.match.frame(frame_number)
        if moment is None:
            return []
        out = []
        for entry in moment.players:
            frame = entry["frame"]
            if frame is None or frame.coordinates is None:
                continue
            norm = pitch.to_normalised(self.source, frame.x, frame.y)
            out.append((str(entry["object"].team), norm))
        return out

    def _formation_set_piece(self, poss):
        """Detect the restart type from player configurations (IFAB laws).

        Samples the dead-ball tail just before ``poss`` starts and checks, in
        FIFA's hierarchy order (kick-off > corner > throw-in > goal kick),
        whether any frame shows the distinctive formation. The trigger is then
        confirmed by a pattern check: the executor's start location must be
        consistent with the restart zone. Returns the subtype or None.
        """
        lookback = max(2, int(round(TRIG_LOOKBACK_S * self.match_fps())))
        step = max(1, lookback // 4)
        sample_frames = list(range(max(1, poss.start_frame - lookback),
                                   poss.start_frame + 1, step))
        start = poss.start_loc

        corners = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
        own_goal = 1.0 - self._attacking_goal(poss.team)

        saw_kickoff = saw_corner = saw_throwin = saw_goalkick = False
        for fn in sample_frames:
            players = self._players_normalised(fn)
            if len(players) < 6:      # too few tracked players to judge
                continue

            # Kick-off: everyone in their own half (tolerance + a couple of
            # stragglers for tracking noise), someone at the centre mark.
            offenders = 0
            centre_present = False
            for team, (x, y) in players:
                goal_x = self._attacking_goal(team)
                in_own_half = (x <= 0.5 + TRIG_OWNHALF_TOL if goal_x == 1.0
                               else x >= 0.5 - TRIG_OWNHALF_TOL)
                if not in_own_half:
                    offenders += 1
                if abs(x - 0.5) <= TRIG_CENTRE_R and abs(y - 0.5) <= TRIG_CENTRE_R:
                    centre_present = True
            if offenders <= TRIG_MAX_OFFENDERS and centre_present:
                saw_kickoff = True

            for _team, (x, y) in players:
                if any(abs(x - cx) <= TRIG_CORNER_R and abs(y - cy) <= TRIG_CORNER_R
                       for cx, cy in corners):
                    saw_corner = True
                if y <= TRIG_TOUCHLINE or y >= 1.0 - TRIG_TOUCHLINE:
                    saw_throwin = True
                if pitch.near_goal((x, y), own_goal, depth=pitch.GOAL_AREA_DEPTH,
                                   half_width=TRIG_GOAL_AREA_HW):
                    saw_goalkick = True

        # Hierarchy + pattern confirmation (executor near the restart zone).
        if saw_kickoff and (start is None or pitch.near_centre(start)):
            return "KICK OFF"
        if saw_corner and start is not None and \
                any(abs(start[0] - cx) <= PATTERN_TOL and abs(start[1] - cy) <= PATTERN_TOL
                    for cx, cy in corners):
            return "CORNER"
        if saw_throwin and start is not None and \
                (start[1] <= PATTERN_TOL or start[1] >= 1.0 - PATTERN_TOL):
            return "THROW IN"
        if saw_goalkick and start is not None and \
                pitch.near_goal(start, own_goal,
                                depth=pitch.GOAL_AREA_DEPTH + PATTERN_TOL,
                                half_width=TRIG_GOAL_AREA_HW + PATTERN_TOL):
            return "GOAL KICK"
        return None

    def _emit_set_piece(self, poss, log, opening=False, restart=None,
                        forced_subtype=None):
        subtype = forced_subtype or \
            self._classify_restart(poss, opening=opening, restart=restart)
        log.add(Event({
            "team": poss.team,
            "type": "SET PIECE",
            "subtype": subtype,
            "start_frame": poss.start_frame,
            "start_time": poss.start_time,
            "end_frame": poss.start_frame,
            "end_time": poss.start_time,
            "from_player": poss.player,
            "to_player": None,
            "start_loc": poss.start_loc,
            "end_loc": poss.start_loc,
        }))

    def _classify_restart(self, poss, opening=False, restart=None):
        loc = poss.start_loc
        if loc is None:
            return "KICK OFF" if opening else ""
        x, y = loc
        own_goal = 1.0 - self._attacking_goal(poss.team)

        if pitch.near_centre(loc):
            return "KICK OFF"

        near_left = x <= pitch.BYLINE_MARGIN
        near_right = x >= 1 - pitch.BYLINE_MARGIN
        near_top = y <= pitch.TOUCHLINE_MARGIN
        near_bottom = y >= 1 - pitch.TOUCHLINE_MARGIN

        # Corner: by a goal line, out near the corner flag.
        if (near_left or near_right) and (near_top or near_bottom):
            return "CORNER"
        # Goal kick: restart from inside the goal area by the defending team.
        if pitch.near_goal(loc, own_goal, depth=pitch.GOAL_AREA_DEPTH + 0.04, half_width=0.25):
            return "GOAL KICK"
        # Throw-in: restart from a touchline.
        if near_top or near_bottom:
            return "THROW IN"
        # Fallback by goal line proximity.
        if near_left or near_right:
            return "GOAL KICK"
        return "KICK OFF" if opening else ""
