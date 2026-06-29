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
# player, lasting no more than this many frames, is treated as a failed tackle
# and removed.
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

    def long_blank_frames(self):
        """Continuous ball-less frames needed to call the ball out of play."""
        return max(1, int(round(self.long_blank_seconds * self.match_fps())))

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

            holder = None
            holder_loc = None
            if ball_norm is not None:
                player, dist, _ = self._nearest_player(moment, ball_norm)
                if player is not None and dist is not None and dist <= self.possession_radius:
                    holder = player
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
        possessions = self._remove_failed_tackles(possessions)
        possessions = self._merge_same_player(possessions)
        possessions = self._drop_short_possessions(possessions)
        # Merging / dropping can expose new adjacencies, so settle once more.
        possessions = self._merge_same_player(possessions)
        return possessions

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
                    curr.duration <= FAILED_TACKLE_MAX_FRAMES
                    and prev.same_player(nxt)
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
        for poss in possessions[1:]:
            last = merged[-1]
            gap = poss.start_frame - last.end_frame
            if last.same_player(poss) and gap <= SAME_PLAYER_MERGE_GAP:
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
            if poss.duration < MIN_POSSESSION_FRAMES and poss.touches < MIN_POSSESSION_FRAMES:
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
                self._emit_set_piece(curr, log, restart=flight)

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
            else:
                # Ball was simply played out of bounds.
                self._emit_ball_out(curr, flight, log)
            return

        # --- Continuous play -------------------------------------------- #
        if curr.same_team(nxt):
            if not curr.same_player(nxt):
                self._emit_pass(curr, nxt, flight, log)
            # same player after merge shouldn't happen; ignore.
        else:
            # Possession changed teams without the ball leaving play.
            if self._is_aerial_challenge(curr, nxt, flight):
                self._emit_challenge(curr, nxt, flight, log)
            elif flight["shot"]:
                # Goalward, gathered by the opposition keeper -> saved shot.
                self._emit_shot(curr, nxt, flight, log, is_goal=False, saved=True)
            else:
                self._emit_turnover(curr, nxt, flight, log)

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

        # Goalward? The ball headed meaningfully towards a's attacking goal.
        goalward = self._is_goalward(start_loc, effective_end, attacking_goal)
        reached_goal_area = (
            pitch.near_goal(effective_end, attacking_goal)
            or (out_norm is not None and pitch.near_goal(out_norm, attacking_goal))
            or self._crossed_byline_at_goal(ball_path, attacking_goal)
        )
        shot = goalward and reached_goal_area

        # A goal: ball reaches the goal mouth and the next restart is a kick-off
        # from the centre by the conceding team.
        goal = False
        if shot and out_of_play:
            if pitch.near_centre(b.start_loc) and not b.same_team(a):
                goal = True

        return {
            "flight_frames": flight_frames,
            "out_of_play": out_of_play,
            "out_norm": out_norm,
            "shot": shot,
            "goal": goal,
            "goalward": goalward,
            "attacking_goal": attacking_goal,
            "start_loc": start_loc,
            "end_loc": end_loc,
        }

    def match_fps(self):
        return 25 if self.source == "metrica" else 24

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
        if flight["flight_frames"] < AERIAL_MIN_FLIGHT_FRAMES:
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
        intercepted = flight["flight_frames"] >= 1
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

    def _emit_set_piece(self, poss, log, opening=False, restart=None):
        subtype = self._classify_restart(poss, opening=opening, restart=restart)
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
