"""
Pitch geometry helpers shared by the event generator.

Tracking data reaches us in two different coordinate spaces:

  * "metrica" - coordinates are already normalised to the [0, 1] range, where
    x runs along the length of the pitch and y along the width. This is the
    format Metrica Sports publishes its sample data in.
  * "raw"     - coordinates are the pitch coordinates produced by the tracking
    pipeline (Roboflow / `sports` SoccerPitchConfiguration). Those are measured
    in centimetres on a 10500 x 6800 cm pitch (ver pitch_config.py).

To reason about the game geometrically (which goal is which, when the ball goes
out of play, how far a pass travelled ...) we convert everything into a single
canonical space: normalised [0, 1] coordinates for positions, and metres on a
standard 105 x 68 m pitch for distances.
"""

# Standard pitch dimensions used for distance calculations (metres).
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Dimensions of the `sports` SoccerPitchConfiguration used by the raw tracking
# pipeline (centimetres). Used to normalise "raw" coordinates to [0, 1].
# Ver data_cleanup/pitch_config.py: hasta el 20-ago esto era 12000x7000, la
# geometria ficticia de ``sports``.
RAW_PITCH_LENGTH_CM = 10500.0
RAW_PITCH_WIDTH_CM = 6800.0

# The two goals sit at the centre of each goal line.
LEFT_GOAL = (0.0, 0.5)
RIGHT_GOAL = (1.0, 0.5)

# Half the width of a goal expressed in normalised pitch-width units
# (~7.32 m goal on a 68 m pitch -> ~0.054, widened a touch for tolerance).
GOAL_HALF_WIDTH = 0.07

# Depth of the goal/six-yard region from the goal line (normalised length).
GOAL_AREA_DEPTH = 0.06

# How close (normalised length) to a goal line the ball has to be before we
# consider a restart to be a corner / goal-kick rather than a throw-in.
BYLINE_MARGIN = 0.04

# How close (normalised width) to a touchline the ball has to be before we
# consider it to have gone out for a throw-in.
TOUCHLINE_MARGIN = 0.04

# A restart happening this close to the centre spot is treated as a kick-off.
CENTRE_RADIUS = 0.12


def to_normalised(source, x, y):
    """Convert native tracking coordinates to normalised [0, 1] pitch space."""
    if x is None or y is None:
        return None
    x = float(x)
    y = float(y)
    if source == "raw":
        return (x / RAW_PITCH_LENGTH_CM, y / RAW_PITCH_WIDTH_CM)
    # metrica (and anything already normalised)
    return (x, y)


def to_metres(norm_point):
    """Convert a normalised [0, 1] point to metres on a 105 x 68 m pitch."""
    return (norm_point[0] * PITCH_LENGTH_M, norm_point[1] * PITCH_WIDTH_M)


def distance_m(norm_a, norm_b):
    """Euclidean distance, in metres, between two normalised points."""
    ax, ay = to_metres(norm_a)
    bx, by = to_metres(norm_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def is_out_of_bounds(norm_point, margin=0.0):
    """True when a normalised point sits outside the playing surface."""
    if norm_point is None:
        return True
    x, y = norm_point
    return (
        x < -margin or x > 1 + margin or y < -margin or y > 1 + margin
    )


def near_goal(norm_point, goal_x, depth=GOAL_AREA_DEPTH, half_width=GOAL_HALF_WIDTH):
    """True when a point is within the mouth of the goal at ``goal_x`` (0 or 1)."""
    if norm_point is None:
        return False
    x, y = norm_point
    if goal_x == 0:
        in_depth = x <= depth
    else:
        in_depth = x >= 1 - depth
    return in_depth and abs(y - 0.5) <= half_width


def near_centre(norm_point, radius=CENTRE_RADIUS):
    """True when a point is near the centre spot (used for kick-off detection)."""
    if norm_point is None:
        return False
    x, y = norm_point
    return abs(x - 0.5) <= radius and abs(y - 0.5) <= radius * (PITCH_LENGTH_M / PITCH_WIDTH_M)
