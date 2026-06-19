"""
A Possession represents an uninterrupted spell of one player being the closest
player to the ball while the ball is within the possession radius.

The event generator works exclusively in terms of possessions: it builds a list
of possessions for the whole match and then slides a (previous, current, next)
window across them to decide what footballing event happened between them.
"""


class Possession():
    def __init__(self, player, start_frame, start_time, start_loc):
        self.player = player
        # ``team`` is whatever the tracking source stored ("Home"/"Away" for
        # Metrica, 0.0/1.0 for the raw pipeline). It is normalised on export.
        self.team = player.team
        self.start_frame = start_frame
        self.end_frame = start_frame
        self.start_time = start_time
        self.end_time = start_time
        # Ball location (normalised [0, 1]) when the player won and lost the ball.
        self.start_loc = start_loc
        self.end_loc = start_loc
        # Number of frames the player was actually the closest in-radius player.
        self.touches = 1

    def extend(self, frame_number, time, loc):
        """Grow this possession to include another in-radius frame."""
        self.end_frame = frame_number
        self.end_time = time
        if loc is not None:
            self.end_loc = loc
        self.touches += 1

    @property
    def duration(self):
        return self.end_frame - self.start_frame + 1

    def same_player(self, other):
        return other is not None and str(self.player.id) == str(other.player.id)

    def same_team(self, other):
        return other is not None and str(self.team) == str(other.team)

    def __repr__(self):
        return (
            f"Possession(player={self.player.name}, team={self.team}, "
            f"frames={self.start_frame}-{self.end_frame})"
        )
