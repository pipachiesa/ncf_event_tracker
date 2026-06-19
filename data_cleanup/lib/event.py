"""
Event data structures and export.

An ``Event`` mirrors a row of Metrica Sports event data. An ``EventLog`` is the
ordered collection of events produced for a match and knows how to print itself
and export to the Metrica event-data CSV format:

    Team, Type, Subtype, Period, Start Frame, Start Time [s],
    End Frame, End Time [s], From, To, Start X, Start Y, End X, End Y
"""

import csv
import os


def team_label(team):
    """Normalise a tracking-source team value to "Home"/"Away"."""
    if team is None:
        return ""
    s = str(team).strip().lower()
    if s in ("0", "0.0", "home", "h"):
        return "Home"
    if s in ("1", "1.0", "away", "a"):
        return "Away"
    # Already a readable label (e.g. an actual club name) - keep it as-is.
    return str(team)


class Event():
    def __init__(self, data):
        self.team = data.get("team")
        self.type = data.get("type")
        self.subtype = data.get("subtype", "")
        self.period = data.get("period", 1)
        self.start_frame = data.get("start_frame")
        self.start_time = data.get("start_time")
        self.end_frame = data.get("end_frame")
        self.end_time = data.get("end_time")
        self.from_player = data.get("from_player")
        self.to_player = data.get("to_player")
        # Start/end locations are stored normalised [0, 1].
        self.start_loc = data.get("start_loc")
        self.end_loc = data.get("end_loc")

    def _name(self, player):
        return player.name if player is not None else ""

    def _time(self, value):
        if value is None or value == "":
            return ""
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return value

    def row(self):
        start_x, start_y = self.start_loc if self.start_loc else ("", "")
        end_x, end_y = self.end_loc if self.end_loc else ("", "")
        return [
            team_label(self.team),
            self.type,
            self.subtype or "",
            self.period,
            self.start_frame,
            self._time(self.start_time),
            self.end_frame,
            self._time(self.end_time),
            self._name(self.from_player),
            self._name(self.to_player),
            round(start_x, 5) if start_x != "" else "",
            round(start_y, 5) if start_y != "" else "",
            round(end_x, 5) if end_x != "" else "",
            round(end_y, 5) if end_y != "" else "",
        ]

    def __repr__(self):
        subtype = f"/{self.subtype}" if self.subtype else ""
        frm = self._name(self.from_player)
        to = self._name(self.to_player)
        link = f" {frm} -> {to}" if to else (f" {frm}" if frm else "")
        return (
            f"[{self.start_frame:>5}] {team_label(self.team):>4} "
            f"{self.type}{subtype}{link}"
        )


class EventLog():
    HEADER = [
        "Team", "Type", "Subtype", "Period",
        "Start Frame", "Start Time [s]", "End Frame", "End Time [s]",
        "From", "To", "Start X", "Start Y", "End X", "End Y",
    ]

    def __init__(self, events=None):
        self.events = events or []

    def add(self, event):
        self.events.append(event)

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def filter(self, event_type):
        """Return the events of a given type (e.g. ``log.filter("PASS")``)."""
        return [e for e in self.events if e.type == event_type]

    def summary(self):
        """Return a {event type: count} dictionary."""
        counts = {}
        for event in self.events:
            counts[event.type] = counts.get(event.type, 0) + 1
        return counts

    def rows(self):
        return [self.HEADER] + [event.row() for event in self.events]

    def export(self, path="./output/", file_name="events.csv"):
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, file_name)
        with open(out_path, "w", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerows(self.rows())
        return out_path

    def print(self):
        for event in self.events:
            print(repr(event))
