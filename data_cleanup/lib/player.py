from lib.frame import Frame
import matplotlib.pyplot as plt

class Player():
        def __init__(self, player_data):
            self.id = player_data["id"]
            self.team = player_data["team"]
            self.name = player_data["name"]
            self.frames = []
            self.source = None
            self._frame_index = {}  # {str(frame_number): Frame} for O(1) lookup

        def change_team(self, team):
            self.team = team

        def change_name(self, name):
            self.name = name
            
        def add_frame(self, frame_data):
            frame_obj = frame_data if isinstance(frame_data, Frame) else Frame(frame_data)
            self.frames.append(frame_obj)
            self._frame_index[str(frame_obj.frame)] = frame_obj

        def frame(self, frame_number):
            key = str(frame_number)
            hit = self._frame_index.get(key)
            if hit is not None:
                return hit
            # Fallback (and self-heal) in case the index is stale.
            for frame in self.frames:
                if(str(frame.frame) == key):
                    self._frame_index[key] = frame
                    return frame
            return None
        
        def still_exist_after_frame(self, frame_number):
            for frame in self.frames[(frame_number+1):]:
                if frame.exists:
                    return True
            return False
        
        def last_frame(self):
            last_frame = 0
            for i in range(len(self.frames)):
                if self.frames[i].exists:
                    last_frame = i
            return last_frame
        
        def path(self):
            path = []
            frame_number = 1
            for frame in self.frames:
                coor = frame.coordinates
                if coor:
                    path.append((coor[0], coor[1], frame_number))
                else:
                    path.append((0, 0, frame_number))
                frame_number += 1
            return path
        
        def replace_frame(self, frame_number, new_frame):
            frame = self.frame(frame_number)

            if(self.source == "raw"):
                self.frames[frame_number - 1] = Frame({
                    "frame" : frame_number,
                    "time" : frame.time,
                    "x" : new_frame.x,
                    "y" : new_frame.y,
                    "frame_x1" : new_frame.frame_x1,
                    "frame_y1" : new_frame.frame_y1,
                    "frame_x2" : new_frame.frame_x2,
                    "frame_y2" : new_frame.frame_y2,
                    "empty" : new_frame.empty
                })
            else:
                self.frames[frame_number - 1] = Frame({
                    "frame" : frame_number,
                    "time" : frame.time,
                    "x" : new_frame.x,
                    "y" : new_frame.y,
                    "empty" : new_frame.empty
                })

            self._frame_index[str(frame_number)] = self.frames[frame_number - 1]

        def plot(self):
            coordinates = self.path()
            x_values, y_values, z_values = zip(*coordinates) if coordinates else ([], [], [])

            norm = plt.Normalize(min(z_values), max(z_values))
            cmap = plt.cm.viridis

            plt.figure(figsize=(6, 4))
            plt.scatter(x_values, y_values, c=z_values, cmap=cmap, norm=norm, marker='o')

            plt.title("Ball Path")
            plt.grid(True)
            plt.gca().invert_yaxis()
            plt.savefig(self.name + ' Path.png', dpi=300)