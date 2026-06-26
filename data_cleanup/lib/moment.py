class Moment():
    def __init__(self, moment_data):
        self.frame = moment_data["frame"]
        self.time = moment_data["time"]
        self.ball = moment_data["ball"]
        self.players = moment_data["players"]

    def ball_loc(self):
        frame = self.ball["frame"]
        if frame is None:
            return None
        return frame.coordinates

    def player_loc(self, player_id):
        for player_data in self.players:
            player = player_data["object"]
            if(str(player_id) == str(player.id)):
                frame = player_data["frame"]
                if frame is None:
                    return None
                return frame.coordinates
        return None
        