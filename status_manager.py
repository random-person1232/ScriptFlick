# status_manager.py

class VideoStatus:
    def __init__(self):
        self.status = "idle"
        self.progress = 0
        self.error = None
        self.current_step = ""
        self.step_details = ""
        self.start_time = None
        self.elapsed_time = 0

    def update_step(self, step: str, details: str = ""):
        self.current_step = step
        self.step_details = details

    def format_elapsed_time(self):
        if not self.elapsed_time:
            return "0:00"
        minutes = int(self.elapsed_time // 60)
        seconds = int(self.elapsed_time % 60)
        return f"{minutes}:{seconds:02d}"

# Create a global instance
video_status = VideoStatus()