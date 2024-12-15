class VideoStatus:
    def __init__(self):
        self.status = "idle"
        self.progress = 0
        self.error = None
        self.current_step = ""
        self.step_details = ""
        self.task_id = None
        self.video_path = None

    def update(self, status=None, progress=None, error=None, current_step=None, step_details=None, video_path=None):
        if status is not None:
            self.status = status
        if progress is not None:
            self.progress = progress
        if error is not None:
            self.error = error
        if current_step is not None:
            self.current_step = current_step
        if step_details is not None:
            self.step_details = step_details
        if video_path is not None:
            self.video_path = video_path

    def update_step(self, step_name, details="", progress=None):
        """
        Update the current step and optionally progress
        Args:
            step_name (str): Name of the current step
            details (str): Additional details about the step
            progress (float, optional): Current progress percentage
        """
        self.current_step = step_name
        self.step_details = details
        if progress is not None:
            self.progress = progress

video_status = VideoStatus()