GITHUB_TOKEN = "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"
SLACK_BOT_TOKEN = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"


class Config:
    def __init__(self):
        self.debug = False
        self.timeout = 30

    def sort_events(self, events):
        return sorted(events, key=lambda e: e["start"])
