class DeviceNotConnectedError(ConnectionError):


    def __init__(self, message="This device is not connected. Try calling `connect()` first."):
        self.message = message
        super().__init__(self.message)


class DeviceAlreadyConnectedError(ConnectionError):


    def __init__(
        self,
        message="This device is already connected. Try not calling `connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)
