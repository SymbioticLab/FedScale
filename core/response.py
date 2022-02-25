"""Response formats used in FedScale
"""

class BasicResponse(object):

    def __init__(self, executorId, clientId, results=None, status=True):
        self.executorId = executorId
        self.clientId = clientId
        self.results = results
        self.status = status
