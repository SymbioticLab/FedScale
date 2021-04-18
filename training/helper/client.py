
class Client(object):

    def __init__(self, hostId, clientId, dis, size, speed, traces=None):
        self.hostId = hostId
        self.clientId = clientId
        self.compute_speed = speed[0]
        self.bandwidth = speed[1]
        self.distance = dis
        self.size = size
        self.score = dis
        self.traces = traces
        self.behavior_index = 0

    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward

    def isActive(self, cur_time):
        if self.traces is None:
            return True
            
        norm_time = cur_time % self.traces['finish_time']

        if norm_time > self.traces['inactive'][self.behavior_index]:
            self.behavior_index += 1

        self.behavior_index %= len(self.traces['active'])

        if (self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]):
            return True

        return False

    def getCompletionTime(self, batch_size, upload_epoch, model_size):
        return (3.0 * batch_size * upload_epoch/float(self.compute_speed) + model_size/float(self.bandwidth))
        #return (3.0 * batch_size * upload_epoch*float(self.compute_speed)/1000. + model_size/float(self.bandwidth))