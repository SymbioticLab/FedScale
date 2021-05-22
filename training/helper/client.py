
class Client(object):

    def __init__(self, hostId, clientId, speed, traces=None):
        self.hostId = hostId
        self.clientId = clientId
        self.compute_speed = speed['computation']
        self.bandwidth = speed['communication']
        self.score = 0
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

    def getCompletionTime(self, batch_size, upload_epoch, upload_size, download_size, augmentation_factor=3.0):
        """
           Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers, 
                                backward-pass takes around 2x the latency, so we multiple it by 3x;
           Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        #return (3.0 * batch_size * upload_epoch/float(self.compute_speed) + model_size/float(self.bandwidth))
        return {'computation':augmentation_factor * batch_size * upload_epoch*float(self.compute_speed)/1000., \
                'communication': (upload_size+download_size)/float(self.bandwidth)}
        # return (augmentation_factor * batch_size * upload_epoch*float(self.compute_speed)/1000. + \
        #         (upload_size+download_size)/float(self.bandwidth))
