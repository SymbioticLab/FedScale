import redis
import pickle
import sys
import os
import subprocess
import time

class Redis_client():
    '''Create a redis client connected to specified server.'''
    def __init__(self, host='localhost', port=6379, password=''):
        # Set decode_responses=False to get bytes response,
        # now all values get from redis (including TYPE command) is bytes
        self.r = redis.Redis(host=host, port=port, password=password, decode_responses=False)
        try:
            # try a random redis command to check connection
            self.r.randomkey()
        except Exception as e:
            print("exception")
            sys.exit(1)

    def __quit__(self):
        self.r.quit()

    def set_val(self, key, val, bytes=False):
        if not bytes:
            return self.r.set(key, val)
        else:
            return self.r.set(key, model_serialize(val))

    def get_val(self, key, type):
        if type in ['bytes']:
            return model_deserialize(self.r.get(key))
        else:
            ret_val = self.r.get(key).decode('utf-8')
            if type in ['string']:
                return ret_val
            elif type in ['int']:
                return int(ret_val)
            elif type in ['float']:
                return float(ret_val)
            else:
                raise ValueError(f'Unrecognized type: {type}')
    
    def get_val_raw(self, key):
        return self.r.get(key)

    # def get_val_str(self, key):
    #     return self.r.get(key).decode('utf-8')

    def delete_key(self, key):
        return self.r.delete(key)

    def dump_to_disk(self):
        self.r.bgsave()

    def create_list(self, key, lst: list, bytes=False):
        if self.r.exists(key):
            raise ValueError(f'Key {key} already exists in database')
        if len(lst) == 0:
            # do nothing
            return 1
        if not bytes:
            return self.r.rpush(key, *lst)
        else:
            return self.r.rpush(key, [model_serialize(s) for s in lst])
    
    def update_list(self, key, lst: list, bytes=False):
        if self.r.type(key) not in [b'list', b'none']:
            raise ValueError(f'Key {key} is not a list')
        if len(lst) == 0:
            # update to an empty list
            self.r.delete(key)
            return 1
        if not bytes:
            self.r.delete(key)
            return self.r.rpush(key, *lst)
        else:
            self.r.delete(key)
            return self.r.rpush(key, [model_serialize(s) for s in lst])

    def get_list(self, key, type):
        if self.r.type(key) in [b'none']:
            # return empty list if list length is 0
            return []
        elif self.r.type(key) not in [b'list']:
            raise ValueError(f'Key {key} is not a list')
        if type in ['bytes']:
            return [model_deserialize(s) for s in self.r.lrange(key, 0, -1)]
        else:
            ret_list = [s.decode('utf-8') for s in self.r.lrange(key, 0, -1)]
            if type in ['string']:
                return ret_list
            elif type in ['int']:
                return [int(i) for i in ret_list]
            elif type in ['float']:
                return [float(f) for f in ret_list]
            else:
                raise ValueError(f'Unrecognized type: {type}')
    
    def get_list_raw(self, key):
        if self.r.type(key) not in [b'list', b'none']:
            raise ValueError(f'Key {key} is not a list')
        return self.r.lrange(key, 0, -1)

    # def get_list_str(self, key):
    #     if self.r.type(key) not in [b'list', b'none']:
    #         raise ValueError(f'Key {key} is not a list')
    #     return [s.decode('utf-8') for s in self.r.lrange(key, 0, -1)]

    def rpush(self, key, val, bytes=False):
        if self.r.type(key) not in [b'list', b'none']:
            raise ValueError(f'Key {key} is not a list')
        if not bytes:
            return self.r.rpush(key, val)
        else:
            return self.r.rpush(key, model_serialize(val))

    def list_len(self, key):
        if self.r.type(key) in [b'none']:
            # return 0 if list length is 0, i.e. no list/empty list exists
            return 0
        elif self.r.type(key) not in [b'list']:
            raise ValueError(f'Key {key} is not a list')
        return self.r.llen(key)

    def exists_key(self, key):
        return self.r.exists(key)

    def type(self, key):
        return self.r.type(key)

    def get_client(self):
        return self.r


def model_serialize(model):
    return pickle.dumps(model)

def model_deserialize(model_bytes):
    return pickle.loads(model_bytes)

def save_model(red, key, model):
    red.set(key, model_serialize(model))

def load_model(red, key):
    return model_deserialize(red.get(key))

def start_redis_server(
    executable,
    fedscale_home, 
    host='127.0.0.1', 
    port=6379,
    password=None
):
    command = [executable]
    # Hard code data storing position for Redis server
    working_dir = fedscale_home + '/redisdata'
    command += ['--dir', working_dir]
    pidfile = working_dir + '/redis_' + str(port) + '.pid'
    command += ['--pidfile', pidfile]
    logfile = working_dir + '/redis_' + str(port) + '.log'
    command += ['--logfile', logfile]
    # Other configs
    command += ['--bind', host]
    command += ['--port', str(port), '--loglevel', 'warning']
    if password:
        if ' ' in password:
            raise ValueError('Spaces not permitted in redis password.')
        command += ['--requirepass', password]
    # start Redis Server as a subprocess
    # print(command)
    print(f'Starting Redis server at at {host}:{port}')
    subprocess.Popen(command)

def is_redis_server_online(
    host='127.0.0.1', 
    port=6379, 
    password=None, 
    retry=10
):
    client = redis.Redis(host=host, port=port, password=password)
    import time
    for i in range(retry):
        try:
            # try a redis command to check connection
            client.randomkey()
        except Exception as e:
            time.sleep(0.1)
        else:
            print(f'Connected to Redis server at {host}:{port} in {i + 1} attempts')
            return True
    else:
        print(f'Failed to reach Redis server at {host}:{port} after {retry} retries')
        return False
    
def shutdown_server(
    host='127.0.0.1', 
    port=6379, 
    password=None,
    nosave=False
):
    client = redis.Redis(host=host, port=port, password=password)
    try:
        client.shutdown(nosave=nosave)
        print(f'Successfully shutdown Redis server at {host}:{port}')
    except Exception:
        pass

class AnObject():
    def __init__(self):
        self.v1 = 'abc'
        self.v2 = 1
    
    def print(self):
        print(self.v1)
        print(self.v2)

# test server start 
if __name__ == '__main__':
    redis_exec = '/usr/bin/redis-server'
    fedscale_home = os.environ['FEDSCALE_HOME']
    # print(fedscale_home)
    if not is_redis_server_online():
        start_redis_server(redis_exec, fedscale_home)
    
    print("Sleeping")
    time.sleep(5)
    print("Going")


    r = Redis_client()

    print("Shutting down")
    time.sleep(5)
    shutdown_server()
    print("Shut down")

    obj = AnObject()
    # obj_se = model_serialize(obj)
    # print(obj_se)
    # obj_de = model_deserialize(obj_se)
    # obj_de.print()

    # r.set_val('obj', model_serialize(obj))
    # obj_se_red = r.get_val('obj')
    # print(r.type('obj'))
    # print(obj_se_red)

    # obj_de_red = model_deserialize(obj_se_red)
    # obj_de_red.print()

    # lst = ['a', 'bb', 'ccc']
    # r.create_list("lst", lst)
    # print(r.get_list('lst'))

    # r = Redis_client()
    # obj = AnObject()
    # objlist = [obj, obj, obj]
    # objlist_se = [model_serialize(o) for o in objlist]
    # # r.create_list('objlist', objlist_se)
    # try:
    r.type("x")
    # except Exception as e:
    #     print("Caught")
    # print(r.get_list('objlist'))
    # objlist_de = [model_deserialize(o) for o in r.get_list('objlist')]
    # for o in objlist_de:
    #     o.print()

    # shutdown_server()
