import redis
import pickle
import sys
import os
import subprocess
import time

def bytes_serialize(data):
    """Serialize input data into bytes.

    Args:
        data (Any): Input to be serialized.

    Returns:
        bytes: Serialized data in bytes.
    """
    return pickle.dumps(data)

def bytes_deserialize(data_bytes):
    """Deserialize bytes.

    Args:
        data_bytes (bytes): Input to be deserialized.

    Returns:
        Any: Original data.
    """
    return pickle.loads(data_bytes)

def start_redis_server(
    executable,
    fedscale_home, 
    host='127.0.0.1', 
    port=6379,
    password='',
):
    """Start the Redis server with specifications as a subprocess.

    Args:
        executable (string): Absolute path to the Redis executable.
        fedscale_home (string): Absolute path to Fedscale working directory.
        host (string, optional): IP address of the Redis server. Defaults to '127.0.0.1'.
        port (int, optional): Port of the Redis server. Defaults to 6379.
        password (string, optional): Password for server side authentication. Defaults to None.

    Raises:
        ValueError: Raised if password contains space characters.
    """
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
    if password != '':
        if ' ' in password:
            raise ValueError('Spaces not permitted in redis password.')
        command += ['--requirepass', password]
    else:
        print("Disabled protected mode since no password is set")
        command += ['--protected-mode no'] # Not safe on public internet
    # start Redis Server as a subprocess
    print(f'Starting Redis server at at {host}:{port}')
    subprocess.Popen(command)

def is_redis_server_online(
    host='127.0.0.1', 
    port=6379, 
    password='', 
    retry=10
):
    """Test if Redis server is online.

    Args:
        host (string, optional): IP address of the Redis server. Defaults to '127.0.0.1'.
        port (int, optional): Port of the Redis server. Defaults to 6379.
        password (string, optional): Password for server side authentication. Defaults to None.
        retry (int, optional): Number of retry times when connection to server fails. Defaults to 10.

    Returns:
        bool: True if Redis server is online, False otherwise.
    """
    client = redis.Redis(host=host, port=port, password=password)
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
    password='',
):
    """Shutdown the Redis server.

    Args:
        host (string, optional): IP address of the Redis server. Defaults to '127.0.0.1'.
        port (int, optional): Port of the Redis server. Defaults to 6379.
        password (string, optional): Password for server side authentication. Defaults to None.
    """
    client = redis.Redis(host=host, port=port, password=password)
    try:
        client.shutdown()
        print(f'Successfully shutdown Redis server at {host}:{port}')
    except Exception:
        pass

def clear_all_keys(
    host='127.0.0.1', 
    port=6379, 
    password='',
):
    """Delete all keys in the Redis server.

    Args:
        host (string, optional): IP address of the Redis server. Defaults to '127.0.0.1'.
        port (int, optional): Port of the Redis server. Defaults to 6379.
        password (string, optional): Password for server side authentication. Defaults to None.
    """
    client = redis.Redis(host=host, port=port, password=password)
    try:
        client.flushall()
        print(f'Successfully cleared all keys')
    except Exception:
        pass

class Redis_client():
    '''Create a redis client connected to specified server.'''

    def __init__(self, host='localhost', port=6379, password=''):
        # Set decode_responses=False to get bytes response,
        # now all values get from redis (including TYPE command) are bytes
        retry = 100
        while not is_redis_server_online(host, port, password, retry):
            print("Waiting for redis server to get online")
            time.sleep(1) # wait until server is online
        self.r = redis.Redis(host=host, port=port, password=password, decode_responses=False)
        print("Successfully created client object")

    def __quit__(self):
        self.r.quit()

    def set_val(self, key, val, bytes=False):
        """Save the value with specified key into Redis server.

        Args:
            key (string): Key for referencing value.
            val (Any): The value to be saved.
            bytes (bool, optional): Set to True if input needs to be serialized, otherwise set to False. 
                                    Defaults to False.

        Returns:
            bool | None: Status of the set operation.
        """
        if not bytes:
            return self.r.set(key, val)
        else:
            return self.r.set(key, bytes_serialize(val))

    def get_val(self, key, type):
        """Get the value specified by key and decode it.

        Args:
            key (string): Key for referencing value.
            type (string): Type of the value.

        Raises:
            ValueError: Raised if type is not one of four options: string | int | float | bytes.

        Returns:
            string | int | float | Any: Decoded value.
        """
        if type in ['bytes']:
            return bytes_deserialize(self.r.get(key))
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
        """Get the raw value specified by key, i.e. undecoded response.

        Args:
            key (string): Key for referencing value.

        Returns:
            bytes | None: Undecoded value.
        """
        return self.r.get(key)

    def delete_key(self, key):
        """Delete the value specified by key.

        Args:
            key (string): Key for referencing value.

        Returns:
            int: Status of the delete operation.
        """
        return self.r.delete(key)

    def dump_to_disk(self):
        """Save the current in-memory database into disk.
        """
        self.r.bgsave()
    
    def update_list(self, key, lst: list, bytes=False):
        """Update list with specified key in Redis server.

        Args:
            key (string): Key for referencing the list.
            lst (list of any): List to be saved.
            bytes (bool, optional): Set to True if input needs to be serialized, otherwise set to False. 
                                    Defaults to False.

        Raises:
            ValueError: Raised if key does not refer to a list or empty value in Redis.

        Returns:
            int: Status of the push operation.
        """
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
            return self.r.rpush(key, [bytes_serialize(s) for s in lst])

    def get_list(self, key, type):
        """Get the list with specified value, and decode each element in it.

        Args:
            key (string): Key for referencing the list.
            type (string): Type of the elements in the list.

        Raises:
            ValueError: Raised if key refers to an non-empty value which is not a list.
            ValueError: Raised if type is not one of four options: string | int | float | bytes.

        Returns:
            list of string | list of int | list of float | list of Any: Decoded list.
        """
        if self.r.type(key) in [b'none']:
            # return empty list if list length is 0
            return []
        elif self.r.type(key) not in [b'list']:
            raise ValueError(f'Key {key} is not a list')
        if type in ['bytes']:
            return [bytes_deserialize(s) for s in self.r.lrange(key, 0, -1)]
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
        """Get the undecoded list specified by key.

        Args:
            key (string): Key for referencing the list.

        Raises:
            ValueError: Raised if key does not refer to a list or empty value in Redis.

        Returns:
            list of bytes: Undecoded list.
        """
        if self.r.type(key) not in [b'list', b'none']:
            raise ValueError(f'Key {key} is not a list')
        return self.r.lrange(key, 0, -1)

    def rpush(self, key, val, bytes=False):
        """Push the value to the right of the list.

        Args:
            key (string): Key for referencing the list.
            val (Any): The value to be pushed into the list.
            bytes (bool, optional): Set to True if input needs to be serialized, otherwise set to False. 
                                    Defaults to False.

        Raises:
            ValueError: Raised if key does not refer to a list or empty value in Redis.

        Returns:
            int: Status of the push operation.
        """
        if self.r.type(key) not in [b'list', b'none']:
            raise ValueError(f'Key {key} is not a list')
        if not bytes:
            return self.r.rpush(key, val)
        else:
            return self.r.rpush(key, bytes_serialize(val))

    def list_len(self, key):
        """Get the length of the list specified by key.

        Args:
            key (string): Key for referencing the list.

        Raises:
            ValueError: Raised if key refers to an non-empty value which is not a list.

        Returns:
            int: Number of elements in the list.
        """
        if self.r.type(key) in [b'none']:
            # return 0 if list length is 0, i.e. no list/empty list exists
            return 0
        elif self.r.type(key) not in [b'list']:
            raise ValueError(f'Key {key} is not a list')
        return self.r.llen(key)

    def exists_key(self, key):
        """Test if key exists in the database.

        Args:
            key (string): Key to be tested.

        Returns:
            int: 1 if key exists, 0 otherwise.
        """
        return self.r.exists(key)

    def type(self, key):
        """Get the type of the key in string.

        Args:
            key (string): Key for referencing the value.

        Returns:
            string: Type of the queried key.
        """
        return self.r.type(key).decode('utf-8')

    def get_client(self):
        """Get a reference to the redis client object.

        Returns:
            Redis: Redis client object.
        """
        return self.r




# test server start 
if __name__ == '__main__':
    redis_exec = '/usr/bin/redis-server'
    fedscale_home = os.environ['FEDSCALE_HOME']
    # print(fedscale_home)
    if not is_redis_server_online():
        start_redis_server(redis_exec, fedscale_home)