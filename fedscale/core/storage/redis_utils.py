import redis
import pickle
import sys
import os
import subprocess

class Redis_client():
    """Create a redis client connected to specified server."""
    def __init__(self, host="localhost", port=6379, db=0, password=""):
        self.r = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
        try:
            # try a random redis command to check connection
            self.r.randomkey()
        except Exception as e:
            print(e)
            sys.exit(1)

    def set(self, key, val):
        self.r.set(key, val)

    def get(self, key):
        return self.r.get(key)

    def save(self):
        self.r.bgsave()

def model_serialize(model):
    return pickle.dumps(model)

def model_deserialize(model_bytes):
    return pickle.loads(model_bytes)

def save_model(red, key, model):
    red.set(key, model_serialize(model))

def load_model(red, key, model):
    return model_deserialize(red.get(key))

def start_redis_server(
    executable,
    fedscale_home, 
    ip="127.0.0.1", 
    port=6379,
    password=None
):
    command = [executable]
    # Hard code data storing position for Redis server
    working_dir = fedscale_home + "/redisdata"
    command += ["--dir", working_dir]
    pidfile = working_dir + "/redis_" + str(port) + ".pid"
    command += ["--pidfile", pidfile]
    logfile = working_dir + "/redis_" + str(port) + ".log"
    command += ["--logfile", logfile]
    # Other configs
    command += ["--bind", ip]
    command += ["--port", str(port), "--loglevel", "warning"]
    if password:
        if " " in password:
            raise ValueError("Spaces not permitted in redis password.")
        command += ["--requirepass", password]
    # start Redis Server as a subprocess
    # print(command)
    subprocess.Popen(command)

def check_redis_server_online(
    host="127.0.0.1", 
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
            print(f"Connected to Redis server at {host}:{port} in {i + 1} attempts")
            break
    else:
        print(f"Failed to reach Redis server at {host}:{port} after {retry} retries")
    
def shutdown_server(
    host="127.0.0.1", 
    port=6379, 
    password=None
):
    client = redis.Redis(host=host, port=port, password=password)
    try:
        client.shutdown()
        print(f"Successfully shutdown Redis server at {host}:{port}")
    except Exception:
        pass

# test server start 
if __name__ == "__main__":
    redis_exec = "/usr/bin/redis-server"
    fedscale_home = os.environ["FEDSCALE_HOME"]
    # print(fedscale_home)
    start_redis_server(redis_exec, fedscale_home)
    check_redis_server_online()
    r = Redis_client()
    r.set("1", 1)
    print(r.get("1"))
    shutdown_server()
