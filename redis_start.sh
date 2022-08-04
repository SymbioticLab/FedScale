#!/bin/bash

# Redis server start script, start with configuration.
# An alternative is start as a subprocess in python; see core/storage/redis_utils.py.

CONFPATH=$FEDSCALE_HOME/redisdata/conf/redis.conf

redis-server $CONFPATH