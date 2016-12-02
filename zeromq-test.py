#!/usr/bin/env python

import argparse
import time
import numpy as np
import zmq
import msgpack
import msgpack_numpy as m

m.patch()

# https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/client_server.html

# Assume a single consumer and multiple producers.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str)
    parser.add_argument('--task_index', type=int)
    args = parser.parse_args()

    context = zmq.Context()

    is_producer = (args.job_name == "producer")
    is_consumer = (args.job_name == "consumer")

    if is_producer:
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://127.0.0.1:7000")
    elif is_consumer:
        socket = context.socket(zmq.REP)
        socket.bind("tcp://127.0.0.1:7000")
    else:
        raise

    value = np.random.randint(0, 255, size=(84, 84, 4), dtype=np.uint8)

    msg_count = 10**6
    count = 0

    t0 = time.time()

    while True:
        if is_producer:
            msg = msgpack.packb(value)
            socket.send(msg)
            msg = socket.recv()
            count = msgpack.unpackb(msg)
            if count == msg_count:
                break
        if is_consumer:
            msg = socket.recv()
            value = msgpack.unpackb(msg)
            count += 1
            msg = msgpack.packb(count)
            socket.send(msg)
            if count == msg_count:
                break

    t1 = time.time()
    dt = t1 - t0
    if is_consumer:
        print("Consumed {} messages in {} ms".format(msg_count, int(1000*dt)))


if __name__ == "__main__":
    main()
