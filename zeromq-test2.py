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

    is_producer = (args.job_name == "producer")
    is_consumer = (args.job_name == "consumer")

    # http://zguide.zeromq.org/py:asyncsrv

    if is_producer:
        context = zmq.Context()
        identity = u'worker-%d' % args.task_index
        socket = context.socket(zmq.REQ)
        socket.identity = identity.encode('ascii')
        socket.connect("tcp://127.0.0.1:7000")
    elif is_consumer:
        context = zmq.Context(io_threads=1)
        frontend = context.socket(zmq.ROUTER)
        frontend.bind("tcp://127.0.0.1:7000")
    else:
        raise

    value = np.random.randint(0, 255, size=(84, 84, 4), dtype=np.uint8)

    queue_size = 5
    msg_count = 10**6
    count = 0

    t0 = time.time()

    print("Entering main loop")

    idents = 100*[None]

    while True:
        if is_producer:
            msg = msgpack.packb(value)
            socket.send(msg)
            msg = socket.recv()
            count = msgpack.unpackb(msg)
            if count % 1000 == 0:
                print(count)
        if is_consumer:
            for i in range(queue_size):
                ident, _, msg = frontend.recv_multipart()
                msgpack.unpackb(msg)
                idents[i] = ident
            for i in range(queue_size):
                ident = idents[i]
                count += 1
                msg = msgpack.packb(count)
                frontend.send_multipart([ident, b"", msg])
            if count == msg_count:
                break

    t1 = time.time()
    dt = t1 - t0
    if is_consumer:
        bandwidth = (84*84*4 * msg_count / 1024**2) / dt
        print("Consumed {} messages in {:0.0f} ms, {:0.0f} MB/s ({:0.0f} Mbps)."
              .format(msg_count, 1000*dt, bandwidth, 8*bandwidth))


if __name__ == "__main__":
    main()
