# TensorFlow Distributed Queues Are Slow

A small test to examine the slowness of TensorFlow FIFO queues.

## How to run

1. Install tmuxinator and tmux with `sudo apt-et install tmuxinator tmux`.
2. Copy `queue.yml` to `~/.tmuxinator/queue.yml`.
3. Start the example by running `mux start queue`.
4. Detach from the session with `CTRL-b d`.
5. Kill the session with `tmux kill-session queue`.

## Performance analysis

* A state observation is 84x84x4 bytes (28224 bytes).
* 4 producers; one consumer; and one parameter server.
* Each producer pushes one observation at a time.
* The consumer pops many observations ("minibatch" size).

Using `queue_size = 100` and `dequeue_count = 10`:

    consumed 10000 messages in 46.950629234313965 secs.

Increasing, or decreasing, the number of producers does not affect this.

Using a minibatch of 1 by setting `dequeue_count = 1` and keeping 4 producers:

    consumed 10000 messages in 59.1890332698822 secs.

Using `queue_size = 200`; `dequeue_count = 100`; and 4 producers:

    consumed 10000 messages in 36.2074179649353 secs.
