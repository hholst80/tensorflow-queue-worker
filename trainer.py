import time
import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("producer_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("consumer_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'producer'")
tf.app.flags.DEFINE_integer("task_index", -1, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    producer_hosts = FLAGS.producer_hosts.split(",")
    consumer_hosts = FLAGS.consumer_hosts.split(",")

    # Create a cluster from the parameter server and producer hosts.
    cluster = tf.train.ClusterSpec({
        "ps": ps_hosts,
        "producer": producer_hosts,
        "consumer": consumer_hosts,
    })

    # Create and start a server for the local task.
    config = tf.ConfigProto(log_device_placement=True)
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             config=config)

    queue_size = 200
    dequeue_count = 100

    is_ps = (FLAGS.job_name == "ps")
    is_producer = (FLAGS.job_name == "producer")
    is_consumer = (FLAGS.job_name == "consumer")

    if is_ps:
        server.join()
    elif is_producer or is_consumer:

        with tf.device('/job:ps/task:0'):

            queue = tf.FIFOQueue(queue_size, tf.uint8,
                                 shapes=[(84, 84, 4)],
                                 shared_name="shared_queue")

        if is_producer:
            with tf.device('/job:producer/task:%d' % FLAGS.task_index):
                input_value = tf.placeholder(tf.uint8, name="input")
                enqueue = queue.enqueue(input_value, name="enqueue")

        if is_consumer:
            with tf.device('/job:consumer/task:%d' % FLAGS.task_index):
                dequeue = queue.dequeue_many(dequeue_count, name="dequeue")

        # Create a "supervisor", which oversees the training process.
        init_op = tf.initialize_all_variables()
        summary_op = None
        global_step = None
        saver = None

        sess = server.target

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/tmp/train_logs",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.

        count = 0
        with sv.managed_session(server.target) as sess:
            t0 = time.time()
            msg_count = 10**4
            value = np.random.randint(0, 255, size=(84, 84, 4),
                                      dtype=np.uint8)
            while True:
                if is_producer:
                    feed_dict = {
                        input_value: value,
                    }
                    s0 = time.time()
                    sess.run(enqueue, feed_dict=feed_dict)
                    s1 = time.time()
                    ds = s1 - s0
                    if ds > 0.1:
                        logger.info("enqueue was blocked for {}ms"
                                    .format(int(1000*ds)))
                elif is_consumer:
                    sess.run(dequeue)
                    count += dequeue_count
                    if count == msg_count:
                        break
            t1 = time.time()
            dt = t1 - t0
            print("consumed {} messages in {} secs.".format(msg_count, dt))
            sv.stop()

        # Ask for all the services to stop.
        sv.stop()

if __name__ == "__main__":
    tf.app.run()
