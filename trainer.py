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
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

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


    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "producer" or FLAGS.job_name == "consumer":

        with tf.device(tf.DeviceSpec(job="ps",
                                     replica=0,
                                     task=0,
                                     device_type="CPU")):

            queue = tf.FIFOQueue(10, tf.uint8, shared_name="shared_queue")

        with tf.device(tf.DeviceSpec(job="producer",
                                     replica=0,
                                     task=FLAGS.task_index,
                                     device_type="CPU")):

            input_value = tf.placeholder(tf.uint8, name="input")
            enqueue = queue.enqueue(input_value, name="enqueue")

        with tf.device(tf.DeviceSpec(job="consumer",
                                     replica=0,
                                     task=FLAGS.task_index,
                                     device_type="CPU")):

            dequeue = queue.dequeue(name="dequeue")

        # Create a "supervisor", which oversees the training process.
        init_op = tf.initialize_all_variables()
        summary_op = None
        global_step = None
        saver = None

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/tmp/train_logs",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        is_producer = (FLAGS.job_name == "producer")
        is_consumer = (FLAGS.job_name == "consumer")
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
                    value = sess.run(dequeue)
                    count += 1
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
