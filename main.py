from tensorflow.examples.tutorials.mnist import input_data
from generative_adversarial_network import *
from tf_tools import *
from vis_utils import *
import os


def run_training():

    if not os.path.exists('out/'):
        os.makedirs('out/')

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    n_input = 784
    n_generator_units = [200]
    n_discriminator_units = [100]
    n_latent = 100
    lam = 0.0
    lr = 0.001

    max_epoch = 4000
    batch_size = 100
    n_sample, n_dims = mnist.train.images.shape
    n_batch_each_epoch = n_sample // batch_size

    graph = tf.Graph()

    with graph.as_default():

        model = GenerativeAdversarialNetwork(n_input, n_generator_units, n_discriminator_units, n_latent, lr, lr, lam)

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(max_epoch):
                aver_dis_loss = 0.0
                aver_gen_loss = 0.0
                for step in range(n_batch_each_epoch):
                    # batch_data = mnist.train.next_batch(batch_size)
                    # latent_variables = sample_latent_variables(batch_size, n_latent)
                    # feed_dict = fill_feed_dict(batch_data, latent_variables, model)

                    # tr_dis_loss, _ = sess.run(
                    #     fetches=[model.dis_loss, model.dis_train_op],
                    #     feed_dict=feed_dict
                    # )
                    # tr_gen_loss, _ = sess.run(
                    #     fetches=[model.gen_loss, model.gen_train_op],
                    #     feed_dict=feed_dict
                    # )

                    x, y = mnist.train.next_batch(batch_size)
                    tr_dis_loss, _ = sess.run(
                        fetches=[model.dis_loss, model.dis_train_op],
                        feed_dict={model.x_pl: x,
                                   model.z_pl: sample_latent_variables_normal(batch_size, n_latent)}
                    )

                    tr_gen_loss, _ = sess.run(
                        fetches=[model.gen_loss, model.gen_train_op],
                        feed_dict={model.z_pl: sample_latent_variables_normal(batch_size, n_latent)}
                    )

                    aver_dis_loss += tr_dis_loss
                    aver_gen_loss += tr_gen_loss

                print("epoch %d, tr_dis_loss %f, tr_gen_loss %f" %
                      (epoch, aver_dis_loss / n_batch_each_epoch, aver_gen_loss / n_batch_each_epoch))

                samples = sess.run(fetches=[model.generated_sample],
                                   feed_dict={model.z_pl: sample_latent_variables_normal(16, n_latent)})
                # print(samples[0].shape)
                fig = visualize_generate_samples(samples[0])
                plt.savefig('out/{}.png'.format(str(epoch).zfill(4)), bbox_inches='tight')
                plt.close(fig)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
