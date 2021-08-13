import random
import argparse
import numpy as np
import LoadData as data
from SNE import SNE
import tensorflow as tf

# Set random seeds
SEED = 2016
random.seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Run SNE.")
    parser.add_argument('--data_path', nargs='?', default='../UNC/',
                        help='Input data path')
    parser.add_argument('--id_dim', type=int, default=20,
                        help='Dimension for id_part.')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--n_neg_samples', type=int, default=10,
                        help='Number of negative samples.')
    parser.add_argument('--attr_dim', type=int, default=20,
                        help='Dimension for attr_part.')
    return parser.parse_args()

#################### Util functions ####################


def run_SNE( data, id_dim, attr_dim ):
    model = SNE( data, id_embedding_size=id_dim, attr_embedding_size=attr_dim)
    model.train( )


if __name__ == '__main__':
    args = parse_args()
    print("data_path: ", args.data_path)
    path = args.data_path
    Data = data.LoadData( path , SEED)
    print("Total training links: ", len(Data.links))
    print("Total epoch: ", args.epoch)

    print('id_dim :', args.id_dim)
    print('attr_dim :', args.attr_dim)

    # weights:(10,2), biases:10, labels： （3，1）， input:(3,2)
    a = tf.nn.sampled_softmax_loss(weights=tf.constant(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]),
                                   # [num_classes, dim] = [10, 2]
                                   biases=tf.constant([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                                   # [num_classes] = [10]
                                   labels=tf.constant([[2], [3], [5]]),
                                   # [batch_size, num_true] = [3, 1]
                                   inputs=tf.constant([[0.2, 0.1], [0.4, 0.1], [0.22, 0.12]]),
                                   # [batch_size, dim] = [3, 2]

                                   num_sampled=3,
                                   num_classes=10,
                                   # num_true=1,
                                   # seed=2020,
                                   # name="sampled_softmax_loss"
                                   )
    # weights:(18163,40), biases:(18163,) embed_layer:(?,40),label:(?,1), n_neg_samples:10
    # tf.nn.sampled_softmax_loss(self.weights['out_embeddings'], self.weights['biases'], self.embed_layer,
    #                            self.train_labels, self.n_neg_samples, self.node_N)

    run_SNE( Data, args.id_dim, args.attr_dim)



