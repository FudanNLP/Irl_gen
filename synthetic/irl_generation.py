import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from rewarder import Rewarder
from rollout_ppo import ROLLOUT
from target_lstm import TARGET_LSTM
import cPickle
import os
import time

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 50 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 256
G_rate = 0.005

#########################################################################################
#  Reward Hyper-parameters
#########################################################################################
MID_LAYER_G = [128]
MID_LAYER_R = [512]
re_dropout_keep_prob = 0.7
re_l2_reg_lambda = 1e-5
re_batch_size = BATCH_SIZE # must equal
entropy_w = 0.09
R_decay = 16 # SGD learn epoch decay
R_rate = 0.01


#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 100
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample'+str(entropy_w)+'.txt'
eval_file = 'save/eval_file'+str(entropy_w)+'.txt'
generated_num = 10000
off_num = 1536  #off_policy samples(use PPO2)
restore = False


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        samples = trainable_model.generate(sess)
        generated_samples.extend(samples)

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def off_policy_samples(sess, trainable_model, batch_size, generated_num):
    off_policy_sample = []
    off_policy_probs = []
    for _ in range(int(generated_num / batch_size)):
        samples, sample_probs = trainable_model.generate(sess)
        off_policy_sample.append(samples)
        off_policy_probs.append(sample_probs)

    return off_policy_sample, off_policy_probs



def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def sigmoid(x):
  return 1/(1+np.exp(-x))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    vocab_size = 5000
    dis_data_loader = Dis_dataloader(re_batch_size)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, MID_LAYER_G)
    rewarder = Rewarder(vocab_size, BATCH_SIZE, EMB_DIM * 4, HIDDEN_DIM * 4, SEQ_LENGTH, START_TOKEN, MID_LAYER_R, l2_reg_lambda=re_l2_reg_lambda)
    target_params = cPickle.load(open('save/target_params.pkl'))
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)
    # ground_loss = target_loss(sess, target_lstm, gen_data_loader)
    # print 'Ground-Truth:', ground_loss

    log = open('save/experiment-ent'+str(entropy_w), 'w')
    #  pre-train generator
    if restore is False:
        print 'Start pre-training...'
        log.write('pre-training...\n')
        for epoch in xrange(PRE_EPOCH_NUM):
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            if epoch % 5 == 0:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                print 'pre-train epoch ', epoch, 'test_loss ', test_loss
                buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
                log.write(buffer)

        print 'Start pre-training rewarder...'
        start = time.time()
        for _ in range(1):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(1):
                dis_data_loader.reset_pointer()
                r_losses = []
                for it in xrange(dis_data_loader.num_batch):
                    x_text = dis_data_loader.next_batch()
                    _, r_loss = rewarder.reward_train_step(sess, x_text, np.ones(BATCH_SIZE), 1.0, re_dropout_keep_prob, 0.01)
                    r_losses.append(r_loss)
                print 'reward_loss', np.mean(r_losses)
        speed = time.time() - start
        print 'Reward pre_training Speed:{:.3f}'.format(speed)

        checkpoint_path = os.path.join('save', 'exper_40.ckpt')
        saver.save(sess, checkpoint_path)
    else:
        print 'Restore pretrained model ...'
        log.write('Restore pre-trained model...\n')
        ckpt = tf.train.get_checkpoint_state('save')
        saver.restore(sess, ckpt.model_checkpoint_path)

    # by setting the parameters to 0.0 and 1.0, we didn't use the mixed policy RL training in SeqGAN
    rollout = ROLLOUT(generator, 0.0, 1.0)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):

        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            log.write(buffer)

        # Train the generator for one step
        start = time.time()
        g_losses = []
        off_samples, off_probs = off_policy_samples(sess, rollout, BATCH_SIZE, off_num)
        avg_reward = []
        for g_it in range(1):
            for it in range(off_num // BATCH_SIZE):
                rewards = rollout.get_reward(sess, off_samples[it], 8, rewarder)
                avg_reward.append(rewards)
            baseline = np.zeros(SEQ_LENGTH)
            for it in range(1):
                for it2 in range(off_num // BATCH_SIZE):
                    _, g_loss = generator.rl_train_step(sess, off_samples[it2], avg_reward[it2], baseline, off_probs[it2], entropy_w, G_rate)
                    g_losses.append(g_loss)
        speed = time.time() - start
        print 'MaxentPolicy Gradient {} round, Speed:{:.3f}, Loss:{:.3f}'.format(total_batch, speed, np.mean(g_losses))

        # Update roll-out parameters
        rollout.update_params()

        # Train the rewarder
        start = time.time()
        r_loss_list = []
        for _ in range(2):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_text= dis_data_loader.next_batch()
                    weights = rewarder.reward_weight(sess, x_text, generator)
                    _, r_loss = rewarder.reward_train_step(sess, x_text, weights, 1, re_dropout_keep_prob, R_rate * np.exp(-(total_batch // R_decay)))
                    r_loss_list.append(r_loss)
        speed = time.time() - start
        print 'Reward training {} round, Speed:{:.3f}, Loss:{:.3f}'.format(total_batch, speed, np.mean(r_loss_list))

    log.close()


if __name__ == '__main__':
    main()
