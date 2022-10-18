#!/usr/bin/env python
from __future__ import print_function

import tensorflow.compat.v1 as tf  # 对比原项目， 这是更新后的tensorflow需要改的地方

tf.disable_v2_behavior()

import cv2
import sys
import random
import numpy as np
from collections import deque  # 双向队列

import warnings

warnings.filterwarnings("ignore")  # 红字警告看得心烦，给他忽略了

# 环境变量的配置， 方便后续导入原作者写的包
sys.path.append("game/")

# 导入原作者实现flappy bird的游戏包
import wrapped_flappy_bird as game

# 游戏参数的设置，在这里可以修改神经网络中的参数
GAME = 'bird'  # 给这个游戏命名，用于保存日志的文件夹命名 the name of the game being played for log files
ACTIONS = 2  # 两个合法运动操作，小鸟要么不动，要么跳跃 number of valid actions
GAMMA = 0.99  # 观测值衰减率 decay rate of past observations
OBSERVE = 100000.  # 训练前观察的时间步骤 timestep to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
INITIAL_EPSILON = 0.0001  # epsilon 的初始值 starting value of epsilon
FINAL_EPSILON = 0.0001  # epsilon 的最终值 final value of epsilon
REPLAY_MEMORY = 50000  # 数据存储管道长 number of previous transitions to remember
BATCH = 32  # 小批量随机梯度下降的batchsize  size of minibatch
FRAME_PER_ACTION = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
    """
    创建网络
    1. (输入) 80*80*4 经过 8*8*4*32(stride=4) 一次卷积  -> 20*20*32
    2. 20*20*32 经过一次 2*2 最大值池化 -> 10*10*32
    3. 10*10*32 经过 4*4*32*64(stride=2) 一次卷积  -> 5*5*64
    4. 5*5*64 经过一次 2*2 最大值池化 -> 3*3*64
    5. 3*3*64 经过 3*3*64*64(stride=1) 一次卷积  -> 3*3*64
    6. 3*3*64 经过一次 2*2 最大值池化 -> 2*2*64
    7. 2*2*64 经过一次reshape -> 256*1
    8. 256*1 经过一次Relu激活函数 -> 256*1
    9. 256*1 进行一次矩阵乘法 -> 2*1 (输出)
    """
    # 卷积网络对应权重， network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # 输入层，input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # 隐藏层，hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2) # 这里原作者考虑后没有加上池化，

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # 输出层，readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    # 输入层，
    return s, readout, h_fc1


def trainNetwork(s, readout, h_fc1, sess):
    """
    训练网络
    """
    # 损失函数，define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])

    # 输出动作
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)

    # 这里的损失函数是L2损失
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)  # 这里使用tensorflow的Adam算法（梯度下降算法的变形）

    # 打开游戏并与程序连接，open up a game state to communicate with emulator
    game_state = game.GameState()

    # 创建一个双向队列，存储一定长度的训练数据，store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # 获得输入层数据，get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)  # 构造矩阵[1, 0]作为初始动作
    do_nothing[0] = 1

    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    # 对输入层数据进行处理，设定阈值等等操作
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 保存训练的数据(若训练中止了), 载入数据(已有初始数据) saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # 这个是为了找训练次数最多的保存数据
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:  # 这里是载入已训练的数据，方便下次继续训练或者观察
        # 注意这里可以阅读readme后面的FAQ，从而把已训练的数据拿出来跑
        saver.restore(sess, checkpoint.model_checkpoint_path)  # 载入数据
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        # 没有的话就直接进行初始的训练
        print("Could not find old network weights")

    # 开始训练，start training
    epsilon = INITIAL_EPSILON
    t = 0  # 记录次数
    while "flappy bird" != "angry bird":  # 程序主循环
        # 输出层最大得分Q， choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:  # 小鸟飞一下
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # 不操控小鸟，do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 执行选择的动作，并且观测下一个状态，从而进行奖惩反馈 并且run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # 保存当前这次行动数据，塞入队列 store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # 取出batch样本 sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # 获取batch的对应变量 get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # Adam算法迭代后走一步 perform gradient step
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch}
            )

        # 迭代值，记录当前第几步  update the old values
        s_t = s_t1
        t += 1

        # 每10000步保存一下数据，用于之后对网络的分析（调参）等等 save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        # 判断当前所处状态 print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        # 打印每一步迭代数据
        print("TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t))

        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''


def playGame():
    # 开始交互会话（构建计算图）
    sess = tf.InteractiveSession()
    # 创建神经网络
    s, readout, h_fc1 = createNetwork()
    # 开始进行训练
    trainNetwork(s, readout, h_fc1, sess)


def main():
    playGame()


if __name__ == "__main__":
    # IDE中右键运行本文件即可，或者命令行python + 本文件名
    main()
