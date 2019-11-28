#!/usr/bin/env python3
import gym, numpy as np, os, rl, tensorflow as tf, time

from lib import dqn_model, common


# Select hyperparameters
params       = common.HYPERPARAMS['star_gunner']
total_frames = params['total_frames']
rep_init     = params['replay_initial']
tgt_net_sync = params['target_net_sync']
batch_size   = params['batch_size']
HIST_LENGTH = 4
UPDATE_FREQ = 4

# Initialise environment and use dqn wrappers
env = rl.common.wrappers.make_atari(params['env_name'])
env = rl.common.wrappers.wrap_deepmind(env=env, stack_frames=HIST_LENGTH)
makeDQN = dqn_model.DQN(env.action_space.n)

# Placeholders
state  = tf.placeholder(tf.float32, shape=[None, HIST_LENGTH, 84, 84], name='state')
action = tf.placeholder(tf.float32, shape=[None, env.action_space.n], name='action')
reward = tf.placeholder(tf.float32, shape=[None], name='reward')
done   = tf.placeholder(tf.float32, shape=[None], name='done')
state2 = tf.placeholder(tf.float32, shape=[None, HIST_LENGTH, 84, 84], name='next_state')

# Loss function
net_q, net_vars = makeDQN.create_model(state, name='online')
tgt_q, tgt_vars = makeDQN.create_model(state2, name='target')
q = tf.reduce_sum(net_q * action, axis=1)
max_tgt_q = tf.reduce_max(tgt_q, axis=1)
tgt = reward + (1. - done) * params['gamma'] * max_tgt_q
delta = tf.stop_gradient(tgt) - q
loss = tf.where(tf.abs(delta) < 1.0, 0.5 * tf.square(delta), tf.abs(delta) - 0.5)

# Training step
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
train_step = optimizer.minimize(loss, global_step=global_step)

# Operation to copy the online DQN to the target DQN
copy_ops = [tgt_var.assign(net_var) for tgt_var, net_var in zip(tgt_vars, net_vars)]
sync_nets = tf.group(*copy_ops)


# Initialise session
with tf.Session() as sess:
    # Create summary writer and saver
    summ_name = params['run_name']+'_test'
    writer = tf.summary.FileWriter(os.path.join(os.pardir, 'logs', summ_name), sess.graph)
    saver = tf.train.Saver(max_to_keep=100000)
    save_dir = os.path.join('D:', 'Models', summ_name, summ_name)

    # Initialise weights and copy from net to target net
    tf.global_variables_initializer().run()
    sync_nets.run()
    # Action selector
    selector = rl.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    # Epsilon increment
    epsilon_tracker = common.EpsilonTracker(selector, params)
    # DQN agent
    agent = rl.agent.DQNAgent(state, net_q, selector)
    # Experience source
    exp_source = rl.experience_ptan.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'])
    # Memory buffer
    buffer = rl.experience_ptan.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])

    frame_idx = 0
    with common.RewardTracker(writer) as reward_tracker:
        # Initial save
        saver.save(sess, save_dir, global_step=global_step)
        while frame_idx < total_frames:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon)

            # Don't train while filling memory
            if len(buffer) < rep_init:
                continue

            if frame_idx % UPDATE_FREQ == 0:
                # Generate and unpack batch
                batch = buffer.sample(batch_size)
                s, a, r, t, s2 = common.unpack_batch(batch)
                a_1h = np.zeros((batch_size, env.action_space.n), dtype=np.float32)
                a_1h[np.arange(batch_size), a] = 1.0
                # Train
                sess.run(train_step, feed_dict={state : s,
                                                action: a_1h,
                                                reward: r,
                                                done  : t,
                                                state2: s2})

            # Copy current network to target network
            if frame_idx % (UPDATE_FREQ * tgt_net_sync) == 0:
                sync_nets.run()

            # Save current network every 250,000 ATARI frames
            if frame_idx % 62500 == 0:
            	saver.save(sess, save_dir, global_step=global_step)
        # Final save
        saver.save(sess, save_dir, global_step=global_step)
