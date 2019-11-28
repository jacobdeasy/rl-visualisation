import sys
import time
import tensorflow as tf
import numpy as np


HYPERPARAMS = {
    'star_gunner': {
        'env_name':         'StarGunnerNoFrameskip-v4',
        'total_frames':     50000000,
        'run_name':         'star_gunner',
        'replay_size':      5*10**5,
        'replay_initial':   5*10**4,
        'target_net_sync':  10**4,
        'epsilon_frames':   10**6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    },
    'breakout': {
        'env_name':         'BreakoutNoFrameskip-v4',
        'total_frames':     50000000,
        'run_name':         'breakout',
        'replay_size':      5*10**5,
        'replay_initial':   5*10**4,
        'target_net_sync':  10**4,
        'epsilon_frames':   10**6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00005,
        'gamma':            0.99,
        'batch_size':       32
    },
    'space_invaders': {
        'env_name':         'SpaceInvadersNoFrameskip-v4',
        'total_frames':     50000000,
        'run_name':         'space_invaders',
        'replay_size':      5*10**5,
        'replay_initial':   5*10**4,
        'target_net_sync':  10**4,
        'epsilon_frames':   10**6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00005,
        'gamma':            0.99,
        'batch_size':       32
    },
    'freeway': {
        'env_name':         'FreewayNoFrameskip-v4',
        'total_frames':     12500000,
        'run_name':         'freeway',
        'replay_size':      500000,  # Half memory size
        'replay_initial':   5*10**4,
        'target_net_sync':  10**4,
        'epsilon_frames':   10**6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00005,
        'gamma':            0.99,
        'batch_size':       32
    }
}


def unpack_batch(batch):
    states, actions, rewards, dones, state2s = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            state2s.append(state)  # appending this state does not matter
        else:
            state2s.append(np.array(exp.last_state, copy=False))
    return np.array(states,  dtype=np.float32, copy=False),  \
           np.array(actions, dtype=np.uint8),  \
           np.array(rewards, dtype=np.float32),  \
           np.array(dones,   dtype=np.float32),  \
           np.array(state2s, dtype=np.float32, copy=False)


class RewardTracker:
    """
    Track reward and other statistics through training.
    """
    def __init__(self, writer):
        self.writer = writer

    def __enter__(self):
        """
        Initialise time, 0 frame number and empty list for game rewards.
        """
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        # Add new reward
        self.total_rewards.append(reward)
        # Record FPS before resetting counter and timer
        speed = 4 * (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        # Rolling mean
        mean_reward = np.mean(self.total_rewards[-100:])
        # Print statement
        epsilon_str = '' if epsilon is None else ', eps %.2f' % epsilon
        print('%d: done %d games, mean reward %.2f, speed %.2f f/s%s' % (
            4 * frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        # Summaries
        summ1 = tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=reward)])
        self.writer.add_summary(summ1, global_step=4*frame)
        summ2 = tf.Summary(value=[tf.Summary.Value(tag='reward_100', simple_value=mean_reward)])
        self.writer.add_summary(summ2, global_step=4*frame)
        summ3 = tf.Summary(value=[tf.Summary.Value(tag='speed', simple_value=speed)])
        self.writer.add_summary(summ3, global_step=4*frame)


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.replay_initial = params['replay_initial']
        self.eps_start = params['epsilon_start']
        self.eps_end = params['epsilon_final']
        self.eps_endt = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            self.eps_end + max(0, (self.eps_start - self.eps_end) * (self.eps_endt - \
            max(0, frame - self.replay_initial)) / self.eps_endt)
