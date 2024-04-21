import torch
from DQN_Agent import DQN_Agent
from Preprocesing import get_dataset
from PowerTrading import PowerTrading
from DO_NOT_TOUCH.environment import BatteryEnv
from torch.utils.tensorboard import SummaryWriter
import time
writer = SummaryWriter(f'./logs/{time.time()}')

EPISODES = 50
EPISODES_PER_SAVE = 5
EPISODES_PER_EVAL = 1
STEPS_PER_UPDATE = 100
DATASET_PATH = '../data/validation_data.csv'
CHECKPOINT_PATH = './checkpoints'
LOAD_PATH = './checkpoints'


def train_agent(continue_training):
    data = get_dataset(DATASET_PATH)
    # print(len(data))
    env = PowerTrading(BatteryEnv(data))
    agent = DQN_Agent(
        state_size=7,
        replay_mem_size=50000,
        gamma=0.99,
        epsilon=1,
        batch_size=256,
        n_actions=2,
        min_eps=0.05,
        lr=0.0001,
        eps_dec=0.0001
    )
    if continue_training is True:
        prev_ep = agent.load_agent(LOAD_PATH)
    else:
        prev_ep = 0

    for ep in range(prev_ep, EPISODES + 1):
        if ep != 0 and ep % EPISODES_PER_SAVE == 0:
            agent.save_agent(current_episode=ep, checkpoint_dir=CHECKPOINT_PATH)

        score_sum = 0
        loss_sum = 0
        current_state = env.preprocessed_data.iloc[env.battery_env.current_step].values
        for i in range(env.battery_env.episode_length - 1):
            # print(i, row)
            t_current_state = torch.tensor(current_state, dtype=torch.float32).to(agent.model.device)
            # print(t_current_state.size())
            # print(t_current_state)
            action = agent.choose_action(t_current_state)
            # action = 1
            new_state, reward = env.step(action)
            done = False
            score_sum += reward
            # print(current_state, env.actions[action], reward, new_state)
            agent.update_replay_mem(current_state, action, reward, new_state, done)
            loss_sum += agent.train()
            current_state = new_state
            if i % STEPS_PER_UPDATE == 0:
                print(f'Reward: {reward}, Balance: {env.internal_state["total_profit"]}, '
                      f'Charge: {env.internal_state["battery_soc"]}, Step: {env.battery_env.current_step}')
                agent.update_target_model()
                writer.add_scalar('reward', score_sum, ep)
                writer.add_scalar('train loss', loss_sum, ep)

        if ep % EPISODES_PER_EVAL == 0:
            agent.model.eval()
            val_agent(agent)
            agent.model.train()
        print('episode', ep, 'score %.2f' % score_sum, 'loss %.2f' % loss_sum, 'epsilon %.2f' % agent.epsilon)
        env.reset()
    writer.close()

def val_agent(agent: DQN_Agent):
    test_data = get_dataset(DATASET_PATH, test_data=True)
    test_env = PowerTrading(BatteryEnv(test_data))

    current_state = test_env.preprocessed_data.iloc[test_env.battery_env.current_step].values
    for i in range(test_env.battery_env.episode_length - 1):
        t_current_state = torch.tensor(current_state, dtype=torch.float32).to(agent.model.device)
        action = agent.choose_action(t_current_state)
        new_state, reward = test_env.step(action)
        done = False
        agent.update_replay_mem(current_state, action, reward, new_state, done)
        current_state = new_state
    print(f'Validation balance: {test_env.internal_state["total_profit"]}, Validation charge: {test_env.internal_state["battery_soc"]}')

if __name__ == '__main__':
    train_agent(continue_training=False)




