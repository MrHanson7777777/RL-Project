from reinforce import ReinforceAgent
from ac import ActorCriticAgent
from ppo import PPO
from sac import SAC, ReplayBuffer
from ddpg import DDPG
from ppo import PPO

# 导入OpenAI Gym环境
import gym

# 主函数
def main():

    # 创建环境
    env = gym.make('CartPole-v0')
    gamma = 0.98
    epochs = 10
    lamda = 0.95
    eps = 0.2
    tau = 0.005  # 软更新参数
    target_entropy = -1
    buffer_size = 10000

    # 创建多个代理
    # reinforce_agent = ReinforceAgent(env.observation_space.shape[0], 16, env.action_space.n, gamma)
    # actor_critic_agent = ActorCriticAgent(env.observation_space.shape[0], 128, env.action_space.n, gamma)
    # ppo_agent = PPO(env.observation_space.shape[0], 128, env.action_space.n, gamma, lamda, epochs, eps)
    # sac_agent = SAC(env.observation_space.shape[0], 128, env.action_space.n, target_entropy, tau, gamma)
    # ddpg_agent = DDPG(env.observation_space.shape[0], 1, 16, env.action_space.n)
    # ppo_agent = PPO(env.observation_space.shape[0], 1, 16, env.action_space.n)

    # 设置训练次数
    num_episodes = 1000

    # 训练并记录每个代理的总奖励
    reinforce_rewards = []
    actor_critic_rewards = []
    ppo_rewards = []
    sac_rewards = []
    ddpg_rewards = []
    ppo_rewards = []

    for episode in range(num_episodes):
        # 训练 Reinforce 代理
        # reinforce_reward = reinforce_agent.train(env)
        # reinforce_rewards.append(reinforce_reward)

        # 训练 Actor-Critic 代理
        # actor_critic_reward = actor_critic_agent.train(env)
        # actor_critic_rewards.append(actor_critic_reward)

        # 训练 PPO 代理
        # ppo_reward = ppo_agent.train(env)
        # ppo_rewards.append(ppo_reward)

        # 训练 SAC 代理
        # sac_reward = sac_agent.train(env, replay_buffer)
        # sac_rewards.append(sac_reward)

        # 训练 DDPG 代理
        # ddpg_reward = ddpg_agent.train(env)
        # ddpg_rewards.append(ddpg_reward)

        # 打印每个代理的奖励
        # print("Reinforce Agent 奖励:", reinforce_reward)
        # print("Actor-Critic Agent 奖励:", actor_critic_reward)
        # print("PPO Agent 奖励:", ppo_reward)
        # print("SAC Agent 奖励:", sac_reward)
        # print("DDPG Agent 奖励:", ddpg_reward)

        # 画图对比实验结果


if __name__ == "__main__":
    main()
