from algorithms.reinforce import ReinforceAgent
from algorithms.actor_critic import ActorCriticAgent
from algorithms.ppo import PPOAgent
from algorithms.sac import SACAgent
from algorithms.ddpg import DDPGAgent

AGENTS = {
    "reinforce": ReinforceAgent,
    "actor_critic": ActorCriticAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
    "ddpg": DDPGAgent,
}
