from algorithms.actor_critic import ActorCriticAgent
from algorithms.ddpg import DDPGAgent
from algorithms.ppo import PPOAgent
from algorithms.reinforce import ReinforceAgent
from algorithms.sac import SACAgent


AGENTS = {
    "reinforce": ReinforceAgent,
    "actor_critic": ActorCriticAgent,
    "ppo": PPOAgent,
    "ddpg": DDPGAgent,
    "sac": SACAgent,
}
