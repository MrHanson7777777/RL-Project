import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="本地批量运行多个算法")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--algos", nargs="+", default=["reinforce", "actor_critic", "ppo"])
    args = parser.parse_args()

    for algo in args.algos:
        subprocess.run(
            ["python", "train.py", "--algo", algo, "--env", args.env, "--episodes", str(args.episodes)],
            check=True,
        )


if __name__ == "__main__":
    main()
