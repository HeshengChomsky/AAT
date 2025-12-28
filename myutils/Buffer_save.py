import numpy as np
import gzip
import os


class SimpleReplayBuffer:
    def __init__(self, path):
        data = np.load(path, mmap_mode='r')

        self.observation = data['observation']
        self.action = data['action']
        self.reward = data['reward']
        self.terminal = data['terminal']
        self.next_observation = data['next_observation']
        self.deltas = data['deltas']

        self.size = len(self.action)

    def sample(self, batch_size, indices=None):
        if indices is None:
            indices = np.random.randint(0, self.size, batch_size)

        return dict(
            observation=self.observation[indices],
            action=self.action[indices],
            reward=self.reward[indices],
            terminal=self.terminal[indices],
            next_observation=self.next_observation[indices],
            deltas=self.deltas[indices],
            indices=indices,
        )



def save_large_fixed_replay_buffer(data_dir, observations, actions, rewards, terminals, deltas,
                                   suffix=0, prefix="replay_buffer", chunk_size=10000):
    """
    分块保存大型 FixedReplayBuffer 数据，保证即使 observations 很大也能保存。

    Args:
        data_dir: str, 保存目录
        observations: np.array, shape (N, ...), dtype uint8 或 float32
        actions: np.array, shape (N,), dtype int32
        rewards: np.array, shape (N,), dtype float32
        terminals: np.array, shape (N,), dtype bool
        suffix: int, replay buffer 后缀
        prefix: str, 文件名前缀
        chunk_size: int, 每块大小
    """
    os.makedirs(data_dir, exist_ok=True)

    N = len(actions)
    num_chunks = int(np.ceil(N / chunk_size))

    def save_array_gz(array, filename):
        with gzip.open(filename, 'wb') as f:
            np.save(f, array)

    # add_count 和 invalid_range
    add_count = np.array(N, dtype=np.int32)
    invalid_range = np.zeros(N, dtype=np.int32)

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, N)

        save_array_gz(observations[start:end], os.path.join(data_dir, f"{prefix}.{suffix}.observation.{chunk_id}.gz"))
        save_array_gz(actions[start:end], os.path.join(data_dir, f"{prefix}.{suffix}.action.{chunk_id}.gz"))
        save_array_gz(rewards[start:end], os.path.join(data_dir, f"{prefix}.{suffix}.reward.{chunk_id}.gz"))
        save_array_gz(terminals[start:end], os.path.join(data_dir, f"{prefix}.{suffix}.terminal.{chunk_id}.gz"))
        save_array_gz(deltas[start:end], os.path.join(data_dir, f"{prefix}.{suffix}.deltas.{chunk_id}.gz"))

    # add_count 和 invalid_range 仍然保存为一份
    save_array_gz(add_count, os.path.join(data_dir, f"{prefix}.{suffix}.add_count.gz"))
    save_array_gz(invalid_range, os.path.join(data_dir, f"{prefix}.{suffix}.invalid_range.gz"))

    print(f"Saved large FixedReplayBuffer in {num_chunks} chunks to {data_dir} with suffix {suffix}")


# ===========================
# 使用示例
# ===========================
if __name__ == "__main__":
    # 模拟大数组
    observations = np.random.randint(0, 256, size=(10000, 84, 84, 4), dtype=np.uint8)
    actions = np.random.randint(0, 4, size=(10000,), dtype=np.int32)
    rewards = np.random.randn(10000).astype(np.float32)
    terminals = np.zeros(10000, dtype=bool)

    save_large_fixed_replay_buffer(
        data_dir="../attacks/large_data_dir",
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        suffix=0,
        chunk_size=20000
    )
