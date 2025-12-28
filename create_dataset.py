import numpy as np
import os


def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer,action_type='atari'):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []
    deltas = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0
    data_dir = data_dir_prefix
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    num_trajectories = 0
    for fp in files:
        if len(obss) >= num_steps:
            break
        data = np.load(fp, allow_pickle=True)
        states_list = list(data['states'])
        actions_list = list(data['actions'])
        rewards_list = (-np.array(data['rewards'], dtype=np.float32)).tolist()
        dones_list = list(data['dones'])
        deltas_list = list(data['deltas'])
        T = len(actions_list)
        for t in range(T):
            if len(obss) >= num_steps:
                break
            s = states_list[t]
            d = deltas_list[t]
            if s.ndim == 3 and s.shape[-1] == 4:
                s = np.transpose(s, (2, 0, 1))
            if d.ndim == 3 and d.shape[-1] == 4:
                d = np.transpose(d, (2, 0, 1))
            obss.append(s)
            deltas.append(d)
            if action_type=='mujoco':
                actions.append(actions_list[t])
            else:
                actions.append(int(actions_list[t]))
            stepwise_returns.append(float(rewards_list[t]))
            returns[-1] += float(rewards_list[t])
            if dones_list[t]:
                done_idxs.append(len(obss))
                returns.append(0)
                num_trajectories += 1
        print(f'loaded {fp}, transitions now {len(obss)} divided into {num_trajectories} trajectories')

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    deltas = np.array(deltas)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return obss, actions, returns, done_idxs, rtg, timesteps, deltas

if __name__ == '__main__':
    obss, actions, returns, done_idxs, rtg, timesteps, deltas=create_dataset(num_buffers=None,num_steps=100000,game='Pong',data_dir_prefix='attacks/FGSM_Pong_data/',trajectories_per_buffer=100)
    print(len(obss))