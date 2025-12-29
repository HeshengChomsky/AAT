from time import gmtime
import numpy as np
import torch
import gym

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UniversalAdversarialAgent:
    def __init__(self, model, epsilon=0.001, norm_type='inf', model_type='dqn', action_type='atari'):
        """
        model: 预训练模型 (DQN, PPO, 或 A2C)
        epsilon: 攻击预算
        norm_type: 'inf', 'l2', 'l1'
        model_type: 'dqn', 'policy_based' (用于 PPO/A2C)
        """
        self.model = model
        self.epsilon = epsilon
        self.norm_type = norm_type
        self.model_type = model_type
        self.action_type=action_type

    def _get_model_output(self, state_t):
        """处理不同模型输出格式的辅助函数"""
        if self.model_type == 'dqn':
            output=self.model(state_t)
        elif self.model_type == 'D4PG':
            output = self.model(state_t)
            output = output.clamp(-1, 1)
            # return output.squeeze(0).cpu().numpy()
        else:
            output = self.model(state_t)
            # PPO/A2C 通常返回 (policy_logits, value) 或类似结构
            if isinstance(output, tuple):
                return output[0]  # 返回 policy 部分
        return output

    def _get_loss_gradient(self, state):
        # 1. 预处理
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # if state_t.max() > 1.0:
        #     state_t = state_t / 255.0
        state_t.requires_grad_(True)

        if self.action_type == 'atari':
            # 2. 获取输出 (Q-values 或 Logits)
            logits = self._get_model_output(state_t)

            # 3. 确定攻击目标：让模型远离当前认为“最好”的动作
            # 对于 DQN，这是 Q 值最大的动作；对于 PPO，这是概率最大的动作
            target_action = torch.argmax(logits, dim=1)

            # 4. 计算交叉熵损失
            # 增加该损失意味着让模型在扰动后，预测该“正确动作”的概率/值降低
            loss = torch.nn.functional.cross_entropy(logits, target_action)

            # 5. 反向传播
            self.model.zero_grad()
            loss.backward()
        elif self.action_type == 'mujoco':
            current_action = self._get_model_output(state_t)
            # 3. 确定攻击目标：记录下原始状态下的“正常动作”
            # 我们使用 detach() 确保在计算梯度时，目标值被视为常量
            target_action = current_action.detach().clone()

            # 4. 计算损失
            # 我们的目标是最大化扰动后的动作与原始动作之间的“距离”
            # 在 FGSM 中，我们通过对 loss 求正向梯度来实现
            loss = torch.nn.functional.mse_loss(current_action, target_action)

            # 5. 反向传播获取梯度
            self.model.zero_grad()
            loss.backward()
        
        grad = state_t.grad.data.cpu().numpy()
        return grad

    def craft_adversarial_input(self, state):
        grad = self._get_loss_gradient(state)
        d = np.prod(state.shape) # 整体维度

        if self.norm_type == 'inf':
            perturbation = self.epsilon * np.sign(grad)
        elif self.norm_type == 'l2':
            norm = np.linalg.norm(grad) + 1e-10
            perturbation = (self.epsilon * np.sqrt(d)) * (grad / norm)
        elif self.norm_type == 'l1':
            perturbation = np.zeros_like(grad)
            max_idx = np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
            perturbation[max_idx] = (self.epsilon * d) * np.sign(grad[max_idx])

        # 应用扰动并恢复范围 (假设输入是 [0, 255])
        adv_state = state + perturbation.squeeze(0) * 255.0
        return np.clip(adv_state, 0, 255)

    def select_action(self, state, is_under_attack=True):
        current_state = state
        if is_under_attack:
            current_state = self.craft_adversarial_input(state)
        
        state_t = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
        # if state_t.max() > 1.0: state_t /= 255.0
        
        with torch.no_grad():
            logits = self._get_model_output(state_t)
            if self.model_type == 'D4PG':
                action=logits.cpu().numpy()
            else:
                action = torch.argmax(logits, dim=1).item()
        
        return action, current_state

# --- 模拟 main 函数中的调用 ---
def collected_data(ppo_net,env,max_eposides,path=None,game='Pong',action_type=None,model_type='PPO',action_spec=None):
    # 假设这里是你的模型加载逻辑
    # 对于 PPO: model_type='policy_based'
    # 对于 DQN: model_type='dqn'
    # from algorithms.PPO import PPOAgent
    # from algorithms.Atarienv import AtariWrapper
    # env = AtariWrapper(gym.make("PongNoFrameskip-v4"))
    # max_eposides = 5
    # # 示例：初始化 PPO 攻击者
    # ppo_agent = PPOAgent(env.action_space.n)
    # ppo_agent.load('../algorithms/models/ppo_pong_3450.pth')
    # ppo_net=ppo_agent.model.cpu()
    agent = UniversalAdversarialAgent(ppo_net, epsilon=0.01, model_type=model_type,action_type=action_type)
    
    # 示例：初始化 DQN 攻击者
    # agent = UniversalAdversarialAgent(dqn_net, epsilon=0.001, model_type='dqn')
    for i in range(max_eposides):

        states = []

        actions = []

        rewards = []

        dones = []

        deltas = []

        state = env.reset()
        trans=False
        if action_type=='mujoco':
            state=state.numpy()
            trans=True

        total_reward = 0

        while True:

            action, ad_s = agent.select_action(state, is_under_attack=True)

            if action_type=='mujoco':
                ev_action = action_spec.minimum + (action + 1.0) * 0.5 * (action_spec.maximum - action_spec.minimum)
                next_state, reward, done, _ = env.step(ev_action)
            else:
                next_state, reward, done, _ = env.step(action)

            total_reward += reward
            if action_type=='atari':
                # env.render()
                states.append(state.astype(np.uint8))

                actions.append(action)

                rewards.append(reward)

                dones.append(done)

                deltas.append((ad_s.astype(np.uint8) - state.astype(np.uint8)))
            else:
                # env.render()
                if trans:
                    states.append(state.astype(np.uint8))
                    trans=False
                    state=torch.from_numpy(state)
                    ad_s=torch.from_numpy(ad_s)
                else:
                    states.append(state.numpy().astype(np.uint8))

                actions.append(action)

                rewards.append(reward)

                dones.append(done)

                deltas.append((ad_s - state).numpy().astype(np.uint8))

            if done:
                break

            state = next_state

        if action_type=='atari':
            env.close()

        print(f"Total reward: {total_reward}")

        np.savez_compressed(

            path+f"FGSM_{game}_attack_{i + 1}.npz",

            states=states,

            # adv_states=adv_states,

            actions=actions,

            rewards=rewards,

            dones=dones,

            deltas=deltas

        )

if __name__ == '__main__':
    pass