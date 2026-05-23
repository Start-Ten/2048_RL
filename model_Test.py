import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import torch.optim as optim


# 设置设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.prev_score = 0
        self.add_tile()
        self.add_tile()
        self.game_over = False
        return self.get_state()
    
    def add_tile(self):
        empty_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    empty_cells.append((i, j))
        
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4
    
    def move(self, direction):
        # 0: 上, 1: 右, 2: 下, 3: 左
        moved = False
        original_board = self.board.copy()
        old_score = self.score
        
        # 根据方向执行移动
        if direction == 0:  # 上
            for j in range(self.size):
                column = self.board[:, j].copy()
                new_column, moved_col = self.slide(column)
                if moved_col:
                    moved = True
                self.board[:, j] = new_column
        
        elif direction == 1:  # 右
            for i in range(self.size):
                row = self.board[i, :].copy()[::-1]
                new_row, moved_row = self.slide(row)
                if moved_row:
                    moved = True
                self.board[i, :] = new_row[::-1]
        
        elif direction == 2:  # 下
            for j in range(self.size):
                column = self.board[::-1, j].copy()
                new_column, moved_col = self.slide(column)
                if moved_col:
                    moved = True
                self.board[:, j] = new_column[::-1]
        
        elif direction == 3:  # 左
            for i in range(self.size):
                row = self.board[i, :].copy()
                new_row, moved_row = self.slide(row)
                if moved_row:
                    moved = True
                self.board[i, :] = new_row
        
        # 如果发生了移动，添加新方块
        if moved:
            self.add_tile()
            self.check_game_over()
        
        reward = self.calculate_reward(old_score, original_board)
        return self.get_state(), reward, self.game_over
    
    def slide(self, line):
        # 移除零并合并相同数字
        non_zero = line[line != 0]
        new_line = np.zeros_like(line)
        idx = 0
        score_inc = 0
        moved = False
        
        # 检查是否移动
        if not np.array_equal(non_zero, line[:len(non_zero)]):
            moved = True
        
        # 合并相同数字
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i+1]:
                new_val = non_zero[i] * 2
                new_line[idx] = new_val
                score_inc += new_val
                i += 2
                idx += 1
            else:
                new_line[idx] = non_zero[i]
                i += 1
                idx += 1
        
        self.score += score_inc
        return new_line, moved or (score_inc > 0)
    
    def calculate_reward(self, old_score, original_board):
        """改进的奖励函数"""
        # 1. 基本分数奖励
        score_reward = (self.score - old_score) * 0.1
        
        # 2. 空格子数量变化奖励
        empty_before = np.count_nonzero(original_board == 0)
        empty_after = np.count_nonzero(self.board == 0)
        empty_reward = (empty_after - empty_before) * 0.1
        empty_reward -= np.square(5-empty_after) * 0.15 if empty_after < 5 and empty_after!=empty_before else 0
        
        # 3. 最大方块奖励
        max_before = np.max(original_board)
        max_after = np.max(self.board)
        max_tile_reward = 0
        if max_after > max_before:
            max_tile_reward = np.sqrt(max_after) * 1

        
        # 5. 单调性惩罚（鼓励有序排列）
        #monotonicity_penalty = self.calculate_monotonicity_penalty() * 0.01
        
        # 6. 游戏结束惩罚
        game_over_penalty = 0
        if self.game_over:
            game_over_penalty = -10
        
        # 7. 平滑度奖励（鼓励相邻方块值接近）
        #smoothness_reward = self.calculate_smoothness() * 0.01

        # 总奖励
        total_reward = (
            score_reward + 
            empty_reward + 
            max_tile_reward + 
            game_over_penalty
        )
        return total_reward
    
    def calculate_monotonicity_penalty(self):
        """计算单调性惩罚（值越低越好）"""
        penalty = 0
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i][j] > self.board[i][j+1]:
                    penalty += self.board[i][j] - self.board[i][j+1]
                else:
                    penalty += self.board[i][j+1] - self.board[i][j]
        return penalty
    
    def calculate_smoothness(self):
        """计算平滑度（值越高越好）"""
        smoothness = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0:
                    value = np.log2(self.board[i][j])
                    # 检查右侧邻居
                    if j < self.size - 1 and self.board[i][j+1] != 0:
                        neighbor_value = np.log2(self.board[i][j+1])
                        smoothness -= abs(value - neighbor_value)
                    # 检查下方邻居
                    if i < self.size - 1 and self.board[i+1][j] != 0:
                        neighbor_value = np.log2(self.board[i+1][j])
                        smoothness -= abs(value - neighbor_value)
        return smoothness
    
    def check_game_over(self):
        # 检查是否还有空格子
        if np.any(self.board == 0):
            self.game_over = False
            return
        
        # 检查水平和垂直方向是否有可合并的方块
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i][j] == self.board[i][j+1]:
                    self.game_over = False
                    return
        
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i][j] == self.board[i+1][j]:
                    self.game_over = False
                    return
        
        self.game_over = True
    
    def get_state(self):
        """改进的状态表示"""
        # 创建4个通道的状态表示
        state = np.zeros((4, self.size, self.size), dtype=np.float32)
        
        # 通道0: 当前方块值的对数（归一化）
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] > 0:
                    state[0, i, j] = np.log2(self.board[i][j]) / 16.0  # 支持到65536 (2^16)
        
        # 通道1: 空格子指示器
        state[1] = (self.board == 0).astype(np.float32)
        
        # 通道2: 可合并的邻居指示器
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] > 0:
                    # 检查右侧
                    if j < self.size - 1 and self.board[i][j] == self.board[i][j+1]:
                        state[2, i, j] = 1.0
                        state[2, i, j+1] = 1.0
                    # 检查下方
                    if i < self.size - 1 and self.board[i][j] == self.board[i+1][j]:
                        state[2, i, j] = 1.0
                        state[2, i+1, j] = 1.0
        
        # 通道3: 最大值位置（归一化）
        max_value = np.max(self.board)
        if max_value > 0:
            max_positions = np.argwhere(self.board == max_value)
            for pos in max_positions:
                state[3, pos[0], pos[1]] = 1.0
        
        return state
    
    def get_valid_moves(self):
        """更高效的有效移动检测"""
        valid_moves = []
        #test_board = np.zeros_like(self.board)
        
        # 检查上移是否有效
        for j in range(self.size):
            column = self.board[:, j].copy()
            new_column, _ = self.slide(column)
            if not np.array_equal(new_column, self.board[:, j]):
                valid_moves.append(0)
                break
        
        # 检查右移是否有效
        for i in range(self.size):
            row = self.board[i, :].copy()[::-1]
            new_row, _ = self.slide(row)
            if not np.array_equal(new_row[::-1], self.board[i, :]):
                valid_moves.append(1)
                break
        
        # 检查下移是否有效
        for j in range(self.size):
            column = self.board[::-1, j].copy()
            new_column, _ = self.slide(column)
            if not np.array_equal(new_column[::-1], self.board[:, j]):
                valid_moves.append(2)
                break
        
        # 检查左移是否有效
        for i in range(self.size):
            row = self.board[i, :].copy()
            new_row, _ = self.slide(row)
            if not np.array_equal(new_row, self.board[i, :]):
                valid_moves.append(3)
                break
        
        return valid_moves

# 改进的深度Q网络（使用Dueling DQN架构）
class DQN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(DQN, self).__init__()
        self.input_channels = input_channels
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Dueling DQN架构
        # 价值流
        self.value_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.value_fc1 = nn.Linear(4 * 4 * 4, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        # 优势流
        self.advantage_conv = nn.Conv2d(128, 16, kernel_size=1)
        self.advantage_fc1 = nn.Linear(16 * 4 * 4, 128)
        self.advantage_fc2 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 价值流
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)
        
        # 优势流
        advantage = F.relu(self.advantage_conv(x))
        advantage = advantage.view(advantage.size(0), -1)
        advantage = F.relu(self.advantage_fc1(advantage))
        advantage = self.advantage_fc2(advantage)
        
        # 合并价值流和优势流
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.pos = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        # 初始优先级设置为最大优先级
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            return None, None, None
        
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states)), 
            torch.tensor(actions, dtype=torch.long), 
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(np.array(next_states)),
            torch.tensor(dones, dtype=torch.float),
            indices,
            torch.tensor(weights)
        )
    
    def update_priorities(self, indices, priorities):
        # 确保 priorities 是一个数组
        if isinstance(priorities, np.ndarray) and priorities.ndim == 1:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority
        else:
            # 处理标量情况（虽然不应该发生）
            if not isinstance(priorities, (list, np.ndarray)):
                priorities = [priorities] * len(indices)
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        return self.size

class DQNAgent:
    def __init__(self, input_channels, action_size, lr=3e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999, 
                 target_update_freq=1000, batch_size=128):
        self.input_channels = input_channels
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 主网络和目标网络
        self.policy_net = DQN(input_channels, action_size).to(device)
        self.target_net = DQN(input_channels, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = PrioritizedReplayBuffer(50000)
        self.steps_done = 0
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
    
    def select_action(self, state, valid_moves):
        self.steps_done += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if random.random() < self.epsilon:
            # 随机选择有效动作
            return random.choice(valid_moves)
        else:
            # 使用策略网络选择动作
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
                
                # 只考虑有效动作
                valid_q_values = np.full(self.action_size, -np.inf)
                for move in valid_moves:
                    valid_q_values[move] = q_values[move]
                
                return np.argmax(valid_q_values)
    
    def optimize_model(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return 0
        
        # 从回放缓冲区采样
        sample = self.memory.sample(self.batch_size, beta)
        if sample is None:
            return 0
            
        states, actions, rewards, next_states, dones, indices, weights = sample
        
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        weights = weights.to(device)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1]
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        losses = self.loss_fn(current_q, target_q)
        loss = (losses * weights).mean()
        
        # 更新优先级（使用每个样本的损失绝对值）
        with torch.no_grad():
            priorities = losses.abs().cpu().numpy() + 1e-5
            self.memory.update_priorities(indices, priorities)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load_model(self, path):
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return
            
        try:
            # 尝试使用 weights_only=False 加载模型
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
            self.policy_net.eval()
            self.target_net.eval()
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # 尝试使用旧版加载方式作为备选
            try:
                warnings.warn("Trying legacy load method without weights_only")
                checkpoint = torch.load(path, map_location=device)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                self.steps_done = checkpoint['steps_done']
                self.policy_net.eval()
                self.target_net.eval()
                print(f"Model loaded successfully using legacy method")
            except Exception as e2:
                print(f"Failed to load model: {e2}")
# 新的推理函数：运行一次游戏并返回结果
def run_single_game(agent, env):
    state = env.reset()
    done = False
    steps = 0
    
    while not done:
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            done = True
            continue
            
        # 选择动作
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            q_values = agent.policy_net(state_tensor).cpu().numpy().flatten()
            
            # 只考虑有效动作
            valid_q_values = np.full(4, -np.inf)
            for move in valid_moves:
                valid_q_values[move] = q_values[move]
            
            action = np.argmax(valid_q_values)
        
        # 执行动作
        next_state, reward, done = env.move(action)
        state = next_state
        steps += 1
    
    return env.score, np.max(env.board), steps

# 新的推理函数：连续运行多次游戏并收集结果
def run_inference(agent, env, num_games=100):
    scores = []
    max_tiles = []
    steps_list = []
    
    print(f"Running inference for {num_games} games...")
    
    for i in tqdm(range(num_games)):
        score, max_tile, steps = run_single_game(agent, env)
        scores.append(score)
        max_tiles.append(max_tile)
        steps_list.append(steps)
    
    return scores, max_tiles, steps_list

# 结果可视化函数
def visualize_results(scores, max_tiles, steps_list):
    # 创建结果目录
    os.makedirs("inference_results", exist_ok=True)
    
    # 绘制分数分布
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制最大方块分布
    plt.subplot(2, 2, 2)
    unique_tiles = sorted(set(max_tiles))
    tile_counts = [max_tiles.count(t) for t in unique_tiles]
    plt.bar(unique_tiles, tile_counts)
    plt.title('Max Tile Distribution')
    plt.xlabel('Max Tile')
    plt.ylabel('Frequency')
    #plt.xticks(unique_tiles)
    
    # 绘制步数分布
    plt.subplot(2, 2, 3)
    plt.hist(steps_list, bins=20, color='salmon', edgecolor='black')
    plt.title('Steps Distribution')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制分数与最大方块的关系
    plt.subplot(2, 2, 4)
    plt.scatter(max_tiles, scores, alpha=0.6)
    plt.title('Score vs Max Tile')
    plt.xlabel('Max Tile')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('inference_results/inference_results.png')
    plt.close()
    
    # 保存统计数据
    with open('inference_results/stats.txt', 'w') as f:
        f.write(f"=== Inference Results (100 games) ===\n")
        f.write(f"Average Score: {np.mean(scores):.2f}\n")
        f.write(f"Median Score: {np.median(scores):.2f}\n")
        f.write(f"Max Score: {max(scores)}\n")
        f.write(f"Min Score: {min(scores)}\n")
        f.write(f"Standard Deviation: {np.std(scores):.2f}\n\n")
        
        f.write(f"Average Max Tile: {np.mean(max_tiles):.2f}\n")
        f.write(f"Most Common Max Tile: {max(set(max_tiles), key=max_tiles.count)}\n")
        f.write(f"Max Tile Achieved: {max(max_tiles)}\n")
        f.write(f"Min Tile Achieved: {min(max_tiles)}\n\n")
        
        f.write(f"Average Steps: {np.mean(steps_list):.2f}\n")
        f.write(f"Max Steps: {max(steps_list)}\n")
        f.write(f"Min Steps: {min(steps_list)}\n")
        
        f.write("\n=== Detailed Max Tile Distribution ===\n")
        tile_dist = {}
        for tile in max_tiles:
            tile_dist[tile] = tile_dist.get(tile, 0) + 1
        for tile, count in sorted(tile_dist.items()):
            f.write(f"Tile {tile}: {count} games\n")
    
    print("Results saved in inference_results/ directory")

# 主程序
if __name__ == "__main__":
    # 初始化环境和智能体
    env = Game2048(size=4)
    input_channels = 4  # 状态表示的通道数
    action_size = 4  # 上、右、下、左
    
    agent = DQNAgent(
        input_channels, 
        action_size,
        lr=1e-4,
        epsilon_decay=0.999,
        target_update_freq=1000,
        batch_size=256
    )
    
    # 加载训练好的模型
    model_path = 'models/dqn_2048_best_tile.pth'
    if not os.path.exists(model_path):
        model_path = 'models/dqn_2048.pth'
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load_model(model_path)
        
        # 运行100次推理测试
        scores, max_tiles, steps_list = run_inference(agent, env, num_games=100)
        
        # 可视化结果
        visualize_results(scores, max_tiles, steps_list)
    else:
        print("No trained model found. Please train the model first.")
