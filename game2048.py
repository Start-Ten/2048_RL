import numpy as np
import random

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.add_tile()
        self.add_tile()
        self.game_over = False
        return self.board.copy()
    
    def add_tile(self):
        """在随机空位置添加新方块(90%概率为2,10%概率为4)"""
        empty_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    empty_cells.append((i, j))
        
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4
    
    def move(self, direction):
        """
        执行移动操作
        0: 上, 1: 右, 2: 下, 3: 左
        返回: (新棋盘状态, 游戏是否结束)
        """
        moved = False
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
        
        # 如果发生了移动，添加新方块并检查游戏结束
        if moved:
            self.add_tile()
            self.check_game_over()
        
        return self.board.copy(), self.game_over
    
    def slide(self, line):
        """处理单行/列的移动和合并逻辑"""
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
    
    def check_game_over(self):
        """检查游戏是否结束"""
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
    
    def get_valid_moves(self):
        """获取当前所有有效移动方向"""
        valid_moves = []
        
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
    
    def get_state(self):
        """获取当前游戏状态表示（用于AI模型）"""
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
