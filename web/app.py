import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.engine import Game2048

# 检测可用设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
# 创建游戏实例
game = Game2048(size=4)

# 方块颜色映射（根据数字值）
TILE_COLORS = {
    0: "#cdc1b4",     # 空白格子
    2: "#eee4da",     # 2
    4: "#ede0c8",     # 4
    8: "#f2b179",     # 8
    16: "#f59563",    # 16
    32: "#f67c5f",    # 32
    64: "#f65e3b",    # 64
    128: "#edcf72",   # 128
    256: "#edcc61",   # 256
    512: "#edc850",   # 512
    1024: "#edc53f",  # 1024
    2048: "#edc22e",  # 2048
    4096: "#3c3a32",  # 4096+,
    8192: "#3c3a32",  # 8192+
    16384: "#3c3a32", # 16384+
}

# 文本颜色映射（根据背景深浅）
TEXT_COLORS = {
    0: "#776e65",     # 空白格子
    2: "#776e65",     # 2
    4: "#776e65",     # 4
    8: "#f9f6f2",     # 8+
    16: "#f9f6f2",    # 16+
    32: "#f9f6f2",    # 32+
    64: "#f9f6f2",    # 64+
    128: "#f9f6f2",   # 128+
    256: "#f9f6f2",   # 256+
    512: "#f9f6f2",   # 512+
    1024: "#f9f6f2",  # 1024+
    2048: "#f9f6f2",  # 2048+
    4096: "#f9f6f2",  # 4096+
    8192: "#f9f6f2",  # 8192+
    16384: "#f9f6f2", # 16384+
}

# 定义DQN网络结构（与训练时相同）
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

# 加载模型（支持CUDA）
def load_model(model_path, device):
    model = DQN(4, 4).to(device)  # 将模型移动到指定设备
    try:
        # 尝试加载模型
        checkpoint = torch.load(model_path, map_location=device)
        
        # 检查检查点是否包含完整的模型状态
        if 'policy_net_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['policy_net_state_dict'])
        else:
            # 如果检查点不包含policy_net_state_dict，尝试直接加载
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"模型成功加载到: {device}")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        # 尝试备选模型路径
        alt_path = model_path.replace('_best_tile', '')
        try:
            checkpoint = torch.load(alt_path, map_location=device)
            if 'policy_net_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['policy_net_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print(f"备选模型成功加载: {alt_path}")
            return model
        except Exception as e2:
            print(f"备选模型加载失败: {e2}")
            return None

# 尝试加载模型
model_paths = [
    "models/dqn_2048_best_tile.pth",
    "models/dqn_2048.pth",
    "dqn_2048_best_tile.pth",
    "dqn_2048.pth"
]

model = None
for path in model_paths:
    model = load_model(path, device)
    if model:
        break

if not model:
    print("警告: 未加载任何模型，AI功能将不可用")

def render_board(board):
    html = "<div style='background-color:#bbada0; padding:10px; border-radius:6px;'>"
    html += "<table style='border-spacing:10px; border-collapse:separate;'>"
    
    for i in range(game.size):
        html += "<tr>"
        for j in range(game.size):
            value = board[i][j]
            color = TILE_COLORS.get(value, "#3c3a32")  # 默认深色
            text_color = TEXT_COLORS.get(value, "#f9f6f2")  # 默认浅色
            font_size = "36px" if value < 100 else "30px" if value < 1000 else "24px"
            
            html += f"""
            <td style='background-color:{color}; 
                        width:80px; height:80px; 
                        border-radius:4px; 
                        text-align:center; 
                        font-weight:bold; 
                        font-size:{font_size};
                        color:{text_color};'>
                {value if value > 0 else ''}
            </td>
            """
        html += "</tr>"
    
    html += "</table></div>"
    return html

def make_move(direction):
    """执行移动操作并更新界面"""
    direction_names = ["上", "右", "下", "左"]
    
    # 执行移动
    new_board, game_over = game.move(direction)
    
    # 渲染棋盘
    board_html = render_board(new_board)
    
    # 更新状态信息
    status = f"<b>移动方向:</b> {direction_names[direction]}"
    status += f"<br><b>当前分数:</b> {game.score}"
    status += f"<br><b>最大方块:</b> {np.max(game.board)}"
    
    if game.game_over:
        status += "<br><br><div style='color:#ff0000; font-weight:bold;'>游戏结束!</div>"
        status += f"<br><b>最终分数:</b> {game.score}"
    
    return board_html, status

def reset_game():
    """重置游戏"""
    global game
    game = Game2048(size=4)
    board = game.reset()
    
    # 渲染棋盘
    board_html = render_board(board)
    
    # 初始状态信息
    status = "<b>游戏已重置!</b>"
    status += f"<br><b>当前分数:</b> {game.score}"
    status += f"<br><b>最大方块:</b> {np.max(game.board)}"
    
    return board_html, status

def ai_move():
    """使用AI模型进行一步移动"""
    if model is None:
        return render_board(game.board), "<b>错误:</b> 未加载AI模型"
    
    # 获取当前状态
    state = game.get_state()
    
    # 获取有效移动
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        return render_board(game.board), "<b>游戏结束!</b> 没有有效移动"
    
    try:
        # 转换状态为模型输入并移动到设备
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        
        # 模型预测
        with torch.no_grad():
            q_values = model(state_tensor).cpu().numpy().flatten()
        
        # 只考虑有效动作
        valid_q_values = np.full(4, -np.inf)
        for move in valid_moves:
            valid_q_values[move] = q_values[move]
        
        # 选择最佳动作
        action = np.argmax(valid_q_values)
        
        # 执行移动
        direction_names = ["上", "右", "下", "左"]
        new_board, game_over = game.move(action)
        
        # 渲染棋盘
        board_html = render_board(new_board)
        
        # 更新状态信息
        status = f"<b>AI移动方向:</b> {direction_names[action]}"
        status += f"<br><b>当前分数:</b> {game.score}"
        status += f"<br><b>最大方块:</b> {np.max(game.board)}"
        status += f"<br><b>设备:</b> {'GPU' if device.type == 'cuda' else 'CPU'}"
        
        if game.game_over:
            status += "<br><br><div style='color:#ff0000; font-weight:bold;'>游戏结束!</div>"
            status += f"<br><b>最终分数:</b> {game.score}"
        
        return board_html, status
    
    except Exception as e:
        print(f"AI移动出错: {e}")
        return render_board(game.board), f"<b>错误:</b> AI移动失败 - {str(e)}"

# 创建Gradio界面
with gr.Blocks(title="2048游戏", theme="soft") as demo:
    gr.Markdown("# 🎮 2048游戏")
    gr.Markdown(f"当前运行设备: **{'GPU' if device.type == 'cuda' else 'CPU'}**")
    gr.Markdown("使用方向键或下方的按钮移动方块，相同数字的方块相撞时会合并!")
    
    with gr.Row():
        with gr.Column(scale=2):
            board_html = gr.HTML(render_board(game.board))
            status_display = gr.HTML("<b>当前分数:</b> 0<br><b>最大方块:</b> 2")
            
        with gr.Column():
            gr.Markdown("## 手动操作")
            with gr.Row():
                up_btn = gr.Button("上 ↑", elem_id="up-btn")
                left_btn = gr.Button("左 ←", elem_id="left-btn")
            with gr.Row():
                down_btn = gr.Button("下 ↓", elem_id="down-btn")
                right_btn = gr.Button("右 →", elem_id="right-btn")
            with gr.Row():
                reset_btn = gr.Button("🔄 重置游戏", elem_id="reset-btn")
                
            gr.Markdown("## AI操作")
            with gr.Row():
                ai_btn = gr.Button("🤖 AI移动一步", elem_id="ai-btn")
                auto_btn = gr.Button("🚀 连续AI模式", elem_id="auto-btn")
    
    # 连接按钮事件
    up_btn.click(lambda: make_move(0), outputs=[board_html, status_display])
    right_btn.click(lambda: make_move(1), outputs=[board_html, status_display])
    down_btn.click(lambda: make_move(2), outputs=[board_html, status_display])
    left_btn.click(lambda: make_move(3), outputs=[board_html, status_display])
    reset_btn.click(reset_game, outputs=[board_html, status_display])
    ai_btn.click(ai_move, outputs=[board_html, status_display])
    
    # 连续AI模式
    def auto_play():
        """连续AI移动直到游戏结束"""
        if model is None:
            return render_board(game.board), "<b>错误:</b> 未加载AI模型"
        
        moves = 0
        max_moves = 200  # 防止无限循环
        
        while not game.game_over and moves < max_moves:
            # 获取当前状态
            state = game.get_state()
            
            # 获取有效移动
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
                
            # 转换状态为模型输入并移动到设备
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            
            # 模型预测
            with torch.no_grad():
                q_values = model(state_tensor).cpu().numpy().flatten()
            
            # 只考虑有效动作
            valid_q_values = np.full(4, -np.inf)
            for move in valid_moves:
                valid_q_values[move] = q_values[move]
            
            # 选择最佳动作
            action = np.argmax(valid_q_values)
            
            # 执行移动
            game.move(action)
            moves += 1
        
        # 渲染棋盘
        board_html = render_board(game.board)
        
        # 更新状态信息
        status = f"<b>连续AI完成!</b>"
        status += f"<br><b>移动次数:</b> {moves}"
        status += f"<br><b>当前分数:</b> {game.score}"
        status += f"<br><b>最大方块:</b> {np.max(game.board)}"
        status += f"<br><b>设备:</b> {'GPU' if device.type == 'cuda' else 'CPU'}"
        
        if game.game_over:
            status += "<br><br><div style='color:#ff0000; font-weight:bold;'>游戏结束!</div>"
            status += f"<br><b>最终分数:</b> {game.score}"
        
        return board_html, status
    
    auto_btn.click(auto_play, outputs=[board_html, status_display])
    
    # 添加键盘快捷键支持
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => {
            document.addEventListener('keydown', function(e) {
                if (e.key === 'ArrowUp') {
                    document.getElementById('up-btn').click();
                } else if (e.key === 'ArrowRight') {
                    document.getElementById('right-btn').click();
                } else if (e.key === 'ArrowDown') {
                    document.getElementById('down-btn').click();
                } else if (e.key === 'ArrowLeft') {
                    document.getElementById('left-btn').click();
                } else if (e.key === 'r' || e.key === 'R') {
                    document.getElementById('reset-btn').click();
                } else if (e.key === 'a' || e.key === 'A') {
                    document.getElementById('ai-btn').click();
                } else if (e.key === 's' || e.key === 'S') {
                    document.getElementById('auto-btn').click();
                }
            });
        }"""
    )
    
    gr.Markdown("### 📚 使用说明")
    gr.Markdown("1. 使用方向键或下方的按钮移动方块")
    gr.Markdown("2. 相同数字的方块相撞时会合并")
    gr.Markdown("3. **快捷键说明**:")
    gr.Markdown("   - ↑/↓/←/→: 移动方块")
    gr.Markdown("   - R: 重置游戏")
    gr.Markdown("   - A: AI移动一步")
    gr.Markdown("   - S: 连续AI模式(自动玩到游戏结束)")

# 启动界面
if __name__ == "__main__":
    demo.launch()
