#!/usr/bin/env python
# coding: utf-8

# In[4]:

# get_ipython().system('apt-get install -y xvfb python-opengl > /dev/null 2>&1')
# get_ipython().system('pip install gym pyvirtualdisplay > /dev/null 2>&1')
# get_ipython().system('pip install JSAnimation')
# get_ipython().system('pip install gym==0.10.5')
# get_ipython().system('pip install pyglet==1.3.2')
from make_env import make_env  # 自作のmake_envモジュールからmake_env関数をインポート
import numpy as np  # 数値計算用のライブラリNumPyをインポートし、別名npで利用
import copy  # Pythonのcopyモジュールをインポート
from collections import deque  # Pythonのcollectionsモジュールからdequeクラスをインポート
import gym  # OpenAI Gymのライブラリをインポート
import random  # Pythonのrandomモジュールをインポート
import torch  # PyTorchのライブラリをインポート
import torch.nn as nn  # PyTorchのニューラルネットワーク関連のモジュールをインポート
import torch.nn.functional as F  # PyTorchのニューラルネットワーク関連の関数群をインポート
import torch.optim as optim  # PyTorchの最適化関連のモジュールをインポート
from torch.nn.utils import clip_grad_norm_  # PyTorchの勾配クリッピング関数をインポート
import matplotlib  # データ可視化のライブラリMatplotlibをインポート
import matplotlib.animation as animation  # Matplotlibのアニメーション機能をインポート
import matplotlib.pyplot as plt  # Matplotlibのプロット機能をインポート

from JSAnimation.IPython_display import display_animation  # IPythonでのアニメーション表示機能をインポート
from IPython.display import HTML  # IPythonでのHTML表示機能をインポート
from pyvirtualdisplay import Display  # 仮想ディスプレイのライブラリをインポート
pydisplay = Display(visible=0, size=(400, 300))
pydisplay.start()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # CUDAが利用可能な場合はGPUデバイス（'cuda:0'）、そうでない場合はCPUデバイス（'cpu'）を選択
print(device)  # 選択されたデバイスを表示
class ReplayBuffer:
    def __init__(self, memory_size=1e+6):
        self.memory = deque([], maxlen=memory_size)  # 空のメモリを作成
        self.is_gpu = torch.cuda.is_available  # CUDAが利用可能かどうかをチェック

    def cache(self, state, next_state, action, reward, done):
        if self.is_gpu:
            # state = torch.tensor(state, dtype=torch.float).cuda()  # 状態をテンソルに変換し、GPU上に配置
            # next_state = torch.tensor(next_state, dtype=torch.float).cuda()  # 次の状態をテンソルに変換し、GPU上に配置
            # action = torch.tensor(action, dtype=torch.float).cuda()  # 行動をテンソルに変換し、GPU上に配置
            # reward = torch.tensor(reward).cuda()  # 報酬をテンソルに変換し、GPU上に配置
            # done = torch.tensor([done]).cuda()  # 終了フラグをテンソルに変換し、GPU上に配置
            state = torch.tensor(state, dtype=torch.float, device=torch.device('cpu')) # 状態をテンソルに変換し、GPU上に配置
            next_state = torch.tensor(next_state, dtype=torch.float, device=torch.device('cpu'))# 次の状態をテンソルに変換し、GPU上に配置
            action = torch.tensor(action, dtype=torch.float, device=torch.device('cpu')) # 行動をテンソルに変換し、GPU上に配置
            reward = torch.tensor(reward, device=torch.device('cpu')) # 報酬をテンソルに変換し、GPU上に配置
            done = torch.tensor([done], device=torch.device('cpu')) # 終了フラグをテンソルに変換し、GPU上に配置
            
        else:
            state = torch.tensor(state, dtype=torch.float)  # 状態をテンソルに変換
            next_state = torch.tensor(next_state, dtype=torch.float)  # 次の状態をテンソルに変換
            action = torch.tensor(action, dtype=torch.float)  # 行動をテンソルに変換
            reward = torch.tensor(reward)  # 報酬をテンソルに変換
            done = torch.tensor([done])  # 終了フラグをテンソルに変換
        self.memory.append((state, next_state, action, reward, done))  # メモリにデータを追加

    def sample(self, batch_size=64):
        batch = random.sample(self.memory, batch_size)  # メモリからランダムにバッチサイズ分のデータをサンプリング
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))  # サンプリングしたデータをテンソルにまとめる
        return state, next_state, action, reward.squeeze(), done.squeeze()  # テンソルを返す

class PolicyNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)  # 入力層から隠れ層への全結合層
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 隠れ層から隠れ層への全結合層
        self.fc3 = nn.Linear(hidden_size, num_action)  # 隠れ層から出力層への全結合層

    def forward(self, x, index):
        x = x[:, index]  # 入力の特定の次元を選択
        h = F.relu(self.fc1(x))  # 隠れ層1への活性化関数ReLUを適用
        h = F.relu(self.fc2(h))  # 隠れ層2への活性化関数ReLUを適用
        action = self.fc3(h)  # 出力層への線形変換（行動の予測または生成）
        return action

class QNetwork(nn.Module):
    def __init__(self,num_state,num_action,agent_num,hidden_size=64,init_w=3e-3):
        super(QNetwork,self).__init__()
        input_size1 = num_state * agent_num  # 入力サイズ1を計算（状態数 * エージェント数）
        input_size2 = hidden_size + num_action * agent_num  # 入力サイズ2を計算（隠れ層サイズ + 行動数 * エージェント数）
        self.fc1 = nn.Linear(input_size1,hidden_size)  # 全結合層1を定義（入力サイズ1 -> 隠れ層サイズ）
        self.fc2 = nn.Linear(input_size2,hidden_size)  # 全結合層2を定義（入力サイズ2 -> 隠れ層サイズ）
        self.fc3 = nn.Linear(hidden_size,1)  # 全結合層3を定義（隠れ層サイズ -> 1）
        self.fc3.weight.data.uniform_(-init_w, init_w)  # 全結合層3の重みを一様分布のランダム値で初期化
        self.fc3.bias.data.uniform_(-init_w, init_w)  # 全結合層3のバイアスを一様分布のランダム値で初期化

    def forward(self,states,actions):
        states = states.view(states.size()[0],-1)  # 状態テンソルを2次元に変形
        actions = actions.view(actions.size()[0],-1)  # 行動テンソルを2次元に変形
        h = F.relu(self.fc1(states))  # 入力状態を隠れ層に入力し、活性化関数ReLUを適用
        x = torch.cat([h,actions],1)  # 隠れ層の出力と行動を結合
        h = F.relu(self.fc2(x))  # 結合されたテンソルを隠れ層に入力し、活性化関数ReLUを適用
        q = self.fc3(h)  # 隠れ層の出力を全結合層3に入力し、Q値を出力
        return q

class OrnsteinUhlenbeckProcess:
    def __init__(self, theta=0.15, mu=0.0, sigma=0.2, dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
          self.theta = theta  # Ornstein-Uhlenbeck過程のθパラメータ
          self.mu = mu  # Ornstein-Uhlenbeck過程のμパラメータ
          self.sigma = sigma  # Ornstein-Uhlenbeck過程のσパラメータ（標準偏差）
          self.dt = dt  # 時間ステップの大きさ
          self.x0 = x0  # 初期状態
          self.size = size  # 出力の次元数
          self.num_steps = 0  # 現在のステップ数を保持する変数

          self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)  # 直前の状態を保持する変数（初期状態で初期化）

          if sigma_min is not None:
              self.m = -float(sigma - sigma_min) / float(n_steps_annealing)  # ステップ数に応じて減少するσの勾配を計算
              self.c = sigma  # 初期のσ値
              self.sigma_min = sigma_min  # 最小のσ値
          else:
              self.m = 0  # 減少しない場合の勾配
              self.c = sigma  # 初期のσ値
              self.sigma_min = sigma  # 最小のσ値（初期値と同じ）

    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.num_steps) + self.c)  # 現在のσ値を計算（ステップ数に応じて減少）
        return sigma

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma() * np.sqrt(self.dt) * np.random.normal(size=self.size)  # Ornstein-Uhlenbeck過程から次の状態をサンプリング
        self.x_prev = x  # 直前の状態を更新
        self.num_steps += 1  # ステップ数をインクリメント
        return x

class MaddpgAgents:
    def __init__(self,observation_space,action_space,num_agent,gamma=0.95,lr=0.01,batch_size=1024,memory_size=int(1e6),tau=0.01,grad_norm_clipping = 0.5):
        self.num_state = observation_space  # 状態空間の次元数
        self.num_action = action_space  # 行動空間の次元数
        self.n = num_agent  # エージェントの数
        self.gamma = gamma  # 割引率
        self.actor_group = [PolicyNetwork(self.num_state,self.num_action).to(device) for _ in range(self.n)]  # アクターネットワークのリスト
        self.target_actor_group = copy.deepcopy(self.actor_group)  # ターゲットアクターネットワークのリスト（初期状態はアクターネットワークと同じ）
        self.actor_optimizer_group = [optim.Adam(self.actor_group[i].parameters(),lr=0.001) for i in range(self.n)]  # アクターネットワークの最適化手法
        self.critic_group = [QNetwork(self.num_state,self.num_action,self.n).to(device) for _ in range(self.n)]  # クリティックネットワークのリスト
        self.target_critic_group = copy.deepcopy(self.critic_group)  # ターゲットクリティックネットワークのリスト（初期状態はクリティックネットワークと同じ）
        self.critic_optimizer_group = [optim.Adam(self.critic_group[i].parameters(),lr=lr) for i in range(self.n)]  # クリティックネットワークの最適化手法
        self.buffer = ReplayBuffer(memory_size=memory_size)  # リプレイバッファ
        self.loss_fn = torch.nn.MSELoss()  # 損失関数（平均二乗誤差）
        self.batch_size = batch_size  # バッチサイズ
        self.is_gpu = torch.cuda.is_available  # GPUの利用可否
        self.noise = OrnsteinUhlenbeckProcess(size=self.num_action)
        self.grad_norm_clipping = grad_norm_clipping
        self.tau = tau

    @torch.no_grad()
    def td_targeti(self, reward, state, next_state, done, agent_index):
        next_actions = []
        for i in range(self.n):
            actionsi = torch.tanh(self.target_actor_group[i](state, i))  # ターゲットアクターネットワークを使用して次の行動を求める
            actionsi = actionsi[:, np.newaxis, :]  # 次元を追加して行動の形状を調整
            next_actions.append(actionsi)
        next_actions = torch.cat(next_actions, dim=1)  # 全エージェントの行動を連結
        next_q = self.target_critic_group[agent_index](next_state, next_actions)  # ターゲットクリティックネットワークを使用して次の状態での行動価値を求める
        if self.n != 1:
            reward = reward[:, agent_index]  # エージェントごとの報酬を選択
            done = done[:, agent_index]  # エージェントごとの終了フラグを選択
        reward = reward[:, np.newaxis]  # 次元を追加して報酬の形状を調整
        done = done[:, np.newaxis]  # 次元を追加して終了フラグの形状を調整
        done = torch.tensor(done, dtype=torch.int)  # Tensorに変換
        td_targeti = reward + self.gamma * next_q * (1. - done.data)  # TDターゲットを計算
        return td_targeti.float()

    def update(self):
        for i in range(self.n):
            state, next_state, action, reward, done = self.buffer.sample(self.batch_size)  # リプレイバッファからミニバッチをサンプリング
            td_targeti = self.td_targeti(reward, state, next_state, done, i)  # TDターゲットを計算
            current_q = self.critic_group[i](state, action)  # 現在の行動価値を計算
            critic_loss = self.loss_fn(current_q, td_targeti)  # クリティックネットワークの損失を計算
            self.critic_optimizer_group[i].zero_grad()  # クリティックネットワークの勾配を初期化
            critic_loss.backward()  # クリティックネットワークの勾配を計算
            clip_grad_norm_(self.critic_group[i].parameters(), max_norm=self.grad_norm_clipping)  # 勾配のクリッピング
            self.critic_optimizer_group[i].step()  # クリティックネットワークのパラメータを更新
            ac = action.clone()
            ac_up = self.actor_group[i](state, i)
            ac[:, i, :] = torch.tanh(ac_up)
            pr = -self.critic_group[i](state, ac).mean()
            pg = (ac[:, i, :].pow(2)).mean()
            actor_loss = pr + pg * 1e-3
            self.actor_optimizer_group[i].zero_grad()  # アクターネットワークの勾配を初期化
            clip_grad_norm_(self.actor_group[i].parameters(), max_norm=self.grad_norm_clipping)  # 勾配のクリッピング
            actor_loss.backward()  # アクターネットワークの勾配を計算
            self.actor_optimizer_group[i].step()  # アクターネットワークのパラメータを更新
        for i in range(self.n):
            for target_param, local_param in zip(self.target_actor_group[i].parameters(), self.actor_group[i].parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)  # ターゲットネットワークのパラメータをソフト更新
            for target_param, local_param in zip(self.target_critic_group[i].parameters(), self.critic_group[i].parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)  # ターゲットネットワークのパラメータをソフト更新

    def get_action(self, state, greedy=False):
        # state = torch.tensor(state, dtype=torch.float).cuda()  # Tensorに変換してGPUに移動
        state = torch.tensor(state, dtype=torch.float, device=torch.device('cpu'))
        state = state[np.newaxis, :, :]  # 次元を追加して状態の形状を調整
        actions = []
        for i in range(self.n):
            action = torch.tanh(self.actor_group[i](state, index=i))  # アクターネットワークを使用して行動を求める
            if not greedy:
                # action += torch.tensor(self.noise.sample(), dtype=torch.float).cuda()  # ノイズを加えることで探索を促進
                action += torch.tensor(self.noise.sample(), dtype=torch.float, device=torch.device('cpu'))
            actions.append(action)
        actions = torch.cat(actions, dim=0)  # 全エージェントの行動を連結
        return np.clip(actions.detach().cpu().numpy(), -1.0, 1.0)  # 行動をNumpy配列に変換してクリッピングして返す


# In[ ]:


# 各種設定
num_episode = 2  # 学習エピソード数（論文では25000になっています）
memory_size = 100000  # replay bufferの大きさ
initial_memory_size = 100000  # 最初に貯めるデータ数

# ログ用の設定
episode_rewards = []
num_average_epidodes = 100

# 環境の作成
env = make_env('simple_spread')
max_steps = 25  # エピソードの最大ステップ数

# MADDPGエージェントの作成
agent = MaddpgAgents(18, 5, num_agent=env.n, memory_size=memory_size)

# 最初にreplay bufferにノイズのかかった行動をしたときのデータを入れる
state = env.reset()  # 環境をリセットして初期状態を取得
for step in range(initial_memory_size):
    if step % max_steps == 0:
        state = env.reset()  # エピソードの開始時に環境をリセットして初期状態を取得
    actions = agent.get_action(state)  # MADDPGエージェントから行動を取得
    next_state, reward, done, _ = env.step(actions)  # 環境に行動を与えて次の状態と報酬を取得
    agent.buffer.cache(state, next_state, actions, reward, done)  # リプレイバッファにデータを追加
    state = next_state  # 状態を更新
print('%d Data collected' % (initial_memory_size))
# In[ ]:


for episode in range(num_episode):
    state = env.reset()  # 環境をリセットして初期状態を取得
    episode_reward = 0  # エピソードの累積報酬を初期化
    for t in range(max_steps):
        actions = agent.get_action(state)  # MADDPGエージェントから行動を取得
        next_state, reward, done, _ = env.step(actions)  # 環境に行動を与えて次の状態と報酬を取得
        episode_reward += sum(reward)  # 報酬を累積
        agent.buffer.cache(state, next_state, actions, reward, done)  # リプレイバッファにデータを追加
        state = next_state  # 状態を更新
        if all(done):  # 全てのエージェントが終了したら
            break  # エピソードを終了
    if episode % 4 == 0:
        agent.update()  # 一定間隔でエージェントを学習
    episode_rewards.append(episode_reward)  # エピソードの累積報酬を記録
    if episode % 20 == 0:
        print("Episode %d finished | Episode reward %f" % (episode, episode_reward))

# 累積報酬の移動平均を表示
moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)),moving_average)
plt.title('MADDPG: average rewards in %d episodes' % num_average_epidodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

env.close()


# In[ ]:


state = env.reset()  # 環境をリセットして初期状態を取得
episode_reward = 0  # エピソードの累積報酬を初期化
frames = []  # フレームを保存するリスト
env.render()  # 環境を描画
screen = env.render(mode='rgb_array')  # 描画画面を取得
frames.append(screen[0])  # 最初のフレームを保存
for t in range(max_steps):
    actions = agent.get_action(state, greedy=True)  # グリーディーな行動をエージェントから取得
    next_state, reward, done, _ = env.step(actions)  # 環境に行動を与えて次の状態と報酬を取得
    episode_reward += sum(reward)  # 報酬を累積
    agent.buffer.cache(state, next_state, actions, reward, done)  # リプレイバッファにデータを追加
    env.render()  # 環境を描画
    screen = env.render(mode='rgb_array')  # 描画画面を取得
    frames.append(screen[0])  # フレームを保存
    state = next_state  # 状態を更新
print(episode_reward)  # エピソードの累積報酬を表示


# In[ ]:


# 結果の確認
plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)  # 描画領域のサイズを設定
patch = plt.imshow(frames[0])  # 最初のフレームを表示するパッチを作成
plt.axis('off')  # 軸を非表示に設定

def animate(i):
    patch.set_data(frames[i])  # フレームを更新

anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)  # アニメーションを作成

HTML(anim.to_jshtml())  # アニメーションを表示

