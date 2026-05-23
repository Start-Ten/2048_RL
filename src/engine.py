"""2048 game engine (vectorized NumPy implementation)."""
import numpy as np
import random
import math


class Game2048:
    def __init__(self, size=4):
        self.size = size
        self._state_buf = np.zeros((8, size, size), dtype=np.float32)
        self._slide_bufs = [np.zeros(size, dtype=np.int32) for _ in range(4)]
        self._board = np.zeros((size, size), dtype=np.int32)
        self._score = 0
        self.game_over = False
        self.reset()

    @property
    def score(self): return self._score

    @property
    def board(self): return self._board

    def reset(self):
        self._board.fill(0)
        self._score = 0
        self.game_over = False
        self._add_tile(); self._add_tile()
        return self.get_state()

    def _add_tile(self):
        flat = self._board.ravel()
        zeros = np.flatnonzero(flat == 0)
        if len(zeros) > 0:
            idx = zeros[random.randint(0, len(zeros) - 1)]
            flat[idx] = 2 if random.random() < 0.9 else 4

    def _slide_row_left(self, row):
        nz = row[row != 0]
        if len(nz) == 0: return False, 0
        buf = self._slide_bufs[0]; idx = 0; merged = False; sc = 0
        i = 0; n = len(nz)
        while i < n:
            if i + 1 < n and nz[i] == nz[i + 1]:
                buf[idx] = nz[i] * 2; sc += nz[i] * 2; merged = True; i += 2
            else:
                buf[idx] = nz[i]; i += 1
            idx += 1
        buf[idx:] = 0
        pos = np.flatnonzero(row)
        moved = (len(pos) == 0 or pos[0] != 0 or merged)
        return moved or merged, sc

    def move(self, direction):
        old = self._board.copy(); old_score = self._score; moved = False
        if direction == 0:
            for j in range(4):
                mv, sc = self._slide_row_left(self._board[:, j])
                if mv: moved = True; self._board[:, j] = self._slide_bufs[0]
                self._score += sc
        elif direction == 1:
            for i in range(4):
                mv, sc = self._slide_row_left(self._board[i, ::-1].copy())
                if mv: moved = True; self._board[i, :] = self._slide_bufs[0][::-1]
                self._score += sc
        elif direction == 2:
            for j in range(4):
                mv, sc = self._slide_row_left(self._board[::-1, j].copy())
                if mv: moved = True; self._board[:, j] = self._slide_bufs[0][::-1]
                self._score += sc
        else:
            for i in range(4):
                mv, sc = self._slide_row_left(self._board[i, :])
                if mv: moved = True; self._board[i, :] = self._slide_bufs[0]
                self._score += sc
        if moved: self._add_tile(); self.game_over = self._is_game_over()
        return self.get_state(), self._calc_reward(old_score, old), self.game_over

    def _is_game_over(self):
        b = self._board
        return not (np.any(b == 0) or np.any(b[:, :-1] == b[:, 1:]) or
                    np.any(b[:-1, :] == b[1:, :]))

    def _calc_reward(self, old_score, old_board):
        sr = (self._score - old_score) * 0.1
        eb = np.count_nonzero(old_board == 0); ea = np.count_nonzero(self._board == 0)
        er = (ea - eb) * 0.1
        if ea < 5 and ea != eb: er -= (5 - ea) ** 2 * 0.15
        mb = old_board.max(); ma = self._board.max()
        mr = math.sqrt(ma) * 1.5 if ma > mb else 0.0
        cr = 0.5 if ma >= 256 and (self._board[0, 0] == ma or self._board[0, -1] == ma or
                                    self._board[-1, 0] == ma or self._board[-1, -1] == ma) else 0.0
        mon = 0.0
        for i in range(4):
            nz = self._board[i][self._board[i] > 0]
            if len(nz) >= 3 and (np.all(np.diff(nz) >= 0) or np.all(np.diff(nz) <= 0)): mon += 0.02
            nz = self._board[:, i][self._board[:, i] > 0]
            if len(nz) >= 3 and (np.all(np.diff(nz) >= 0) or np.all(np.diff(nz) <= 0)): mon += 0.02
        return sr + er + mr + cr + mon + (-15.0 if self.game_over else 0.0)

    def get_state(self):
        s = self._state_buf; b = self._board; s.fill(0.0)
        pos = b > 0; s[0][pos] = np.log2(b[pos]) * (1/17)
        s[1] = (b == 0).astype(np.float32)
        horiz = (b[:, :-1] > 0) & (b[:, :-1] == b[:, 1:])
        s[2, :, :-1] += horiz.astype(np.float32); s[2, :, 1:] += horiz.astype(np.float32)
        vert = (b[:-1, :] > 0) & (b[:-1, :] == b[1:, :])
        s[2, :-1, :] += vert.astype(np.float32); s[2, 1:, :] += vert.astype(np.float32)
        mx = b.max()
        if mx > 0:
            s[3] = (b == mx).astype(np.float32)
            sm = (b > 0) & (b < mx)
            if sm.any(): s[4] = (b == b[sm].max()).astype(np.float32)
            if b[0,0]==mx or b[0,-1]==mx or b[-1,0]==mx or b[-1,-1]==mx:
                mp = np.unravel_index(b.argmax(), (4,4)); s[7, mp[0], mp[1]] = 1.0
        for i in range(4):
            nz = b[i][b[i] > 0]
            if len(nz) >= 2 and np.all(np.diff(nz) <= 0): s[5, i, :] = 1.0
            nz = b[:, i][b[:, i] > 0]
            if len(nz) >= 2 and np.all(np.diff(nz) <= 0): s[6, :, i] = 1.0
        return s.copy()

    def get_valid_moves(self):
        b = self._board; cu=cd=cr=cl=False
        has_empty = b.min() == 0
        for i in range(4):
            for j in range(4):
                v = b[i,j]
                if v == 0: continue
                if not cu and i > 0 and (b[i-1,j]==0 or b[i-1,j]==v): cu = True
                if not cd and i < 3 and (b[i+1,j]==0 or b[i+1,j]==v): cd = True
                if not cl and j > 0 and (b[i,j-1]==0 or b[i,j-1]==v): cl = True
                if not cr and j < 3 and (b[i,j+1]==0 or b[i,j+1]==v): cr = True
                if cu and cd and cl and cr: return [0,1,2,3]
        if not has_empty:
            cu=cd=cr=cl=False
            for i in range(4):
                for j in range(4):
                    v = b[i,j]
                    if not cu and i > 0 and b[i-1,j]==v: cu = True
                    if not cd and i < 3 and b[i+1,j]==v: cd = True
                    if not cl and j > 0 and b[i,j-1]==v: cl = True
                    if not cr and j < 3 and b[i,j+1]==v: cr = True
        valid = []
        if cu: valid.append(0)
        if cr: valid.append(1)
        if cd: valid.append(2)
        if cl: valid.append(3)
        return valid
