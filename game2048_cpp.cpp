/**
 * game2048_cpp.cpp - 2048 Game Engine (C++ Accelerated + Batch Parallel)
 *
 * Build: pip install pybind11 && python setup.py build_ext --inplace
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <cstring>
#include <cmath>
#include <vector>

namespace py = pybind11;

static constexpr int SIZE = 4;
static constexpr int CHANNELS = 8;
static constexpr int N_CELLS = SIZE * SIZE;
static constexpr int N_STATE = CHANNELS * N_CELLS;

// ---- slide a row left, return score increment ----
inline int slide_row_left(int row[4]) {
    int out = 0, sc = 0;
    for (int i = 0; i < 4; i++) {
        int val = row[i];
        if (val == 0) continue;
        if (out > 0 && row[out - 1] == val) {
            row[out - 1] = val * 2;
            sc += val * 2;
        } else {
            row[out] = val;
            out++;
        }
    }
    for (int i = out; i < 4; i++) row[i] = 0;
    return sc;
}

inline bool boards_differ(const int a[4][4], const int b[4][4]) {
    return std::memcmp(a, b, sizeof(int) * N_CELLS) != 0;
}

// ============================================================
// Single Environment
// ============================================================
class Game2048 {
public:
    int board[SIZE][SIZE]{};
    int score = 0;
    bool game_over = false;

    Game2048() { init_tiles(); }

    void reset() {
        std::memset(board, 0, sizeof(board));
        score = 0;
        game_over = false;
        init_tiles();
    }

    void move(int direction, float* out_state, float& reward, bool& done) {
        int old[SIZE][SIZE];
        std::memcpy(old, board, sizeof(board));
        int old_score = score;
        bool moved = false;

        switch (direction) {
        case 0: // up
            for (int c = 0; c < SIZE; c++) {
                int col[4] = {board[0][c], board[1][c], board[2][c], board[3][c]};
                score += slide_row_left(col);
                if (!moved) {
                    for (int r = 0; r < SIZE; r++)
                        if (col[r] != old[r][c]) { moved = true; break; }
                }
                for (int r = 0; r < SIZE; r++) board[r][c] = col[r];
            }
            break;
        case 1: // right
            for (int r = 0; r < SIZE; r++) {
                int row[4] = {board[r][3], board[r][2], board[r][1], board[r][0]};
                score += slide_row_left(row);
                for (int c = 0; c < SIZE; c++) board[r][c] = row[3 - c];
            }
            moved = boards_differ(board, old);
            break;
        case 2: // down
            for (int c = 0; c < SIZE; c++) {
                int col[4] = {board[3][c], board[2][c], board[1][c], board[0][c]};
                score += slide_row_left(col);
                for (int r = 0; r < SIZE; r++) board[r][c] = col[3 - r];
            }
            moved = boards_differ(board, old);
            break;
        case 3: // left
            for (int r = 0; r < SIZE; r++)
                score += slide_row_left(board[r]);
            moved = boards_differ(board, old);
            break;
        }

        if (moved) { add_tile(); game_over = check_gameover(); }
        reward = calc_reward(old_score, old);
        done = game_over;
        write_state(out_state);
    }

    void write_state(float* state) const {
        std::memset(state, 0, N_STATE * sizeof(float));
        int max_val = 0, max_r = 0, max_c = 0;

        // channel 0: log2 values, channel 1: empty
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                int v = board[r][c];
                int idx = r * SIZE + c;
                if (v > 0) {
                    state[idx] = std::log2f((float)v) / 16.0f;
                    if (v > max_val) { max_val = v; max_r = r; max_c = c; }
                } else {
                    state[N_CELLS + idx] = 1.0f;
                }
            }
        }

        // channel 2: mergeable neighbours
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                int v = board[r][c];
                if (v == 0) continue;
                int idx = r * SIZE + c;
                if (c < SIZE - 1 && board[r][c + 1] == v) {
                    state[2 * N_CELLS + idx] = 1.0f;
                    state[2 * N_CELLS + idx + 1] = 1.0f;
                }
                if (r < SIZE - 1 && board[r + 1][c] == v) {
                    state[2 * N_CELLS + idx] = 1.0f;
                    state[2 * N_CELLS + (r + 1) * SIZE + c] = 1.0f;
                }
            }
        }

        if (max_val == 0) return;

        // channel 3: max position
        state[3 * N_CELLS + max_r * SIZE + max_c] = 1.0f;

        // channel 4: second max
        int second_val = 0;
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++) {
                int v = board[r][c];
                if (v > 0 && v < max_val && v > second_val) second_val = v;
            }
        if (second_val > 0)
            for (int r = 0; r < SIZE; r++)
                for (int c = 0; c < SIZE; c++)
                    if (board[r][c] == second_val)
                        state[4 * N_CELLS + r * SIZE + c] = 1.0f;

        // channel 7: max at corner
        if ((max_r == 0 && max_c == 0) || (max_r == 0 && max_c == 3) ||
            (max_r == 3 && max_c == 0) || (max_r == 3 && max_c == 3))
            state[7 * N_CELLS + max_r * SIZE + max_c] = 1.0f;

        // channel 5: row monotonic
        for (int r = 0; r < SIZE; r++) {
            int nz[4], n = 0;
            for (int c = 0; c < SIZE; c++) if (board[r][c] > 0) nz[n++] = board[r][c];
            if (n >= 2) {
                bool ok = true;
                for (int k = 1; k < n; k++) if (nz[k] > nz[k - 1]) { ok = false; break; }
                if (ok) for (int c = 0; c < SIZE; c++) state[5 * N_CELLS + r * SIZE + c] = 1.0f;
            }
        }

        // channel 6: col monotonic
        for (int c = 0; c < SIZE; c++) {
            int nz[4], n = 0;
            for (int r = 0; r < SIZE; r++) if (board[r][c] > 0) nz[n++] = board[r][c];
            if (n >= 2) {
                bool ok = true;
                for (int k = 1; k < n; k++) if (nz[k] > nz[k - 1]) { ok = false; break; }
                if (ok) for (int r = 0; r < SIZE; r++) state[6 * N_CELLS + r * SIZE + c] = 1.0f;
            }
        }
    }

    void get_valid_moves(int* out, int& count) const {
        count = 0;
        bool can[4] = {};
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                int v = board[r][c];
                if (v == 0) { can[0] = can[1] = can[2] = can[3] = true; goto done; }
                if (r > 0 && (board[r-1][c] == 0 || board[r-1][c] == v)) can[0] = true;
                if (r < 3 && (board[r+1][c] == 0 || board[r+1][c] == v)) can[2] = true;
                if (c > 0 && (board[r][c-1] == 0 || board[r][c-1] == v)) can[3] = true;
                if (c < 3 && (board[r][c+1] == 0 || board[r][c+1] == v)) can[1] = true;
                if (can[0] && can[1] && can[2] && can[3]) goto done;
            }
        }
    done:
        if (can[0]) out[count++] = 0;
        if (can[1]) out[count++] = 1;
        if (can[2]) out[count++] = 2;
        if (can[3]) out[count++] = 3;
    }

private:
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> udist{0.0f, 1.0f};

    void init_tiles() { add_tile(); add_tile(); }

    void add_tile() {
        int empty[N_CELLS], n = 0;
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++)
                if (board[r][c] == 0) empty[n++] = r * SIZE + c;
        if (n > 0) {
            int idx = empty[(int)(udist(rng) * n) % n];
            board[idx / SIZE][idx % SIZE] = (udist(rng) < 0.9f) ? 2 : 4;
        }
    }

    bool check_gameover() const {
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++)
                if (board[r][c] == 0) return false;
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE - 1; c++)
                if (board[r][c] == board[r][c + 1]) return false;
        for (int c = 0; c < SIZE; c++)
            for (int r = 0; r < SIZE - 1; r++)
                if (board[r][c] == board[r + 1][c]) return false;
        return true;
    }

    float calc_reward(int old_score, const int old[4][4]) const {
        float sr = (float)(score - old_score) * 0.1f;
        int eb = 0, ea = 0, mb = 0, ma = 0;
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                if (old[r][c] == 0) eb++; if (board[r][c] == 0) ea++;
                if (old[r][c] > mb) mb = old[r][c];
                if (board[r][c] > ma) ma = board[r][c];
            }
        }
        float er = (float)(ea - eb) * 0.1f;
        if (ea < 5 && ea != eb) er -= (float)((5 - ea) * (5 - ea)) * 0.15f;
        float mr = (ma > mb) ? std::sqrt((float)ma) * 1.5f : 0.0f;
        float cr = 0.0f;
        if (ma >= 256 && (board[0][0] == ma || board[0][3] == ma ||
                          board[3][0] == ma || board[3][3] == ma))
            cr = 0.5f;
        float mon = 0.0f;
        for (int r = 0; r < SIZE; r++) {
            int nz[4], n = 0;
            for (int c = 0; c < SIZE; c++) if (board[r][c] > 0) nz[n++] = board[r][c];
            if (n >= 3) {
                bool d = true, inc = true;
                for (int k = 1; k < n; k++) {
                    if (nz[k] > nz[k - 1]) d = false;
                    if (nz[k] < nz[k - 1]) inc = false;
                }
                if (d || inc) mon += 0.02f;
            }
            n = 0;
            for (int c = 0; c < SIZE; c++) if (board[c][r] > 0) nz[n++] = board[c][r];
            if (n >= 3) {
                bool d = true, inc = true;
                for (int k = 1; k < n; k++) {
                    if (nz[k] > nz[k - 1]) d = false;
                    if (nz[k] < nz[k - 1]) inc = false;
                }
                if (d || inc) mon += 0.02f;
            }
        }
        float gp = game_over ? -15.0f : 0.0f;
        return sr + er + mr + cr + mon + gp;
    }
};

// ============================================================
// Batch Parallel Environment
// ============================================================

struct EnvState {
    int board[4][4];
    int score;
    bool done;
    float reward;
};

class BatchGame2048 {
public:
    const int n_envs;

    explicit BatchGame2048(int n) : n_envs(n), envs(n), rngs(n) {
        std::random_device rd;
        for (int i = 0; i < n; i++) {
            rngs[i].seed(rd());
            reset_env(i);
        }
    }

    py::array_t<float> reset() {
        auto result = py::array_t<float>({n_envs, CHANNELS, SIZE, SIZE});
        float* data = result.mutable_data();
        for (int i = 0; i < n_envs; i++) {
            reset_env(i);
            write_state_of(i, data + i * N_STATE);
        }
        return result;
    }

    py::tuple step(py::array_t<int> actions) {
        auto act = actions.unchecked<1>();
        if (act.size() != n_envs)
            throw std::runtime_error("actions size must match n_envs");

        auto states = py::array_t<float>({n_envs, CHANNELS, SIZE, SIZE});
        auto rewards = py::array_t<float>({n_envs});
        auto dones = py::array_t<bool>({n_envs});

        float* sdata = states.mutable_data();
        float* rdata = rewards.mutable_data();
        bool*  ddata = dones.mutable_data();

        for (int i = 0; i < n_envs; i++) {
            step_env(i, act(i));
            write_state_of(i, sdata + i * N_STATE);
            rdata[i] = envs[i].reward;
            ddata[i] = envs[i].done;
        }

        return py::make_tuple(states, rewards, dones);
    }

    py::array_t<bool> get_valid_masks() {
        auto mask = py::array_t<bool>({n_envs, SIZE});
        bool* data = mask.mutable_data();
        for (int i = 0; i < n_envs; i++) {
            int moves[4], cnt;
            get_valid_of(i, moves, cnt);
            for (int a = 0; a < 4; a++) data[i * 4 + a] = false;
            for (int k = 0; k < cnt; k++) data[i * 4 + moves[k]] = true;
        }
        return mask;
    }

    py::array_t<int> get_scores() {
        auto arr = py::array_t<int>({n_envs});
        int* data = arr.mutable_data();
        for (int i = 0; i < n_envs; i++) data[i] = envs[i].score;
        return arr;
    }

    py::array_t<int> get_max_tiles() {
        auto arr = py::array_t<int>({n_envs});
        int* data = arr.mutable_data();
        for (int i = 0; i < n_envs; i++) {
            int mx = 0;
            for (int r = 0; r < SIZE; r++)
                for (int c = 0; c < SIZE; c++)
                    if (envs[i].board[r][c] > mx) mx = envs[i].board[r][c];
            data[i] = mx;
        }
        return arr;
    }

    int get_score(int i) const { return envs[i].score; }
    bool get_done(int i) const { return envs[i].done; }

private:
    std::vector<EnvState> envs;
    std::vector<std::mt19937> rngs;

    void reset_env(int i) {
        auto& e = envs[i];
        std::memset(e.board, 0, sizeof(e.board));
        e.score = 0; e.done = false; e.reward = 0.0f;
        add_tile(i); add_tile(i);
    }

    void add_tile(int i) {
        auto& e = envs[i];
        std::uniform_real_distribution<float> ud(0, 1);
        int empty[N_CELLS], n = 0;
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++)
                if (e.board[r][c] == 0) empty[n++] = r * SIZE + c;
        if (n > 0) {
            int idx = empty[(int)(ud(rngs[i]) * n) % n];
            e.board[idx / SIZE][idx % SIZE] = (ud(rngs[i]) < 0.9f) ? 2 : 4;
        }
    }

    void step_env(int i, int action) {
        auto& e = envs[i];
        if (e.done) return;

        int old[4][4];
        std::memcpy(old, e.board, sizeof(old));
        int old_score = e.score;
        bool moved = false;

        switch (action) {
        case 0:
            for (int c = 0; c < SIZE; c++) {
                int col[4] = {e.board[0][c], e.board[1][c], e.board[2][c], e.board[3][c]};
                e.score += slide_row_left(col);
                if (!moved)
                    for (int r = 0; r < SIZE; r++)
                        if (col[r] != old[r][c]) { moved = true; break; }
                for (int r = 0; r < SIZE; r++) e.board[r][c] = col[r];
            }
            break;
        case 1:
            for (int r = 0; r < SIZE; r++) {
                int row[4] = {e.board[r][3], e.board[r][2], e.board[r][1], e.board[r][0]};
                e.score += slide_row_left(row);
                for (int c = 0; c < SIZE; c++) e.board[r][c] = row[3 - c];
            }
            moved = boards_differ(e.board, old);
            break;
        case 2:
            for (int c = 0; c < SIZE; c++) {
                int col[4] = {e.board[3][c], e.board[2][c], e.board[1][c], e.board[0][c]};
                e.score += slide_row_left(col);
                for (int r = 0; r < SIZE; r++) e.board[r][c] = col[3 - r];
            }
            moved = boards_differ(e.board, old);
            break;
        case 3:
            for (int r = 0; r < SIZE; r++)
                e.score += slide_row_left(e.board[r]);
            moved = boards_differ(e.board, old);
            break;
        }

        if (moved) { add_tile(i); e.done = check_gameover_of(i); }
        e.reward = calc_reward_of(i, old_score, old);
    }

    bool check_gameover_of(int i) const {
        auto& b = envs[i].board;
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++)
                if (b[r][c] == 0) return false;
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE - 1; c++)
                if (b[r][c] == b[r][c + 1]) return false;
        for (int c = 0; c < SIZE; c++)
            for (int r = 0; r < SIZE - 1; r++)
                if (b[r][c] == b[r + 1][c]) return false;
        return true;
    }

    float calc_reward_of(int i, int old_score, const int old[4][4]) const {
        auto& e = envs[i];
        float sr = (float)(e.score - old_score) * 0.1f;
        int eb = 0, ea = 0, mb = 0, ma = 0;
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                if (old[r][c] == 0) eb++; if (e.board[r][c] == 0) ea++;
                if (old[r][c] > mb) mb = old[r][c];
                if (e.board[r][c] > ma) ma = e.board[r][c];
            }
        }
        float er = (float)(ea - eb) * 0.1f;
        if (ea < 5 && ea != eb) er -= (float)((5 - ea) * (5 - ea)) * 0.15f;
        float mr = (ma > mb) ? std::sqrt((float)ma) * 1.5f : 0.0f;
        float cr = 0.0f;
        if (ma >= 256 && (e.board[0][0] == ma || e.board[0][3] == ma ||
                          e.board[3][0] == ma || e.board[3][3] == ma))
            cr = 0.5f;
        float mon = 0.0f;
        for (int r = 0; r < SIZE; r++) {
            int nz[4], n = 0;
            for (int c = 0; c < SIZE; c++) if (e.board[r][c] > 0) nz[n++] = e.board[r][c];
            if (n >= 3) { bool d=true, inc=true; for (int k=1;k<n;k++) {if(nz[k]>nz[k-1])d=false;if(nz[k]<nz[k-1])inc=false;} if(d||inc) mon+=0.02f; }
            n = 0;
            for (int c = 0; c < SIZE; c++) if (e.board[c][r] > 0) nz[n++] = e.board[c][r];
            if (n >= 3) { bool d=true, inc=true; for (int k=1;k<n;k++) {if(nz[k]>nz[k-1])d=false;if(nz[k]<nz[k-1])inc=false;} if(d||inc) mon+=0.02f; }
        }
        float gp = e.done ? -15.0f : 0.0f;
        return sr + er + mr + cr + mon + gp;
    }

    void get_valid_of(int i, int* out, int& cnt) const {
        cnt = 0;
        auto& b = envs[i].board;
        bool can[4] = {};
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                int v = b[r][c];
                if (v == 0) { can[0]=can[1]=can[2]=can[3]=true; goto done2; }
                if (r > 0 && (b[r-1][c]==0||b[r-1][c]==v)) can[0]=true;
                if (r < 3 && (b[r+1][c]==0||b[r+1][c]==v)) can[2]=true;
                if (c > 0 && (b[r][c-1]==0||b[r][c-1]==v)) can[3]=true;
                if (c < 3 && (b[r][c+1]==0||b[r][c+1]==v)) can[1]=true;
                if (can[0]&&can[1]&&can[2]&&can[3]) goto done2;
            }
        }
    done2:
        if (can[0]) out[cnt++]=0; if (can[1]) out[cnt++]=1;
        if (can[2]) out[cnt++]=2; if (can[3]) out[cnt++]=3;
    }

    void write_state_of(int i, float* state) const {
        auto& b = envs[i].board;
        std::memset(state, 0, N_STATE * sizeof(float));
        int max_val = 0, max_r = 0, max_c = 0;

        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                int v = b[r][c];
                int idx = r * SIZE + c;
                if (v > 0) {
                    state[idx] = std::log2f((float)v) / 16.0f;
                    if (v > max_val) { max_val = v; max_r = r; max_c = c; }
                } else {
                    state[N_CELLS + idx] = 1.0f;
                }
            }
        }
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                int v = b[r][c];
                if (v == 0) continue;
                int idx = r * SIZE + c;
                if (c < SIZE-1 && b[r][c+1]==v) { state[2*N_CELLS+idx]=1.0f; state[2*N_CELLS+idx+1]=1.0f; }
                if (r < SIZE-1 && b[r+1][c]==v) { state[2*N_CELLS+idx]=1.0f; state[2*N_CELLS+(r+1)*SIZE+c]=1.0f; }
            }
        }
        if (max_val == 0) return;
        state[3*N_CELLS+max_r*SIZE+max_c] = 1.0f;
        int second_val = 0;
        for (int r=0;r<SIZE;r++) for(int c=0;c<SIZE;c++) {
            int v=b[r][c]; if(v>0&&v<max_val&&v>second_val)second_val=v;
        }
        if(second_val>0) for(int r=0;r<SIZE;r++) for(int c=0;c<SIZE;c++)
            if(b[r][c]==second_val) state[4*N_CELLS+r*SIZE+c]=1.0f;
        if((max_r==0&&max_c==0)||(max_r==0&&max_c==3)||(max_r==3&&max_c==0)||(max_r==3&&max_c==3))
            state[7*N_CELLS+max_r*SIZE+max_c]=1.0f;

        for (int r=0;r<SIZE;r++) {
            int nz[4], n=0;
            for(int c=0;c<SIZE;c++) if(b[r][c]>0)nz[n++]=b[r][c];
            if(n>=2){ bool ok=true; for(int k=1;k<n;k++)if(nz[k]>nz[k-1]){ok=false;break;} if(ok)for(int c=0;c<SIZE;c++)state[5*N_CELLS+r*SIZE+c]=1.0f; }
            n=0;
            for(int c=0;c<SIZE;c++) if(b[c][r]>0)nz[n++]=b[c][r];
            if(n>=2){ bool ok=true; for(int k=1;k<n;k++)if(nz[k]>nz[k-1]){ok=false;break;} if(ok)for(int c=0;c<SIZE;c++)state[6*N_CELLS+c*SIZE+r]=1.0f; }
        }
    }
};

// ============================================================
// pybind11 bindings
// ============================================================

PYBIND11_MODULE(game2048_cpp, m) {
    m.doc() = "2048 Game Engine (C++ accelerated)";

    py::class_<Game2048>(m, "Game2048")
        .def(py::init<>())
        .def("reset", [](Game2048& g) {
            g.reset();
            auto arr = py::array_t<float>({CHANNELS, SIZE, SIZE});
            g.write_state(arr.mutable_data());
            return arr;
        })
        .def("move", [](Game2048& g, int dir) {
            auto arr = py::array_t<float>({CHANNELS, SIZE, SIZE});
            float reward; bool done;
            g.move(dir, arr.mutable_data(), reward, done);
            return py::make_tuple(arr, reward, done);
        })
        .def("get_state", [](Game2048& g) {
            auto arr = py::array_t<float>({CHANNELS, SIZE, SIZE});
            g.write_state(arr.mutable_data());
            return arr;
        })
        .def("get_valid_moves", [](const Game2048& g) {
            int m[4], c; g.get_valid_moves(m, c);
            py::list lst;
            for (int i = 0; i < c; i++) lst.append(m[i]);
            return lst;
        })
        .def_readonly("score", &Game2048::score)
        .def_readonly("game_over", &Game2048::game_over)
        .def("get_board", [](const Game2048& g) {
            auto arr = py::array_t<int>({SIZE, SIZE});
            auto* p = arr.mutable_data();
            for (int i=0;i<SIZE;i++) for(int j=0;j<SIZE;j++) p[i*SIZE+j]=g.board[i][j];
            return arr;
        });

    py::class_<BatchGame2048>(m, "BatchGame2048")
        .def(py::init<int>(), py::arg("n_envs"))
        .def("reset", &BatchGame2048::reset)
        .def("step", &BatchGame2048::step, py::arg("actions"))
        .def("get_valid_masks", &BatchGame2048::get_valid_masks)
        .def("get_scores", &BatchGame2048::get_scores)
        .def("get_max_tiles", &BatchGame2048::get_max_tiles)
        .def("get_score", &BatchGame2048::get_score)
        .def("get_done", &BatchGame2048::get_done)
        .def_readonly("n_envs", &BatchGame2048::n_envs);
}
