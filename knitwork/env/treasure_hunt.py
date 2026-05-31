# knitwork/env/treasure_hunt.py
"""
TreasureHunt: векторизованная среда для тестирования
ассоциативной памяти рекуррентных агентов.

Суть задачи:
  - N×N сетка, агент движется (4 действия: URDL)
  - На карте n_pairs пар (ключ_c, дверь_c) для каждого цвета c
  - Агент видит только локальное окно 3×3 вокруг себя (частичная наблюдаемость)
  - Поднял ключ_c → запомни где. Нашёл дверь_c → нужен ключ_c → открываешь
  - Reward: +1 за каждую открытую дверь, -0.01/шаг (time penalty)
  - Эпизод заканчивается при все двери открыты или max_steps

Для тестирования памяти:
  - easy: ключ и дверь близко, немного пар
  - hard: карта большая, много пар, ключи появляются задолго до дверей

API совместим с StoreDistractQueryGenerator:
  - next() → dict(tokens, targets, reset_mask, sq_gaps)
  - n_tokens: размер словаря наблюдений
  - V: размер пространства действий
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Константы токенов/действий ──────────────────────────────────────────────
ACT_UP    = 0
ACT_DOWN  = 1
ACT_LEFT  = 2
ACT_RIGHT = 3
N_ACTIONS = 4

# Типы клеток на карте
CELL_EMPTY    = 0
CELL_WALL     = 1
CELL_KEY_BASE = 2          # CELL_KEY_BASE + c = ключ цвета c
CELL_DOOR_BASE = 2 + 8     # CELL_DOOR_BASE + c = дверь цвета c
CELL_AGENT    = 2 + 8 + 8  # агент
CELL_VISITED_KEY_BASE = CELL_AGENT + 1  # уже собранный ключ (отсутствует)

# Размер словаря: пустая + стена + 8 ключей + 8 дверей + агент
VOCAB_SIZE = CELL_AGENT + 1  # = 19


@dataclass
class TreasureHuntConfig:
    grid_size: int = 10         # N×N сетка
    n_colors: int = 4           # количество пар ключ-дверь
    view_radius: int = 1        # агент видит (2r+1)×(2r+1) окно
    max_steps: int = 200        # максимум шагов в эпизоде
    wall_density: float = 0.1   # вероятность стены
    # Расстояние: None = случайное, иначе ключ и дверь ~dist клеток
    min_key_door_dist: Optional[int] = None
    reward_open_door: float = 1.0
    reward_step: float = -0.01
    reward_revisit: float = -0.0  # штраф за повторное посещение


class TreasureHuntEnv:
    """
    Векторизованная среда TreasureHunt для n_envs параллельных эпизодов.

    Выход next() совместим с форматом StoreDistractQueryGenerator,
    используемым в run_sdq2.py.
    """

    # Константы для совместимости с run_sdq2.py
    @property
    def n_tokens(self) -> int:
        """Размер входного словаря (наблюдений)."""
        obs_dim = (2 * self.cfg.view_radius + 1) ** 2
        # Каждая клетка — один токен из VOCAB_SIZE вариантов
        # Кодируем всё наблюдение как один целочисленный токен
        # через хэш: obs_flat -> единственный индекс
        # Для совместимости с nn.Embedding используем линейное кодирование
        # obs[i] ∈ [0, VOCAB_SIZE), obs_size = view_size²
        # Итого: VOCAB_SIZE^obs_size — слишком много.
        # Используем упрощение: возвращаем obs_flat как ПОСЛЕДОВАТЕЛЬНОСТЬ,
        # но для совместимости с однотокенным входом кодируем центральную клетку
        # + направленческий токен в один индекс.
        # Полноценно: входной тензор имеет форму [batch, obs_flat_size].
        # Но run_sdq2 ожидает [batch, 1].
        # Решение: наблюдение = одна числовая "сводная" метка,
        # кодирующая (central_cell, has_key_c, inv_bitmask).
        return self._n_tokens

    @property
    def V(self) -> int:
        """Размер словаря выходов (действий)."""
        return N_ACTIONS

    def __init__(
        self,
        cfg: TreasureHuntConfig,
        n_envs: int,
        seed: int = 42,
        ignore_index: int = -100,
    ):
        self.cfg       = cfg
        self.n_envs    = n_envs
        self.rng       = np.random.default_rng(seed)
        self.ignore_index = ignore_index

        # Вычисляем размер словаря наблюдений:
        # obs = (центральная клетка [VOCAB_SIZE вариантов])
        #       × (инвентарь агента: 2^n_colors вариантов)
        # → VOCAB_SIZE × 2^n_colors
        self._n_tokens = VOCAB_SIZE * (2 ** cfg.n_colors)

        self.G   = cfg.grid_size
        self.R   = cfg.view_radius

        # Состояния всех сред
        self._grids     : np.ndarray  # [n_envs, G, G]
        self._agent_pos : np.ndarray  # [n_envs, 2]  (row, col)
        self._inventory : np.ndarray  # [n_envs, n_colors] bool
        self._opened    : np.ndarray  # [n_envs, n_colors] bool
        self._steps     : np.ndarray  # [n_envs]
        self._done      : np.ndarray  # [n_envs] bool

        # Целевые действия (метки для supervised loss)
        # В TreasureHunt для recurrent агента используем
        # следующее действие оптимальной политики (BFS-based)
        # как обучающий сигнал — IMITATION обучение.
        # Если оптимальный путь неизвестен — ignore_index.
        self._targets   : np.ndarray  # [n_envs]

        # sq_gaps: "сложность" текущего шага
        # = количество ещё не открытых дверей / n_colors ∈ [0, 1]
        self._sq_gaps   : np.ndarray  # [n_envs]

        self._reset_all()

    # ── Инициализация ────────────────────────────────────────────────────────

    def _reset_all(self):
        self._grids     = np.zeros((self.n_envs, self.G, self.G), dtype=np.int32)
        self._agent_pos = np.zeros((self.n_envs, 2), dtype=np.int32)
        self._inventory = np.zeros((self.n_envs, self.cfg.n_colors), dtype=bool)
        self._opened    = np.zeros((self.n_envs, self.cfg.n_colors), dtype=bool)
        self._steps     = np.zeros(self.n_envs, dtype=np.int32)
        self._done      = np.ones(self.n_envs, dtype=bool)  # форс-ресет на старте
        self._targets   = np.full(self.n_envs, self.ignore_index, dtype=np.int64)
        self._sq_gaps   = np.zeros(self.n_envs, dtype=np.float32)
        self._key_pos   = np.zeros((self.n_envs, self.cfg.n_colors, 2), dtype=np.int32)
        self._door_pos  = np.zeros((self.n_envs, self.cfg.n_colors, 2), dtype=np.int32)

        for i in range(self.n_envs):
            self._reset_env(i)

    def _reset_env(self, idx: int):
        G   = self.G
        cfg = self.cfg
        rng = self.rng

        # Генерируем карту
        grid = np.zeros((G, G), dtype=np.int32)

        # Стены
        for r in range(G):
            for c in range(G):
                if rng.random() < cfg.wall_density:
                    grid[r, c] = CELL_WALL

        # Позиция агента (не на стене)
        while True:
            ar, ac = rng.integers(0, G, size=2)
            if grid[ar, ac] == CELL_EMPTY:
                break
        self._agent_pos[idx] = [ar, ac]

        # Размещаем ключи и двери
        key_pos  = []
        door_pos = []
        for color in range(cfg.n_colors):
            # Ключ
            while True:
                kr, kc = rng.integers(0, G, size=2)
                if (grid[kr, kc] == CELL_EMPTY
                        and [kr, kc] != [ar, ac]):
                    break
            grid[kr, kc] = CELL_KEY_BASE + color
            key_pos.append([kr, kc])

            # Дверь (по возможности далеко от ключа)
            attempts = 0
            while True:
                dr, dc = rng.integers(0, G, size=2)
                dist = abs(dr - kr) + abs(dc - kc)
                min_dist = cfg.min_key_door_dist or 0
                if (grid[dr, dc] == CELL_EMPTY
                        and [dr, dc] != [ar, ac]
                        and [dr, dc] != [kr, kc]
                        and dist >= min_dist):
                    break
                attempts += 1
                if attempts > 1000:
                    # fallback: игнорируем min_dist
                    while True:
                        dr, dc = rng.integers(0, G, size=2)
                        if grid[dr, dc] == CELL_EMPTY and [dr, dc] != [ar, ac]:
                            break
                    break
            grid[dr, dc] = CELL_DOOR_BASE + color
            door_pos.append([dr, dc])

        self._grids[idx]    = grid
        self._key_pos[idx]  = key_pos
        self._door_pos[idx] = door_pos
        self._inventory[idx] = False
        self._opened[idx]    = False
        self._steps[idx]     = 0
        self._done[idx]      = False

        # sq_gaps = 1.0 (все двери закрыты в начале)
        self._sq_gaps[idx] = 1.0

        # Вычисляем целевое действие (imitation)
        self._targets[idx] = self._optimal_action(idx)

    def _get_obs_token(self, idx: int) -> int:
        """
        Кодирует наблюдение в один токен:
        token = central_cell_type * 2^n_colors + inventory_bitmask
        """
        ar, ac = self._agent_pos[idx]
        central_cell = self._grids[idx, ar, ac]
        # Инвентарь как битовая маска
        inv_mask = 0
        for c in range(self.cfg.n_colors):
            if self._inventory[idx, c]:
                inv_mask |= (1 << c)
        token = int(central_cell) * (2 ** self.cfg.n_colors) + inv_mask
        return min(token, self._n_tokens - 1)  # safety clamp

    def _optimal_action(self, idx: int) -> int:
        """
        BFS до ближайшей цели:
        - Если несём ключ цвета c → иди к двери c
        - Иначе иди к ближайшему ключу, который ещё не подобран
        Возвращает оптимальное действие или ignore_index если нет пути.
        """
        G   = self.G
        grid = self._grids[idx]
        ar, ac = self._agent_pos[idx]
        inv  = self._inventory[idx]
        opened = self._opened[idx]

        # Определяем цель
        targets = []
        for c in range(self.cfg.n_colors):
            if opened[c]:
                continue
            if inv[c]:
                # У нас есть ключ → нужна дверь
                targets.append(tuple(self._door_pos[idx, c]))
            else:
                # Нужен ключ
                targets.append(tuple(self._key_pos[idx, c]))

        if not targets:
            return self.ignore_index

        # BFS
        from collections import deque
        target_set = set(targets)
        queue = deque([(ar, ac, [])])
        visited = {(ar, ac)}
        dr_arr = [-1, 1, 0, 0]
        dc_arr = [0, 0, -1, 1]
        actions = [ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT]

        while queue:
            r, c, path = queue.popleft()
            if (r, c) in target_set:
                return path[0] if path else self.ignore_index
            for action, (dr, dc) in enumerate(zip(dr_arr, dc_arr)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < G and 0 <= nc < G
                        and grid[nr, nc] != CELL_WALL
                        and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc, path + [action]))

        return self.ignore_index

    # ── Step ─────────────────────────────────────────────────────────────────

    def _step_env(self, idx: int, action: int):
        G   = self.G
        ar, ac = self._agent_pos[idx]
        dr_arr = [-1, 1, 0, 0]
        dc_arr = [0, 0, -1, 1]

        dr = dr_arr[action]
        dc = dc_arr[action]
        nr, nc = ar + dr, ac + dc

        reward = self.cfg.reward_step

        # Движение
        if 0 <= nr < G and 0 <= nc < G:
            cell = self._grids[idx, nr, nc]
            if cell != CELL_WALL:
                self._agent_pos[idx] = [nr, nc]
                # Подбираем ключ
                for c in range(self.cfg.n_colors):
                    if cell == CELL_KEY_BASE + c and not self._inventory[idx, c]:
                        self._inventory[idx, c] = True
                        self._grids[idx, nr, nc] = CELL_EMPTY  # ключ исчез
                        break
                # Открываем дверь
                for c in range(self.cfg.n_colors):
                    if (cell == CELL_DOOR_BASE + c
                            and self._inventory[idx, c]
                            and not self._opened[idx, c]):
                        self._opened[idx, c] = True
                        self._inventory[idx, c] = False
                        reward += self.cfg.reward_open_door
                        break

        self._steps[idx] += 1
        n_closed = self.cfg.n_colors - self._opened[idx].sum()
        self._sq_gaps[idx] = float(n_closed) / self.cfg.n_colors

        if (self._opened[idx].all()
                or self._steps[idx] >= self.cfg.max_steps):
            self._done[idx] = True

        # Обновляем целевое действие ПОСЛЕ шага
        self._targets[idx] = self._optimal_action(idx)

    # ── Главный метод API ─────────────────────────────────────────────────────

    def next(self) -> dict:
        """
        Возвращает батч наблюдений и меток.
        Совместим с форматом StoreDistractQueryGenerator.

        Ключи:
          tokens     : [n_envs, 1]  int64
          targets    : [n_envs]     int64 (оптимальное действие или ignore_index)
          reset_mask : [n_envs]     bool  (True = среда только что была сброшена)
          sq_gaps    : [n_envs]     float32 (доля ещё незакрытых дверей)
        """
        reset_mask = self._done.copy()

        # Сбрасываем завершённые среды
        for i in range(self.n_envs):
            if self._done[i]:
                self._reset_env(i)

        # Собираем токены
        tokens = np.array(
            [self._get_obs_token(i) for i in range(self.n_envs)],
            dtype=np.int64,
        ).reshape(-1, 1)

        # Делаем шаг: используем случайные действия на этапе сбора данных
        # (в реальном обучении действия приходят из политики,
        #  но для имитационного обучения просто делаем шаги по optimal policy)
        for i in range(self.n_envs):
            action = self._targets[i]
            if action == self.ignore_index:
                action = self.rng.integers(0, N_ACTIONS)
            self._step_env(i, int(action))

        targets   = self._targets.copy()
        sq_gaps   = self._sq_gaps.copy()

        return {
            "tokens"    : tokens,
            "targets"   : targets,
            "reset_mask": reset_mask.astype(np.float32),
            "sq_gaps"   : sq_gaps,
        }

    def observe(self) -> dict:
        """Return current observation without stepping. Resets done envs first.

        Keys: tokens [n_envs,1], reset_mask [n_envs] float32.
        """
        reset_mask = self._done.copy()
        for i in range(self.n_envs):
            if self._done[i]:
                self._reset_env(i)
        tokens = np.array(
            [self._get_obs_token(i) for i in range(self.n_envs)], dtype=np.int64
        ).reshape(-1, 1)
        return {"tokens": tokens, "reset_mask": reset_mask.astype(np.float32)}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Step all envs with given actions. Returns (rewards [n_envs], dones [n_envs])."""
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        prev_opened = self._opened.sum(axis=1).copy()
        for i in range(self.n_envs):
            r_before = self._opened[i].sum()
            self._step_env(i, int(actions[i]))
            r_after = self._opened[i].sum()
            rewards[i] = (r_after - r_before) * self.cfg.reward_open_door + self.cfg.reward_step
        return rewards, self._done.copy()

    def get_stats(self) -> dict:
        return {
            "mean_opened"  : float(self._opened.mean()),
            "mean_steps"   : float(self._steps.mean()),
            "pct_done"     : float(self._done.mean()),
        }

    def set_metaparams(self, **kwargs):
        """Поддержка curriculum (совместимость с run_sdq2)."""
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)