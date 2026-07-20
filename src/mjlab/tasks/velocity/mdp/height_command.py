"""Height command term for pelvis/root absolute height.

Homework TODOs in this file: 1, 2  (of 10 total)
Index: docs/HOMEWORK_TODO.md · grep: 【作业 TODO
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformBaseHeightCommand(CommandTerm):
  """Uniformly sampled absolute pelvis/root height command above terrain."""

  cfg: UniformBaseHeightCommandCfg

  def __init__(self, cfg: UniformBaseHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["target_height_mean"] = torch.zeros(self.num_envs, device=self.device)

    self._height_slider: viser.GuiSliderHandle | None = None
    self._height_enabled: viser.GuiCheckboxHandle | None = None
    self._height_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    # >>> HOMEWORK_TODO_2_START
    # ==============================================================================
    # 【作业 TODO 2/10】高度跟踪误差指标
    # 位置: height_command.py · UniformBaseHeightCommand._update_metrics
    # 提示: 记录 |指令高度 - 实际高度| 的累积均值，用于训练日志与调试。
    # 概念: metrics["error_height"]、root_link_pos_w[:, 2]
    # 索引: docs/HOMEWORK_TODO.md
    # ==============================================================================
    # 计算目标高度与当前骨盆高度的绝对误差。
    height_error = torch.abs(
      self.height_command[:, 0] - self.robot.data.root_link_pos_w[:, 2]
    )
    # 框架在重置时会把它按环境取平均再清零
    self.metrics["error_height"] += height_error / max_command_step
    # 记录当前目标高度，便于训练日志观察。
    self.metrics["target_height_mean"] = self.height_command[:, 0]

    # --- 实现提示 ---
    # - 从 self.robot.data 读取骨盆世界坐标 z（root_link_pos_w[:, 2]）
    # - 计算 |指令高度 - 实际高度|，累加进 metrics["error_height"]（除以 max_command_step）
    # - 同步更新 metrics["target_height_mean"]
    # <<< HOMEWORK_TODO_2_END

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    # >>> HOMEWORK_TODO_1_START
    # ==============================================================================
    # 【作业 TODO 1/10】高度指令均匀采样
    # 位置: height_command.py · UniformBaseHeightCommand._resample_command
    # 提示: 在 cfg.ranges.height 范围内均匀随机采样新高度指令。
    # 概念: torch.uniform_、UniformBaseHeightCommandCfg.ranges
    # 索引: docs/HOMEWORK_TODO.md
    # ==============================================================================
    # 为本次重采样的环境写入新的高度指令。
    self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
    # --- 实现提示 ---
    # - 使用已创建的 r 张量，对 height_command[env_ids, 0] 做均匀随机采样
    # - 采样上下界来自 self.cfg.ranges.height
    # <<< HOMEWORK_TODO_1_END

  def _update_command(self) -> None:
    pass

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create a height slider in the Viser viewer."""
    height_range = self.cfg.ranges.height

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      slider = server.gui.add_slider(
        "height",
        min=height_range[0],
        max=height_range[1],
        step=0.01,
        initial_value=0.5 * (height_range[0] + height_range[1]),
      )

    self._height_enabled = enabled
    self._height_slider = slider
    self._height_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._height_enabled is not None and self._height_enabled.value:
      assert self._height_get_env_idx is not None
      idx = self._height_get_env_idx()
      assert self._height_slider is not None
      self.height_command[idx, 0] = self._height_slider.value

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw target and actual pelvis height gauges to the robot's right side.

    Each gauge is a sphere at the target/actual height with a vertical stem from
    the ground (z=0) up to the sphere, so the stem length reads as the height.
    The actual gauge is shifted 0.02m to the robot's right (body -y) of the
    target gauge so the two stay distinguishable when heights are close. xy
    follows the robot; z is the absolute target/actual height.
    """
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_mat_ws = matrix_from_quat(self.robot.data.root_link_quat_w).cpu().numpy()
    sphere_radius = 0.01
    # Stem diameter = 2/3 of the sphere diameter; radius scales by the same ratio.
    line_radius = (2.0 / 3.0) * sphere_radius
    side_offset = self.cfg.viz.side_offset
    actual_shift = 0.02  # actual gauge sits this far right of the target gauge

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      target_height = cmds[batch, 0]
      actual_height = base_pos_w[2]

      # Body -y is the robot's right; the offset rotates with the robot. The
      # actual gauge is `actual_shift` further right so it doesn't overlap the
      # target gauge. Only xy follows the robot; z is set per gauge below.
      right_vec = base_mat_ws[batch] @ np.array([0.0, -1.0, 0.0])
      tx = base_pos_w[0] + side_offset * right_vec[0]
      ty = base_pos_w[1] + side_offset * right_vec[1]
      ax = base_pos_w[0] + (side_offset + actual_shift) * right_vec[0]
      ay = base_pos_w[1] + (side_offset + actual_shift) * right_vec[1]

      target_center = np.array([tx, ty, target_height], dtype=np.float64)
      actual_center = np.array([ax, ay, actual_height], dtype=np.float64)
      target_ground = np.array([tx, ty, 0.0], dtype=np.float64)
      actual_ground = np.array([ax, ay, 0.0], dtype=np.float64)

      # Target gauge: sphere at target height + vertical stem from ground up to it.
      visualizer.add_sphere(
        center=target_center,
        radius=sphere_radius,
        color=self.cfg.viz.target_color,
        label="base_height_target",
      )
      visualizer.add_cylinder(
        start=target_ground,
        end=target_center,
        radius=line_radius,
        color=self.cfg.viz.target_color,
        label="base_height_target_stem",
      )

      # Actual gauge: sphere at real pelvis height + stem, shifted right.
      visualizer.add_sphere(
        center=actual_center,
        radius=sphere_radius,
        color=self.cfg.viz.actual_color,
        label="base_height_actual",
      )
      visualizer.add_cylinder(
        start=actual_ground,
        end=actual_center,
        radius=line_radius,
        color=self.cfg.viz.actual_color,
        label="base_height_actual_stem",
      )


@dataclass(kw_only=True)
class UniformBaseHeightCommandCfg(CommandTermCfg):
  entity_name: str

  @dataclass
  class Ranges:
    height: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (0.60, 0.20, 0.85, 0.90)
    """Target height marker color (purple)."""
    actual_color: tuple[float, float, float, float] = (1.00, 0.40, 0.70, 0.90)
    """Actual pelvis height marker color (pink)."""
    side_offset: float = 0.4
    """Lateral offset (m) along the robot's body -y (right) for height markers."""

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformBaseHeightCommand:
    return UniformBaseHeightCommand(self, env)
