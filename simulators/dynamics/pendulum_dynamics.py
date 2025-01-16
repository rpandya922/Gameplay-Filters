from typing import Tuple, Any, Optional
import numpy as np
from jaxlib.xla_extension import DeviceArray

from .base_dynamics import BaseDynamics
from .base_dstb_dynamics import BaseDstbDynamics

class PendulumDynamics(BaseDstbDynamics):
  def __init__(self, cfg : Any, action_space : np.ndarray):
    super().__init__(cfg, action_space)
    self.dim_x = 2
    self.max_speed = 8
    self.g = 9.81
    self.m = 1.0
    self.l = 1.0

    discretize_actions = bool(cfg.discretize_actions)
    if discretize_actions:
      n_discrete = int(cfg.n_discrete_actions)
      self.ctrl_space = np.linspace(-2.0, 2.0, n_discrete).reshape(-1, 1)
      # TODO: finish implementing

  def reset(self):
    # sample until outside the target set
    while True:
      state = np.random.uniform(low=[-np.pi, -self.max_speed], high=[np.pi, self.max_speed])
      if min(self.get_target_margin(state).values()) < 0:
        break

    self.state = state
    return state

  def cont_deriv(self, state : np.ndarray, control : np.ndarray, disturbance : np.ndarray) -> np.ndarray:
    """
    Computes the continuous-time dynamics of the pendulum.

    Args:
        state (np.ndarray): [theta, theta_dot]
        control (np.ndarray): control torque
        disturbance (np.ndarray): disturbance torque input

    Returns:
        np.ndarray: [theta_dot, theta_ddot].
    """

    # clip control and disturbance
    ctrl_clip = np.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])
    dstb_clip = np.clip(disturbance, self.dstb_space[:, 0], self.dstb_space[:, 1])

    theta, theta_dot = self.state
    theta_ddot = (
        -3 * self.g / (2 * self.l) * np.sin(theta + np.pi)
        + 3.0 / (self.m * self.l ** 2) * (ctrl_clip + dstb_clip).item()
    )

    return np.array([theta_dot, theta_ddot])
  
  def integrate_forward(self, state: np.ndarray, control: np.ndarray, noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif', adversary: Optional[np.ndarray] = None,
      transform_mtx: Optional[np.ndarray] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    self, state: np.ndarray, control: np.ndarray, num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif', adversary: Optional[np.ndarray] = None, **kwargs
    Computes the next state of the pendulum.

    Args:
        state (np.ndarray): [theta, theta_dot].
        control (np.ndarray): control torque.
        disturbance (np.ndarray): disturbance torque input.

    Returns:
        np.ndarray: [theta, theta_dot].
    """

    # clip control and disturbance
    ctrl_clip = np.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])
    dstb_clip = np.clip(adversary, self.dstb_space[:, 0], self.dstb_space[:, 1])

    # integrate with RK4
    k1 = self.cont_deriv(self.state, ctrl_clip, dstb_clip)
    k2 = self.cont_deriv(self.state + k1*self.dt/2, ctrl_clip, dstb_clip)
    k3 = self.cont_deriv(self.state + k2*self.dt/2, ctrl_clip, dstb_clip)
    k4 = self.cont_deriv(self.state + k3*self.dt, ctrl_clip, dstb_clip)
    state_next = self.state + (k1 + 2*k2 + 2*k3 + k4) * self.dt / 6

    # clip speed
    state_next[1] = np.clip(state_next[1], -self.max_speed, self.max_speed)

    self.state = state_next

    return state_next, ctrl_clip, dstb_clip
  
  def angle_normalize(self, x : np.ndarray):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

  def get_target_margin(self, state : np.ndarray = None):
    if state is None:
      state = self.state
    # positive -> inside the target set
    theta_margin = 0.174533 # 10 degrees
    speed_margin = 0.5

    theta = self.angle_normalize(state[0])
    theta_l = theta_margin - np.abs(theta)

    speed_l = speed_margin - np.abs(state[1])

    # l_x = min(l_x, speed_l)
    # if l_x > speed_l:
    #   l_x = speed_l

    return {"theta": theta_l, "speed": speed_l}
  
  def get_safety_margin(self, state : np.ndarray = None):
    # NOTE: for now, only using a target set (easier to specify margin fn for pendulum)
    # negative -> inside unsafe set
    return {"theta_g": np.inf}
  
  def integrate_forward_jax(self, state: DeviceArray, control: DeviceArray) -> Tuple[DeviceArray, DeviceArray]:
    return super().integrate_forward_jax(state, control)

  def _integrate_forward(self, state: DeviceArray, control: DeviceArray) -> DeviceArray:
    return super()._integrate_forward(state, control)