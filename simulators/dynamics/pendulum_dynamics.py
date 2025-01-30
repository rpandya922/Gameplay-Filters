from typing import Tuple, Any, Optional
import numpy as np
from jaxlib.xla_extension import DeviceArray

from .base_dynamics import BaseDynamics
from .base_dstb_dynamics import BaseDstbDynamics

class PendulumDynamics(BaseDstbDynamics):
  def __init__(self, cfg : Any, action_space : np.ndarray):
    super().__init__(cfg, action_space)
    self.dim_x = 2
    self.max_speed = 3
    self.g = 9.81
    self.m = 1.0
    self.l = 1.0

    discretize_actions = bool(cfg.discretize_actions)
    if discretize_actions:
      n_discrete = int(cfg.n_discrete_actions)
      self.ctrl_space = np.linspace(-2.0, 2.0, n_discrete).reshape(-1, 1)
      # TODO: finish implementing
    self.gui = bool(cfg.gui)

    if self.gui:
      import matplotlib.pyplot as plt
      from matplotlib.patches import Wedge
      self.plt = plt
      self.Wedge = Wedge
      plt.figure(figsize=(12,6))
      self.pend_ax = plt.subplot(121)
      self.margin_ax = plt.subplot(222)
      self.ctrl_ax = plt.subplot(224)

      # data saving
      self.traj_data = {"state": [], "ctrl": [], "dstb": [], "l_x": [], "l_g": []}

  def reset(self):
    # sample until outside the target set
    while True:
      state = np.random.uniform(low=[-np.pi, -self.max_speed], high=[np.pi, self.max_speed])
      if min(self.get_target_margin(state).values()) < 0:
        break
    # state = np.array([0.2, -0.05])

    self.state = state

    if self.gui:
      self.pend_ax.cla()
      # compute pendulum start and end points
      start = np.array([0, 0])
      end = np.array([self.l * np.cos(state[0] + np.pi/2), self.l * np.sin(state[0] + np.pi/2)])
      self.pend_ax.plot([start[0], end[0]], [start[1], end[1]], 'r')
      self.pend_ax.set_xlim(-1.5, 1.5)
      self.pend_ax.set_ylim(-1.5, 1.5)
      self.pend_ax.set_aspect('equal')
      plt = self.plt
      plt.pause(0.001)

      # data saving
      self.traj_data = {"state": [self.state], "ctrl": [], "dstb": [], "l_x": [], "g_x": []}

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

    if self.gui:
      Wedge = self.Wedge
      # data saving
      self.traj_data["state"].append(self.state)
      self.traj_data["ctrl"].append(ctrl_clip)
      self.traj_data["dstb"].append(dstb_clip)
      self.traj_data["l_x"].append(min(self.get_target_margin().values()))
      self.traj_data["g_x"].append(min(self.get_safety_margin().values()))

      self.pend_ax.cla()
      # plotting target set as Wedge
      wedge = Wedge((0,0), 1.5, 80, 100, facecolor='green', alpha=0.3)
      self.pend_ax.add_patch(wedge)

      # compute pendulum start and end points
      start = np.array([0, 0])
      end = np.array([self.l * np.cos(state_next[0] + np.pi/2), self.l * np.sin(state_next[0] + np.pi/2)])
      self.pend_ax.plot([start[0], end[0]], [start[1], end[1]], 'r')

      self.pend_ax.set_xlim(-1.5, 1.5)
      self.pend_ax.set_ylim(-1.5, 1.5)
      self.pend_ax.set_aspect('equal')

      # plotting target margin function
      self.margin_ax.cla()
      self.margin_ax.plot(self.traj_data["l_x"])
      self.margin_ax.set_title("Target Margin")
      zeros = np.zeros(150)
      self.margin_ax.plot(zeros, 'k--')
      self.margin_ax.set_ylim(-3, 0.5)
      self.margin_ax.set_xlim(0, 150)

      # plotting state
      states = np.array(self.traj_data["state"])
      self.ctrl_ax.cla()
      self.ctrl_ax.plot(states[:,0], label="theta")
      self.ctrl_ax.plot(states[:,1], label="theta_dot")
      self.ctrl_ax.plot(zeros, 'k--')
      self.ctrl_ax.legend()
      self.ctrl_ax.set_title("State")
      self.ctrl_ax.set_xlim(0, 150)

      # plotting control inputs
      # self.ctrl_ax.cla()
      # ctrl = np.array(self.traj_data["ctrl"])
      # dstb = np.array(self.traj_data["dstb"])
      # self.ctrl_ax.plot(ctrl, label="ctrl")
      # self.ctrl_ax.plot(dstb, label="dstb")
      # self.ctrl_ax.plot(zeros, 'k--')
      # self.ctrl_ax.legend()
      # self.ctrl_ax.set_title("Control Inputs")
      # self.ctrl_ax.set_xlim(0, 150)

      plt = self.plt
      plt.pause(0.001)

    return state_next, ctrl_clip, dstb_clip
  
  def angle_normalize(self, x : np.ndarray):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)
  
  def distance_function(self, x, neg=1, pos=1):
    """
    Taken from Belief Games repo
        if x<0:
            y = neg*x
        else:
            y = pos*atan(x)
    """
    x_pos_mask = x >= 0
    x_neg_mask = np.logical_not(x_pos_mask)
    
    y_pos = pos*np.arctan(x)*x_pos_mask
    y_neg = neg*x*x_neg_mask
    
    return y_pos + y_neg

  def get_target_margin(self, state : np.ndarray = None):
    if state is None:
      state = self.state
    # positive -> inside the target set
    theta_margin = 0.174533 # 10 degrees
    speed_margin = 0.5

    theta = self.angle_normalize(state[0])
    theta_l = theta_margin - np.abs(theta)
    theta_l = -self.distance_function(-theta_l, neg=10, pos=1)

    speed_l = speed_margin - np.abs(state[1])
    speed_l = -self.distance_function(-speed_l, neg=5, pos=1)

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