# coding: utf8
# fix for pyinstaller
import FileDialog
import mpl_toolkits
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import art3d
import scipy.special._ufuncs_cxx
import numpy
# end of fix

import csv
import scipy.linalg

from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint

from sympy import symbols, Dummy, lambdify
from sympy.physics.mechanics import *

from numpy import matrix, dot, rank, asarray, array
from numpy import hstack, zeros, linspace, pi, ones
from numpy import zeros, cos, sin, arange, around
from numpy.linalg import solve, matrix_rank

from matplotlib import pyplot as plt, animation
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import cm

import json
# font = FontProperties(fname=r"WenQuanYiMicroHeiMono.ttf", size=14)

# 假设
# 所有臂长度相同且质量为 0
# 所有摆质量相同

def chunker(seq = [], size = 0):
  return (seq[pos:pos + size]
    for pos in xrange(0, len(seq), size))

class Shape(object):

  def __init__(self, ax, x = 0., y = 0., z = 0., a = 1., **kwargs) :

    self.ax = ax
    self.position = [x, y, z, a]
    self.surfaces = []
    self.current_position = [x, y, z, a]

  @property
  def x(self):
    return self.position[0]

  @x.setter
  def x(self, value):
    self.position[0] = value

  @property
  def y(self):
    return self.position[1]

  @y.setter
  def y(self, value):
    self.position[1] = value

  @property
  def z(self):
    return self.position[2]

  @z.setter
  def z(self, value):
    self.position[2] = value

  @property
  def a(self):
    return self.position[3]

  @a.setter
  def a(self, value):
    self.position[3] = value

  @property
  def alpha(self):
    return self.position[3]

  @alpha.setter
  def alpha(self, value):
    self.position[3] = value

  _dimension_dict = {'x': 0, 'y': 1, 'z': 2, 'a': 3, 'alpha': 3}

  def _modify_dimension(self, new_value, dimension = 0) :

    if dimension not in [0, 1, 2, 3] :
      dimension = Shape._dimension_dict[dimension.lower()]

    diff = new_value - self.position[dimension]

    for surface in self.surfaces :
      for i, __ in enumerate(surface._vec[dimension]) :
        surface._vec[dimension][i] += diff

    self.position[dimension] = new_value

  def modify_x(self, new_x) :
    self._modify_dimension(new_x, dimension = 0)

  def modify_y(self, new_y) :
    self._modify_dimension(new_y, dimension = 1)

  def modify_z(self, new_z) :
    self._modify_dimension(new_z, dimension = 2)

  def modify_alpha(self, new_alpha) :
    self._modify_dimension(new_alpha, dimension = 3)

  def modify_position(self, *position) :
    self.modify_x(position[0])
    self.modify_y(position[1])
    self.modify_z(position[2])

class Cube(Shape):

  def __init__(self, ax, x = 0, y = 0, z = 0,
    height = 1., width = 1., depth = 1., **kwargs) :

    super(Cube, self).__init__(ax, x, y, z, **kwargs)

    self.initiate(ax, x, y, z, height, width, depth, **kwargs)

  def set_size(self, height = None, width = None, depth = None, update = False) :

    self.height = height or self.height
    self.width = width or self.width
    self.depth = depth or self.depth

    if update : # To do: update size using _vec

      self.front_bot_left  = [x,         y,          z]
      self.front_bot_right = [x + width, y,          z]
      self.front_top_left  = [x,         y + height, z]
      self.front_top_right = [x + width, y + height, z]

      self.rear_bot_left   = [x,         y,          z + depth]
      self.rear_bot_right  = [x + width, y,          z + depth]
      self.rear_top_left   = [x,         y + height, z + depth]
      self.rear_top_right  = [x + width, y + height, z + depth]

  def change_size(self, height = None, width = None, depth = None) :
    raise NotImplementedError('Change size is not implemented for Cube.')

  def create_Ploy3DCollection(self, *two_lines, **kwargs) :

    x, y, z = zip(two_lines[0], two_lines[1], two_lines[2], two_lines[3])

    X = [ __ for __ in chunker(x, 2) ]
    Y = [ __ for __ in chunker(y, 2) ]
    Z = [ __ for __ in chunker(z, 2) ]

    return self.ax.plot_surface(X, Y, Z, **kwargs)

  def initiate(self, ax, x = 0, y = 0, z = 0,
    height = 1., width = 1., depth = 1., **kwargs) :

    self.ax = ax
    self.x, self.y, self.z = x, y, z
    self.set_size(height, width, depth)

    if 'initiate points' :

      self.points = [ numpy.zeros(3, dtype = 'float') for __ in xrange(8) ]

      self.front_bot_left  = self.points[0]
      self.front_bot_right = self.points[1]
      self.front_top_left  = self.points[2]
      self.front_top_right = self.points[3]

      self.rear_bot_left   = self.points[4]
      self.rear_bot_right  = self.points[5]
      self.rear_top_left   = self.points[6]
      self.rear_top_right  = self.points[7]

      self.front_bot_left  = [x,         y,          z]
      self.front_bot_right = [x + width, y,          z]
      self.front_top_left  = [x,         y + height, z]
      self.front_top_right = [x + width, y + height, z]

      self.rear_bot_left   = [x,         y,          z + depth]
      self.rear_bot_right  = [x + width, y,          z + depth]
      self.rear_top_left   = [x,         y + height, z + depth]
      self.rear_top_right  = [x + width, y + height, z + depth]

    if 'initiate surfaces' :

      self.top = self.create_Ploy3DCollection(
        self.front_top_left, self.front_top_right,
        self.rear_top_left,  self.rear_top_right, **kwargs)

      self.bot = self.create_Ploy3DCollection(
        self.front_bot_left, self.front_bot_right,
        self.rear_bot_left,  self.rear_bot_right, **kwargs)

      self.front = self.create_Ploy3DCollection(
        self.front_bot_left, self.front_bot_right,
        self.front_top_left, self.front_top_right, **kwargs)

      self.rear = self.create_Ploy3DCollection(
        self.rear_top_left, self.rear_top_right,
        self.rear_bot_left, self.rear_bot_right, **kwargs)

      self.left = self.create_Ploy3DCollection(
        self.front_bot_left, self.front_top_left,
        self.rear_bot_left, self.rear_top_left, **kwargs)

      self.right = self.create_Ploy3DCollection(
        self.front_bot_right, self.front_top_right,
        self.rear_bot_right, self.rear_top_right, **kwargs)

    self.surfaces = [ self.top,   self.bot,
                     self.front, self.rear,
                     self.left,  self.right ]

class Sphere(Shape):

  def __init__(self, ax, x = 0, y = 0, z = 0,
    radius = 1., detail_level = 16, rstride = 1, cstride = 1, **kwargs) :

    super(Sphere, self).__init__(ax, x, y, z, **kwargs)

    self.initiate(ax, x, y, z, radius, detail_level, rstride, cstride, **kwargs)

  def set_size(self, radius = None, update = False) :

    self.radius = radius or self.radius

    if update :
      self.change_size(radius)

  def change_size(self, radius) :

    current_radius = self.radius
    for i, center_value in enumerate(self.position[:3]) :
      for j, value in enumerate(self.surfaces[0]._vec[i]) :
        self.surfaces[0]._vec[i][j] = center_value + \
          (value - center_value) * float(radius) / current_radius

  def create_Ploy3DCollection(self, **kwargs) :

    u = numpy.linspace(0, 2 * numpy.pi, self.detail_level)
    v = numpy.linspace(0, numpy.pi, self.detail_level)

    X = self.x + \
        self.radius * numpy.outer(numpy.cos(u), numpy.sin(v))

    Y = self.y + \
        self.radius * numpy.outer(numpy.sin(u), numpy.sin(v))

    Z = self.z + \
        self.radius * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))

    return self.ax.plot_surface(X, Y, Z,
      rstride = self.rstride, cstride = self.cstride, **kwargs)

  def initiate(self, ax, x = 0, y = 0, z = 0, radius = 1., detail_level = 16,
    rstride = 1, cstride = 1, **kwargs) :

    self.ax = ax
    self.x, self.y, self.z = x, y, z
    self.detail_level = detail_level
    self.rstride = rstride
    self.cstride = cstride
    self.set_size(radius)

    self.surface = self.create_Ploy3DCollection(**kwargs)
    self.surfaces = [ self.surface ]

def inverted_pendulum (
  N = 2,
  total_arm_length = 1.,
  cart_mass = 0.01,
  bob_mass = 0.01,
  g_value = 9.81,
  simulation_time = 10,
  simulation_step = 0.01,
  Q_value_input = [1,1,1,1,1,1],
  R_input = 1,
  AngleInput = [pi / 2.2, pi /1.7, pi / 2.2, pi /1.7, pi / 2.2],
  HPosition = 1,
  animation_filename = 'inverted_pendulum.avi',
  draw = False, animate = False, save_csv = False):

  # print N, 'inverted pendulum'
  # print

  arm_length = total_arm_length/N

  # LQR 线性二次型调节器
  def lqr(A, B, Q, R):

    """线性二次型调节器
       基于 scipy.linalg.solve_continuous_are
       兼容 control.lqr
    """

    r = matrix(R)

    # Riccati 方程 X
    X = matrix(scipy.linalg.solve_continuous_are(A, B, Q, r))

    # 计算 LQR 系数
    # K = R^{-1} B^T X
    # 加入 asarray 以兼容 control.lqr
    K = asarray(scipy.linalg.inv(r)*(B.T*X))

    # 求特征向量
    # 兼容 control.lqr
    E, __ = scipy.linalg.eig(A-B*K)

    return K, X, E

  # 创建动画
  def create_animation(t, states, length, filename = None,
    cart_width = 0.4, cart_height = 0.1,
    cart_color = 'red', cart_edge_color = 'black',
    bob_color = 'black', arm_color = 'blue' ):

    """创建系统动画

      必要参数:

      t : 时间数组 shape(m)
      states: 状态矩阵 shape(m, q)
      length: 摆臂长度

      可选参数:

      filename: 可选文件名
      cart_width: 小车宽度
      cart_height: 小车高度

      返回值:

      fig : matplotlib.Figure
      anim : matplotlib.FuncAnimation

    """

    # 摆头的数目 + 1 小车
    n = states.shape[1] / 2

    # 创建新图
    fig = plt.figure(figsize=(8,6), dpi=200)
    fig.set_size_inches(8, 6, forward=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # 图宽度
    x_min = around(states[:, 0].min() - cart_width / 2.0, 1)
    x_max = around(states[:, 0].max() + cart_width / 2.0, 1)

    x_min = min(-0.5, x_min)
    x_max = max(x_max, 0.5)

    # 创建坐标系
    ax = plt.axes(
      xlim = (x_min, x_max),
      ylim = (0, total_arm_length * 1.1),
      # ylim = (-arm_length * 2.2, arm_length * 2.2),
      zlim = (-0.2, 0.2),
      projection = '3d')
    ax.view_init(elev = -80., azim = 90)
    ax.set_zticks([])

    cart_width = (x_max - x_min) / 6
    cart_height = total_arm_length / 16

    # 时间
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    time_label = ax.text(right, bottom, 0, '', zdir = 'x')

      # horizontalalignment='left',
      # verticalalignment='bottom',
      # transform=ax.transAxes)

    levels = 4
    scale = 0.8
    line_width = 8.

    def color_map (level, max_levels, scale = 0.8) :
      value = scale * (max_levels - level) / max_levels
      return [value, value, value]

    each_line_width = line_width / levels
    each_line_width_point = each_line_width * 0.01/ 2.5

    # 加入小车
    cart = Cube(ax, x = - cart_width / 2, y = - cart_height / 2,
      z = - cart_height / 2,
      height = cart_height, width = cart_width, depth = cart_height,
      color = 'red')

    for surface in cart.surfaces :
      ax.add_collection3d(surface)

    # 摆
    bobs = [ Sphere(ax, x = 0, y = 0, z = 0, lw = 0.1, cmap = cm.binary,
      zorder = 101, alpha = 1,
      radius = cart_width / 6 , color = 'black') for i in xrange(n-1) ]

    arm_3d = [ ax.plot([], [], [], lw = each_line_width, zorder = 10, color = color_map(lvl, levels) )[0]
               for lvl in xrange(levels) ]
    # 初始化
    def init():
      # arm.set_data([], [0.1])
      for arm in arm_3d :
        arm.set_3d_properties(zs=-0.05, zdir=u'z')

    # animation function: update the objects
    def animate(i):

      time_label.set_text(u'{:2.3f} sec.'.format(t[i]))
      new_value = states[i, 0] - cart_width / 2
      cart.modify_x(new_value)

      x = hstack((states[i, 0], zeros((n - 1))))
      y = zeros((n))

      for j in arange(1, n):
        x[j] = x[j - 1] + length * cos(states[i, j])
        y[j] = y[j - 1] + length * sin(states[i, j])

      ya = [i for i in y]
      ya[0]  += cart_height
      ya[-1] -= cart_height/4

      for lvl in xrange(levels) :
        arm_3d[lvl].set_data(x + each_line_width_point*(-float(levels)/2 + float(lvl)) + cart_height/8, ya)
      for arm in arm_3d :
        arm.set_3d_properties(zs=-0.05, zdir=u'z')

      for i in xrange(n-1) :
        bobs[i].modify_x(x[i+1])
        bobs[i].modify_y(y[i+1])

    # call the animator function
    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
            interval=t[-1] / len(t) * 1000, blit=True, repeat=False)

    # save the animation if a filename is given
    if filename is not None:
      anim.save(filename, writer='ffmpeg', fps=int(1.0/simulation_step),
        bitrate = 100000)

  # 求 t 时 x 的导数值
  def compute_derivative(x, t, constants):

    """dx/dt at t
    """

    # 反馈
    v = dot(K, equilibrium_point - x)

    # 状态, 输入, 参数
    arguments = hstack((x, v, constants))

    # 求导
    dx = array(solve(M_func(*arguments),
      F_func(*arguments))).T[0]

    return dx

  if '建立模型' :

    # q, v, f = 位置, 速度, 小车受力
    q, v, f = map(dynamicsymbols, ['q:'+str(N+1), 'v:'+str(N+1), 'f'])

    # m, l, g, t = 摆的质量, 杆的长度, 重力, 时间
    m, l, g, t = map(symbols, ['m:'+str(N+1), 'l:'+str(N), 'g', 't'])

    # 惯性参照系
    I = ReferenceFrame('I')

    # 原点
    O = Point('O')

    # 原点速度为 0
    O.set_vel(I, 0)

    if '建立摆的运动模型' :

      # 顶端摆
      P0 = Point('P0')
      P0.set_pos(O, q[0] * I.x) # 位置
      P0.set_vel(I, v[0] * I.x) # 速度
      Pa0 = Particle('Pa0', P0, m[0]) # 摆的重心

      # 惯性参考系
      frames = [I]
      # 点
      points = [P0]
      # 受力点
      particles = [Pa0]
      # 受力, 包括小车受力 f
      forces = [(P0, f * I.x - m[0] * g * I.y)]
      # 运动方程 (dq / dt) - v = 0
      kindiffs = [q[0].diff(t) - v[0]]

      for i in xrange(N):
        # 创建新参考系
        Bi = I.orientnew('B' + str(i), 'Axis', [q[i + 1], I.z])
        # 设置角速度
        Bi.set_ang_vel(I, v[i + 1] * I.z)
        # 添加参考系
        frames.append(Bi)

        # 创建新点
        Pi = points[-1].locatenew('P' + str(i + 1), l[i] * Bi.x)
        # 设置速度
        Pi.v2pt_theory(points[-1], I, Bi)
        # 添加点
        points.append(Pi)

        # 创建新受力点
        Pai = Particle('Pa' + str(i + 1), Pi, m[i + 1])
        # 添加受力点
        particles.append(Pai)

        # 添加受力
        forces.append((Pi, -m[i + 1] * g * I.y))

        # 添加运动方程 (dq / dt) - v = 0
        kindiffs.append(q[i + 1].diff(t) - v[i + 1])

      # 新建运动模型
      kane = KanesMethod(I, q_ind=q, u_ind=v, kd_eqs=kindiffs)

      # 创建运动方程组: fr, frstar
      fr, frstar = kane.kanes_equations(forces, particles)

    if '建立臂的运动模型' :

      # 创建参数和参数值相关数组
      # 参数, 参数值, 臂长参数, 臂长参数值
      # 加入重力 g 和重力值 g_value
      parameters, parameter_vals, l_name, l_value, m_value = [], [], [g], [g_value], [cart_mass]

      # 臂长参数
      l_name += l
      m_value += [ bob_mass for __ in xrange(N) ]
      # 臂长参数值
      l_value += [ arm_length for __ in l ]

      # 给参数和参数值赋值
      for length, mass, length_value, b_mass in zip(l_name, m, l_value, m_value):
        parameters += [ length, mass ]
        parameter_vals += [length_value, b_mass]

      # 创建建动态列表: [坐标, 速度, 受力]
      dynamic = q + v + [f]

      # 创建状态标识
      dummy_symbols = [ Dummy() for __ in dynamic ]

      # 创建状态字典
      dummy_dict = dict(zip(dynamic, dummy_symbols))

      # 求解运动微分方程组
      kindiff_dict = kane.kindiffdict()

      # 带入质量
      M = kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)

      # 带入受力
      F = kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)

      # 创建质量函数
      M_func = lambdify(dummy_symbols + parameters, M)

      # 创建受力函数
      F_func = lambdify(dummy_symbols + parameters, F)

    if '创建控制系统' :

      # 平衡点
      equilibrium_point = hstack(( 0, pi / 2 * ones(len(q) - 1), zeros(len(v)) ))

      # 平衡点字典
      equilibrium_dict = dict(zip(q + v, equilibrium_point))

      # 参数字典
      parameter_dict = dict(zip(parameters, parameter_vals))

      # 生成线性方程组
      linear_state_matrix, linear_input_matrix, inputs = kane.linearize()

      # 带入平衡点
      # 状态
      f_A_lin = linear_state_matrix.subs(parameter_dict).subs(equilibrium_dict)
      # 反馈
      f_B_lin = linear_input_matrix.subs(parameter_dict).subs(equilibrium_dict)
      # 质量
      m_mat = kane.mass_matrix_full.subs(parameter_dict).subs(equilibrium_dict)

      # 计算 LQR 中的 A 和 B
      # dx/dt = Ax + Bu
      # u = Fx 是反馈控制
      A = matrix(m_mat.inv() * f_A_lin)
      B = matrix(m_mat.inv() * f_B_lin)

      # Q 输入 N = 2 => Q.shape = (6, 6)
      if len(q_value_input) == A.shape[0] :
        Q_input = zeros((A.shape[0],A.shape[0]))
        for i in xrange(A.shape[0]):
          Q_input[i, i] = q_value_input[i]
      else :
        # print 'Warning: the shape of Q should be {0}, {0}'.format(A.shape[0])
        # print 'Default value Q = ones( {0}, {0} ) is used.'.format(A.shape[0])
        # print
        Q_input = ones(A.shape)

      K, __, __ = lqr(A, B, Q_input, R_input);

      print json.dumps(dict((i, round(k,2))  for i, k in enumerate(K[0])))
      # print

      if len(AngleInput) != N :
        # print 'Warning: the length of angle_input should be', N
        # print 'Default value pi/2 * ones(N) is used.'
        # print
        AngleInput = pi / 2 * ones(N)

      # 初始状态 q v
      # x0 = hstack(( 0, pi / 2 * ones(len(q) - 1), 1 * ones(len(v)) ))
      x0 = hstack(( HPosition, array(AngleInput), 1 * ones(len(v)) ))

      # 时间向量
      t = linspace(0, simulation_time, int(simulation_time/simulation_step))

      # 系统
      y = odeint(compute_derivative, x0, t, args=(parameter_vals,))

  if draw :

    # 水平 = sin(角度) * 角速度 * L

    def angle(cn, y, ct) :
      return y[:, 1: y.shape[1] / 2][ct][cn]

    def cart_position(y, ct) :
      return y[:, 0][ct]

    def angle_velocity(cn, y, ct) :
      return y[:, (y.shape[1] / 2 + 1):][ct][cn]

    def velocity_x(cn, r, y, ct) :
      return sin(angle(cn,y,ct)) * angle_velocity(cn,y,ct) * r

    def h_position(a, r) :
      return

    if '水平位置' :

      # 新建绘图
      plt.figure(u'horizontal position')
      cart_positions = [ cart_position(y, ct) for ct in xrange(int(simulation_time/simulation_step)) ]
      hp = [[] for __ in xrange(N+1)]
      hp[0] = cart_positions

      for i in xrange(1, N+1) :
        hp[i] = array([ hp[i-1][ct] + cos(angle(i-1, y, ct)) * arm_length
                  for ct in xrange(int(simulation_time/simulation_step)) ])
        plt.plot(t, hp[i])

      # 速度曲线

      # 标签
      v_x_x_label = plt.xlabel(u'time (s)') #, fontproperties=font)
      v_x_y_label = plt.ylabel(u'position (s)') #, fontproperties=font)
      q_leg = plt.legend(xrange(N))

      plt.savefig('position.png', bbox_inches='tight')

    if '水平速度' :
      # 新建绘图
      plt.figure(u'horizontal speed')
      # 速度曲线
      v_x_lines = plt.plot(t, [[ velocity_x(b, arm_length, y, ct) for b in xrange(N) ]
        for ct in xrange(int(simulation_time/simulation_step))])
      # 标签
      v_x_x_label = plt.xlabel(u'time (s)') #, fontproperties=font)
      v_x_y_label = plt.ylabel(u'speed (m/s)') #, fontproperties=font)
      q_leg = plt.legend(xrange(N))

      plt.savefig('horizontal_velocity.png', bbox_inches='tight')

    if '角度' :

      # 新建绘图
      plt.figure(u'angle')
      # 位置曲线
      q_lines = plt.plot(t, y[:, 1:y.shape[1]/2])
      # 标签
      q_x_label = plt.xlabel(u'time (s)') # , fontproperties=font)
      q_y_label = plt.ylabel(u'rad') #, fontproperties=font)
      # 图例
      # q_leg = plt.legend(dynamic[1:y.shape[1] / 2])
      leg = plt.legend(xrange(N))
      # 保存
      plt.savefig('angle.png', bbox_inches='tight')

    if '线速度' :
      # 新建绘图
      plt.figure(u'speed')
      # 速度曲线
      lines = plt.plot(t, y[:, (y.shape[1] / 2 + 1):])
      # 标签
      x_label = plt.xlabel(u'time (s)') #, fontproperties=font)
      y_label = plt.ylabel(u'speed (m/s)') #, fontproperties=font)
      # 图例
      # leg = plt.legend(dynamic[y.shape[1] / 2:])
      leg = plt.legend(xrange(N))
      # 保存
      plt.savefig('speed.png', bbox_inches='tight')

  if save_csv :

    # 水平 = sin(角度) * 角速度 * L

    def angle(cn, y, ct) :
      return y[:, 1: y.shape[1] / 2][ct][cn]

    def cart_position(y, ct) :
      return y[:, 0][ct]

    def angle_velocity(cn, y, ct) :
      return y[:, (y.shape[1] / 2 + 1):][ct][cn]

    def velocity_x(cn, r, y, ct) :
      return sin(angle(cn,y,ct)) * angle_velocity(cn,y,ct) * r

    def h_position(a, r) :
      return

    if '水平位置' :

      cart_positions = [ cart_position(y, ct) for ct in xrange(int(simulation_time/simulation_step)) ]
      hp = [[] for __ in xrange(N+1)]
      hp[0] = cart_positions

      for i in xrange(1, N+1) :
        hp[i] = array([ hp[i-1][ct] + cos(angle(i-1, y, ct)) * arm_length
                  for ct in xrange(int(simulation_time/simulation_step)) ])

      with open('position.csv', 'wb') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(['time(s)', 'bob_0(m)', 'bob_1(m)'])
        for i, __ in enumerate(t) :
          row = [t[i]]
          for j in xrange(N) :
            row.append(hp[j][i])
          csv_writer.writerow(row)


    if '水平速度' :

      v_x = [ [ velocity_x(b, arm_length, y, ct) for b in xrange(N) ]
                for ct in xrange(int(simulation_time/simulation_step)) ]

      with open('horizontal_velocity.csv', 'wb') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(['time(s)', 'bob_0(m/s)', 'bob_1(m/s)'])
        for i, __ in enumerate(t) :
          row = [t[i]]
          for j in xrange(N) :
            row.append(v_x[i][j])
          csv_writer.writerow(row)


    if '角度' :

      q_lines = y[:, 1:y.shape[1]/2]

      with open('angle.csv', 'wb') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(['time(s)', 'bob_0(rad)', 'bob_1(rad)'])
        for i, __ in enumerate(t) :
          row = [t[i]]
          for j in xrange(N) :
            row.append(q_lines[i][j])
          csv_writer.writerow(row)

    if '线速度' :

      lines = y[:, (y.shape[1] / 2 + 1):]

      with open('speed.csv', 'wb') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(['time(s)', 'bob_0(m/s)', 'bob_1(m/s)'])
        for i, __ in enumerate(t) :
          row = [t[i]]
          for j in xrange(N) :
            row.append(lines[i][j])
          csv_writer.writerow(row)

  if animate :

    # 创建动画
    create_animation(t, y, arm_length, filename = animation_filename)

if __name__ == '__main__':

  if '默认值' :
    # 摆的数目
    N = 2

    # 臂的长度
    total_arm_length = 1.

    # 小车的质量
    cart_mass = 0.01/N

    # 摆的质量
    bob_mass = 0.01/N

    # 重力
    g_value = 9.81

    # 仿真时间
    simulation_time = 10
    simulation_step = 0.01

    # Q 输入
    q_value_input = [1,1,1,1,1,1]

    # R 输入
    r_value_input = 1

    # 角度输入
    angle_input = [pi / 2.2, pi /1.7]

    # 水平位置如数
    h_position = 1

    # 动画文件名
    # 设为 None 不画动画
    # animation_filename = None
    animation_filename = 'inverted_pendulum.avi'

  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--order', '-N', help='order 1 or 2', default=1)
  parser.add_argument('--total_arm_length', '-l', help='Total arm length (m)', default=total_arm_length)
  parser.add_argument('--cart_mass', '-M', help='Cart mass (kg)', default=cart_mass)
  parser.add_argument('--bob_mass', '-m', help='Bob mass (kg)', default=bob_mass)
  parser.add_argument('--g_value', '-g', help='gravity (m/s²)', default=g_value)
  parser.add_argument('--simulation_time', '-st', help='simulation time (s)', default=simulation_time)
  parser.add_argument('--simulation_step', '-ss', help='simulation step (s)', default=simulation_step)
  parser.add_argument('--q_value_input', '-q', help='Q values, e.g. "[1, 2, 3, 4]" for N = 1, "[1, 2, 3, 4, 5, 6]" for N = 2', default=str(q_value_input))
  parser.add_argument('--r_value_input', '-r', help='R value, integer', default=r_value_input)
  parser.add_argument('--angle_input', '-a', help='Angle input (rad), 2 elements array, e.g."[1.5, 1.5]"', default=str(angle_input))
  parser.add_argument('--h_position', '-hp', help='Horizontal position (m)', default=h_position)
  parser.add_argument('--animation_filename', '-af', help='Animation file name, should be end with .avi', default=animation_filename)

  parser.add_argument('--draw', help='Set to draw result charts.', action='store_true')
  parser.add_argument('--save_csv', help='Set to produce csv files for Excel.', action='store_true')
  parser.add_argument('--animate', help='Set to render animation.', action='store_true')

  args = parser.parse_args()
  N =  int(args.order)
  total_arm_length = float(args.total_arm_length)
  cart_mass = float(args.cart_mass)
  bob_mass = float(args.bob_mass)
  g_value = float(args.g_value)
  simulation_time = float(args.simulation_time)
  simulation_step = float(args.simulation_step)
  q_value_input = eval(args.q_value_input)
  r_value_input = int(args.r_value_input)
  angle_input = eval(args.angle_input)
  h_position = float(args.h_position)

  draw = args.draw
  animate = args.animate
  save_csv = args.save_csv

  animation_filename = args.animation_filename

  if animation_filename.lower() == 'none' :
    animation_filename = None

  inverted_pendulum(N, total_arm_length, cart_mass, bob_mass, g_value,
    simulation_time, simulation_step, q_value_input, r_value_input,
    angle_input, h_position, animation_filename, draw, animate, save_csv)
