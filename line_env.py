from parameter import args
from utils import trapezoidal_function
import numpy as np
import gym
import matplotlib.pyplot as plt
from random import *
from pid_controller import PID_Controller,PID_Info
from utils import sin_curve,step_function,horizontal_line
import math


class LineEnv(gym.Env):
    def __init__(self):
        super(LineEnv, self).__init__()
        np.random.seed(args.seed)
        # 环境的一些状态信息
        self.last_last_error = 0.0
        self.last_error = 0.0
        self.now_error = 0.0
        self.now_pos = 0.0
        self.last_speed = 0.0
        self.now_speed = 0.0
        # 计数器
        self.cnt = 0
        # 时间步长
        self.times = np.arange(args.start_time, args.end_time, args.dt)

        # 环境状态： [上上次误差、上次误差、当前误差、当前位置、上次速度、当前速度]
        self.state = np.array([self.last_last_error,self.last_error,self.now_error,self.now_pos,self.last_speed,self.now_speed])
        # 环境状态维度
        self.state_dim = len(self.state)

        # 奖励函数的一些常量信息
        self.k = 2
        self.c = -200

        #TODO 待修改  一些阈值常量
        self.error_limit = 150
        self.quick_error_limit = 50

        self.speed_limie = 100
        # PID控制器
        self.pid = PID_Controller(kp=args.kp, ki=args.ki, kd=args.kd)
        self.control_info = PID_Info("Trick Line",100)  # 一些控制信息

        self.height = 100
        self.line_y = step_function(list(self.times))
        self.horizontal_line = horizontal_line(self.times,self.height)

    def reset(self):
        # 环境的一些状态信息 TODO 是全部初始化为0还是？
        # self.now_pos = random() * 10  # 当前随机一个位置
        # self.now_error = self.now_pos - sin_curve(0)  # 当前误差
        # self.last_error = min(self.now_error * 1.3,self.error_limit)  # 上次误差
        # self.last_last_error = min(self.now_error * 1.6,self.error_limit) # 上上次误差

        self.now_pos = 0  # 当前随机一个位置
        self.now_error = 0  # 当前误差
        self.last_error = 0  # 上次误差
        self.last_last_error = 0 # 上上次误差

        self.last_speed = 0.0
        self.now_speed  = 0.0
        self.state = np.array([self.last_last_error, self.last_error, self.now_error, self.now_pos, self.last_speed, self.now_speed])
        # 计数器
        self.cnt = 0

        # PID控制器  还需要重置，因为要初始化pid的本身的误差信息
        self.pid = PID_Controller(kp=args.kp, ki=args.ki, kd=args.kd)
        return self.state

    def step(self,action):
        """
        :param action: [_kp, _ki, _kd, _out]
        :return: state : [self.state, reward, done]
        """
        # 获取当前环境的状态
        last_last_error, last_error, now_error, now_pos, last_speed, now_speed = self.state
        _kp, _ki, _kd, _out = action
        # 计算误差
        # error = sin_curve(self.times[self.cnt]) - now_pos

        error = self.horizontal_line[self.cnt] - now_pos
        # 如果已经出于稳态，则不再进行补偿
        if error <= self.height * 0.02:
            _out = 0
        out_speed = self.pid.update_up(_kp,_ki,_kd,error) + _out

        # 更新状态
        self.last_last_error = self.last_error
        self.last_error = self.now_error
        self.now_error = error
        self.now_pos += out_speed * args.dt
        self.last_speed = now_speed
        self.now_speed = out_speed

        # 获得即时奖励
        reward = self._get_immediate_reward(error)
        # 是否到达终止状态
        done = self._isDone()

        self.cnt += 1  # TODO 应该加在Done后面
        self.state = np.array([self.last_last_error,self.last_error,self.now_error,self.now_pos,self.last_speed,self.now_speed])

        return self.state, reward, done

    def _get_immediate_reward(self,error):
        """
        获取当前的即时奖励
        :param error:
        :return: int (reward)
        """
        if math.fabs(error) >= self.error_limit:
            return self.c
        elif math.fabs(error) >= self.quick_error_limit:
            return math.fabs(self.last_error) - math.fabs(error)
        else:
            return math.exp(self.k - math.fabs(error))

    def _isDone(self):
        """
        判断是否到达了终止状态
        :return:
        """
        if math.fabs(self.now_error) >= self.error_limit:  # 提前结束
            return True
        if self.cnt == len(self.times) - 1:  # 到达终止状态
            return True
        return False


if __name__ == '__main__':
    x_limit = 2 * np.pi
    x = np.arange(args.start_time, args.end_time, args.dt)
    print(len(x))
    y = sin_curve(x)
    plt.plot(x, y)
    plt.show()