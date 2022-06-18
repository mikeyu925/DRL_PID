import math

import numpy as np
from line_env import LineEnv
from random import random
from  pid_controller import PID_Controller,PID_Info
from parameter import args
from utils import  sin_curve,show_resutlt,line_curve,step_function,horizontal_line
from matplotlib import pyplot as plt

def track_sin_test():
    x = list(np.arange(args.start_time, args.end_time, args.dt))
    y = sin_curve(x)
    p1 = []
    # p2 = []
    now1 = random() * 50
    now2 = now1
    # pid = PID_Controller(kp=args.kp, ki=args.ki, kd=args.kd) # 2 0.1 1
    p,i,d = 8,0.6,0
    pid1 = PID_Controller(p,i,d)
    pid2 = PID_Controller(p,i,d)
    for i,sim_time in enumerate(x):
        error1 = sin_curve(sim_time) - now1
        now1 += pid1.update(0, 0, 0, error1) * args.dt

        # error2 = sin_curve(sim_time) - now2
        # now2 += pid2.update_up(0,0,0,error2) * args.dt
        p1.append(now1)
        # p2.append(now2)

    plt.plot(x, y, 'r-', linewidth=0.5)
    plt.plot(x, p1, 'b-', linewidth=0.5)
    # plt.plot(x, p2, 'g-', linewidth=0.5)
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend(["target_pos", "real_pos1"], loc='lower right')
    # plt.legend(["target_pos", "real_pos1", "real_pos2"], loc='lower right')
    plt.show()

# 阶跃曲线
def step_response():
    pidinfo = PID_Info("Classic PID",100)
    x = list(np.arange(args.start_time, args.end_time, args.dt))
    y = step_function(x)
    pidinfo.set_start_time(x[int(len(x) / 5)])
    p1 = []
    now1 = 0
    p,i,d = 10,0.5,4
    pid1 = PID_Controller(p,i,d)
    for i,sim_time in enumerate(x):
        error1 = y[i] - now1
        print(error1)
        pidinfo.check_stable(error1,now1,sim_time,i) # 更新相关信息
        now1 += pid1.update(0, 0, 0, error1) * args.dt
        p1.append(now1)
    pidinfo.showPIDControlInfo()

    plt.plot(x, y, 'r-', linewidth=0.5)
    plt.plot(x, p1, 'b-', linewidth=0.5)
    # plt.scatter(pidinfo.stable_point.x, pidinfo.stable_point.y,color=(0.7,0.,0.6))
    plt.scatter(x[pidinfo.stable_idx], p1[pidinfo.stable_idx], color=(0.7, 0., 0.6))
    plt.scatter(pidinfo.top_point.x, pidinfo.top_point.y,color=(0.,0.5,0.))

    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend(["target_pos", "real_pos"], loc='lower right')

    plt.show()

# 一个水平线
def horizontal_line_response():
    pidinfo = PID_Info("Classic PID",100)
    x = list(np.arange(args.start_time, args.end_time, args.dt))
    y = horizontal_line(x,100)
    pidinfo.set_start_time(x[0])
    p1 = []
    now1 = 0
    p,i,d = 10,1,6
    pid1 = PID_Controller(p,i,d)
    for i,sim_time in enumerate(x):
        error1 = y[i] - now1
        # print(error1)
        pidinfo.check_stable(error1,now1,sim_time,i) # 更新相关信息
        # now1 += pid1.update_up(0, 0, 0, error1) * args.dt
        now1 += pid1.update(0, 0, 0, error1) * args.dt
        p1.append(now1)
    pidinfo.showPIDControlInfo()

    plt.plot(x, y, 'r-', linewidth=0.5)
    plt.plot(x, p1, 'b-', linewidth=0.5)
    # plt.scatter(pidinfo.stable_point.x, pidinfo.stable_point.y,color=(0.7,0.,0.6)) # stable_idx
    plt.scatter(x[pidinfo.stable_idx], p1[pidinfo.stable_idx], color=(0.7, 0., 0.6))
    plt.scatter(pidinfo.top_point.x, pidinfo.top_point.y,color=(0.,0.5,0.))

    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend(["target_pos", "real_pos"], loc='lower right')

    plt.show()

def track_line_test():
    x = list(np.arange(args.start_time, args.end_time, args.dt))
    y = line_curve(x,100)
    p1 = []
    p2 = []
    now1 = random() * 30
    now2 = now1
    # pid = PID_Controller(kp=args.kp, ki=args.ki, kd=args.kd) # 2 0.1 1
    p,i,d = 40,1,60
    pid1 = PID_Controller(p,i,d)
    pid2 = PID_Controller(p,i,d)
    for i,sim_time in enumerate(x):
        # error = sin_curve(sim_time) - now
        error1 = y[i] - now1
        error2 = y[i] - now2
        now1 += pid1.update(0,0,0,error1) * args.dt
        now2 += pid2.update_up(0,0,0,error2) * args.dt
        p1.append(now1)
        p2.append(now2)

    plt.plot(x, y, 'r-', linewidth=0.5)
    plt.plot(x, p1, 'b-', linewidth=0.5)
    plt.plot(x, p2, 'g-', linewidth=0.5)
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend(["target_pos", "real_pos1", "real_pos2"], loc='lower right')
    plt.show()

if __name__ == '__main__':
    horizontal_line_response()


