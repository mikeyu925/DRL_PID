#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/11 17:27   xxx      1.0         None
"""
import numpy as np
from parameter import args
x = list(np.arange(args.start_time, args.end_time, args.dt))
print(x)