import numpy as np
import pandas as pd


def iterate_pd_discrete(x, mu, dict_coeffs, supercrit=True):
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * x ** order

    cubic_coeff = 1 if supercrit else -1
    x_next = -(1 + mu) * x + cubic_coeff * x ** 3 + sum_hot

    return x_next


def iterate_ns_discrete(s, mu, theta, dict_coeffs_x, dict_coeffs_y, supercrit=True):
    x, y = s
    sum_hot_x = 0
    sum_hot_y = 0
    for order in dict_coeffs_x.keys():
        for index in np.arange(0, order + 1):
            sum_hot_x += dict_coeffs_x[order][index] * x ** (order - index) * y ** index
            sum_hot_y += dict_coeffs_y[order][index] * x ** (order - index) * y ** index

    v_hot = np.array([sum_hot_x, sum_hot_y])

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    cubic_coeff = -1 if supercrit else 1

    s_next = np.matmul((1 + mu) * R, s) + cubic_coeff * np.matmul((x ** 2 + y ** 2) * R, s) + v_hot

    return s_next


def iterate_fold_discrete(x, mu, dict_coeffs):
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * (x - np.sqrt(max(-mu, 0))) ** order

    x_next = x - mu - x ** 2 + sum_hot

    return x_next


def iterate_tc_discrete(x, mu, dict_coeffs):
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * x ** order

    x_next = x * (1 + mu) - x ** 2 + sum_hot

    return x_next


def iterate_tc_discrete_two(x, mu, dict_coeffs):
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * (x - mu) ** order

    x_next = x * (1 - mu) + x ** 2 + sum_hot

    return x_next


def iterate_pf_discrete(x, mu, dict_coeffs, supercrit=True):
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * x ** order

    cubic_coeff = -1 if supercrit else 1

    x_next = x * (1 + mu) + cubic_coeff * x ** 3 + sum_hot

    return x_next


def simulate_pd_discrete(bl=-1, bh=0, tmax=500, tburn=100, sigma=0.01,
                max_order=10, dev_thresh=0.4, supercrit=True):
    x0 = 0
    t = np.arange(0, tmax, 1)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs = {order: np.random.normal(0, 1) for \
                   order in np.arange(4, max_order + 1)}

    dW_burn = np.random.normal(loc=0, scale=sigma, size=int(tburn))
    dW = np.random.normal(loc=0, scale=sigma, size=len(t))

    for i in range(int(tburn)):
        x0 = iterate_pd_discrete(x0, bl, dict_coeffs, supercrit) + dW_burn[i]
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t)) * np.nan
            return pd.DataFrame({'time': t, 'x': x})

    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = iterate_pd_discrete(x[i], b.iloc[i], dict_coeffs, supercrit) + dW[i]
        if abs(x[i + 1]) > dev_thresh:
            x[i:] = np.nan
            break

    return pd.DataFrame({'time': t, 'x': x})


def simulate_ns_discrete(bl=-1, bh=0, theta=np.pi / 2, tmax=500, tburn=100,
                sigma=0.01, max_order=10, dev_thresh=0.4, supercrit=True):
    s0 = np.array([0, 0])
    t = np.arange(0, tmax, 1)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs_x = {}
    dict_coeffs_y = {}
    for order in np.arange(4, max_order + 1):
        dict_coeffs_x[order] = np.random.normal(0, 1, size=order + 1)
        dict_coeffs_y[order] = np.random.normal(0, 1, size=order + 1)

    dW_burn = np.random.normal(loc=0, scale=sigma, size=(int(tburn), 2))
    dW = np.random.normal(loc=0, scale=sigma, size=(len(t), 2))

    for i in range(int(tburn)):
        s0 = iterate_ns_discrete(s0, bl, theta, dict_coeffs_x, dict_coeffs_y, supercrit) + dW_burn[i]
        if np.linalg.norm(s0) > 1e6:
            print('Model diverged during burn in period')
            s = np.ones([len(t), 2]) * np.nan
            return pd.DataFrame({'time': t, 'x': s[:, 0], 'y': s[:, 1]})

    s = np.zeros([len(t), 2])
    s[0] = s0
    for i in range(len(t) - 1):
        s[i + 1] = iterate_ns_discrete(s[i], b.iloc[i], theta, dict_coeffs_x, dict_coeffs_y, supercrit) + dW[i]

        if abs(s[i + 1][0]) > dev_thresh:
            s[i:] = np.nan
            break

    return pd.DataFrame({'time': t, 'x': s[:, 0], 'y': s[:, 1]})


def simulate_fold_discrete(bl=-0.5, bh=0, tmax=500, tburn=100, sigma=0.01,
                  max_order=10, dev_thresh=0.4, return_dev=True):
    x0 = np.sqrt(-bl)
    t = np.arange(0, tmax, 1)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs = {order: np.random.normal(0, 1) for \
                   order in np.arange(3, max_order + 1)}

    dW_burn = np.random.normal(loc=0, scale=sigma, size=int(tburn))
    dW = np.random.normal(loc=0, scale=sigma, size=len(t))

    for i in range(int(tburn)):
        x0 = iterate_fold_discrete(x0, bl, dict_coeffs) + dW_burn[i]
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t)) * np.nan
            return pd.DataFrame({'time': t, 'x': x})

    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = iterate_fold_discrete(x[i], b.iloc[i], dict_coeffs) + dW[i]
        if abs(x[i + 1] - np.sqrt(max(-b.loc[i + 1], 0))) > dev_thresh:
            x[i:] = np.nan
            break

    df_traj = pd.DataFrame(
        {'time': t, 'x_raw': x, 'b': b})

    if return_dev:
        df_traj['x'] = df_traj['x_raw'] - np.sqrt((-b).apply(lambda x: max(x, 0)))
    else:
        df_traj['x'] = df_traj['x_raw']

    return df_traj[['time', 'x']]


def simulate_tc_discrete(bl=-1, bh=0, tmax=500, tburn=100, sigma=0.01,
                max_order=10, dev_thresh=0.4):
    x0 = 0
    t = np.arange(0, tmax)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs = {order: np.random.normal(0, 1) for \
                   order in np.arange(3, max_order + 1)}

    dW_burn = np.random.normal(loc=0, scale=sigma, size=int(tburn))
    dW = np.random.normal(loc=0, scale=sigma, size=len(t))

    for i in range(int(tburn)):
        x0 = iterate_tc_discrete(x0, bl, dict_coeffs) + dW_burn[i]
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t)) * np.nan
            return pd.DataFrame({'time': t, 'x': x})

    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = iterate_tc_discrete(x[i], b.iloc[i], dict_coeffs) + dW[i]
        if abs(x[i + 1]) > dev_thresh:
            x[i:] = np.nan
            break

    return pd.DataFrame({'time': t, 'x': x})


def simulate_tc_discrete_two(bl=-1, bh=0, tmax=500, tburn=100, sigma=0.01,
                max_order=10, dev_thresh=0.4, return_dev=True):
    x0 = bl
    t = np.arange(0, tmax)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs = {order: np.random.normal(0, 1) for \
                   order in np.arange(3, max_order + 1)}

    dW_burn = np.random.normal(loc=0, scale=sigma, size=int(tburn))
    dW = np.random.normal(loc=0, scale=sigma, size=len(t))

    for i in range(int(tburn)):
        x0 = iterate_tc_discrete_two(x0, bl, dict_coeffs) + dW_burn[i]
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t)) * np.nan
            return pd.DataFrame({'time': t, 'x': x})

    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = iterate_tc_discrete_two(x[i], b.iloc[i], dict_coeffs) + dW[i]
        if abs(x[i + 1] - b.iloc[i + 1]) > dev_thresh:
            x[i:] = np.nan
            break

    df_traj = pd.DataFrame(
        {'time': t, 'x_raw': x, 'b': b})

    if return_dev:
        df_traj['x'] = df_traj['x_raw'] - b
    else:
        df_traj['x'] = df_traj['x_raw']

    return df_traj[['time', 'x']]


def simulate_pf_discrete(bl=-1, bh=0, tmax=500, tburn=100, sigma=0.01,
                max_order=10, dev_thresh=0.4, supercrit=True):
    x0 = 0
    t = np.arange(0, tmax, 1)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs = {order: np.random.normal(0, 1) for \
                   order in np.arange(4, max_order + 1)}

    dW_burn = np.random.normal(loc=0, scale=sigma, size=int(tburn))
    dW = np.random.normal(loc=0, scale=sigma, size=len(t))

    for i in range(int(tburn)):
        x0 = iterate_pf_discrete(x0, bl, dict_coeffs, supercrit) + dW_burn[i]
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t)) * np.nan
            return pd.DataFrame({'time': t, 'x': x})

    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = iterate_pf_discrete(x[i], b.iloc[i], dict_coeffs, supercrit) + dW[i]
        if abs(x[i + 1]) > dev_thresh:
            x[i:] = np.nan
            break

    return pd.DataFrame({'time': t, 'x': x})


def iterate_hopf_con(s, mu, dict_coeffs_x, dict_coeffs_y, supercrit=True):
    x, y = s
    sum_hot_x = 0
    sum_hot_y = 0
    for order in dict_coeffs_x.keys():
        for index in np.arange(0, order + 1):
            sum_hot_x += dict_coeffs_x[order][index] * x ** (order - index) * y ** index
            sum_hot_y += dict_coeffs_y[order][index] * x ** (order - index) * y ** index

    v_hot = np.array([sum_hot_x, sum_hot_y])

    R = np.array([[mu, -1], [1, mu]])

    cubic_coeff = -1 if supercrit else 1

    s_next = np.matmul(R, s) + cubic_coeff * (x ** 2 + y ** 2) * s + v_hot

    return s_next


def iterate_fold_con(x, mu, dict_coeffs):
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * (x - np.sqrt(max(-mu, 0))) ** order

    x_next = - mu - x ** 2 + sum_hot

    return x_next


def iterate_tc_con(x, mu, dict_coeffs):
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * x ** order

    x_next = x * mu - x ** 2 + sum_hot

    return x_next


def iterate_tc_con_two(x, mu, dict_coeffs):
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * (x - mu) ** order

    x_next = - x * mu + x ** 2 + sum_hot

    return x_next


def iterate_pf_con(x, mu, dict_coeffs, supercrit=True):
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * x ** order

    cubic_coeff = -1 if supercrit else 1

    x_next = x * mu + cubic_coeff * x ** 3 + sum_hot

    return x_next


def simulate_hopf_con(bl=-1, bh=0, tmax=500, tburn=100,
                  sigma=0.01, max_order=10, dev_thresh=0.4, supercrit=True):
    dt = 0.01
    dt2 = 1
    s0 = np.array([0, 0])
    t = np.arange(0, tmax, dt)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs_x = {}
    dict_coeffs_y = {}
    for order in np.arange(4, max_order + 1):
        dict_coeffs_x[order] = np.random.normal(0, 1, size=order + 1)
        dict_coeffs_y[order] = np.random.normal(0, 1, size=order + 1)

    dW_burn = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=(int(tburn / dt), 2))
    dW = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=(len(t), 2))

    for i in range(int(tburn)):
        s0 = s0 + iterate_hopf_con(s0, bl, dict_coeffs_x, dict_coeffs_y, supercrit) * dt + dW_burn[i]
        if np.linalg.norm(s0) > 1e6:
            print('Model diverged during burn-in period')
            s = np.ones([len(t), 2]) * np.nan
            return pd.DataFrame({'time': t, 'x': s[:, 0], 'y': s[:, 1]})

    s = np.zeros([len(t), 2])
    s[0] = s0
    for i in range(len(t) - 1):
        s[i + 1] = s[i] + iterate_hopf_con(s[i], b.iloc[i], dict_coeffs_x, dict_coeffs_y, supercrit) * dt + dW[i]

        if abs(s[i + 1][0]) > dev_thresh:
            s[i:] = np.nan
            break

    df_traj = pd.DataFrame({'time': t, 'x': s[:, 0], 'y': s[:, 1]})

    df_traj = df_traj.loc[::int(dt2 / dt)]

    return df_traj


def simulate_fold_con(bl=-0.5, bh=0, tmax=500, tburn=100, sigma=0.01,
                  max_order=10, dev_thresh=0.4, return_dev=True):
    dt = 0.01
    dt2 = 1
    x0 = np.sqrt(-bl)
    t = np.arange(0, tmax, dt)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs = {order: np.random.normal(0, 1) for \
                   order in np.arange(3, max_order + 1)}

    dW_burn = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=int(tburn/dt))
    dW = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=len(t))

    for i in range(int(tburn)):
        x0 = x0 + iterate_fold_con(x0, bl, dict_coeffs) * dt + dW_burn[i]
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t)) * np.nan
            return pd.DataFrame({'time': t, 'x': x})

    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = x[i] + iterate_fold_con(x[i], b.iloc[i], dict_coeffs) * dt + dW[i]
        if abs(x[i + 1] - np.sqrt(max(-b.iloc[i + 1], 0))) > dev_thresh:
            x[i:] = np.nan
            break

    df_traj = pd.DataFrame(
        {'time': t, 'x_raw': x, 'b': b})

    if return_dev:
        df_traj['x'] = df_traj['x_raw'] - np.sqrt((-b).apply(lambda x: max(x, 0)))
    else:
        df_traj['x'] = df_traj['x_raw']

    df_traj = df_traj.loc[::int(dt2 / dt)]

    return df_traj[['time', 'x']]


def simulate_tc_con(bl=-1, bh=0, tmax=500, tburn=100, sigma=0.01,
                max_order=10, dev_thresh=0.4):
    dt = 0.01
    dt2 = 1
    x0 = 0
    t = np.arange(0, tmax, dt)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs = {order: np.random.normal(0, 1) for \
                   order in np.arange(3, max_order + 1)}

    dW_burn = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=int(tburn/dt))
    dW = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=len(t))

    for i in range(int(tburn)):
        x0 = x0 + iterate_tc_con(x0, bl, dict_coeffs) * dt + dW_burn[i]
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t)) * np.nan
            return pd.DataFrame({'time': t, 'x': x})

    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = x[i] + iterate_tc_con(x[i], b.iloc[i], dict_coeffs) * dt + dW[i]
        if abs(x[i + 1]) > dev_thresh:
            x[i:] = np.nan
            break

    df_traj = pd.DataFrame({'time': t, 'x': x})

    df_traj = df_traj.loc[::int(dt2 / dt)]

    return df_traj


def simulate_tc_con_two(bl=-1, bh=0, tmax=500, tburn=100, sigma=0.01,
                max_order=10, dev_thresh=0.4, return_dev=True):
    dt = 0.01
    dt2 = 1
    x0 = bl
    t = np.arange(0, tmax, dt)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs = {order: np.random.normal(0, 1) for \
                   order in np.arange(3, max_order + 1)}

    dW_burn = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=int(tburn/dt))
    dW = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=len(t))

    for i in range(int(tburn)):
        x0 = x0 + iterate_tc_con_two(x0, bl, dict_coeffs) * dt + dW_burn[i]
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t)) * np.nan
            return pd.DataFrame({'time': t, 'x': x})

    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = x[i] + iterate_tc_con_two(x[i], b.iloc[i], dict_coeffs) * dt + dW[i]
        if abs(x[i + 1] - b.iloc[i + 1]) > dev_thresh:
            x[i:] = np.nan
            break

    df_traj = pd.DataFrame(
        {'time': t, 'x_raw': x, 'b': b})

    if return_dev:
        df_traj['x'] = df_traj['x_raw'] - b
    else:
        df_traj['x'] = df_traj['x_raw']

    df_traj = df_traj[['time', 'x']]

    df_traj = df_traj.loc[::int(dt2 / dt)]

    return df_traj


def simulate_pf_con(bl=-1, bh=0, tmax=500, tburn=100, sigma=0.01,
                max_order=10, dev_thresh=0.4, supercrit=True):
    dt = 0.01
    dt2 = 1
    x0 = 0
    t = np.arange(0, tmax, dt)
    b = pd.Series(np.linspace(bl, bh, len(t)), index=t)

    dict_coeffs = {order: np.random.normal(0, 1) for \
                   order in np.arange(4, max_order + 1)}

    dW_burn = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=int(tburn/dt))
    dW = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=len(t))

    for i in range(int(tburn)):
        x0 = x0 + iterate_pf_con(x0, bl, dict_coeffs, supercrit) * dt + dW_burn[i]
        if abs(x0) > 1e6:
            print('Model diverged during burn in period')
            x = np.ones(len(t)) * np.nan
            return pd.DataFrame({'time': t, 'x': x})

    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = x[i] + iterate_pf_con(x[i], b.iloc[i], dict_coeffs, supercrit) * dt + dW[i]
        if abs(x[i + 1]) > dev_thresh:
            x[i:] = np.nan
            break

    df_traj = pd.DataFrame({'time': t, 'x': x})

    df_traj = df_traj.loc[::int(dt2 / dt)]

    return df_traj