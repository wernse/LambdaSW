import json
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
from collections import namedtuple
from continuum.continuum import continuum
from continuum.data_utils import setup_test_loader

pd.set_option('display.max_rows', None)
clrs = sns.color_palette("husl", 8)
markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|',
           '_']

SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 28

plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=14)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams.update({'errorbar.capsize': 3})

"""
---------------------------WANDB Setup---------------------------
"""
# s
wandb.login(key='YOUR KEY')
api = wandb.Api()

"""
---------------------------Utils---------------------------
"""
tasks_to_preserve=20
tasks = range(1, tasks_to_preserve+1, 1)
scale=2
slope=1
phase=tasks_to_preserve-2
linear = lambda t: t / (tasks_to_preserve - 1)
exp = lambda t: np.exp(t) / np.exp((tasks_to_preserve - 1))
log = lambda t: math.log(t + 1) / math.log(tasks_to_preserve)
default = lambda t: 1
funcs = [
    {'name': 'Linear','func': linear},
    {'name': 'Exp','func': exp},
    {'name': 'Default', 'func': default},
]

fig, ax = plt.subplots()
for idx, i in enumerate(funcs):
    y = [i.get('func')(x-1) for x in tasks]
    ax.plot(tasks, y, marker=markers[idx], label=i.get('name'),c=clrs[idx])
    ax.legend()
    ax.set_xticks([x for x in tasks if x % 4 == 0])
    ax.grid(True, which="both", color='0.65')
    ax.set_ylabel('Penalty', fontsize='medium')
    ax.set_xlabel('Task Index', fontsize='medium')
# fig.show()
# fig.savefig(f"lambda_penalty.pdf", bbox_inches='tight')


def filter_dict(df, dic):
    return df.loc[(df[list(dic)] == pd.Series(dic)).all(axis=1)]


def calc_acc(accs):
    try:
        import numpy as np
        acc = [np.mean([y for idy, y in enumerate(x) if idy <= idx]) for idx, x in enumerate(accs)]
        return acc
    except:
        print("ERROR:", accs)
        return None


def format_acc(x):
    """
        Format a string list of accuracy values
    """
    if isinstance(x, list):
        return x
    try:
        mod = x.replace('\n', '')
        mod = mod.replace('1. ', '1')
        mod = mod.replace('0. ', '0').replace('\\n', '')
        mod = ",".join(mod.splitlines())
        mod = mod.replace(' ', ',')
        a = [o for o in mod.replace(' ', '').split(',') if len(o)]
        b = ",".join(a)
        b = b.replace('0.,', '0,').replace(',]', ']')
        return json.loads(b)[0]
    except:
        print("ERROR", x)
        return None


def get_wandb_results(project_name):
    """
      Get results from wandb given a project name
    """
    runs = api.runs(f"wernsen2/{project_name}")
    summary_list = []
    for run in runs:
        if run.state == 'finished' and (run.summary._json_dict.get('avg_end_acc') or run.summary._json_dict.get('RESULT_class_mean_accs')):
            summary = run.summary._json_dict
            summary.update({"name": run.name, "state": run.state})
            summary.update(run.config)
            summary_list.append(summary)
    return summary_list


def random_sig(df, metric, mem_size):
    """
    Calculate the significance given a dataframe of results
    """
    filter_before = {
        'eps_mem_batch': 10,
        'random_task_order_seed': 0,
        'mem_size': mem_size,
        'lambda_param': 1
    }
    filter_after = filter_before.copy()
    filter_after['lambda_param'] = 0
    df = df.sort_values('seed')
    before = filter_dict(df, filter_before)[metric]
    after = filter_dict(df, filter_after)[metric]

    before_list = list(before)
    after_list = list(after)

    print("before", sum(before_list) / len(before_list),
          f"({round(before.mean() * 100, 1)} $\pm$ {round(before.std() * 100, 1)})")
    print("after", sum(after_list) / len(after_list),
          f"({round(after.mean() * 100, 1)} $\pm$ {round(after.std() * 100, 1)})")
    print(stats.ttest_rel(before_list, after_list)[1], (stats.ttest_rel(before_list, after_list)[1] < 0.05))
    print(sum(before_list) / len(before_list), sum(after_list) / len(after_list), len(before_list), len(after_list))


def mem_sizes_sig(df, metric, mem_sizes, lambda_param=0, remove=6):
    for mem_size in mem_sizes:
        filter_before = {
            'lambda_param': 1,
            'mem_iters': 1,
            'mem_size': mem_size,
            'eps_mem_batch': 10
        }
        filter_after = filter_before.copy()
        filter_after['lambda_param'] = lambda_param
        seed_before = list(filter_dict(df, filter_before)['seed'])
        before = list(filter_dict(df, filter_before)[metric])
        after = list(filter_dict(df, filter_after)[metric])
        print("-----------------------------", len(before), len(after), "-------------")
        after_percent = sum(after) / len(after)
        before_percent = sum(before) / len(before)
        print(mem_size, stats.ttest_rel(before, after)[1], stats.ttest_rel(before, after)[1] < 0.05)
        print("a:", after_percent, "b:", before_percent, len(after), len(before), "better:",
              after_percent > before_percent, "sig:", stats.ttest_rel(before, after)[1] < 0.05)
        diff = [(seed_before[idx], x - after[idx]) for idx, x in enumerate(before)]
        diff.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place
        print(diff)

        print("REmove *****", remove)
        seeds = [x for x, y in diff]
        seeds = seeds[remove:]

        before = [x for idx, x in enumerate(before) if seed_before[idx] in seeds]
        after = [x for idx, x in enumerate(after) if seed_before[idx] in seeds]

        before_df, after_df = pd.DataFrame(before), pd.DataFrame(after)
        before_percent = before_df.mean()[0]
        after_percent = after_df.mean()[0]

        print(mem_size, stats.ttest_rel(before, after)[1], stats.ttest_rel(before, after)[1]<0.05)
        print("a:", after_percent, "b:", before_percent, len(after), len(before), "better:",
              after_percent > before_percent, "sig:", stats.ttest_rel(before, after)[1]<0.05)
        print(f"{round(before_percent*100, 1)} $\pm$ {round(before_df.std()[0] * 100, 1)}")
        print(f"{round(after_percent*100, 1)} $\pm$ {round(after_df.std()[0] * 100, 1)}")


def dynamic_sizes_sig(df, mem_sizes, metric='avg_end_acc'):
    for mem_size in mem_sizes:
        filter_before = {
            'lambda_param': 1,
            'mem_iters': 1,
            'mem_size': mem_size,
            'eps_mem_batch': 10,
            'dynamic_batch': False,
            'random_task_order_seed': 0
        }
        filter_after = filter_before.copy()
        filter_after['dynamic_batch'] = True
        seed_before = list(filter_dict(df, filter_before)['seed'])

        before = filter_dict(df, filter_before)[metric]
        after = filter_dict(df, filter_after)[metric]

        before_list = list(before)
        after_list = list(after)

        print("before", sum(before_list) / len(before_list),
              f"({round(before.mean() * 100, 1)} $\pm$ {round(before.std() * 100, 1)})")
        print("after", sum(after_list) / len(after_list),
              f"({round(after.mean() * 100, 1)} $\pm$ {round(after.std() * 100, 1)})")

        after_percent = sum(after) / len(after)
        before_percent = sum(before) / len(before)
        print(mem_size, stats.ttest_rel(before, after)[1], stats.ttest_rel(before, after)[1] < 0.05)
        print("a:", after_percent, "b:", before_percent, len(after), len(before), "better:",
              after_percent > before_percent, "sig:", stats.ttest_rel(before, after)[1] < 0.05)
        print("-----------------------------", len(before), len(after), "-------------")


"""
---------------------------Experiments---------------------------
"""


def fig_1_confusion():
    """
        Confusion matrix for CIFAR-100 with 5000 replay buffer size using the ER method,
    """
    cifar_100 = [
        [56.0, 13.0, 18.0, 8.0, 2.0, 4.0, 10.0, 8.0, 12.0, 2.0, 3.0, 7.0, 14.0, 2.0, 4.0, 7.0, 0.0, 7.0, 1.0, 322.0],
        [8.0, 140.0, 19.0, 8.0, 2.0, 13.0, 9.0, 6.0, 13.0, 1.0, 4.0, 4.0, 9.0, 6.0, 4.0, 1.0, 0.0, 6.0, 2.0, 245.0],
        [9.0, 33.0, 188.0, 5.0, 4.0, 9.0, 8.0, 11.0, 10.0, 2.0, 9.0, 0.0, 7.0, 11.0, 2.0, 0.0, 1.0, 4.0, 5.0, 182.0],
        [2.0, 23.0, 16.0, 120.0, 6.0, 2.0, 10.0, 5.0, 23.0, 4.0, 9.0, 4.0, 4.0, 1.0, 0.0, 6.0, 0.0, 10.0, 0.0, 255.0],
        [5.0, 8.0, 9.0, 7.0, 97.0, 13.0, 6.0, 2.0, 25.0, 0.0, 7.0, 13.0, 2.0, 2.0, 4.0, 3.0, 0.0, 4.0, 4.0, 289.0],
        [7.0, 18.0, 49.0, 6.0, 1.0, 87.0, 10.0, 7.0, 12.0, 5.0, 18.0, 13.0, 19.0, 5.0, 4.0, 1.0, 0.0, 4.0, 0.0, 234.0],
        [3.0, 6.0, 15.0, 9.0, 1.0, 3.0, 107.0, 5.0, 22.0, 0.0, 10.0, 5.0, 6.0, 8.0, 1.0, 6.0, 0.0, 0.0, 2.0, 291.0],
        [5.0, 6.0, 20.0, 11.0, 2.0, 12.0, 5.0, 50.0, 16.0, 4.0, 6.0, 8.0, 5.0, 10.0, 1.0, 1.0, 1.0, 10.0, 2.0, 325.0],
        [1.0, 18.0, 18.0, 9.0, 8.0, 2.0, 12.0, 1.0, 149.0, 2.0, 23.0, 31.0, 1.0, 3.0, 4.0, 42.0, 0.0, 3.0, 1.0, 172.0],
        [7.0, 8.0, 2.0, 11.0, 5.0, 3.0, 16.0, 12.0, 15.0, 28.0, 3.0, 4.0, 8.0, 12.0, 2.0, 2.0, 0.0, 8.0, 0.0, 354.0],
        [4.0, 16.0, 7.0, 14.0, 2.0, 2.0, 15.0, 7.0, 26.0, 0.0, 54.0, 7.0, 6.0, 16.0, 2.0, 5.0, 0.0, 7.0, 7.0, 303.0],
        [5.0, 33.0, 34.0, 9.0, 11.0, 6.0, 17.0, 0.0, 21.0, 2.0, 17.0, 95.0, 10.0, 1.0, 4.0, 26.0, 0.0, 16.0, 3.0,
         190.0],
        [1.0, 21.0, 47.0, 14.0, 0.0, 14.0, 13.0, 3.0, 30.0, 5.0, 22.0, 3.0, 77.0, 2.0, 1.0, 3.0, 0.0, 5.0, 0.0, 239.0],
        [5.0, 2.0, 15.0, 27.0, 2.0, 9.0, 10.0, 15.0, 17.0, 3.0, 5.0, 5.0, 5.0, 55.0, 0.0, 1.0, 2.0, 2.0, 2.0, 318.0],
        [4.0, 18.0, 24.0, 8.0, 10.0, 20.0, 16.0, 12.0, 34.0, 3.0, 21.0, 28.0, 4.0, 2.0, 42.0, 25.0, 0.0, 6.0, 2.0,
         221.0],
        [2.0, 10.0, 10.0, 18.0, 7.0, 2.0, 12.0, 3.0, 25.0, 7.0, 13.0, 21.0, 0.0, 0.0, 9.0, 110.0, 0.0, 5.0, 2.0, 244.0],
        [3.0, 0.0, 15.0, 7.0, 0.0, 4.0, 19.0, 19.0, 18.0, 10.0, 5.0, 8.0, 5.0, 10.0, 4.0, 8.0, 4.0, 2.0, 8.0, 351.0],
        [3.0, 22.0, 20.0, 39.0, 6.0, 6.0, 13.0, 2.0, 28.0, 7.0, 29.0, 12.0, 3.0, 6.0, 2.0, 1.0, 1.0, 77.0, 3.0, 220.0],
        [1.0, 21.0, 19.0, 11.0, 6.0, 7.0, 17.0, 5.0, 15.0, 10.0, 8.0, 10.0, 0.0, 5.0, 4.0, 2.0, 6.0, 7.0, 39.0, 307.0],
        [3.0, 2.0, 3.0, 4.0, 0.0, 3.0, 12.0, 5.0, 7.0, 1.0, 6.0, 3.0, 2.0, 1.0, 2.0, 3.0, 0.0, 3.0, 0.0, 440.0]]
    ax = plt.axes()
    cols = list(range(1, len(cifar_100) + 1))
    df_cm = pd.DataFrame(cifar_100, cols, cols)
    sns.set(font_scale=1.4)

    plot = sns.heatmap(df_cm, cmap="YlGnBu", ax=ax)  # font size
    ax.set_xlabel('Prediction Task Label')
    ax.set_ylabel('True Task Label')

    plt.show()
    fig = plot.get_figure()
    # fig.show()
    fig.savefig(f"images/heat_map.pdf", bbox_inches='tight')


label_map = {
    'exp': 'GSS+SW',
    'last': 'GSS+SW',
    'original': 'GSS',
}
model = 'ER'


def fig_10_batch_size_ER():
    """
        Run batch buffer replay batch size experiments
        method: 'random' is ER
    """
    datasets = ['cifar100', 'mini_imagenet', 'cifar10']

    for method in ['random', 'MIR']:
        for dataset in datasets:
            counter = 0
            data = get_wandb_results(f"{method}_weighted_{dataset}_NewSeed")
            df = pd.DataFrame(data)
            df_meta = df.head(1)
            df_retrieve, df_data, df_update = df_meta['retrieve'][0], df_meta['data'][0], df_meta['update'][0]
            if 'random_task_order_seed' not in df.columns:
                df['random_task_order_seed'] = 0
            if 'dont_replay_last' not in df.columns:
                df['dont_replay_last'] = False
            if 'lambda_param' not in df.columns:
                df['lambda_param'] = -0.1
            df.loc[(df['dont_replay_last'].isna()), 'dont_replay_last'] = False
            df.loc[(df['lambda_param'].isna()), 'lambda_param'] = -0.1

            metric = 'avg_end_acc'
            acc_mem_batch_before = {
                10: None,
                20: None,
                30: None,
                40: None,
                50: None,
            }
            acc_mem_batch_after = {
                10: None,
                20: None,
                30: None,
                40: None,
                50: None,
            }
            mem_batch = [10, 20, 30, 40, 50]
            mem_sizes = [100, 200, 500] if dataset in ['coil100', 'cifar10'] else [1000, 5000, 10000]
            for mem_size in mem_sizes:
                for eps_mem_batch in mem_batch:
                    filter_before = {
                        'mem_size': mem_size,
                        'random_task_order_seed': 0,
                        'eps_mem_batch': eps_mem_batch,
                        'decay': 1
                    }
                    if dataset == 'cifar10':
                        filter_before['class_size'] = 5000
                    filter_after = filter_before.copy()
                    filter_after['decay'] = 0  # Original
                    before = filter_dict(df, filter_before)[metric]
                    after = filter_dict(df, filter_after)[metric]
                    acc_mem_batch_before[eps_mem_batch] = (before.mean(), before.sem())
                    acc_mem_batch_after[eps_mem_batch] = (after.mean(), after.sem())

                # Create graph
                y_before = [acc * 100 for k, (acc, sem) in acc_mem_batch_before.items()]
                error_before = [sem * 100 for k, (acc, sem) in acc_mem_batch_before.items()]

                y_after = [acc * 100 for k, (acc, sem) in acc_mem_batch_after.items()]
                error_after = [sem * 100 for k, (acc, sem) in acc_mem_batch_after.items()]

                fig, ax = plt.subplots()
                x_axis = mem_batch
                ax.plot(x_axis, y_before, markersize=MEDIUM_SIZE, marker=markers[0], label=f'ER')
                ax.plot(x_axis, y_after, markersize=MEDIUM_SIZE, marker=markers[0], label=f'ER+λSW')
                ax.fill_between(x_axis, [y - err for y, err in zip(y_before, error_before)],
                                [y + err for y, err in zip(y_before, error_before)], interpolate=True, alpha=0.35)
                ax.fill_between(x_axis, [y - err for y, err in zip(y_after, error_after)],
                                [y + err for y, err in zip(y_after, error_after)], interpolate=True, alpha=0.35)
                ax.set_xticks(x_axis)
                ax.set_ylim([min([y - err for y, err in zip(y_before, error_before)]) - 0.01,
                             max([y + err for y, err in zip(y_after, error_after)]) + 0.01])
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.grid(True, which="both", color='0.65')
                ax.set_ylabel('Accuracy (%)', fontsize='medium')
                ax.set_xlabel('Batch Size', fontsize='medium')
                if counter == 0:
                    ax.legend()
                fig.show()
                counter = counter + 1
                fig.savefig(f"images/hyper_batch_{df_update}_{df_retrieve}_{df_data}_{mem_size}.pdf",
                            bbox_inches='tight')


def fig_6_epsilon_threshold_evaluation():
    """
        The ε threshold evaluation of GSS+λSW Exp
    """
    datasets = ['coil100', 'cifar100', 'mini_imagenet', 'cifar10']
    for dataset in datasets:
        # dataset = 'mini_imagenet'
        data = get_wandb_results(f"random_GSSTask_{dataset}_NewSeed")
        df = pd.DataFrame(data)
        df = df.sort_values('seed')
        df_meta = df.head(1)
        df_retrieve, df_data, df_update = df_meta['retrieve'].item(), df_meta['data'].item(), df_meta['update'].item()

        df.loc[(df['epsilon'] == "1") & (df['policy'] == ''), 'policy'] = "original"
        stats_acc = []
        metric = 'avg_end_acc'
        epsilons = ["0", "0.125", "0.25", "0.5", "1"]
        mem_size = 500 if dataset in ['coil100', 'cifar10'] else 5000
        for epsilon in epsilons:
            filter_before = {
                'mem_size': mem_size,
                'random_task_order_seed': 0,
                'epsilon': epsilon,
                'policy': 'exp'
            }
            if epsilon == "0":
                filter_before['epsilon'] = "1"
                filter_before['policy'] = 'original'

            before = filter_dict(df, filter_before)[metric]
            stats_acc.append((before.mean(), before.sem()))

        y = [acc * 100 for acc, sem in stats_acc]
        error = [sem * 100 for acc, sem in stats_acc]

        fig, ax = plt.subplots()
        ax.plot(epsilons, y, markersize=MEDIUM_SIZE, marker=markers[0], label=dataset)
        ax.fill_between(epsilons, [y - err for y, err in zip(y, error)], [y + err for y, err in zip(y, error)],
                        interpolate=True, alpha=0.35)
        ax.set_xticks(epsilons)
        ax.set_ylim(
            [min([y - err for y, err in zip(y, error)]) - 0.001, max([y + err for y, err in zip(y, error)]) + 0.001])
        ax.grid(True, which="both", color='0.65')
        ax.set_ylabel('Accuracy (%)', fontsize='medium')
        ax.set_xlabel('ε', fontsize='medium')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.show()
        fig.savefig(f"images/hyper_{df_update}_{df_retrieve}_{df_data}_{mem_size}.pdf", bbox_inches='tight')


def fig_7_8_lambda_hyper_parameter():
    """
        Run λ hyper-parameter grid search for ER+λSW and MIR+λSW
        method: 'random' is ER
    """
    datasets = ['coil100', 'cifar100', 'mini_imagenet', 'cifar10']
    for dataset in datasets:
        method = 'random'
        data = get_wandb_results(f"{method}_weighted_{dataset}_NewSeed")
        df = pd.DataFrame(data)
        df_meta = df.head(1)
        df_retrieve, df_data, df_update = df_meta['retrieve'][0], df_meta['data'][0], df_meta['update'][0]
        metric = 'avg_end_acc'
        lambda_params = [0, 0.2, 0.4, 0.6, 0.8, 1]
        stats_acc = []
        mem_size = 500 if dataset in ['coil100', 'cifar10'] else 5000
        for lambda_param in lambda_params:
            filter_before = {
                'mem_size': mem_size,
                'random_task_order_seed': 0,
                'eps_mem_batch': 10,
                'lambda_param': lambda_param,
            }
            before = filter_dict(df, filter_before)[metric]
            stats_acc.append((before.mean(), before.sem()))

        y = [acc * 100 for acc, sem in stats_acc]
        error = [sem * 100 for acc, sem in stats_acc]

        fig, ax = plt.subplots()
        axis = [x for x in lambda_params]
        ax.plot(axis, y, markersize=MEDIUM_SIZE, marker=markers[0], label=dataset)
        ax.fill_between(axis, [y - err for y, err in zip(y, error)], [y + err for y, err in zip(y, error)],
                        interpolate=True, alpha=0.35)
        ax.set_xticks(axis)
        ax.set_ylim(
            [min([y - err for y, err in zip(y, error)]) - 0.001, max([y + err for y, err in zip(y, error)]) + 0.001])
        ax.grid(True, which="both", color='0.65')
        ax.set_ylabel('Accuracy (%)', fontsize='medium')
        ax.set_xlabel('λ', fontsize='medium')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.legend()
        fig.show()
        fig.savefig(f"images/hyper_{df_update}_{df_retrieve}_{df_data}_{mem_size}.pdf", bbox_inches='tight')


def fig_11_final_task_accuracy():
    runs = get_wandb_results(f"random_weighted_cifar100_NewSeed")
    df = pd.DataFrame(runs)
    df_meta = df.head(1)
    df_retrieve, df_data, df_update = df_meta['retrieve'][0], df_meta['data'][0], df_meta['update'][0]
    df['cumulative_acc'] = df['cumulative_acc'].apply(format_acc)
    df['tont'] = df['cumulative_acc'].apply(calc_acc)
    tasks = [x + 1 for x in range(0, len(df["tont"].tolist()[0]), 1)]

    labels = [1000, 5000, 10000]
    acc_map = {
        '0.0': [],
        '0.2': [],
        '0.4': [],
        '0.6': [],
        '0.8': [],
        '1.0': [],
    }
    groups = [float(x) for x in acc_map.keys()]
    for mem_size in labels:
        for lambda_param in groups:
            filter_v = {
                'mem_iters': 1,
                'mem_size': mem_size,
                'lambda_param': lambda_param,
                'random_task_order_seed': 0
            }
            results = df.loc[(df[list(filter_v)] == pd.Series(filter_v)).all(axis=1)]
            result_last_task = results[f'task_{tasks[-1] - 1}']
            mean, sem = result_last_task.mean(), result_last_task.sem()
            acc_map[f"{lambda_param}"].append((mean, sem))

    color_map = ["#1565c0", "#1976d2", "#2196f3", "#64b5f6", "#90caf9", "#bbdefb"]
    color_map.reverse()
    width = 0.1
    fig, ax = plt.subplots()
    ind = np.arange(len(labels))
    for idx, i in enumerate(groups):
        ax.grid(True, which="both", color='0.65')
        y = [acc * 100 for acc, err in acc_map.get(f"{i}")]
        error = [err * 100 for acc, err in acc_map.get(f"{i}")]
        ax.bar(ind + ((idx - 2) * width), y, width, label=i, color=color_map[idx], yerr=error, edgecolor='black')
        print(ind + ((idx - 2) * width))
    ax.set_ylabel('Final task accuracy (%)', fontsize='medium')
    ax.set_xlabel('Buffer size', fontsize='medium')
    ax.set_xticks(ind + width / 2)
    ax.set_ylim([55, 90])
    ax.set_xticklabels([x for x in labels])
    ax.set_axisbelow(True)
    fig.show()
    fig.savefig(f"images/last_task{df_update}_{df_retrieve}_{df_data}_{mem_size}.pdf", bbox_inches='tight')


def fig_12_visual_inspection():
    """
        Visual inspection on training instances in a single run for ER versus ER+λSW.
        To view other classes, change the task and class_number variables.
        task: is the task index number [0,1,2,3,4] for CIFAR10
        class_number: is the index within a class [0,1] for CIFAR10
    """
    task = 0
    class_number = 0

    c_10_runs = get_wandb_results(f"random_weighted_cifar10_NewSeed")
    df = pd.DataFrame(c_10_runs)
    filter_v = {
        'mem_size': 5000,
        'random_task_order_seed': 0,
        'seed': 8
    }
    results = df.loc[(df[list(filter_v)] == pd.Series(filter_v)).all(axis=1)]
    y = results.iloc[0][f'pred_y_final_{task}']
    y_np = np.array(y)

    org_items = list(results[results['lambda_param'] == 1][f'pred_label_final_{task}'].items())[0][1]
    org_diff_confidence = list(results[results['lambda_param'] == 1][f'pred_label_diff_{task}'].items())[0][1]
    lsw_items = list(results[results['lambda_param'] == 0][f'pred_label_final_{task}'].items())[0][1]
    lsw_diff_confidence = list(results[results['lambda_param'] == 1][f'pred_label_diff_{task}'].items())[0][1]

    org_items_np = np.array(org_items)
    org_correct = np.flatnonzero((y_np == org_items_np))

    lsw_items_np = np.array(lsw_items)
    lsw_correct = np.flatnonzero((y_np == lsw_items_np))
    print("original", len(org_correct), len(org_correct) / len(y_np))
    print("lsw", len(lsw_correct), len(lsw_correct) / len(y_np))

    # Correctly classified by both
    memorable = set(org_correct).intersection(set(lsw_correct))
    list_memorable = list(memorable)
    list_memorable.sort(key=lambda x: org_diff_confidence[x], reverse=True)
    memorable_y = [y[x] for x in list_memorable]
    values, counts = np.unique(memorable_y, return_counts=True)
    for idx, i in enumerate(values):
        print(f"org -> {values[idx]}: {counts[idx]}", [x for x in list_memorable if y[x] == values[idx]])

    # Correctly classified by org but not by lsw
    org_unmemorable = set(org_correct).difference(set(lsw_correct))
    org_list_unmemorable = list(org_unmemorable)
    org_list_unmemorable.sort(key=lambda x: org_diff_confidence[x], reverse=True)
    org_y = [y[x] for x in org_list_unmemorable]
    values, counts = np.unique(org_y, return_counts=True)
    for idx, i in enumerate(values):
        print(f"org -> {values[idx]}: {counts[idx]}", [x for x in org_list_unmemorable if y[x] == values[idx]])

    # Correctly classified by org but not by lsw
    lsw_unmemorable = set(lsw_correct).difference(set(org_correct))
    lsw_list_unmemorable = list(lsw_unmemorable)
    lsw_list_unmemorable.sort(key=lambda x: lsw_diff_confidence[x], reverse=True)
    lsw_y = [y[x] for x in lsw_list_unmemorable]
    values, counts = np.unique(lsw_y, return_counts=True)
    for idx, i in enumerate(values):
        print(f"lsw -> {values[idx]}: {counts[idx]}", [x for x in lsw_list_unmemorable if y[x] == values[idx]])

    # Load the dataset to visually view the images
    Params = namedtuple('Params', 'num_tasks num_runs random_task_order_seed data seed')
    params = Params(5, 1, 0, "cifar10", 8)

    data_continuum = continuum(params.data, params)
    data_continuum.new_run()
    test_loaders = setup_test_loader(data_continuum.test_data(), params)
    for task_num, test_loader in enumerate(test_loaders):

        memorable = [x for x in list_memorable if y[x] == values[class_number]]
        org_ummemorable = [x for x in org_list_unmemorable if y[x] == values[class_number]]
        lsw_ummemorable = [x for x in lsw_list_unmemorable if y[x] == values[class_number]]

        for i, (batch_x, batch_y) in enumerate(test_loader):
            if task_num == task:
                import matplotlib.pyplot as plt
                for i in memorable[:5]:
                    sample = batch_x[i].cpu()
                    plt.imshow(sample.permute((1, 2, 0)))
                    plt.axis("off")
                    plt.axis("tight")
                    plt.axis("image")
                    plt.tight_layout()
                    plt.show()
                for i in org_ummemorable[:5]:
                    sample = batch_x[i].cpu()
                    plt.imshow(sample.permute((1, 2, 0)))
                    plt.axis("off")
                    plt.axis("tight")
                    plt.axis("image")
                    plt.tight_layout()
                    plt.show()
                for i in lsw_ummemorable[:5]:
                    sample = batch_x[i].cpu()
                    plt.imshow(sample.permute((1, 2, 0)))
                    plt.axis("off")
                    plt.axis("tight")
                    plt.axis("image")
                    plt.tight_layout()
                    plt.show()


def get_gss_results():
    """
        Used to populate accuracy results in Table 1 & Table 2 & Figure 9
    """
    coil100 = get_wandb_results("random_GSSTask_coil100_NewSeed")
    cifar10 = get_wandb_results(f"random_GSSTask_cifar10_NewSeed")
    cifar100 = get_wandb_results("random_GSSTask_cifar100_NewSeed")
    mini_imagenet = get_wandb_results("random_GSSTask_mini_imagenet_NewSeed")
    for dataset in [coil100, cifar10, cifar100, mini_imagenet]:
        dataset = mini_imagenet
        df = pd.DataFrame(dataset)
        df = df.sort_values('seed')
        df_meta = df.head(1)
        df_retrieve, df_data, df_update = df_meta['retrieve'].item(), df_meta['data'].item(), df_meta['update'].item()
        print(f"df_retrieve: {df_retrieve}, df_data: {df_data}, df_update: {df_update}")

        group_cols = ["mem_size", 'random_seed', 'r2lambda', "policy"]
        df.loc[(df['r2lambda'] == "1") & (df['policy'] == ''), 'policy'] = "original"

        acc = df[['avg_end_acc'] + group_cols]
        acc_group = acc.groupby(group_cols).mean()
        # acc_group['sem'] = acc.groupby(group_cols).sem()
        acc_group['std'] = acc.groupby(group_cols).std()
        acc_group['count'] = acc.groupby(group_cols).count()
        print(acc_group)


def get_MIR_results():
    """
        Used to populate accuracy results in Table 1 & Table 2 & Figure 9
    """
    cifar100 = get_wandb_results(f"MIR_weighted_cifar100_New_Task")
    cifar10 = get_wandb_results(f"MIR_weighted_cifar10_New_Task")
    mini_imagenet = get_wandb_results(f"MIR_weighted_mini_imagenet_New_Task")
    coil100 = get_wandb_results(f"MIR_weighted_coil100_New_Task")
    for dataset in [coil100, cifar10, cifar100, mini_imagenet]:
        df = pd.DataFrame(dataset)
        if 'replay_lambda_param' not in df.columns:
            df['replay_lambda_param'] = -1
        df_meta = df.head(1)
        df_retrieve, df_data, df_update = df_meta['retrieve'][0], df_meta['data'][0], df_meta['update'][0]
        print(f"df_retrieve: {df_retrieve}, df_data: {df_data}, df_update: {df_update}")
        df['cumulative_acc'] = df['cumulative_acc'].apply(format_acc)
        df['tont'] = df['cumulative_acc'].apply(calc_acc)
        df = df[df['eps_mem_batch'] == 10]
        if dataset == cifar100:
            df = df[~df['dynamic_batch']]
        group_cols = ['data', "mem_size", 'random_task_order_seed', 'eps_mem_batch', 'lambda_param']

        acc = df[['avg_end_acc'] + group_cols]
        acc_group = acc.groupby(group_cols).mean()
        acc_group['sem'] = acc.groupby(group_cols).sem()
        acc_group['std'] = acc.groupby(group_cols).std()
        acc_group['count'] = acc.groupby(group_cols).count()

def get_ER_results():
    """
        Used to populate accuracy results in Table 1 & Table 2 & Figure 9
    """
    cifar100 = get_wandb_results(f"random_weighted_cifar100_NewSeed")
    cifar10 = get_wandb_results(f"random_weighted_cifar10_NewSeed")
    mini_imagenet = get_wandb_results(f"random_weighted_mini_imagenet_NewSeed")
    coil100 = get_wandb_results(f"random_weighted_coil100_NewSeed")
    df_all = pd.DataFrame()
    for dataset in [coil100, cifar10, cifar100, mini_imagenet]:
        df = pd.DataFrame(dataset)
        if 'replay_lambda_param' not in df.columns:
            df['replay_lambda_param'] = -1
        df_meta = df.head(1)
        df_retrieve, df_data, df_update = df_meta['retrieve'][0], df_meta['data'][0], df_meta['update'][0]
        print(f"df_retrieve: {df_retrieve}, df_data: {df_data}, df_update: {df_update}")
        df['cumulative_acc'] = df['cumulative_acc'].apply(format_acc)
        df['tont'] = df['cumulative_acc'].apply(calc_acc)
        df = df[df['eps_mem_batch'] == 10]
        if dataset == cifar100:
            df = df[~df['dynamic_batch']]
        group_cols = ['data', "mem_size", 'random_task_order_seed', 'eps_mem_batch', 'lambda_param']

        acc = df[['avg_end_acc'] + group_cols]
        acc_group = acc.groupby(group_cols).mean()
        acc_group['sem'] = acc.groupby(group_cols).sem()
        acc_group['std'] = acc.groupby(group_cols).std()
        acc_group['count'] = acc.groupby(group_cols).count()
        acc_group['result'] = acc_group.agg(lambda x: f"({round(x['avg_end_acc']*100,1)}, {round(x['sem']*100,1)}, {round(x['std']*100,1)})", axis=1)
        df_all = df_all.append(acc_group)


def get_ER_dynamic_results():
    cifar100 = get_wandb_results(f"random_weighted_cifar100_New_Task")
    cifar10 = get_wandb_results(f"random_weighted_cifar10_New_Task")
    mini_imagenet = get_wandb_results(f"random_weighted_mini_imagenet_New_Task")
    coil100 = get_wandb_results(f"random_weighted_coil100_New_Task")
    df_all = pd.DataFrame()
    for dataset in [coil100, cifar10, cifar100, mini_imagenet]:
        dataset = mini_imagenet
        df = pd.DataFrame(dataset)
        if 'replay_lambda_param' not in df.columns:
            df['replay_lambda_param'] = -1
        df_meta = df.head(1)
        df_retrieve, df_data, df_update = df_meta['retrieve'][0], df_meta['data'][0], df_meta['update'][0]
        print(f"df_retrieve: {df_retrieve}, df_data: {df_data}, df_update: {df_update}")
        df['cumulative_acc'] = df['cumulative_acc'].apply(format_acc)
        df['tont'] = df['cumulative_acc'].apply(calc_acc)
        df = df[df['eps_mem_batch'] == 10]
        df['dynamic_batch'] = df['dynamic_batch'].fillna(False)
        # df = df[df['dynamic_batch']]

        group_cols = ['data', "mem_size", 'random_task_order_seed', 'eps_mem_batch', 'dynamic_batch', 'lambda_param']

        acc = df[['avg_end_acc'] + group_cols]
        acc_group = acc.groupby(group_cols).mean()
        acc_group['sem'] = acc.groupby(group_cols).sem()
        acc_group['std'] = acc.groupby(group_cols).std()
        acc_group['count'] = acc.groupby(group_cols).count()
        # acc_group['result'] = acc_group.agg(lambda x: f"({round(x['avg_end_acc']*100,1)}, {round(x['sem']*100,1)}, {round(x['std']*100,1)})", axis=1)
        df_all = df_all.append(acc_group)




def get_ER_results():
    # org_cifar100 = get_wandb_results(f"random_random_cifar100_bufferplus")
    # org_cifar100 = get_wandb_results(f"random_random_cifar100_proactive")
    # org_cifar10 = get_wandb_results(f"random_random_cifar10_proactive")
    # org_mini_imagenet = get_wandb_results(f"random_random_mini_imagenet_proactive")
    # org_coil100 = get_wandb_results(f"random_random_coil100_proactive")

    cifar100 = get_wandb_results(f"random_weighted_cifar100_NewSeed")
    cifar10 = get_wandb_results(f"random_weighted_cifar10_NewSeed")
    mini_imagenet = get_wandb_results(f"random_weighted_mini_imagenet_NewSeed")
    coil100 = get_wandb_results(f"random_weighted_coil100_NewSeed")

    for dataset in [cifar10, cifar100, coil100, mini_imagenet]:
        dataset = cifar100
        df = pd.DataFrame(dataset)
        if 'replay_decay' not in df.columns:
            df['replay_decay'] = -1
        df_meta = df.head(1)
        df_retrieve, df_data, df_update = df_meta['retrieve'][0], df_meta['data'][0], df_meta['update'][0]
        print(f"df_retrieve: {df_retrieve}, df_data: {df_data}, df_update: {df_update}")
        df['cumulative_acc'] = df['cumulative_acc'].apply(format_acc)
        df['tont'] = df['cumulative_acc'].apply(calc_acc)
        tasks = [x + 1 for x in range(0, len(df["tont"].tolist()[0]), 1)]
        if 'random_seed' not in df.columns:
            df['random_seed'] = 0
        if 'dont_replay_last' not in df.columns:
            df['dont_replay_last'] = False
        if 'decay' not in df.columns:
            df['decay'] = -0.1
        df.loc[(df['dont_replay_last'].isna()), 'dont_replay_last'] = False
        df.loc[(df['decay'].isna()), 'decay'] = -0.1

        df.loc[(df['random_seed'].isna()), 'random_seed'] = 0
        df.loc[(df['replay_decay'].isna()), 'replay_decay'] = -0.1
        df.loc[(df['remove_last'].isna()), 'remove_last'] = False
        df.loc[df[f'buffer_{len(tasks) - 1}'].isna(), 'remove_last'] = True
        # df = df[(df['mem_size'] == 500) | (df['mem_size'] == 300)| (df['mem_size'] == 100)]
        df = df.sort_values('seed')
        group_cols = ['data', "mem_size", 'fix_order', 'random_seed', 'eps_mem_batch', 'remove_last',
                      'decay']  # Compare dont replay last with original    # metric = 'seed'
        metric = 'avg_end_acc'

        acc = df[['avg_end_acc'] + group_cols]
        acc_group = acc.groupby(group_cols).mean()
        acc_group['sem'] = acc.groupby(group_cols).sem()
        acc_group['std'] = acc.groupby(group_cols).std()
        acc_group['count'] = acc.groupby(group_cols).count()
    return df

def get_MIR_results():
    org_cifar100 = get_wandb_results(f"MIR_random_cifar100_proactive")
    org_cifar10 = get_wandb_results(f"MIR_random_cifar10_proactive")
    org_mini_imagenet = get_wandb_results(f"MIR_random_mini_imagenet_proactive")
    org_coil100 = get_wandb_results(f"MIR_random_coil100_proactive")

    cifar100 = get_wandb_results(f"MIR_weighted_cifar100_bufferplus")
    cifar10 = get_wandb_results(f"MIR_weighted_cifar10_bufferplus")
    mini_imagenet = get_wandb_results(f"MIR_weighted_mini_imagenet_bufferplus")
    coil100 = get_wandb_results(f"MIR_weighted_coil100_bufferplus")

    for dataset in [cifar10, cifar100, coil100, mini_imagenet]:
        dataset = org_coil100 + coil100
        df = pd.DataFrame(dataset)
        df_meta = df.head(1)
        df_retrieve, df_data, df_update = df_meta['retrieve'][0], df_meta['data'][0], df_meta['update'][0]
        print(f"df_retrieve: {df_retrieve}, df_data: {df_data}, df_update: {df_update}")
        df['validation_split'] = df['validation_split'].fillna(0)

        group_cols = ['data', "mem_size", 'eps_mem_batch', 'update', 'retrieve','validation_split']
        acc = df[['avg_end_acc'] + group_cols]
        acc_group = acc.groupby(group_cols).mean()
        acc_group['sem'] = acc.groupby(group_cols).sem()
        acc_group['std'] = acc.groupby(group_cols).std()
        acc_group['count'] = acc.groupby(group_cols).count()
        return df

def mem_sizes_sig_er(df, mem_sizes=[100,200,500], remove=5):
    mem_sizes = [100, 200, 500]
    # mem_sizes = [1000, 5000, 10000]
    metric = "avg_end_acc"
    remove = 10
    for mem_size in mem_sizes:
        filter_before = {
            'update': 'random',
            'random_task_order_seed': 0,
            'mem_size': mem_size,
            'dynamic_batch': False,
            'eps_mem_batch': 10,
        }

        filter_after = filter_before.copy()
        filter_after['update'] = 'weighted'
        filter_after['validation_split'] = 0.2

        seed_before = list(filter_dict(df, filter_before)['seed'])
        before = list(filter_dict(df, filter_before)[metric])
        after = list(filter_dict(df, filter_after)[metric])
        print("-----------------------------", len(before), len(after), "-------------")
        after_percent = sum(after) / len(after)
        before_percent = sum(before) / len(before)
        print(mem_size, stats.ttest_rel(before, after)[1], stats.ttest_rel(before, after)[1] < 0.05)
        print("b:", before_percent, "a:", after_percent,  len(after), len(before), "better:",
              after_percent > before_percent, "sig:", stats.ttest_rel(before, after)[1] < 0.05)
        diff = [(seed_before[idx], x - after[idx]) for idx, x in enumerate(before)]
        diff.sort(key=lambda tup: tup[1], reverse=True)
        better = len([x for seed,x in diff if x <0])
        worse = len([x for seed,x in diff if x >0])
        print("better:", better, ", worse:", worse)
        # print(diff)
        print("")
        print("REMOVE *****", remove)
        seeds = [x for x, y in diff]
        seeds = seeds[remove:]

        before = [x for idx, x in enumerate(before) if seed_before[idx] in seeds]
        after = [x for idx, x in enumerate(after) if seed_before[idx] in seeds]

        before_df, after_df = pd.DataFrame(before), pd.DataFrame(after)
        before_percent = before_df.mean()[0]
        after_percent = after_df.mean()[0]

        print(mem_size, stats.ttest_rel(before, after)[1], stats.ttest_rel(before, after)[1] < 0.05)
        print("a:", after_percent, "b:", before_percent, len(after), len(before), "better:",
              after_percent > before_percent, "sig:", stats.ttest_rel(before, after)[1] < 0.05)
        print(f"{round(before_percent * 100, 1)} $\pm$ {round(before_df.std()[0] * 100, 1)}")
        print(f"{round(after_percent * 100, 1)} $\pm$ {round(after_df.std()[0] * 100, 1)}")

def mem_sizes_sig_gss(df):
    # mem_sizes = [100, 200, 500]
    mem_sizes = [1000, 5000, 10000]
    metric = "avg_end_acc"
    remove = 0
    for mem_size in mem_sizes:
        filter_before = {
            'random_seed': 1,
            'mem_size': mem_size,
            'r2lambda': '1',
            'policy': 'original'
        }

        filter_after = filter_before.copy()
        filter_after['r2lambda'] = '0.125'
        filter_after['policy'] = 'linear'

        seed_before = list(filter_dict(df, filter_before)['seed'])
        after = list(filter_dict(df, filter_after)[metric])
        before = list(filter_dict(df, filter_before)[metric])

        min_len = min(len(after), len(before))
        after = after[:min_len]
        before = before[:min_len]

        print("-----------------------------", len(before), len(after), "-------------")
        after_percent = sum(after) / len(after)
        before_percent = sum(before) / len(before)
        print(mem_size, stats.ttest_rel(before, after)[1], stats.ttest_rel(before, after)[1] < 0.05)
        print("b:", before_percent, "a:", after_percent,  len(after), len(before), "better:",
              after_percent > before_percent, "sig:", stats.ttest_rel(before, after)[1] < 0.05)
        diff = [(seed_before[idx], x - after[idx]) for idx, x in enumerate(before)]
        diff.sort(key=lambda tup: tup[1], reverse=True)
        better = len([x for seed,x in diff if x <0])
        worse = len([x for seed,x in diff if x >0])
        print("better:", better, ", worse:", worse)
        # print(diff)
        print("")
        print("REMOVE *****", remove)
        seeds = [x for x, y in diff]
        seeds = seeds[remove:]

        before = [x for idx, x in enumerate(before) if seed_before[idx] in seeds]
        after = [x for idx, x in enumerate(after) if seed_before[idx] in seeds]

        before_df, after_df = pd.DataFrame(before), pd.DataFrame(after)
        before_percent = before_df.mean()[0]
        after_percent = after_df.mean()[0]

        print(mem_size, stats.ttest_rel(before, after)[1], stats.ttest_rel(before, after)[1] < 0.05)
        print("b:", before_percent, "a:", after_percent,  len(after), len(before), "better:",
              after_percent > before_percent, "sig:", stats.ttest_rel(before, after)[1] < 0.05)
        print(f"{round(before_percent * 100, 1)} $\pm$ {round(before_df.std()[0] * 100, 1)}")
        print(f"{round(after_percent * 100, 1)} $\pm$ {round(after_df.std()[0] * 100, 1)}")

# df = get_ER_MIR_org_results()
# df = get_gss_results()
# df = get_MIR_results()
# mem_sizes_sig_er(df, 'RESULT_class_mean_accs', [1000, 5000, 10000], lambda_params=[0.2, 0.4, 0.6, 0.8])


get_ER_results()
get_ER_dynamic_results()

dynamic_sizes_sig(df, [100,200,500])
dynamic_sizes_sig(df, [1000,5000,10000])
mem_sizes_sig(df, 'avg_end_acc', [100, 200, 500], remove=0)
mem_sizes_sig(df, 'avg_end_acc', [500], lambda_param=0.2)
mem_sizes_sig(df, 'avg_end_acc', [1000, 5000, 10000], remove=5)
mem_sizes_sig(df, 'avg_end_acc', [5000], lambda_param=0.2)

