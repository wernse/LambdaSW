# import neptune.new as neptune
# neptune_run = neptune.init(
#     project="wwon129/continual",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNTdmOThkZS00NjNiLTQ1OGEtODhkMi0zZGNiZTM2YjI5M2EifQ==",
# )  # your credentials
#

# Import comet_ml at the top of your file
# from comet_ml import Experiment
#
# # Create an experiment with your api key
# experiment = Experiment(
#     api_key="y5Jeu3ca4j0EZX7ixIWxeoGsm",
#     project_name="general",
#     workspace="wwon129",
# )
import time

import numpy as np
import matplotlib.pyplot as plt

# Add the following two lines to your code, to have ClearML automatically log your experiment
# from clearml import Task

# task = Task.init(project_name='My Project', task_name='My Experiment4')
# experiment = task.get_logger()
# # Create a plot using matplotlib, or you can also use plotly
# plt.scatter(np.random.rand(50), np.random.rand(50), s=1, c=np.random.rand(50), alpha=0.5)
# # Plot will be reported automatically to clearml
# plt.show()

# # Report some scalars
# for i in range(100):
#     time.sleep(5)
#     task.get_logger().report_scalar(title="mem", series="mem0", value=i * 2, iteration=i)
#     task.get_logger().report_scalar(title="mem", series="mem1", value=i * 2, iteration=i)
#     task.get_logger().report_scalar(title="graph title", series="linear", value=i * 2, iteration=i)
