import time

import pandas as pd
import seaborn as sns

from lib.models import MPNNModel
from lib.engine import run
from lib.datasets import train_loader, val_loader, test_loader, std

results = {}
results_df = pd.DataFrame(columns=["Test MAE", "Val MAE", "Epoch", "Model"])

model = MPNNModel(num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1)
model_name = type(model).__name__
best_val_error, test_error, train_time, perf_per_epoch = run(
    model,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    std,
    n_epochs=100
)

results[model_name] = (best_val_error, test_error, train_time)
print(results)

df_temp = pd.DataFrame(perf_per_epoch, columns=[
                       "Test MAE", "Val MAE", "Epoch", "Model"])
results_df = results_df.append(df_temp, ignore_index=True)

results_df.to_csv(f'./results/run-{time.time()}')

p = sns.lineplot(x="Epoch", y="Val MAE", hue="Model", data=results_df)
p.set(ylim=(0, 2))
