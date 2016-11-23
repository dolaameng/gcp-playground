import calendar
import time
import numpy as np
import pandas as pd

n_samples = 3000
groups = np.array(["A", "B", "C"])

start = time.strptime("1 1 2015", "%m %d %Y")
end = time.strptime("11 1 2016", "%m %d %Y")
start = calendar.timegm(start)*1e9
end = calendar.timegm(end)*1e9

ts = np.random.randint(start, end+1, n_samples)
ts = pd.to_datetime(ts)
group = groups[np.random.randint(0, len(groups), n_samples)]
success = np.random.randint(0, 2, n_samples).astype(np.bool)

df = pd.DataFrame({"time": ts, "group": group, "success": success})
df = df[["time", "group", "success"]]
i = df[df["group"].isin(["B", "C"]) & df.success].sample(500).index
df.loc[i, "success"] = False
df.to_csv("static/ab_result.csv", header=True, index=False)