import matplotlib.pyplot as plt
import pandas as pd

all_betas = ['Uncertainty', 'Diversity', 'Representative', 'Uncertainty_Diversity']
axes_ = [all_betas[i] for i in [0,1,2]]

filenames = ["df objective model_UCI.csv",
             "df objective model_checkerboard.csv",
             "df objective model_Vision.csv",
             "df objective model_bAbI"]
filename = "evaluations/"+filenames[0]

df = pd.read_csv(filename)
if True:
    sum = df[all_betas].sum(axis=1).add(1)
    for beta_str in all_betas:
        df[beta_str] = df[beta_str] / sum

fig = plt.figure()
x = df[axes_[0]]
if len(axes_) >= 2:
    y = df[axes_[1]]
c = df['accuracy']
if len(axes_) == 3:
    ax = fig.add_subplot(111, projection='3d')
    z = df[axes_[2]]
    img = ax.scatter(x, y, z, c=c, cmap='RdYlGn')
    ax.set_xlabel(axes_[0])
    ax.set_ylabel(axes_[1])
    ax.set_zlabel(axes_[2])
    fig.colorbar(img)
elif len(axes_) == 2:
    ax = fig.add_subplot()
    img = ax.scatter(x, y, c=c, cmap='RdYlGn')
    ax.set_xlabel(axes_[0])
    ax.set_ylabel(axes_[1])
    fig.colorbar(img)
elif len(axes_) == 1:
    ax = fig.add_subplot()
    img = ax.scatter(x, c)
    ax.set_xlabel(axes_[0])
    ax.set_ylabel('accuracy')


plt.title(f"Accuracy dependent on beta values")


plt.show()
