import pandas as pd

from AL_agents.visualize_objective_function.plot_objective_function import plot_objective_function

filenames = ["df objective model_UCI.csv", "df objective model_checkerboard.csv", "df objective model_Vision.csv",
             "df objective model_bAbI.csv"]
filename_df = "evaluations/" + filenames[3]
directory = filename_df[:-4]
df: pd.DataFrame = pd.read_csv(filename_df)
all_features = ["Uncertainty", "Diversity", "Representative", "Uncertainty_Diversity"]

# plot over single feature
for axes_index in range(len(all_features)):
    feature_name = all_features[axes_index]
    filename_to_save = directory+f"/{feature_name}.png"
    plot_objective_function(df, [feature_name], filename_to_save, plot=False)

# plot over two features
for axes_index_1 in range(len(all_features)):
    for axes_index_2 in range(axes_index_1+1, len(all_features)):
        feature_names = [all_features[i] for i in [axes_index_1, axes_index_2]]
        filename_to_save = directory+f"/{' '.join(feature_names)}.png"
        plot_objective_function(df, feature_names, filename_to_save, plot=False)

# plot over three features
feature_names = [all_features[i] for i in [0, 1, 2]]
filename_to_save = directory+f"/{' '.join(feature_names)}.png"
plot_objective_function(df, feature_names, filename_to_save, plot=False)
