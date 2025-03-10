import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_preprocessed.pickle")
# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"] == 1] 
plt.plot(set_df["acc_y"].reset_index(drop=True))
# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
df["label"].unique()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True),label = label)
    plt.legend()
    plt.show()
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df[(df["label"] == "row") & (df["participant"] == "C")].reset_index(drop=True)

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_xlabel(" Number of Samples")
ax.set_ylabel(" Acc_y ")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

# participant_df = df[df["label"] == "bench"].reset_index()
# participant_df = participant_df.sort_values(by=["participant"])
participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()
fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_xlabel(" Number of Samples")
ax.set_ylabel(" Acc_y ")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"

all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop=True)
fig, ax = plt.subplots()
all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax = ax)
ax.set_xlabel(" Number of Samples")
ax.set_ylabel(" Acc_y ")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (df.query(f"label == '{label}'")
                       .query(f"participant == '{participant}'")
                       .reset_index(drop=True))
        if len(all_axis_df) > 0:
            
            fig, ax = plt.subplots()
            all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax = ax)
            ax.set_xlabel(" Number of Samples")
            ax.set_ylabel(" Acc_y ")
            plt.legend()




labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (df.query(f"label == '{label}'")
                       .query(f"participant == '{participant}'")
                       .reset_index(drop=True))
        if len(all_axis_df) > 0:
            
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x","gyr_y","gyr_z"]].plot(ax = ax)
            ax.set_xlabel(" Number of Samples")
            ax.set_ylabel(" gyr_y ")
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "squat"
participant = "A"

combined_plot_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop=True)
fig, ax = plt.subplots(nrows=2, sharex = True, figsize = (20,10) )
combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax = ax[0])
combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax = ax[1])
ax[0].legend(loc = "upper right", bbox_to_anchor = (0.5,1.15) ,title = "Accelerometer", fancybox = True, shadow = True, ncols = 3)
ax[1].legend(loc = "upper right", bbox_to_anchor = (0.5,1.15), title = "Gyrometer", fancybox = True, shadow = True, ncols = 3)
ax[1].set_xlabel("Samples")
plt.show()

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (df.query(f"label == '{label}'")
                       .query(f"participant == '{participant}'")
                       .reset_index(drop=True))
        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex = True, figsize = (20,10) )
            combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax = ax[0])
            combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax = ax[1])
            ax[0].legend(loc = "upper right", bbox_to_anchor = (0.5,1.15) ,title = "Accelerometer", fancybox = True, shadow = True, ncols = 3)
            ax[1].legend(loc = "upper right", bbox_to_anchor = (0.5,1.15), title = "Gyrometer", fancybox = True, shadow = True, ncols = 3)
            ax[1].set_xlabel("Samples")
            plt.savefig(f"../../reports/figures/{label}_{participant}.png")
            plt.show()

