import pandas as pd
import matplotlib.pyplot as plt



test_data = pd.read_csv("file_path.csv")


def plot_data(data, save_path, title):

    timesteps = data['Episode'] * 1000
    y_mean = data.iloc[:, 1:].mean(axis=1)
    y_std = data.iloc[:, 1:].std(axis=1)
    window = 1000
    # calculate rolling average
    rolling_avg = y_mean.rolling(window=window).mean()


    plt.figure(figsize=(20, 12))
    plt.plot(timesteps, rolling_avg, label=f"Rolling Average ({window})", color="#3380FF")
    plt.fill_between(timesteps, y_mean - y_std, y_mean + y_std, color="#66B2FF", alpha=0.3)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(save_path)
    plt.show()
    plt.close()
    return plt




plot_data(test_data, 'file_path.png', 'Humanoid Stand Up Rewards')

