import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


if __name__ == '__main__':
    data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    data = pd.DataFrame(data, columns=['x', 'y'])
    # for col in 'xy':
    #     sns.kdeplot(data[col], shade=True)
    sns.kdeplot(data['x'], data['y'])
    plt.show()
