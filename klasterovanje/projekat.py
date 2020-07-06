import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore

if __name__ == '__main__':
    data = pd.read_csv('credit_card_data.csv')  # ucitavanje csv fajla
    # print(data.head())
    # print(data.describe())
    # print(data.isnull().sum()) |  uocili smo da u kolonama MINIMUM_PAYMENTS i CREDIT_LIMIT ima mnogo null vrednosti
    data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].mean(), inplace=True)  # zamenjivanje null vrednosti prosecnim vrednostima za datu kolonu
    data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean(), inplace=True)
    # print(data.isnull().sum())
    data.drop('CUST_ID', axis=1, inplace=True)  # ignorisemo datu kolonu jer nisu bitni za analizu
    data_scaled = data.apply(zscore)    # skaliranje kako bi podaci bili u odredenom rangu
    # print(data_scaled.head())

    cluster_range = range(1, 15)    # elbow metoda za odredivanje broja klastera
    cluster_errors = []
    for i in cluster_range:
        clusters = KMeans(i)
        clusters.fit(data_scaled)
        labels = clusters.labels_
        centroids = clusters.cluster_centers_, 3
        cluster_errors.append(clusters.inertia_)
    clusters_df = pd.DataFrame({'num_clusters': cluster_range, 'cluster_errors': cluster_errors})
    # print(clusters_df)

    f, ax = plt.subplots(figsize=(15, 6))   # graficki prikaz date metode
    plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker='o')
    # plt.show()

    kmean = KMeans(4)   # izabran k=4 na osnovu grafickog prikaza
    kmean.fit(data_scaled)
    labels = kmean.labels_
    clusters = pd.concat([data, pd.DataFrame({'cluster': labels})], axis=1) # konverzija podataka za panda biblioteku
    # print(clusters.head())

    for c in clusters:  # graficki prikaz svake kolone po klasterima
        grid = sns.FacetGrid(clusters, col='cluster')
        grid.map(plt.hist, c)
    # plt.show()
    # print(clusters.groupby('cluster').mean())

    pca = PCA(n_components=2)   # PCA algoritam za pretvaranje podataka u 2 dimenzije za vizuelizaciju
    principalComponents = pca.fit_transform(data_scaled)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    # print(principalDf.head(2))
    finalDf = pd.concat([principalDf, pd.DataFrame({'cluster': labels})], axis=1)
    # print(finalDf.head())

    plt.figure(figsize=(15, 10))    # vizuelni prikaz klastera u 2D prostoru
    ax = sns.scatterplot(x="principal component 1", y="principal component 2", hue="cluster", data=finalDf,
                         palette=['red', 'blue', 'green', 'yellow'])
    plt.show()
