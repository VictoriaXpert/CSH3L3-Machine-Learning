from fun import load_data, visualize_data, write_to_file
import KMeans as kmeans

if __name__ == '__main__':
    # Train model K-Means dan visualisasi hasil prediksi train set
    X_train = load_data("TrainsetTugas2.txt")
    centroids, clusters = kmeans.train(5, X_train, 300)
    y_train = kmeans.predict(X_train, centroids)
    visualize_data(X_train, y_train, centroids)
    # print(kmeans.getSumSquaredError(centroids, clusters))

    # Prediksi class test set dan menuliskan ke file hasil.csv
    X_test = load_data("TestsetTugas2.txt")
    y_test = kmeans.predict(X_test, centroids)
    write_to_file("hasil_running_nanang.csv", y_test)
    visualize_data(X_test, y_test, centroids)
