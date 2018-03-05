import matplotlib.pyplot as plt
import matplotlib


def scatter3d_visualize(X, y, title=""):
    # Soal A nomer 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(X)):
        xs = X[i][0]
        ys = X[i][1]
        zs = X[i][2]

        if y[i] == 0:
            bundar = ax.scatter(xs, ys, zs, c="r", marker="o", label="Kelas 0")
        elif y[i] == 1:
            kotak = ax.scatter(xs, ys, zs, c="g", marker="s", label="Kelas 1")
        elif y[i] == 2:
            segitiga = ax.scatter(
                xs, ys, zs, c="b", marker="^", label="Kelas 2")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.legend(handles=[bundar, kotak, segitiga])
    plt.title(title)
    plt.show()


def count_accuracy(y_pred, y_test):
    correctness = 0
    for yp, yt in zip(y_pred, y_test):
        if yp == yt:
            correctness += 1
    return correctness / len(y_pred) * 100
