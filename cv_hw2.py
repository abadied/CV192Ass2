
import numpy as np
import matplotlib.pyplot as plt


def G(row_s, Temp):
    return np.exp((1/Temp)*np.matmul(row_s[:-1], row_s[1:]))


def F(row_s, row_t, Temp):
    return np.exp((1/Temp)*np.matmul(row_s, row_t))


def Z_3():
    Temp = {1, 1.5, 2}
    X_values = {-1, 1}
    Z = np.zeros(len(Temp))
    for i, temp in enumerate(Temp):
        for x1_1 in X_values:
            for x1_2 in X_values:
                for x2_1 in X_values:
                    for x2_2 in X_values:
                        Z[i] += np.exp((1/temp)*(x1_1*x1_2+x1_1*x2_1+x1_2*x2_2+x2_1*x2_2))
    return Z
  

def Z_4():
    Temp = {1, 1.5, 2}
    X_values = {-1,1}
    Z = np.zeros(len(Temp))
    for i,temp in enumerate(Temp):
      for x1_1 in X_values:
          for x1_2 in X_values:
            for x1_3 in X_values:
              for x2_1 in X_values:
                  for x2_2 in X_values:
                    for x2_3 in X_values:
                      for x3_1 in X_values:
                        for x3_2 in X_values:
                          for x3_3 in X_values:
                            Z[i]+=np.exp((1/temp)*
                                         (x1_1*x1_2+x1_1*x2_1+
                                          x1_2*x2_2+x1_2*x1_3+
                                          x1_3*x2_3+
                                          x2_1*x3_1+x2_1*x2_2+
                                          x2_2*x3_2+x2_2*x2_3+
                                          x2_3*x3_3+
                                          x3_1*x3_2+
                                          x3_2*x3_3))
    return Z


def y2row(y, width=8):
    if not 0 <= y <= (2 ** width) - 1:
        raise ValueError(y)
    my_str = np.binary_repr(y, width=width)
    # my_list = map(int,my_str) # Python 2
    my_list = list(map(int, my_str)) # Python 3
    my_array = np.asarray(my_list)
    my_array[my_array == 0] -= 1
    row = my_array
    return row


def Z_5():
    Temp = {1, 1.5, 2}
    Y_values = {0, 1, 2, 3}
    Z = np.zeros(len(Temp))
    for i, temp in enumerate(Temp):
      for y_1 in Y_values:
        row_1 = y2row(y_1,2)
        for y_2 in Y_values:
            row_2 = y2row(y_2,2)

            G_1,G_2 = G(row_1,temp),G(row_2,temp)
            F1_2 = F(row_1, row_2, temp)
            Z[i] += G_1*G_2*F1_2
    return Z


def Z_6():
    Temp = {1, 1.5, 2}
    Y_values = np.arange(8)
    Z = np.zeros(len(Temp))
    for i, temp in enumerate(Temp):
      for y_1 in Y_values:
          row_1 = y2row(y_1,3)
          for y_2 in Y_values:
              row_2 = y2row(y_2,3)
              for y_3 in Y_values:
                  row_3 = y2row(y_3, 3)
                  G_1, G_2, G_3 = G(row_1,temp), G(row_2, temp), G(row_3, temp)
                  F1_2, F2_3 = F(row_1, row_2, temp), F(row_2, row_3, temp)
                  Z[i] += G_1*G_2*G_3*F1_2*F2_3
    return Z


def calc_T(temp, width):
    t_mat = np.zeros((width + 1, 2 ** width))
    t_mat[0] = np.ones((2 ** width,))

    for i in range(1, width):
        for y_i in range(2 ** width):
            first_row = y2row(y_i, width)
            G_y_i = G(first_row, temp)
            for y_j in range(2 ** width):
                sec_row = y2row(y_j, width)
                F_y_ij = F(first_row, sec_row, temp)
                t_mat[i, y_j] += t_mat[i - 1, y_i] * G_y_i * F_y_ij

    for y_i in range(2 ** width):
        first_row = y2row(y_i, width)
        G_y_i = G(first_row, temp)
        t_mat[width] += t_mat[width - 1, y_i] * G_y_i

    return t_mat


def calc_p(t_mat, temp, width):

    Z = t_mat[width, 0]
    p_mat = np.ndarray((width + 1, 2 ** width, 2 ** width))

    for y_i in range(2 ** width):
        row_i = y2row(y_i, width)
        p_mat[width, y_i] = t_mat[width - 1, y_i] * G(row_i, temp) / Z

    for i in range(width - 1, 0, -1):
        for y_i in range(2 ** width):
            row_i = y2row(y_i, width)
            G_i = G(row_i, temp)
            for y_j in range(2 ** width):
                row_j = y2row(y_j, width)
                F_j = F(row_i, row_j, temp)
                p_mat[i, y_i, y_j] = t_mat[i - 1, y_i] * G_i * F_j / t_mat[i, y_j]

    return p_mat[1:]


def sample(p, width):
    y = np.ndarray(width, dtype=int)
    y[width-1] = np.random.choice(2 ** width, p=p[width - 1, :, 0])
    for i in range(width - 2, -1, -1):
        y[i] = np.random.choice(2 ** width, p=p[i, :, y[i + 1]])
    return y


def ex_7(temps, p_mat, width):
    samples_per_temp = 10

    f = plt.figure(figsize=(15, 6))
    f.suptitle("ex7")
    for idx in range(0, len(temps) * samples_per_temp):
        temp = temps[int(idx / samples_per_temp)]
        plt.subplot(3, 10, idx+1)
        y = np.array([y2row(y_k) for y_k in sample(p_mat[temp], width)])
        plt.imshow(y, cmap="Greys", interpolation="None")

    plt.subplots_adjust(hspace=0.7)  # make subplots farther from each other.
    plt.show()


def ex_8(p, width):
    for temp in [1., 1.5, 2.]:
        samples = []
        for i in range(10000):
            y = np.array([y2row(y_k) for y_k in sample(p[temp], width)])
            samples.append(y)

        x11_x22 = [im[0, 0] * im[1, 1] for im in samples]
        x11_x88 = [im[0, 0] * im[7, 7] for im in samples]

        e_1122 = np.sum(x11_x22, dtype=float) / 10000
        e_1188 = np.sum(x11_x88, dtype=float) / 10000

        print("Temp {}:".format(temp))
        print("E(X11 * X22): {}".format(e_1122))
        print("E(X11 * X88): {}".format(e_1188))


def main():
    width = 8
    temps = [1., 1.5, 2.]
    t_mat = {temp: calc_T(temp, width) for temp in temps}
    p_mat = {temp: calc_p(t_mat[temp], temp, width) for temp in temps}
    ex_7(temps, p_mat, width)
    ex_8(p_mat, width)


if __name__ == '__main__':
    main()
