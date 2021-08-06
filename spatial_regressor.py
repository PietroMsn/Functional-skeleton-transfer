import numpy as np
import tensorflow as tf

optimizer = tf.optimizers.Adam(0.1)
epochs = 1000
Rs = []

def get_GTR_loc(F, X, J):
    R_loc = np.zeros((len(J), len(X)))
    for i in range(len(J)):
        newF = np.tile(F[i, :], (3, 1)).T

        temp = np.multiply(newF, X).T

        pinv_temp = np.linalg.pinv(temp)
        R_loc[i, :] = np.matmul(pinv_temp, J[i, :].T)

    return R_loc

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

# function to minimize
def loss_spatial(R, X, J, F):

    eq = np.ones((len(J), 1))
    en = np.ones((len(X), 1))

    a, b, c = 1, 1000, 10  # coefficients for each optimization term

    loss1 = a * tf.reduce_sum(tf.reduce_sum(tf.square(tf.linalg.matmul(R, X) - J)))

    lossF = 100 * tf.reduce_sum(((np.abs(F - 1) * R)) ** 2)

    loss2b = b * tf.reduce_sum(tf.reduce_sum(tf.square(tf.multiply(R, F) - R)))

    loss3 = c * tf.reduce_sum(tf.reduce_sum(tf.square(tf.linalg.matmul(R, en) - eq)))

    return loss1 + loss2b + loss3 + lossF

def train_spatial(R, X, J, F):
    with tf.GradientTape() as t:
        current_loss = loss_spatial(R, X, J, F)

    dR = t.gradient(current_loss, [R])
    optimizer.apply_gradients(zip(dR, [R]))

def regressor_single_joint_spatial(Joints, verts, skW, parents, params):

    F = np.zeros((len(Joints), len(verts)))

    for i in range(len(Joints)):
        temp1 = np.nonzero(skW[:, i] > 0.1)
        temp2 = np.nonzero(skW[:, parents[i]] > 0.1)
        ind = intersection(temp2[0].tolist(), temp1[0].tolist())

        F[i, ind] = 1

    X = verts
    J = Joints
    R = tf.Variable(np.zeros((len(Joints), len(verts))))

    for epoch in range(epochs):
        Rs.append(R)

        current_loss = loss_spatial(R, X, J, F)

        train_spatial(R, X, J, F)
        if not(epoch % 20):
            print(f"Epoch {epoch}: Loss: {current_loss.numpy()}")

    return R.numpy(), F