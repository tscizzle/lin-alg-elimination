import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter


def tyler_elimination_solver(A, b):
    """ Solve the equation Ax=b by elimination.

    :param ndarray A: (n x n) coefficient matrix
    :param ndarray b: (n x 1) constant vector

    :return ndarray x: (n x 1) solution vector (None if no solution exists)
    """
    n = len(A)

    ## Through row operations, find upper-triangular problem Ux=c with same solution x.

    # Combine U and c into Uc (n x n+1) so row operations get applied to both at once.
    Uc = np.hstack((A, b))

    # Go to each pivot, column by column.
    for pivot_j in range(n):
        pivot_i = pivot_j
        pivot_val = Uc[pivot_i][pivot_j]

        # If the pivot is 0, swap this row with a later row that has a non-zero value in
        # the current pivot column.
        if pivot_val == 0:
            for candidate_i in range(pivot_i + 1, n):
                candidate_pivot = Uc[candidate_i][pivot_j]
                if candidate_pivot != 0:
                    # Swap rows so the new pivot is non-zero.
                    Uc[[pivot_i, candidate_i]] = Uc[[candidate_i, pivot_i]]
                    pivot_val = candidate_pivot
                    break
        # If we found no rows to swap with that provide a non-zero pivot, the pivot is
        # still 0 and there is no solution.
        if pivot_val == 0:
            return None

        # For each target row beneath the pivot row, eliminate the entry in the pivot
        # column by subtracting a multiple of the pivot row from the target row.
        for target_i in range(pivot_i + 1, n):
            target_val = Uc[target_i][pivot_j]
            multiplier = target_val / pivot_val
            Uc[target_i] -= Uc[pivot_i] * multiplier

    ## With the problem being upper-triangular, solve via back-propagation.

    x = np.empty((n, 1))

    # Solve for each unknown, starting at the bottom.
    for i in range(n - 1, -1, -1):
        # To understand this line, manually do an example of back-propogation on an
        # upper-triangular matrix problem and see the operations you do at each line.
        x[i][0] = (Uc[i][-1] - sum(Uc[i][j] * x[j] for j in range(i + 1, n))) / Uc[i][i]

    return x


def random_matrix(m, n, bounds=(-100, 100)):
    """ Generate a matrix with all random elements.

    :param int m: Number of rows of the matrix to create.
    :param int n: Number of columns of the matrix to create.
    :param (float, float) bounds: Lower and upper bound the random values are in.

    :return ndarray:
    """
    # For each value, get a random value between 0 and 1, scale it to the size of the
    # bounds, then shift it to start at the lower bound.
    # E.g. For bounds (-100, 100), a random number is between 0 and 1, then multiplied
    # so it's between 0 and 200, then shifted so it's between -100 and 100.
    return np.random.rand(m, n) * (bounds[1] - bounds[0]) + bounds[0]


def evaluate_solver_runtime(solver, up_to_n=100, trials_per_n=10):
    """ Time the runtime of the elimination solver as the size of the problem increases,
        and chart the result.

    :param func(A,b)->x solver: Function that takes A as an ndarray (n x n) and b as an
        ndarray (n x 1) and returns x as an ndarray (n x 1)
    :param int up_to_n: Time the solver for matrix sizes up to this.
    :param int trials_per_n: Number of solves at each matrix size, to increase accuracy.
    """
    ## Run the solver at various matrix sizes.

    sizes = range(2, up_to_n, int(up_to_n / 10))
    avg_runtimes = []
    for n in sizes:
        runtimes = []
        for _ in range(trials_per_n):
            # Create some random inputs.
            A = random_matrix(n, n)
            b = random_matrix(n, 1)
            # Perform the work, and measure its runtime.
            start = perf_counter()
            solver(A, b)
            end = perf_counter()
            # Log the runtime.
            runtime = end - start
            runtimes.append(runtime)
        # Log the average runtime for this matrix size.
        avg_runtime = sum(runtimes) / len(runtimes)
        avg_runtimes.append(avg_runtime)

    ## Chart the runtimes to guess the algorithm's complexity.

    # Include a best fit polynomial to help guess the algorithm's complexity.
    # (We actually know the complexity should be O(n^3).)
    best_fit_coeffs = np.polyfit(sizes, avg_runtimes, 3)
    best_fit_poly = np.poly1d(best_fit_coeffs)
    best_fit_x = np.linspace(sizes[0], sizes[-1])
    best_fit_y = best_fit_poly(best_fit_x)
    # Show the chart.
    fig, ax = plt.subplots()
    ax.plot(sizes, avg_runtimes, "o", best_fit_x, best_fit_y)
    plt.show()


def main():
    # Check that Tyler's solver gets the same answer as numpy's, on some random problem.
    A = random_matrix(10, 10)
    b = random_matrix(10, 1)
    x0 = tyler_elimination_solver(A, b)
    x1 = np.linalg.solve(A, b)
    np.testing.assert_allclose(x0, x1)

    # Chart runtime growth of Tyler's solver and numpy's solver.
    evaluate_solver_runtime(np.linalg.solve, up_to_n=300, trials_per_n=5)
    evaluate_solver_runtime(tyler_elimination_solver, up_to_n=300, trials_per_n=5)


if __name__ == "__main__":
    main()
