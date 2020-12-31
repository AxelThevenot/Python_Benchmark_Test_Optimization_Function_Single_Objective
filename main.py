import numpy as np
import matplotlib.pyplot as plt

import pybenchfunction as bench


if __name__ == '__main__':
    # get all the available functions accepting ANY dimension
    any_dim_functions = bench.get_functions(None)
    # get all the available continuous and non-convex functions accepting 2D
    continous_nonconvex_2d_functions = bench.get_functions(
        2,  # dimension
        continuous=True,
        convex=False,
        separable=None,
        differentiable=None,
        mutimodal=None,
        randomized_term=None
    )
    print(len(any_dim_functions))  # --> 40
    print(len(continous_nonconvex_2d_functions))  # --> 41


    # Import specific function
    # set the dimension of the input for the function
    sphere = bench.function.Sphere(3)
    # get results
    X = np.array([1, 3, 0])
    print(sphere(X))  # --> 10

    # Plot 2d or plot 3d contours
    # Warning ! Only working on 2d functions objects !
    # Warning 2! change n_space to reduce the computing time
    thevenot = bench.function.Thevenot(2)
    bench.plot_2d(thevenot, n_space=1000, ax=None)
    bench.plot_3d(thevenot, n_space=1000, ax=None)

    # access/change the parameters of parametrics functions :
    print(thevenot.get_param())  # --> {'m': 5, 'beta': 15}
    thevenot.beta = 42
    print(thevenot.get_param())  # --> {'m': 5, 'beta': 42}


    # get the global minimum for a specific dimension
    # it only gives the global minimum for defaut parameters
    X_min, minimum = sphere.get_global_minimum(3)
    print(X_min)  # --> [0 0 0]
    print(minimum)  # --> 0

    # access the latex formulas. You can also convert it into images
    latex = bench.function.Thevenot.latex_formula
    # latex = bench.function.Thevenot.latex_formula_dimension
    # latex = bench.function.Thevenot.latex_formula_input_domain
    # latex = bench.function.Thevenot.latex_formula_global_minimum
    print(latex)  # --> f(\mathbf{x}) = exp(-\sum_{i=1}^{d}(x_i / \beta)^{2m}) - 2exp(-\prod_{i=1}^{d}x_i^2) \prod_{i=1}^{d}cos^ 2(x_i)
    latex_img = bench.latex_img(latex)
    # convert the latex formula to image
    plt.imshow(latex_img)
    plt.show()
