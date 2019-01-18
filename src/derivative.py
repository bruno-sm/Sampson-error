def partial_derivative(f, x, axis):
    h = 0.00001
    near_x = x.copy()
    near_x[axis] = near_x[axis] + h
    return (f(near_x) - f(x))/h
