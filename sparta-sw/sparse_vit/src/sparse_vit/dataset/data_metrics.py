def divide_w_zero_division(a, b):
    """
    Division that handles cases where divisor is zero.
    """
    return a / b if b else 0


class OnlineMean:
    """
    A class to represent an online-calculated average.
    """

    def __init__(self):
        """
        Initialize `OnlineMean` calculation.
        """
        self.n = 0
        self.mean = 0.0

    def update(self, x):
        """
        Update `mean` estimate with input value.
        """
        self.n += 1
        self.mean += (x - self.mean) / self.n

    def get_val(self):
        """
        Return `mean` estimate (current).
        """
        return self.mean


class OnlineVariance:
    """
    A class to represent an online-calculated variance.
    """

    def __init__(self):
        """
        Initialize `OnlineVariance` calculation.
        """
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        """
        Update `variance` estimate with input value.
        """
        self.n += 1
        delta = x - self.mean

        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def get_val(self):
        """
        Return `variance` estimate (current).
        """
        if self.n < 2:
            return float("nan")
        else:
            return self.M2 / (self.n - 1)


def gather_mean_std_statistics(loader, model, N=64):
    """
    Gather dataset statistics, mean, std from mini batches.
    """
    means = [[OnlineMean() for j in range(N)] for i in range(3)]
    mean_of_vars = [[OnlineMean() for j in range(N)] for i in range(3)]
    var_of_means = [[OnlineVariance() for j in range(N)] for i in range(3)]

    for batch in loader:

        imgs, _ = batch

        xs = model.embed(imgs)

        mean = xs.mean(dim=(0, 3, 4))  # (1, C, P, 1, 1)
        vars = xs.var(dim=(0, 3, 4)).clamp_min(1e-6)

        for c in range(3):
            for i in range(N):
                means[c][i].update(mean[c][i])
                mean_of_vars[c][i].update(vars[c][i])
                var_of_means[c][i].update(mean[c][i])

    means = [float(means[c][i].get_val()) for c in range(3) for i in range(N)]

    varss = [
        float(mean_of_vars[c][i].get_val()) + float(var_of_means[c][i].get_val())
        for c in range(3)
        for i in range(N)
    ]

    stdss = [_var**0.5 for _var in varss]

    return means, stdss
