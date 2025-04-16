# pylint: disable=too-many-lines
"""
Convenient encapsulation of distributions  and sampling from distributions not
directly available in scipy or numpy. 

Useful for simulation.

Each distribution has its own random number stream
that can be set by a seed.
"""

import math

import numpy as np
from numpy.typing import NDArray, ArrayLike

from typing import Protocol, Optional, Union, Tuple, Any, runtime_checkable



# pylint: disable=too-few-public-methods
@runtime_checkable
class Distribution(Protocol):
    """
    Distribution protocol defining the interface for probability distributions.
    
    Any class implementing this protocol should provide a sampling mechanism
    that generates random values according to a specific probability distribution.
    """
    
    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        
        Examples
        --------
        >>> dist = SomeDistribution(params)
        >>> single_sample = dist.sample()  # Returns a float
        >>> array_1d = dist.sample(10)  # Returns 1D array with 10 samples
        >>> array_2d = dist.sample((2, 3))  # Returns 2×3 array of samples
        """
        ...



# pylint: disable=too-few-public-methods
class Exponential:
    """
    Exponential distribution implementation.
    
    A probability distribution that models the time between events in a Poisson process,
    where events occur continuously and independently at a constant average rate.
    
    This class conforms to the Distribution protocol and provides methods to sample
    from an exponential distribution with a specified mean.
    """

    def __init__(self, mean: float, random_seed: Optional[int] = None):
        """
        Initialize an exponential distribution.
        
        Parameters
        ----------
        mean : float
            The mean of the exponential distribution.
            Must be positive.
        
        random_seed : Optional[int], default=None
            A random seed to reproduce samples. If None, a unique
            sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)
        self.mean = mean

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the exponential distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the exponential distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.exponential(self.mean, size=size)



# pylint: disable=too-few-public-methods
class Bernoulli:
    """
    Bernoulli distribution implementation.
    
    A discrete probability distribution that takes value 1 with probability p
    and value 0 with probability 1-p.
    
    This class conforms to the Distribution protocol and provides methods to sample
    from a Bernoulli distribution with a specified probability.
    """

    def __init__(self, p: float, random_seed: Optional[int] = None):
        """
        Initialize a Bernoulli distribution.
        
        Parameters
        ----------
        p : float
            Probability of drawing a 1. Must be between 0 and 1.
        
        random_seed : Optional[int], default=None
            A random seed to reproduce samples. If None, a unique
            sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)
        self.p = p

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Bernoulli distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Bernoulli distribution:
            - A single float (0 or 1) when size is None
            - A numpy array of floats (0s and 1s) with shape determined by size parameter
        """
        return self.rng.binomial(n=1, p=self.p, size=size)


class Lognormal:
    """
    Lognormal distribution implementation.
    
    A continuous probability distribution where the logarithm of a random variable
    is normally distributed. It is useful for modeling variables that are the product
    of many small independent factors.
    
    This class conforms to the Distribution protocol and provides methods to sample
    from a lognormal distribution with a specified mean and standard deviation.
    """

    def __init__(
        self,
        mean: float,
        stdev: float,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a lognormal distribution.
        
        Parameters
        ----------
        mean : float
            Mean of the lognormal distribution.
        
        stdev : float
            Standard deviation of the lognormal distribution.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma

    def normal_moments_from_lognormal(self, m: float, v: float) -> Tuple[float, float]:
        """
        Calculate mu and sigma of the normal distribution underlying 
        a lognormal with mean m and variance v.
        
        Parameters
        ----------
        m : float
            Mean of lognormal distribution.
        v : float
            Variance of lognormal distribution.
        
        Returns
        -------
        Tuple[float, float]
            The mu and sigma parameters of the underlying normal distribution.
            
        Notes
        -----
        Formula source: https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-with-specified-mean-and-variance.html
        """
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2 / phi)
        sigma = math.sqrt(math.log(phi**2 / m**2))
        return mu, sigma

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the lognormal distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the lognormal distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.lognormal(self.mu, self.sigma, size=size)


class Normal:
    """
    Normal distribution implementation with optional truncation.
    
    A continuous probability distribution that follows the Gaussian bell curve.
    This implementation allows truncating the distribution at a minimum value.
    
    This class conforms to the Distribution protocol and provides methods to sample
    from a normal distribution with specified mean and standard deviation.
    """

    def __init__(
        self,
        mean: float,
        sigma: float,
        minimum: Optional[float] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize a normal distribution.
        
        Parameters
        ----------
        mean : float
            The mean (μ) of the normal distribution.
        
        sigma : float
            The standard deviation (σ) of the normal distribution.
        
        minimum : Optional[float], default=None
            If provided, truncates the distribution to this minimum value.
            Any sampled values below this minimum will be set to this value.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
        """
        self.rng = np.random.default_rng(seed=random_seed)
        self.mean = mean
        self.sigma = sigma
        self.minimum = minimum

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the normal distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the normal distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
            
        Notes
        -----
        If a minimum value was specified during initialization, any samples
        below this value will be truncated (set to the minimum value).
        """
        samples = self.rng.normal(self.mean, self.sigma, size=size)

        if self.minimum is None:
            return samples

        if size is None:
            return max(self.minimum, samples)

        # Truncate samples below minimum
        below_min_idx = np.where(samples < self.minimum)[0]
        samples[below_min_idx] = self.minimum
        return samples


class Uniform:
    """
    Uniform distribution implementation.
    
    A continuous probability distribution where all values in a range have
    equal probability of being sampled.
    
    This class conforms to the Distribution protocol and provides methods to sample
    from a uniform distribution between specified low and high values.
    """

    def __init__(
        self, low: float, high: float, random_seed: Optional[int] = None
    ):
        """
        Initialize a uniform distribution.
        
        Parameters
        ----------
        low : float
            Lower bound of the distribution range.
        
        high : float
            Upper bound of the distribution range.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)
        self.low = low
        self.high = high

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the uniform distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the uniform distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.uniform(low=self.low, high=self.high, size=size)


class Triangular:
    """
    Triangular distribution implementation.
    
    A continuous probability distribution with lower limit, upper limit, and mode,
    forming a triangular-shaped probability density function.
    
    This class conforms to the Distribution protocol and provides methods to sample
    from a triangular distribution with specified parameters.
    """

    def __init__(
        self,
        low: float,
        mode: float,
        high: float,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a triangular distribution.
        
        Parameters
        ----------
        low : float
            Lower limit of the distribution.
        
        mode : float
            Mode (peak) of the distribution. Must be between low and high.
        
        high : float
            Upper limit of the distribution.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)
        self.low = low
        self.high = high
        self.mode = mode

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the triangular distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the triangular distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.triangular(self.low, self.mode, self.high, size=size)


class FixedDistribution:
    """
    Fixed distribution implementation.
    
    A degenerate distribution that always returns the same fixed value.
    Useful for constants or deterministic parameters in models.
    
    This class conforms to the Distribution protocol and provides methods to sample
    a constant value regardless of the number of samples requested.
    """

    def __init__(self, value: float):
        """
        Initialize a fixed distribution.
        
        Parameters
        ----------
        value : float
            The constant value that will be returned by sampling.
        """
        self.value = value

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate "samples" from the fixed distribution (always the same value).
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns the fixed value as a float
            - If int: returns a 1-D array filled with the fixed value
            - If tuple of ints: returns an array with that shape filled with the fixed value
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            The fixed value:
            - A single float when size is None
            - A numpy array filled with the fixed value with shape determined by size parameter
        """
        if size is not None:
            return np.full(size, self.value)
        return self.value


class CombinationDistribution:
    """
    Combination distribution implementation.
    
    A distribution that combines (sums) samples from multiple underlying distributions.
    Useful for modeling compound effects or building complex distributions from simpler ones.
    
    This class conforms to the Distribution protocol and provides methods to sample
    a combination of values from multiple distributions.
    """

    def __init__(self, *dists: Distribution):
        """
        Initialize a combination distribution.
        
        Parameters
        ----------
        *dists : Sequence[Distribution]
            Variable length sequence of Distribution objects to combine.
            The sample method will return the sum of samples from all these distributions.
        """
        self.dists = dists

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the combination distribution.
        
        For each sample drawn, the result is the sum of samples from each
        of the underlying distributions.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single combined sample as a float
            - If int: returns a 1-D array with that many combined samples
            - If tuple of ints: returns an array with that shape of combined samples
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the combination distribution:
            - A single float (sum of component samples) when size is None
            - A numpy array of combined samples with shape determined by size parameter
        """
        total = 0.0 if size is None else np.zeros(size)

        for dist in self.dists:
            total += dist.sample(size)
        return total


class ContinuousEmpirical:
    """
    Continuous Empirical Distribution implementation.
    
    A distribution that performs linear interpolation between upper and lower
    bounds of a discrete distribution. Useful for modeling empirical data with
    a continuous approximation.
    
    This class conforms to the Distribution protocol and provides methods to sample
    from a continuous empirical distribution.
    """

    def __init__(
        self,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        freq: ArrayLike,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize a continuous empirical distribution.
        
        Parameters
        ----------
        lower_bounds : ArrayLike
            Lower bounds of a discrete empirical distribution.
        
        upper_bounds : ArrayLike
            Upper bounds of a discrete empirical distribution.
        
        freq : ArrayLike
            Frequency of observations between bounds.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)
        self.lower_bounds = np.asarray(lower_bounds)
        self.upper_bounds = np.asarray(upper_bounds)
        self.cumulative_probs = self.create_cumulative_probs(freq)

    def create_cumulative_probs(
            self,
            freq: ArrayLike
    ) -> NDArray[np.float64]:
        """
        Calculate cumulative relative frequency from frequency.
        
        Parameters
        ----------
        freq : ArrayLike
            Frequency distribution.
        
        Returns
        -------
        NDArray[np.float64]
            Cumulative relative frequency.
        """
        freq = np.asarray(freq, dtype='float')
        return np.cumsum(freq / freq.sum(), dtype='float')

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Sample from the Continuous Empirical Distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the continuous empirical distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        if size is None:
            size = 1

        # Handle the case where size is a tuple - convert to total number of samples
        total_samples = size if isinstance(size, int) else np.prod(size)
        
        samples = []
        for _ in range(total_samples):
            # Sample a value u from the uniform(0, 1) distribution
            u = self.rng.random()

            # Obtain lower and upper bounds of a sample from the
            # discrete empirical distribution
            idx = np.searchsorted(self.cumulative_probs, u)
            lb, ub = self.lower_bounds[idx], self.upper_bounds[idx]

            # Use linear interpolation of u between
            # the lower and upper bound to obtain a continuous value
            continuous_value = (
                lb + (ub - lb) * (u - self.cumulative_probs[idx - 1]) / (
                    self.cumulative_probs[idx] - self.cumulative_probs[idx - 1]
                )
            )

            samples.append(continuous_value)

        if total_samples == 1:
            # .item() ensures returned as python 'float'
            # as opposed to np.float64
            return samples[0].item()
            
        result = np.asarray(samples)
        # Reshape if size was a tuple
        if isinstance(size, tuple):
            result = result.reshape(size)
        return result


class Erlang:
    """
    Erlang distribution implementation.
    
    A continuous probability distribution that is a special case of the Gamma distribution
    where the shape parameter is an integer. This implementation allows users to specify
    mean and standard deviation rather than shape (k) and scale (theta) parameters.
    
    This class conforms to the Distribution protocol and provides methods to sample
    from an Erlang distribution with specified parameters.
    
    Notes
    -----
    The Erlang is a special case of the gamma distribution where k is an integer.
    Internally this is implemented using numpy Generator's gamma method. The k parameter
    is calculated from the mean and standard deviation and rounded to an integer.
    
    Sources
    -------
    Conversion between mean+stdev to k+theta:
    https://www.statisticshowto.com/erlang-distribution/
    """

    def __init__(
        self,
        mean: float,
        stdev: float,
        location: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize an Erlang distribution.
        
        Parameters
        ----------
        mean : float
            Mean of the Erlang distribution.
        
        stdev : float
            Standard deviation of the Erlang distribution.
        
        location : float, default=0.0
            Offset the origin of the distribution. The returned value 
            will be the sampled value plus this location parameter.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)
        self.mean = mean
        self.stdev = stdev
        self.location = location

        # k also referred to as shape
        self.k = round((mean / stdev) ** 2)

        # theta also referred to as scale
        self.theta = mean / self.k

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Erlang distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Erlang distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.gamma(self.k, self.theta, size) + self.location



import numpy as np
from typing import Optional, Union, Tuple
from numpy.typing import NDArray

class Weibull:
    """
    Weibull distribution implementation.
    
    A continuous probability distribution useful for modeling time-to-failure and
    similar phenomena. Characterized by shape (alpha) and scale (beta) parameters.
    
    This implementation also includes a third parameter "location" (default = 0)
    to shift the distribution if a lower bound is needed.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        location: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize a three-parameter Weibull distribution.
        
        Parameters
        ----------
        alpha : float
            The shape parameter. Must be > 0.
        
        beta : float
            The scale parameter. Must be > 0. The higher the scale parameter,
            the more variance in the samples.
        
        location : float, default=0.0
            An offset to shift the distribution from 0.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
        
        Notes
        -----
        Caution is advised when setting shape and scale parameters as different
        sources use different notations:
        
        - In Law and Kelton, shape=alpha and scale=beta
        - Wikipedia defines shape=k and scale=lambda=1/beta
        - Other sources define shape=beta and scale=eta (η)
        - In Python's random.weibullvariate, alpha=scale and beta=shape!
        
        It's recommended to verify the mean and variance of samples match expectations.
        """

        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0")

        self.rng = np.random.default_rng(random_seed)
        self.shape = alpha
        self.scale = beta
        self.location = location

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Weibull distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Weibull distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.scale * self.rng.weibull(self.shape, size) + self.location



class Gamma(Distribution):
    """
    Gamma distribution

    Gamma distribution set up to accept alpha (scale) and beta (shape)
    parameters as described in Law (2007).

    Also contains functions to compute mean, variance, and a static method
    to computer alpha and beta from specified mean and variance.

    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        location: Optional[float] = 0.0,
        random_seed: Optional[int] = None,
    ):
        """
        Gamma distribution

        Params:
        ------
        alpha: float. Must be > 0

        beta: float
            scale parameter. Must be > 0

        location, float, optional (default=0.0)
            Offset the original of the distribution i.e.
            the returned value = sample[Gamma] + location

        random_seed: int, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.

        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0")

        self.rng = np.random.default_rng(random_seed)
        self.alpha = alpha  # shape
        self.beta = beta  # scale
        self.location = location

    def mean(self) -> float:
        """
        The computed mean of the gamma distribution

        Returns:
        -------
        float
        """
        return self.alpha * self.beta

    def variance(self) -> float:
        """
        The computed varaince of the gamma distribution

        Returns:
        -------
        float
        """
        return self.alpha * (self.beta**2)

    @staticmethod
    def params_from_mean_and_var(
        mean: float,
        var: float
    ) -> Tuple[float, float]:
        """
        Helper static method to get alpha and beta parameters
        from a mean and variance.

        Params:
        ------
        mean: float
            mean of the gamma distribution

        var: float
            variance of the gamma distribution

        Returns:
        -------
        (float, float)
        alpha, beta

        """
        alpha = mean**2 / var
        beta = mean / var
        return alpha, beta

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample fron the Gamma distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.rng.gamma(self.alpha, self.beta, size) + self.location


class Beta:
    """
    Beta distribution implementation.
    
    A flexible continuous probability distribution defined on the interval [0,1],
    which can be rescaled to any arbitrary interval [min, max].
    
    As defined in Simulation Modeling and Analysis (Law, 2007).
    
    Common uses:
    -----------
    1. Useful as a rough model in the absence of data
    2. Distribution of a random proportion
    3. Time to complete a task
    """
    
    def __init__(
        self,
        alpha1: float,
        alpha2: float,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize a Beta distribution.
        
        Parameters
        ----------
        alpha1 : float
            First shape parameter. Must be positive.
        
        alpha2 : float
            Second shape parameter. Must be positive.
        
        lower_bound : float, default=0.0
            Lower bound for rescaling the distribution from [0,1] to [lower_bound, upper_bound].
        
        upper_bound : float, default=1.0
            Upper bound for rescaling the distribution from [0,1] to [lower_bound, upper_bound].
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.min = lower_bound
        self.max = upper_bound

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Beta distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Beta distribution, rescaled to [min, max]:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.min + (
            (self.max - self.min) *
            self.rng.beta(self.alpha1, self.alpha2, size)
        )

class Discrete:
    """
    Discrete distribution implementation.
    
    A probability distribution that samples values with specified frequencies.
    Useful for modeling categorical data or discrete outcomes with known probabilities.
    
    Example uses:
    -------------
    1. Routing percentages
    2. Classes of entity
    3. Batch sizes of arrivals
    """

    def __init__(
        self,
        values: ArrayLike,
        freq: ArrayLike,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize a discrete distribution.
        
        Parameters
        ----------
        values : ArrayLike
            List of possible outcome values. Must be of equal length to freq.
        
        freq : ArrayLike
            List of observed frequencies or probabilities. Must be of equal length to values.
            These will be normalized to sum to 1.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
        
        Raises
        ------
        ValueError
            If values and freq have different lengths.
        """
        if len(values) != len(freq):
            raise ValueError(
                "values and freq arguments must be of equal length")

        self.rng = np.random.default_rng(random_seed)
        self.values = np.asarray(values)
        self.freq = np.asarray(freq)
        self.probabilities = self.freq / self.freq.sum()

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[Any, NDArray]:
        """
        Generate random samples from the discrete distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[Any, NDArray]
            Random samples from the discrete distribution:
            - A single value (of whatever type was in the values array) when size is None
            - A numpy array of values with shape determined by size parameter
        """
        sample = self.rng.choice(self.values, p=self.probabilities, size=size)

        if size is None:
            return sample.item()
        return sample


class TruncatedDistribution:
    """
    Truncated Distribution implementation.
    
    Wraps any distribution conforming to the Distribution protocol and truncates
    samples at a specified lower bound. No resampling is performed; the class simply 
    ensures no values are below the lower bound.
    
    This class itself conforms to the Distribution protocol.
    """

    def __init__(self, dist_to_truncate: Distribution, lower_bound: float):
        """
        Initialize a truncated distribution.
        
        Parameters
        ----------
        dist_to_truncate : Distribution
            Any object conforming to the Distribution protocol that generates samples.
        
        lower_bound : float
            Truncation point. Any samples below this value will be set to this value.
        """
        self.dist = dist_to_truncate
        self.lower_bound = lower_bound

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the truncated distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the truncated distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
            
        Notes
        -----
        All values will be greater than or equal to the specified lower bound.
        """
        if size is None:
            sample = self.dist.sample()
            return max(self.lower_bound, sample)
        
        samples = self.dist.sample(size)
        if isinstance(samples, np.ndarray):
            samples[samples < self.lower_bound] = self.lower_bound
        
        return samples


class RawEmpirical:
    """
    Raw Empirical distribution implementation.
    
    Samples with replacement from a list of empirical values. Useful when no theoretical
    distribution fits the observed data well.
    
    This class conforms to the Distribution protocol.
    """

    def __init__(
        self,
        values: ArrayLike,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a raw empirical distribution.
        
        Parameters
        ----------
        values : ArrayLike
            List of empirical sample values to sample from with replacement.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
            
        Notes
        -----
        If the sample size is small, consider whether the upper and lower limits
        in the raw data are representative of the real-world system.
        """
        self.rng = np.random.default_rng(random_seed)
        self.values = np.asarray(values)

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[Any, NDArray]:
        """
        Generate random samples from the raw empirical data with replacement.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[Any, NDArray]
            Random samples from the empirical data:
            - A single value when size is None
            - A numpy array of values with shape determined by size parameter
        """
        samples = self.rng.choice(self.values, size)
        
        # Ensure we return a scalar if size is None, not an array with one element
        if size is None:
            return samples.item()
        return samples


class PearsonV:
    """
    Pearson Type V distribution implementation (inverse Gamma distribution).
    
    Where alpha = shape, and beta = scale (both > 0).
    
    Law (2007, pg 293-294) defines the distribution as
    PearsonV(alpha, beta) = 1/Gamma(alpha, 1/beta) and notes that the
    PDF is similar to that of lognormal, but has a larger spike
    close to 0. It can be used to model the time to complete a task.
    
    For certain values of the shape parameter the mean and variance can be
    directly computed:
    
    mean = beta / (alpha - 1) for alpha > 1.0
    var = beta^2 / (alpha - 1)^2 × (alpha - 2) for alpha > 2.0
    
    This class conforms to the Distribution protocol.
    
    Alternative Sources:
    --------------------
    [1] https://riskwiki.vosesoftware.com/PearsonType5distribution.php
    [2] https://modelassist.epixanalytics.com/display/EA/Pearson+Type+5
    
    Notes:
    ------
    A good R package for Pearson distributions is PearsonDS
    https://www.rdocumentation.org/packages/PearsonDS/versions/1.3.0
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a Pearson Type V distribution.
        
        Parameters
        ----------
        alpha : float
            Shape parameter. Must be > 0.
        
        beta : float
            Scale parameter. Must be > 0.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
            
        Raises
        ------
        ValueError
            If alpha or beta are not positive.
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0")

        self.rng = np.random.default_rng(random_seed)
        self.alpha = alpha  # shape
        self.beta = beta    # scale

    def mean(self) -> float:
        """
        Calculate the mean of the Pearson Type V distribution.
        
        Returns
        -------
        float
            The theoretical mean of this distribution.
            
        Raises
        ------
        ValueError
            If alpha <= 1.0, as the mean is not defined in this case.
        """
        if self.alpha > 1.0:
            return self.beta / (self.alpha - 1)
        msg = "Cannot directly compute mean when alpha <= 1.0"
        raise ValueError(msg)

    def var(self) -> float:
        """
        Calculate the variance of the Pearson Type V distribution.
        
        Returns
        -------
        float
            The theoretical variance of this distribution.
            
        Raises
        ------
        ValueError
            If alpha <= 2.0, as the variance is not defined in this case.
        """
        if self.alpha > 2.0:
            return (
                self.beta**2) / (((self.alpha - 1) ** 2) * (self.alpha - 2))
        msg = "Cannot directly compute var when alpha <= 2.0"
        raise ValueError(msg)

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Pearson Type V distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Pearson Type V distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return 1 / self.rng.gamma(self.alpha, 1 / self.beta, size)


class PearsonVI:
    """
    Pearson Type VI distribution implementation (inverted beta distribution).
    
    Where:
    - alpha1 = shape parameter 1 (> 0)
    - alpha2 = shape parameter 2 (> 0)
    - beta = scale (> 0)
    
    Law (2007, pg 294-295) notes that PearsonVI can be used to model 
    the time to complete a task.
    
    For certain values of the shape parameters, the mean and variance can be
    directly computed. See functions mean() and var() for details.
    
    Sampling:
    ---------
    Pearson6(a1,a2,b) = b*X/(1-X), where X=Beta(a1,a2)
    
    This class conforms to the Distribution protocol.
    
    Sources:
    --------
    [1] https://riskwiki.vosesoftware.com/PearsonType6distribution.php
    
    Notes:
    ------
    A good R package for Pearson distributions is PearsonDS
    https://www.rdocumentation.org/packages/PearsonDS/versions/1.3.0
    """

    def __init__(
        self,
        alpha1: float,
        alpha2: float,
        beta: float,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize a Pearson Type VI distribution.
        
        Parameters
        ----------
        alpha1 : float
            Shape parameter 1. Must be > 0.
        
        alpha2 : float
            Shape parameter 2. Must be > 0.
        
        beta : float
            Scale parameter. Must be > 0.
        
        random_seed : Optional[int], default=None
            Random seed to control sampling. If None, a unique
            sample sequence is generated.
            
        Raises
        ------
        ValueError
            If any of the parameters are not positive.
        """
        if alpha1 <= 0 or alpha2 <= 0 or beta <= 0:
            raise ValueError("alpha1, alpha2, and beta must all be > 0")
            
        self.rng = np.random.default_rng(random_seed)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta

    def mean(self) -> float:
        """
        Calculate the mean of the Pearson Type VI distribution.
        
        Returns
        -------
        float
            The theoretical mean of this distribution.
            
        Raises
        ------
        ValueError
            If alpha2 <= 1.0, as the mean is not defined in this case.
        """
        if self.alpha2 > 1.0:
            return (self.beta * self.alpha1) / (self.alpha2 - 1)
        raise ValueError("Cannot compute mean when alpha2 <= 1.0")

    def var(self) -> float:
        """
        Calculate the variance of the Pearson Type VI distribution.
        
        Returns
        -------
        float
            The theoretical variance of this distribution.
            
        Raises
        ------
        ValueError
            If alpha2 <= 2.0, as the variance is not defined in this case.
        """
        if self.alpha2 > 2.0:
            return (
                (self.beta**2) * self.alpha1 * (self.alpha1 + self.alpha2 - 1)
            ) / (((self.alpha2 - 1) ** 2) * (self.alpha2 - 2))
        msg = "Cannot directly compute var when alpha2 <= 2.0"
        raise ValueError(msg)

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Pearson Type VI distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Pearson Type VI distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        # Pearson6(a1,a2,b)=b∗X/(1−X), where X=Beta(a1,a2,1)
        x = self.rng.beta(self.alpha1, self.alpha2, size)
        return self.beta * x / (1 - x)


class ErlangK:
    """
    Erlang distribution where k and theta are specified.

    The Erlang is a special case of the gamma distribution where
    k is a positive integer. Internally this is implemented using
    numpy Generator's gamma method.

    Optionally a user can offset the origin of the distribution
    using the location parameter.
    """

    def __init__(
        self,
        k: int,
        theta: float,
        location: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize an Erlang distribution with specified k and theta.
        
        Parameters
        ----------
        k : int
            Shape parameter (positive integer) of the Erlang distribution.
        
        theta : float
            Scale parameter of the Erlang distribution.
        
        location : float, default=0.0
            Offset the origin of the distribution i.e.
            the returned value = sample[Erlang] + location
        
        random_seed : Optional[int], default=None
            A random seed to reproduce samples. If set to None then a unique
            sample sequence is generated.
            
        Raises
        ------
        ValueError
            If k is not a positive integer.
        """
        # Check that k is a positive integer
        if not isinstance(k, int):
            raise ValueError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be > 0")

        self.rng = np.random.default_rng(random_seed)
        self.k = k
        self.theta = theta
        self.location = location

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Erlang distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Erlang distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.gamma(self.k, self.theta, size) + self.location


class Poisson:
    """
    Poisson distribution implementation.
    
    Used to simulate number of events that occur in an interval of time.
    E.g. number of items in a batch.
    
    This class conforms to the Distribution protocol.
    
    Sources:
    --------
    Law (2007 pg. 308) Simulation modelling and analysis.
    """

    def __init__(self, rate: float, random_seed: Optional[int] = None):
        """
        Initialize a Poisson distribution.
        
        Parameters
        ----------
        rate : float
            Mean number of events in time period.
        
        random_seed : Optional[int], default=None
            A random seed to reproduce samples. If set to None then a unique
            sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)
        self.rate = rate

    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[int, NDArray[np.int_]]:
        """
        Generate random samples from the Poisson distribution.
        
        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as an integer
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape
        
        Returns
        -------
        Union[int, NDArray[np.int_]]
            Random samples from the Poisson distribution:
            - A single integer when size is None
            - A numpy array of integers with shape determined by size parameter
        """
        return self.rng.poisson(self.rate, size)

