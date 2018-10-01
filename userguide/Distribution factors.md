---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Factors and Constraints](Factors and Constraints.md)

## Distributions Factors

This page lists the built-in methods for creating random variables with various distribution factors. In many cases, you can pass in random variables as arguments e.g. `Variable<int>` instead of **int**. For compactness, this is not shown in the syntax below.

These methods provide a convenient short alternative to using `Variable<T>.Factor` and passing in the factor method, as described [on this page](Applying functions and operators to variables.md). Note also that you can create a random variable with _any_ prior distribution using `Variable.Random(IDistribution<T> dist)`. This is useful if you already have a distribution object, such as the posterior from a previous inference, and wish to create a new random variable with that distribution as its prior.

#### Discrete Distributions

Distributions where the random variable can take one of a discrete set of states. Where the discrete set of states is indexed by a Range, it should replace a size argument or be passed as an additional parameter.

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| _Bernoulli_ | `Variable.Bernoulli(double probTrue)` | Creates a **boolean** random variable from the probability of being true P(true). |
| | `Variable.BernoulliFromLogOdds(double logOdds)` | Creates a **boolean** random variable from the log odds i.e. log P(true)/P(false). Equivalent to `Variable.Bernoulli(Variable.Logistic(logOdds))`. |
| _Discrete_ | `Variable.Discrete(double[] probs)` `Variable.Discrete(Vector probs)` | Creates an **int** random variable from supplied array or vector of probabilities (which must add up to 1). The variable can take values between 0 and N-1 where N is the length of the array/vector. |
| | `Variable.DiscreteFromLogProbs(double[] logProbs)` | Creates an **int** random variable from supplied array or vector of log probabilities. Equivalent to `Variable.Discrete(Variable.Softmax(logProbs))`. |
| | `Variable.DiscreteUniform(int size)` | Creates an **int** random variable which can take values from 0 to size-1 with equal probability. |
| _Discrete_ _enum_ | `Variable.EnumDiscrete<TEnum>(double[] probs)` `Variable.EnumDiscrete<TEnum>(Vector probs)` | Creates an **enum** random variable where the probability of each enum value is given by the specified array or vector. _TEnum_ specifies the enum type. |
| | `Variable.EnumUniform<TEnum>()` | Creates an **enum** random variable with equal probability of taking each possible value of the enumeration. _TEnum_ specifies the enum type. |
| _Binomial_ | `Variable.Binomial(double probSuccess, int trialCount)` | Creates an **int** random variable which has a Binomial distribution with the specified probability of success per trial and number of trials. |
| _Multinomial_ | `Variable.Multinomial(Vector probs, int trialCount)` |Creates an **int\[\]** random variable array which has a Multinomial distribution with the specified array of probabilities and number of trials. |
| _Poisson_ | `Variable.Poisson(double mean)` | Creates an **int** random variable which has a Poisson distribution with the specified mean. |
| _DiscreteChar_ | `Variable.Char(Vector probs)` | Creates a **char** random variable from a supplied vector of character probabilities. The probabilities much add up to 1 and be provided for every possible character value, from **char.MinValue** to **char.MaxValue**. |
| | `Variable.CharUniform()` `Variable.CharLower()` `Variable.CharUpper()` `Variable.CharLetter()` `Variable.CharDigit()` `Variable.CharLetterOrDigit()` `Variable.CharWord()` `Variable.CharNonWord()` | Creates a **char** random variable having a uniform distribution over all possible (lowercase, uppercase, letters, digits, letters and digits, word, non-word) characters. |

#### Continuous Distributions

Distributions where the random variable can take one of a continuous range of values.

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| _Beta_ | `Variable.Beta(double trueCount, double falseCount)`| Creates a **double** random variable with a Beta distribution with the specified counts. |
| | `Variable.BetaFromMeanAndVariance(double mean, double variance)` | Creates a **double** random variable with a Beta distribution of the specified mean and variance. |
| _Gaussian_ | `Variable.GaussianFromMeanAndPrecision(double mean, double precision)` | Creates a **double** random variable with a Gaussian distribution of the specified mean and precision (inverse variance). |
| | `Variable.GaussianFromMeanAndVariance(double mean, double variance)` | Creates a **double** random variable with a Gaussian distribution of the specified mean and variance. |
| _Gamma_ | `Variable.GammaFromShapeAndScale(double shape, double scale)` | Creates a positive **double** random variable with a Gamma distribution of the specified shape and scale. |
| | `Variable.GammaFromShapeAndRate(double shape, double rate)` | Creates a positive **double** random variable with a Gamma distribution of the specified shape and rate. We now support both stochastic shape and rate, but the support for stochastic shape is experimental. |
| | `Variable.GammaFromMeanAndVariance(double mean, double variance)` | Creates a positive **double** random variable with a Gamma distribution of the specified mean and variance. |

#### Multivariate Distributions

Distributions where the random variable is multivariate.

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| _Dirichlet_ | `Variable.Dirichlet(double[] u)` `Variable.Dirichlet(Vector v)` | Creates a **Vector** random variable with a Dirichlet distribution with the specified array or vector of pseudo-counts. |
| | `Variable.DirichletUniform(int dimension)` | Creates a **Vector** random variable with a Dirichlet distribution whose pseudo-counts are all set to 1. |
| _Wishart_ | `Variable.WishartFromShapeAndScale(double shape, PositiveDefiniteMatrix scale)` | Creates a **PositiveDefiniteMatrix** random variable with a Wishart distribution with the specified shape parameter and scale matrix. |

#### Sequence Distributions

Distributions where the random variable can take one of a set of sequences of a certain type, like strings.

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| _StringDistribution_ | `Variable.StringUniform()` | Creates a uniformly distributed **string** random variable. The distribution of the variable is improper. |
| | `Variable.StringLower()` `Variable.StringLower(int minLength, int maxLength)` `Variable.StringUpper()` `Variable.StringUpper(int minLength, int maxLength)` `Variable.StringLetters()` `Variable.StringLetters(int minLength, int maxLength)` `Variable.StringDigits()` `Variable.StringDigits(int minLength, int maxLength)` `Variable.StringLettersOrDigits()` `Variable.StringLettersOrDigits(int minLength, int maxLength)` `Variable.StringWhitespace()` `Variable.StringWhitespace(int minLength, int maxLength)` `Variable.StringWord()` `Variable.StringWord(int minLength, int maxLength)` `Variable.StringNonWord()` `Variable.StringNonWord(int minLength, int maxLength)` | Creates a **string** random variable uniformly distributed over all either non-empty strings or strings with length in given bounds, containing lowercase (uppercase, letters, digits, letters and digits, word, non-word) characters only. If the upper length bound is not specified, the distribution of the variable is improper. |
| | `Variable.StringCapitalized()` `Variable.StringCapitalized(int minLength, int maxLength)` | Creates a **string** random variable uniformly distributed over all strings starting from an uppercase letter, followed by one or more lowercase letters. If the upper length bound is not specified, the distribution of the variable is improper. |
| | `Variable.StringOfLength(int length)` `Variable.StringOfLength(int length, DiscreteChar allowedCharacters)` | Creates a **string** random variable uniformly distributed over all strings of given length. String characters are restricted to be non zero probability characters under a given character distribution, if it is provided. |
| | `Variable.String(int minLength, int maxLength)` `Variable.String(int minLength, int maxLength, DiscreteChar allowedCharacters)` | Creates a **string** random variable uniformly distributed over all strings with length in given bounds. String characters are restricted to be non zero probability characters under a given character distribution, if it is provided. If the upper length bound is not specified, the distribution of the variable is improper. |
