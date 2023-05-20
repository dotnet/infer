// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Math
{
    using System.Collections.Generic;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <summary>
    /// This class provides a source of non-uniform random numbers.
    /// It cannot be instantiated and consists of only static functions.
    /// </summary>
    /// <remarks>A thread-static <c>System.Random</c> object provides the underlying random numbers.</remarks>
    public static class Rand
    {
        /// <summary>
        /// Generate random seeds for new threads.
        /// </summary>
        private static Random threadGenerator = new Random();

        /// <summary>
        /// Lock for threadGenerator.
        /// </summary>
        private static object threadGeneratorLock = new object();

        /// <summary>
        /// Supplies uniform random numbers for this thread.
        /// </summary>
        [ThreadStatic] private static Random _gen;

        /// <summary>
        /// This must be set up as a property to allow correct initialisation across threads.
        /// </summary>
        private static Random gen
        {
            get
            {
                if (_gen == null)
                {
                    lock (threadGeneratorLock)
                    {
                        // This approach avoids the problem of using a time-based seed as documented at:
                        // http://msdn.microsoft.com/en-us/library/vstudio/h343ddh9(v=vs.100).aspx
                        _gen = new Random(threadGenerator.Next());
                    }
                }
                return _gen;
            }
            set { _gen = value; }
        }

        /// <summary>
        /// If true, Normal() returns previousSample.
        /// </summary>
        [ThreadStatic] private static bool usePreviousSample; // = false;

        /// <summary>
        /// If usePreviousSample = true, this is the next value Normal() 
        /// will return.  Otherwise its value is unspecified.
        /// </summary>
        [ThreadStatic] private static double previousSample;

        /// <summary>
        /// Restarts the random number sequence for this thread (and influence future threads)
        /// </summary>
        /// <param name="seed">A number used to calculate a starting value for the pseudo-random number sequence.
        /// If a negative number is specified, the absolute value of the number is used.</param>
        public static void Restart(int seed)
        {
            gen = new Random(seed);
            lock (threadGeneratorLock)
            {
                threadGenerator = new Random(seed);
            }
            usePreviousSample = false;
        }

        /// <summary>
        /// Generates a non-negative random integer.
        /// </summary>
        /// <returns>A random integer >= 0.</returns>
        /// <remarks>Same as <see cref="System.Random.Next()"/>.</remarks>
        [Stochastic]
        public static int Int()
        {
            return gen.Next();
        }

        /// <summary>
        /// Generates a random integer x, 0 &lt;= x &lt; <paramref name="maxPlus1"/>
        /// </summary>
        /// <param name="maxPlus1">Upper bound.  Must be >= 0.</param>
        /// <returns>A random integer x, 0 &lt;= x &lt; <paramref name="maxPlus1"/>.  If <paramref name="maxPlus1"/> is zero, zero is returned.</returns>
        /// <remarks>Same as <see cref="System.Random.Next(int)"/>.</remarks>
        [Stochastic]
        public static int Int(int maxPlus1)
        {
            return gen.Next(maxPlus1);
        }

        /// <summary>
        /// Generates a random integer x, <paramref name="min"/> &lt;= x &lt; <paramref name="maxPlus1"/>.
        /// </summary>
        /// <param name="min">Minimum value.</param>
        /// <param name="maxPlus1">Maximum value plus 1.  Must be > <paramref name="min"/>.</param>
        /// <returns>A random integer x, <paramref name="min"/> &lt;= x &lt; <paramref name="maxPlus1"/>.</returns>
        /// <remarks>Same as <see cref="System.Random.Next(int,int)"/> except the equal-argument case is disallowed.</remarks>
        [Stochastic]
        public static int Int(int min, int maxPlus1)
        {
            if (min == maxPlus1)
                throw new ArgumentException("min == maxPlus1");
            return gen.Next(min, maxPlus1);
        }

        /// <summary>
        /// Generates a random double-precision value in [0,1).
        /// </summary>
        /// <returns>A random double.</returns>
        /// <remarks>Same as <see cref="System.Random.NextDouble"/>.</remarks>
        [Stochastic]
        public static double Double()
        {
            return gen.NextDouble();
        }

        /// <summary>
        /// Generates a random permutation.
        /// </summary>
        /// <param name="n">The length of permutation to make. Must be > 0.</param>
        /// <returns>An array of <paramref name="n"/> unique integers, each in the range [0,<paramref name="n"/>-1].</returns>
        [Stochastic]
        public static int[] Perm(int n)
        {
            int[] p = new int[n];
            for (int i = 0; i < n; i++) p[i] = i;
            Shuffle(p);
            return p;
        }

        /// <summary>
        /// Permute the elements of an array.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="array">The array to shuffle.</param>
        [Stochastic]
        public static void Shuffle<T>(T[] array)
        {
            int n = array.Length;
            // Algorithm: repeatedly draw random integers, without replacement.
            // We fill in the array from back to front. 
            // (would also work the other way)
            for (int i = n - 1; i > 0; i--)
            {
                int j = Rand.Int(i + 1);
                T temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }

        /// <summary>
        /// Permute the elements of a list.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="list">The list to shuffle.</param>
        [Stochastic]
        public static void Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            // Algorithm: repeatedly draw random integers, without replacement.
            // We fill in the array from back to front. 
            // (would also work the other way)
            for (int i = n - 1; i > 0; i--)
            {
                int j = Rand.Int(i + 1);
                T temp = list[i];
                list[i] = list[j];
                list[j] = temp;
            }
        }

        /// <summary>
        /// Draw a random subset of a list.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="list">The source list.</param>
        /// <param name="count">The number of elements to draw</param>
        /// <returns></returns>
        [Stochastic]
        public static HashSet<T> SampleWithoutReplacement<T>(IReadOnlyList<T> list, int count)
        {
            HashSet<T> set = new HashSet<T>();
            foreach (int index in SampleWithoutReplacement(list.Count, count))
            {
                set.Add(list[index]);
            }
            return set;
        }

        /// <summary>
        /// Draw a random subset of the integers between 0 and itemCount-1.
        /// </summary>
        /// <param name="itemCount">The number of elements to draw from.</param>
        /// <param name="count">The number of elements to draw</param>
        /// <returns></returns>
        [Stochastic]
        public static HashSet<int> SampleWithoutReplacement(int itemCount, int count)
        {
            if (count > itemCount)
                throw new ArgumentException("count > itemCount");
            HashSet<int> set = new HashSet<int>();
            if (count > itemCount / 2)
            {
                // use Shuffle
                int[] array = Util.ArrayInit(itemCount, i => i);
                Shuffle(array);
                for (int i = 0; i < count; i++)
                {
                    set.Add(array[i]);
                }
            }
            else
            {
                // use rejection
                for (int i = 0; i < count; i++)
                {
                    while (true)
                    {
                        int item = Rand.Int(itemCount);
                        if (!set.Contains(item))
                        {
                            set.Add(item);
                            break;
                        }
                    }
                }
            }
            return set;
        }

        /// <summary>
        /// Generates a random sample from a finite discrete distribution.
        /// </summary>
        /// <param name="prob">prob[i] >= 0 is the probability of outcome i, times an arbitrary constant.</param>
        /// <returns>An integer from 0 to <c>prob.Count</c>-1.</returns>
        /// <exception cref="AllZeroException">Thrown when prob is all zeros.</exception>
        [Stochastic]
        public static int Sample(Vector prob)
        {
            double sum = prob.Sum();
            if (sum == 0) throw new AllZeroException();
            return Sample(prob, sum);
        }

        /// <summary>
        /// Generates a random sample from a finite discrete distribution.
        /// </summary>
        /// <param name="prob">prob[i] is the probability of outcome i, times <paramref name="sum"/>.  Must be >= 0.</param>
        /// <param name="sum">The sum of the prob array.  Must be > 0.</param>
        /// <returns>An integer from 0 to <c>prob.Count</c>-1.</returns>
        [Stochastic]
        public static int Sample(Vector prob, double sum)
        {
            // Inversion method, Devroye chap 3
            int x = 0;
            double cumsum = prob[x];
            Assert.IsTrue(prob[x] >= 0, "negative probability");
            Assert.IsTrue(sum > 0, "sum is not positive");
            double u = gen.NextDouble() * sum;
            u = System.Math.Max(u, double.Epsilon);
            while (u > cumsum)
            {
                ++x;
                cumsum += prob[x];
                Assert.IsTrue(prob[x] >= 0, "negative probability");
            }
            return x;
        }

        /// <summary>
        /// Generates a random sample from a finite discrete distribution.
        /// </summary>
        /// <param name="prob">prob[i] is the probability of outcome i, times <paramref name="sum"/>.  Must be >= 0.</param>
        /// <param name="sum">The sum of the prob array.  Must be > 0.</param>
        /// <returns>An integer from 0 to <c>prob.Count</c>-1.</returns>
        [Stochastic]
        public static int Sample(IList<double> prob, double sum)
        {
            // Inversion method, Devroye chap 3
            int x = 0;
            double cumsum = prob[x];
            Assert.IsTrue(prob[x] >= 0, "negative probability");
            Assert.IsTrue(sum > 0, "sum is not positive");
            double u = gen.NextDouble() * sum;
            u = System.Math.Max(u, double.Epsilon);
            while (u > cumsum)
            {
                ++x;
                cumsum += prob[x];
                Assert.IsTrue(prob[x] >= 0, "negative probability");
            }
            return x;
        }

        /// <summary>
        /// Generates a random sample from a normal distribution.
        /// </summary>
        /// <returns>A finite real number.</returns>
        [Stochastic]
        public static double Normal()
        {
            double x1, x2;
            double w;

            /* We generate 2 values per iteration, saving one for the next call. */
            if (usePreviousSample)
            {
                usePreviousSample = false;
                return previousSample;
            }
            /* Generate a random point inside the unit circle */
            do
            {
                x1 = 2.0 * gen.NextDouble() - 1.0;
                x2 = 2.0 * gen.NextDouble() - 1.0;
                w = (x1 * x1) + (x2 * x2);
            } while ((w >= 1.0) || (w == 0.0));

            /* Apply the Box-Muller formula */
            w = System.Math.Sqrt(-2.0 * System.Math.Log(w) / w);
            x1 = w * x1;
            x2 = w * x2;

            usePreviousSample = true;
            previousSample = x2;
            return (x1);
        }

        /// <summary>
        /// Generates a random sample from a normal distribution.
        /// </summary>
        /// <param name="mean">The mean of the distribution.  Must be finite.</param>
        /// <param name="stdDev">The standard deviation (sqrt of the variance).</param>
        /// <returns>A finite real number.</returns>
        [Stochastic]
        public static double Normal(double mean, double stdDev)
        {
            return mean + stdDev * Normal();
        }

        /// <summary>
        /// Generates a random sample from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">The mean of the distribution.  Must be finite.</param>
        /// <param name="variance">The covariance matrix.  Must be positive-definite.</param>
        /// <param name="result">Receives the result.  Must be non-null and the correct size.</param>
        [Stochastic]
        public static void Normal(Vector mean, PositiveDefiniteMatrix variance, Vector result)
        {
            LowerTriangularMatrix L = new LowerTriangularMatrix(variance.Rows, variance.Rows);
            bool isPosDef = L.SetToCholesky(variance);
            if (!isPosDef)
            {
                throw new ArgumentException("PROB.Random.Normal(Vector,Matrix,Vector): variance must be positive definite", nameof(variance));
            }

            NormalChol(mean, L, result);
        }

        /// <summary>
        /// Generates a random sample from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">The mean of the distribution.  Must be finite.</param>
        /// <param name="precision">The inverse of the covariance matrix.  Must be positive-definite.</param>
        /// <param name="result">Receives the result.  Must be non-null and the correct size.</param>
        [Stochastic]
        public static void NormalP(Vector mean, PositiveDefiniteMatrix precision, Vector result)
        {
            LowerTriangularMatrix L = new LowerTriangularMatrix(precision.Rows, precision.Rows);
            bool isPosDef = L.SetToCholesky(precision);
            if (!isPosDef)
            {
                throw new ArgumentException("PROB.Random.NormalP(Vector,Matrix,Vector): precision must be positive definite", nameof(precision));
            }
            UpperTriangularMatrix U = UpperTriangularMatrix.TransposeInPlace(L);
            NormalPChol(mean, U, result);
        }

        /// <summary>
        /// Generates a random sample from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">The mean of the distribution.  Must be finite.</param>
        /// <param name="varChol">The lower triangular Cholesky factor of the covariance matrix.  Must be positive-definite.</param>
        /// <param name="result">Receives the result.  Must be non-null and the correct size.</param>
        [Stochastic]
        public static void NormalChol(Vector mean, LowerTriangularMatrix varChol, Vector result)
        {
            if (ReferenceEquals(mean, result))
                throw new ArgumentException("mean and result are the same object");
            Vector temp = Vector.Zero(result.Count);
            for (int i = 0; i < result.Count; i++) temp[i] = Normal();
            result.SetToProduct(varChol, temp);
            result.SetToSum(mean, result);
            // result = mean + varChol*N(0,1)
        }

        /// <summary>
        /// Generates a random sample from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">The mean of the distribution.  Must be finite.</param>
        /// <param name="precCholT">The upper triangular transpose of the Cholesky factor of the precision matrix.  Must be positive-definite.</param>
        /// <param name="result">Receives the result.  Must be non-null and the correct size.</param>
        [Stochastic]
        public static void NormalPChol(Vector mean, UpperTriangularMatrix precCholT, Vector result)
        {
            if (ReferenceEquals(mean, result))
                throw new ArgumentException("mean and result are the same object");
            for (int i = 0; i < result.Count; i++) result[i] = Normal();
            result.PredivideBy(precCholT);
            result.SetToSum(mean, result);
            // result = mean + precChol'\N(0,1)
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// Sample from a Gaussian(0,1) truncated at the given lower bound
        /// </summary>
        /// <param name="lowerBound">The truncation point.  Can be -Infinity.</param>
        /// <returns>A real number &gt;= <paramref name="lowerBound"/></returns>
        [Stochastic]
        public static double NormalGreaterThan(double lowerBound)
        {
            if (lowerBound < 1)
            {
                // simple rejection
                while (true)
                {
                    double x = Rand.Normal();
                    if (x >= lowerBound) return x;
                }
            }
            else
            {
                // lowerBound >= 1
                if (false)
                {
                    // Devroye's (Ch.9) rejection sampler with x ~ Exp(lowerBound)
                    // requires lowerBound > 0
                    double c = 2 * lowerBound * lowerBound;
                    while (true)
                    {
                        // note it is possible to generate two exponential r.v.s with a single logarithm (Devroye Ch.9 sec.2.1)
                        double E = -System.Math.Log(Rand.Double());
                        double E2 = -System.Math.Log(Rand.Double());
                        if (E * E <= c * E2) return lowerBound + E / lowerBound;
                    }
                }
                else
                {
                    // Marsaglia's (1964) rejection sampler (Devroye Ch.9)
                    // Reference: "Non-Uniform Random Variate Generation" by Luc Devroye (1986)
                    double c = lowerBound * lowerBound * 0.5;
                    while (true)
                    {
                        double U = Rand.Double();
                        double V = Rand.Double();
                        double x = c - System.Math.Log(U);
                        if (V * V * x <= c)
                        {
                            return System.Math.Sqrt(2 * x);
                        }
                    }
                }
                throw new InferRuntimeException("failed to sample");
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>
        /// Sample from a uniform distribution truncated at the given upper and lower bounds.
        /// </summary>
        /// <param name="lowerBound">Must be finite.</param>
        /// <param name="upperBound">Must be &gt;= <paramref name="lowerBound"/>.  Must be finite.</param>
        /// <returns>A real number &gt;= <paramref name="lowerBound"/> and &lt; <paramref name="upperBound"/></returns>
        public static double UniformBetween(double lowerBound, double upperBound)
        {
            double delta = upperBound - lowerBound;
            if (delta == 0)
                return lowerBound;
            if (delta < 0)
                throw new ArgumentException("upperBound (" + upperBound + ") < lowerBound (" + lowerBound + ")");
            if (double.IsPositiveInfinity(delta))
                throw new ArgumentException($"lowerBound ({lowerBound}) or upperBound ({upperBound}) is infinite.  Both bounds must be finite.");
            double x = Rand.Double() * delta + lowerBound;
            return x;
        }

        /// <summary>
        /// Sample from a Gaussian(0,1) truncated at the given upper and lower bounds
        /// </summary>
        /// <param name="lowerBound">Can be -Infinity.</param>
        /// <param name="upperBound">Must be &gt;= <paramref name="lowerBound"/>.  Can be Infinity.</param>
        /// <returns>A real number &gt;= <paramref name="lowerBound"/> and &lt; <paramref name="upperBound"/></returns>
        public static double NormalBetween(double lowerBound, double upperBound)
        {
            if (double.IsNaN(lowerBound))
                throw new ArgumentException("lowerBound is NaN");
            if (double.IsNaN(upperBound))
                throw new ArgumentException("upperBound is NaN");
            double delta = upperBound - lowerBound;
            if (delta == 0) return lowerBound;
            if (delta < 0) throw new ArgumentException("upperBound (" + upperBound + ") < lowerBound (" + lowerBound + ")");
            // Switch between the following 3 options:
            // 1. Gaussian rejection, with acceptance rate Z = NormalCdf(upperBound) - NormalCdf(lowerBound)
            // 2. Uniform rejection, with acceptance rate sqrt(2*pi)*Z/delta if the interval contains 0
            // 3. Truncated exponential rejection, with acceptance rate
            //    = sqrt(2*pi)*Z*lambda*exp(-lambda^2/2)/(exp(-lambda*lowerBound)-exp(-lambda*upperBound))
            //    = sqrt(2*pi)*Z*lowerBound*exp(lowerBound^2/2)/(1-exp(-lowerBound*(upperBound-lowerBound)))
            // (3) has the highest acceptance rate under the following conditions:
            //     lowerBound > 0.5 or (lowerBound > 0 and delta < 2.5)
            // (2) has the highest acceptance rate if the interval contains 0 and delta < sqrt(2*pi)
            // (1) has the highest acceptance rate otherwise
            if (lowerBound > 0.5 || (lowerBound > 0 && delta < 2.5))
            {
                // Rejection sampler using truncated exponential proposal
                double lambda = lowerBound;
                double s = MMath.ExpMinus1(-lambda * delta);
                double c = 2 * lambda * lambda;
                while (true)
                {
                    double x = -MMath.Log1Plus(s * Rand.Double());
                    double u = -System.Math.Log(Rand.Double());
                    if (c * u > x * x) return x / lambda + lowerBound;
                }
                throw new InferRuntimeException("failed to sample");
            }
            else if (upperBound < -0.5 || (upperBound < 0 && delta < 2.5))
            {
                return -NormalBetween(-upperBound, -lowerBound);
            }
            else if (lowerBound <= 0 && upperBound >= 0 && delta < MMath.Sqrt2PI)
            {
                // Uniform rejection
                while (true)
                {
                    double x = Rand.Double() * delta + lowerBound;
                    double u = -System.Math.Log(Rand.Double());
                    if (2 * u > x * x) return x;
                }
            }
            else
            {
                // Gaussian rejection
                while (true)
                {
                    double x = Rand.Normal();
                    if (x >= lowerBound && x < upperBound) return x;
                }
            }
        }

        // Samples Gamma assuming a >= 1.0
        // Reference: G. Marsaglia and W.W. Tsang, A simple method for generating gamma
        // variables, ACM Transactions on Mathematical Software, Vol. 26, No. 3,
        // Pages 363-372, September, 2000.
        // http://portal.acm.org/citation.cfm?id=358414
        private static double GammaShapeGE1(double a)
        {
            double d = a - 1.0 / 3, c = 1.0 / System.Math.Sqrt(9 * d);
            double v;
            while (true)
            {
                double x;
                do
                {
                    x = Normal();
                    v = 1 + c * x;
                } while (v <= 0);
                v = v * v * v;
                x = x * x;
                double u = gen.NextDouble();
#if false
    // first version
                if(Math.Log(u) < 0.5*x + d*(1-v+Math.Log(v)))
                {
                    break;
                }
#else
                // faster version
                if ((u < 1 - .0331 * x * x) || (System.Math.Log(u) < 0.5 * x + d * (1 - v + System.Math.Log(v))))
                {
                    break;
                }
#endif
            }
            return d * v;
        }

        /// <summary>
        /// Generates a random sample from a Gamma distribution.
        /// </summary>
        /// <param name="shape">The shape parameter.  Must be finite and > 0.</param>
        /// <returns>A nonnegative finite real number.  May be zero.</returns>
        /// <remarks>The distribution is defined as p(x) = x^(a-1)*exp(-x)/Gamma(a).
        /// To incorporate a scale parameter b, multiply the result by b.</remarks>
        [Stochastic]
        public static double Gamma(double shape)
        {
            if (shape < 1)
                // boost using Marsaglia's (1961) method: gam(a) = gam(a+1)*U^(1/a)
                return GammaShapeGE1(shape + 1) * System.Math.Exp(System.Math.Log(gen.NextDouble()) / shape);
            else
                return GammaShapeGE1(shape);
        }

        [Stochastic]
        private static double LogGamma(double shape)
        {
            if (shape < 1)
                // boost using Marsaglia's (1961) method: gam(a) = gam(a+1)*U^(1/a)
                return System.Math.Log(GammaShapeGE1(shape + 1)) + System.Math.Log(gen.NextDouble()) / shape;
            else
                return System.Math.Log(GammaShapeGE1(shape));
        }

        /// <summary>
        /// Generates a random sample from a Wishart distribution.
        /// </summary>
        /// <param name="a">The shape parameter.  Must be finite and > 0.</param>
        /// <param name="result">Receives the lower triangular Cholesky factor of the sampled matrix.  Must be non-null, square, and already allocated to the desired size.</param>
        /// <remarks><para>
        /// The <a href="http://en.wikipedia.org/wiki/Wishart_distribution">Wishart distribution</a> 
        /// is defined as
        /// </para><para>
        /// p(X) = |X|^((n-d-1)/2)*exp(-tr(X))/Gamma_d(n/2) (using parameter n)
        /// </para><para>
        /// or 
        /// </para><para>
        /// p(X) = |X|^(a-(d+1)/2)*exp(-tr(X))/Gamma_d(a) (using parameter a)
        /// </para><para>
        /// This routine returns chol(X).  To incorporate a scale parameter C, 
        /// set Y = chol(C)*X*chol(C)', which implies chol(Y) = chol(C)*chol(X). 
        /// If you invert and transpose chol(X), then you have chol(inv(X)), 
        /// where inv(X) is a sample from the inverse Wishart distribution: 
        /// </para><para>
        /// p(X) = |X|^(-a-(d+1)/2)*exp(-tr(inv(X)))/Gamma_d(a)
        /// </para></remarks>
        /// <example>
        /// <code>
        /// Matrix L = new Matrix(d,d);
        /// Rand.Wishart(a,L);
        /// Matrix X = new Matrix(d,d);
        /// X.SetToProduct(L, L.Transpose());
        /// </code>
        /// </example>
        [Stochastic]
        public static void Wishart(double a, LowerTriangularMatrix result)
        {
            Assert.IsTrue(result.Rows == result.Cols);
            for (int i = 0; i < result.Rows; ++i)
            {
                for (int j = 0; j < i; ++j)
                {
                    result[i, j] = Normal() * MMath.SqrtHalf;
                    result[j, i] = 0;
                }
                result[i, i] = System.Math.Sqrt(Gamma(a - i * 0.5));
            }
        }

        /// <summary>
        /// Generates a random sample from the Beta distribution with given parameters.
        /// </summary>
        /// <param name="trueCount"></param>
        /// <param name="falseCount"></param>
        /// <returns></returns>
        [Stochastic]
        public static double Beta(double trueCount, double falseCount)
        {
            double gFalse, gTrue;
            if (trueCount < 1 && falseCount < 1)
            {
                // To handle small counts, use Stuart's (1962) theorem:
                // gamma(a) has the same distribution as gamma(a+1)*exp(log(U)/a)
                double boost1 = System.Math.Log(gen.NextDouble()) / trueCount;
                trueCount++;
                double boost2 = System.Math.Log(gen.NextDouble()) / falseCount;
                falseCount++;
                if (boost1 > boost2)
                {
                    // divide by exp(boost1)
                    gTrue = Rand.Gamma(trueCount);
                    gFalse = Rand.Gamma(falseCount) * System.Math.Exp(boost2 - boost1);
                }
                else
                {
                    // divide by exp(boost2)
                    gTrue = Rand.Gamma(trueCount) * System.Math.Exp(boost1 - boost2);
                    gFalse = Rand.Gamma(falseCount);
                }
            }
            else
            {
                gTrue = Rand.Gamma(trueCount);
                gFalse = Rand.Gamma(falseCount);
            }
            return gTrue / (gTrue + gFalse);
        }

        /// <summary>
        /// Generates a random sample from the Dirichlet distribution with the given
        /// pseudo-count
        /// </summary>
        /// <param name="pseudoCount">Pseudo-count</param>
        /// <param name="result">Where to put the result</param>
        /// <returns>The sample</returns>
        /// <remarks>If pseudoCount is a sparse vector and the common
        /// value for the sparse vector is not 0, then the result will be dense; in such
        /// a case it is recommended that 'result' be a dense vector type.</remarks>
        [Stochastic]
        public static Vector Dirichlet(Vector pseudoCount, Vector result)
        {
            // If pseudo-count is sparse and its common value is not 0, we need
            // to process it as a dense vector so that samples from common value
            // are allowed to differ
            if (pseudoCount.IsSparse && ((SparseVector)pseudoCount).CommonValue != 0.0)
                pseudoCount = DenseVector.Copy(pseudoCount);

            if (pseudoCount.Max() < 1)
            {
                // To handle small counts, use Stuart's (1962) theorem:
                // gamma(a) has the same distribution as gamma(a+1)*exp(log(U)/a)
                Vector boost = Vector.Copy(pseudoCount);
                boost.SetToFunction(pseudoCount, a => System.Math.Log(gen.NextDouble()) / a);
                double maxBoost = boost.Max();
                result.SetToFunction(pseudoCount, boost, (a, b) => Rand.Gamma(a + 1) * System.Math.Exp(b - maxBoost));
            }
            else
            {
                result.SetToFunction(pseudoCount, a => Rand.Gamma(a));
            }
            double sum = result.Sum();
            result.Scale(1.0 / sum);
            return result;
        }

        /// <summary>
        /// Generates a random sample from the Binomial distribution with parameters p and n.
        /// </summary>
        /// <param name="n">Number of trials</param>
        /// <param name="p">Probability of success per trial</param>
        /// <remarks>
        /// Reference:
        ///  [1]  L. Devroye, "Non-Uniform Random Variate Generation", 
        ///  Springer-Verlag, 1986.
        ///  [2] Kachitvichyanukul, V., and Schmeiser, B. W. "Binomial Random Variate Generation." 
        ///  Comm. ACM, 31, 2 (Feb. 1988), 216.
        /// </remarks>
        [Stochastic]
        public static int Binomial(int n, double p)
        {
            if (p * (n + 1) == n + 1) return n;
            if (n < 15)
            {
                // coin flip method
                // this takes O(n) time
                int result = 0;
                while (n-- > 0)
                {
                    if (Rand.Double() < p) result++;
                }
                return result;
            }
            else if (n * p < 150)
            {
                // waiting time method
                // this takes O(np) time
                double q = -System.Math.Log(1 - p);
                int r = n;
                double e = -System.Math.Log(Rand.Double());
                double s = e / r;
                while (s <= q)
                {
                    r--;
                    if (r == 0) break;
                    e = -System.Math.Log(Rand.Double());
                    s = s + e / r;
                }
                return n - r;
            }
            else
            {
                // recursive method
                // this makes O(log(log(n))) recursive calls
                int i = (int)(p * (n + 1));
                double b = Rand.Beta(i, n + 1 - i);
                if (b <= p)
                    return i + Rand.Binomial(n - i, (p - b) / (1 - b));
                else
                    return i - 1 - Rand.Binomial(i - 1, (b - p) / b);
            }
        }

        /// <summary>
        /// Sample from a Multinomial distribution with specified probabilities and number of trials.
        /// </summary>
        /// <param name="trialCount">Number of trials, >= 0</param>
        /// <param name="probs">Must sum to 1</param>
        /// <returns>An array of length <c>probs.Count</c> of integers between 0 and trialCount, whose sum is trialCount.</returns>
        [Stochastic]
        public static int[] Multinomial(int trialCount, Vector probs)
        {
            return Multinomial(trialCount, (DenseVector)probs);
        }

        /// <summary>
        /// Sample from a Multinomial distribution with specified dense vector of probabilities and number of trials.
        /// </summary>
        /// <param name="trialCount"></param>
        /// <param name="probs"></param>
        /// <returns></returns>
        [Stochastic]
        public static int[] Multinomial(int trialCount, DenseVector probs)
        {
            int[] result = new int[probs.Count];
            if (probs.Count == 0) return result;
            double remainingProb = 1;
            for (int dim = 0; dim < result.Length - 1; dim++)
            {
                int sample = Binomial(trialCount, probs[dim] / remainingProb);
                result[dim] = sample;
                trialCount -= sample;
                remainingProb -= probs[dim];
                if (remainingProb <= 0) break;
            }
            result[result.Length - 1] = trialCount;
            return result;
        }

        /// <summary>
        /// Sample from a Poisson distribution with specified mean.
        /// </summary>
        /// <param name="mean">Must be >= 0</param>
        /// <returns>An integer in [0,infinity)</returns>
        [Stochastic]
        public static int Poisson(double mean)
        {
            // TODO: There are more efficient samplers
            if (mean < 0) throw new ArgumentException("mean < 0");
            else if (mean == 0.0) return 0;
            else if (mean < 10)
            {
                double L = System.Math.Exp(-mean);
                double p = 1.0;
                int k = 0;
                do
                {
                    k++;
                    p *= Rand.Double();
                } while (p > L);
                return k - 1;
            }
            else
            {
                // mean >= 10
                // Devroye ch10.3, with corrections
                // Reference: "Non-Uniform Random Variate Generation" by Luc Devroye (1986)
                double mu = System.Math.Floor(mean);
                double muLogFact = MMath.GammaLn(mu + 1);
                double logMeanMu = System.Math.Log(mean / mu);
                double delta = System.Math.Max(6, System.Math.Min(mu, System.Math.Sqrt(2 * mu * System.Math.Log(128 * mu / System.Math.PI))));
                double c1 = System.Math.Sqrt(System.Math.PI * mu / 2);
                double c2 = c1 + System.Math.Sqrt(System.Math.PI * (mu + delta / 2) / 2) * System.Math.Exp(1 / (2 * mu + delta));
                double c3 = c2 + 2;
                double c4 = c3 + System.Math.Exp(1.0 / 78);
                double c = c4 + 2 / delta * (2 * mu + delta) * System.Math.Exp(-delta / (2 * mu + delta) * (1 + delta / 2));
                while (true)
                {
                    double u = Rand.Double() * c;
                    double x, w;
                    if (u <= c1)
                    {
                        double n = Rand.Normal();
                        double y = -System.Math.Abs(n) * System.Math.Sqrt(mu) - 1;
                        x = System.Math.Floor(y);
                        if (x < -mu) continue;
                        w = -n * n / 2;
                    }
                    else if (u <= c2)
                    {
                        double n = Rand.Normal();
                        double y = 1 + System.Math.Abs(n) * System.Math.Sqrt(mu + delta / 2);
                        x = System.Math.Ceiling(y);
                        if (x > delta) continue;
                        w = (2 - y) * y / (2 * mu + delta);
                    }
                    else if (u <= c3)
                    {
                        x = 0;
                        w = 0;
                    }
                    else if (u <= c4)
                    {
                        x = 1;
                        w = 0;
                    }
                    else
                    {
                        double v = -System.Math.Log(Rand.Double());
                        double y = delta + v * 2 / delta * (2 * mu + delta);
                        x = System.Math.Ceiling(y);
                        w = -delta / (2 * mu + delta) * (1 + y / 2);
                    }
                    double e = -System.Math.Log(Rand.Double());
                    w -= e + x * logMeanMu;
                    double qx = x * System.Math.Log(mu) - MMath.GammaLn(mu + x + 1) + muLogFact;
                    if (w <= qx) return (int)System.Math.Round(x + mu);
                }
            }
        }
    }

    /// <summary>
    /// Exception type thrown when probability vector = (0,0,0,...,0).
    /// </summary>
    [Serializable]
    public class AllZeroException : Exception
    {
        /// <summary>
        /// Constructs the exception.
        /// </summary>
        public AllZeroException()
            : base("The model has zero probability")
        {
        }

        /// <summary>
        /// Constructs the exception with a message
        /// </summary>
        /// <param name="message"></param>
        public AllZeroException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AllZeroException"/> class with a specified error message 
        /// and a reference to the inner exception that is the cause of this exception. 
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public AllZeroException(string message, Exception inner)
            : base(message, inner)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AllZeroException"/> class.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected AllZeroException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}

namespace Microsoft.ML.Probabilistic.Factors.Attributes
{
    /// <summary>
    /// When applied to a method, indicates that the method is non-deterministic.
    /// </summary>
    /// <remarks>
    /// A method is non-deterministic if its return value is not completely determined by its arguments.
    /// For a void method, this attribute is meaningless.
    /// </remarks>
    public class Stochastic : Attribute
    {
    }
}