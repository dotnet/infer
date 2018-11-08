// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;
    using Math;
    using Utilities;
    using Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// A Poisson distribution over the integers [0,infinity).
    /// </summary>
    /// <remarks><para>
    /// The Poisson is often used as a distribution on counts.
    /// The formula for the distribution is <c>p(x) = rate^x exp(-rate) / x!</c>
    /// where rate is the rate parameter.
    /// This implementation uses a generalization called the Conway-Maxwell Poisson or "Com-Poisson", which
    /// has an extra precision parameter nu.  
    /// The formula for the distribution is <c>p(x) =propto rate^x / x!^nu</c>.
    /// The Com-Poisson can represent a uniform distribution via (rate=1,nu=0) and 
    /// a point mass via rate=0 or nu=infinity.  
    /// This family is closed under multiplication, while the standard Poisson is not.
    /// </para></remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Stable)]
    public struct Poisson : IDistribution<int>,
                            SettableTo<Poisson>, SettableToProduct<Poisson>, SettableToRatio<Poisson>,
                            SettableToPower<Poisson>, SettableToWeightedSum<Poisson>,
                            CanGetLogAverageOf<Poisson>, CanGetLogAverageOfPower<Poisson>,
                            CanGetAverageLog<Poisson>,
                            Sampleable<int>, CanGetMean<double>, CanGetVariance<double>,
                            CanGetMeanAndVarianceOut<double, double>, CanSetMeanAndVariance<double, double>
    {
        /// <summary>
        /// The rate parameter of the COM-Poisson distribution, always >= 0.
        /// </summary>
        /// <remarks>
        /// The natural parameter of the distribution is log(rate).  
        /// However, since rate remains >= 0 under multiplication, there
        /// is no harm in using rate as the parameter.
        /// </remarks>
        [DataMember]
        public double Rate;

        /// <summary>
        /// The precision parameter of the COM-Poisson distribution
        /// </summary>
        [DataMember]
        public double Precision;

        /// <summary>
        /// Gets the expected value E(x)
        /// </summary>
        /// <returns>E(x)</returns>
        public double GetMean()
        {
            if (IsPointMass) return Point;
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else if (Precision == 1) return Rate;
            else return Math.Exp(GetLogPowerSum(Rate, Precision, 1) - GetLogNormalizer(Rate, Precision));
        }

        /// <summary>
        /// Gets the variance
        /// </summary>
        /// <returns>Variance</returns>
        public double GetVariance()
        {
            if (IsPointMass) return 0.0;
            else if (IsUniform()) return Double.PositiveInfinity;
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else if (Precision == 1) return Rate;
            else
            {
                double m, v;
                GetMeanAndVariance(out m, out v);
                return v;
            }
        }

        /// <summary>
        /// Gets the mean and variance
        /// </summary>
        /// <param name="mean">Where to put the mean</param>
        /// <param name="variance">Where to put the variance</param>
        public void GetMeanAndVariance(out double mean, out double variance)
        {
            if (IsPointMass)
            {
                mean = Point;
                variance = 0;
            }
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else if (Precision == 1)
            {
                mean = Rate;
                variance = Rate;
            }
            else
            {
                double logZ = GetLogNormalizer(Rate, Precision);
                double m = Math.Exp(GetLogPowerSum(Rate, Precision, 1) - logZ);
                double m2 = Math.Exp(GetLogPowerSum(Rate, Precision, 2) - logZ);
                mean = m;
                variance = m2 - m*m;
            }
        }

        /// <summary>
        /// Sets the mean and variance
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        public void SetMeanAndVariance(double mean, double variance)
        {
            if (variance == 0.0)
            {
                Point = (int) mean;
                if (mean <= 0 || Point != mean) throw new ArgumentException("mean must be a positive integer");
            }
            else if (Double.IsPositiveInfinity(variance)) SetToUniform();
            else if (mean == variance)
            {
                Rate = mean;
                Precision = 1;
            }
            else
            {
                throw new NotImplementedException("Precision != 1 is not implemented");
            }
        }

        /// <summary>
        /// Clones this COM-Poisson. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a Poisson type</returns>
        public object Clone()
        {
            return new Poisson(this);
        }

        /// <summary>
        /// Sets/gets the instance as a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public int Point
        {
            get { return (int) Rate; }
            set
            {
                Rate = value;
                Precision = Double.PositiveInfinity;
            }
        }

        /// <summary>
        /// Asks whether the instance is a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return (Rate == 0 || Double.IsPositiveInfinity(Precision)); }
        }

        /// <summary>
        /// Returns the maximum difference between the parameters of this COM-Poisson
        /// and another COM-Poisson
        /// </summary>
        /// <param name="that">The other COM-Poisson</param>
        /// <returns>The maximum difference</returns>
        public double MaxDiff(object that)
        {
            if (!(that is Poisson)) return Double.PositiveInfinity;
            Poisson thatd = (Poisson) that;
            return Math.Max(MMath.AbsDiff(Rate, thatd.Rate), MMath.AbsDiff(Precision, thatd.Precision));
        }

        /// <summary>
        /// Override of the Equals method
        /// </summary>
        /// <param name="thatd">The instance to compare to</param>
        /// <returns>True if the two distributions are the same in value, false otherwise</returns>
        /// <exclude/>
        public override bool Equals(object thatd)
        {
            return (MaxDiff(thatd) == 0.0);
        }

        /// <summary>
        /// Override of GetHashCode method
        /// </summary>
        /// <returns>The hash code for this instance</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            return Hash.Combine(Rate.GetHashCode(), Precision.GetHashCode());
        }

        /// <summary>
        /// Equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator ==(Poisson a, Poisson b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Not equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator !=(Poisson a, Poisson b)
        {
            return !(a == b);
        }

        /// <summary>
        /// Sets this COM-Poisson instance to be a uniform distribution
        /// </summary>
        public void SetToUniform()
        {
            Rate = 1;
            Precision = 0;
        }

        /// <summary>
        /// Asks whether this instance is uniform
        /// </summary>
        /// <returns>True if uniform, false otherwise</returns>
        public bool IsUniform()
        {
            return (Rate == 1) && (Precision == 0);
        }

        /// <summary>
        /// Asks whether this instance is proper or not. A COM-Poisson distribution
        /// is proper if Rate >= 0 and (Precision > 0 or (Precision == 0 and Rate &lt; 1)).
        /// </summary>
        /// <returns>True if proper, false otherwise</returns>
        public bool IsProper()
        {
            return (Rate >= 0) && (Precision > 0 || (Precision == 0 && Rate < 1));
        }

        /// <summary>
        /// Evaluates the log of of the density function of this COM-Poisson at the given value
        /// </summary>
        /// <param name="value">The value at which to calculate the density</param>
        /// <returns>log p(x=value)</returns>
        /// <remarks>
        /// The formula for the distribution is <c>p(x) = rate^x exp(-rate) / x!</c>
        /// when nu=1.  In the general case, it is <c>p(x) =propto rate^x / x!^nu</c>.
        /// If the distribution is improper, the normalizer is omitted.
        /// </remarks>
        public double GetLogProb(int value)
        {
            if (IsPointMass) return (value == Point) ? 0.0 : Double.NegativeInfinity;
            else if (value < 0) return Double.NegativeInfinity;
            else if (IsUniform()) return 0.0;
            else if (Precision == 1)
            {
                // we know that Rate > 0
                return value*Math.Log(Rate) - MMath.GammaLn(value + 1) - Rate;
            }
            else
            {
                return value*Math.Log(Rate) - Precision*MMath.GammaLn(value + 1) - GetLogNormalizer(Rate, Precision);
            }
        }

        /// <summary>
        /// Sets this COM-Poisson instance to have the parameter values of another COM-Poisson instance
        /// </summary>
        /// <param name="value">The other COM-Poisson</param>
        public void SetTo(Poisson value)
        {
            Rate = value.Rate;
            Precision = value.Precision;
        }

        /// <summary>
        /// Sets the parameters to represent the product of two COM-Poissons.
        /// </summary>
        /// <param name="a">The first COM-Poisson</param>
        /// <param name="b">The second COM-Poisson</param>
        public void SetToProduct(Poisson a, Poisson b)
        {
            if (a.IsPointMass)
            {
                if (b.IsPointMass && !a.Point.Equals(b.Point))
                {
                    throw new AllZeroException();
                }
                Point = a.Point;
            }
            else if (b.IsPointMass)
            {
                Point = b.Point;
            }
            else
            {
                Rate = a.Rate*b.Rate;
                Precision = a.Precision + b.Precision;
            }
        }

        /// <summary>
        /// Creates a new COM-Poisson which is the product of two other COM-Poissons
        /// </summary>
        /// <param name="a">The first COM-Poisson</param>
        /// <param name="b">The second COM-Poisson</param>
        /// <returns>Result</returns>
        public static Poisson operator *(Poisson a, Poisson b)
        {
            Poisson result = new Poisson();
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the ratio of two COM-Poisson distributions
        /// </summary>
        /// <param name="numerator">The numerator distribution</param>
        /// <param name="denominator">The denominator distribution</param>
        /// <param name="forceProper">If true, the result has precision >= 0 and rate &lt;= 1</param>
        /// <remarks>
        /// The result may not be proper. No error is thrown in this case.
        /// </remarks>
        public void SetToRatio(Poisson numerator, Poisson denominator, bool forceProper = false)
        {
            if (numerator.IsPointMass)
            {
                if (denominator.IsPointMass)
                {
                    if (numerator.Point.Equals(denominator.Point))
                    {
                        SetToUniform();
                    }
                    else
                    {
                        throw new DivideByZeroException();
                    }
                }
                else
                {
                    Point = numerator.Point;
                }
            }
            else if (denominator.IsPointMass)
            {
                throw new DivideByZeroException();
            }
            else
            {
                Rate = numerator.Rate/denominator.Rate;
                Precision = numerator.Precision - denominator.Precision;
                if (forceProper && !IsProper()) throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Creates a new Poisson which is the ratio of two other COM-Poissons
        /// </summary>
        /// <param name="numerator">The numerator distribution</param>
        /// <param name="denominator">The denominator distribution</param>
        /// <returns>Result</returns>
        public static Poisson operator /(Poisson numerator, Poisson denominator)
        {
            Poisson result = new Poisson();
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the power of a COM-Poisson to some exponent.
        /// </summary>
        /// <param name="dist">The source Poisson</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(Poisson dist, double exponent)
        {
            if (dist.IsPointMass)
            {
                if (exponent == 0)
                {
                    SetToUniform();
                }
                else if (exponent < 0)
                {
                    throw new DivideByZeroException("The exponent is negative and the distribution is shape point mass");
                }
                else
                {
                    Point = dist.Point;
                }
                return;
            }
            Rate = Math.Pow(dist.Rate, exponent);
            Precision = dist.Precision*exponent;
        }

        /// <summary>
        /// Raises a COM-Poisson to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static Poisson operator ^(Poisson dist, double exponent)
        {
            Poisson result = new Poisson();
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Set the parameters to match the moments of a mixture distribution.
        /// </summary>
        /// <param name="dist1">The first distribution</param>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist2">The second distribution</param>
        /// <param name="weight2">The second weight</param>
        public void SetToSum(double weight1, Poisson dist1, double weight2, Poisson dist2)
        {
            if (weight1 + weight2 == 0) SetToUniform();
            else if (weight1 + weight2 < 0)
                throw new ArgumentException("weight1 (" + weight1 + ") + weight2 (" + weight2 + ") < 0");
            else if (weight1 == 0) SetTo(dist2);
            else if (weight2 == 0) SetTo(dist1);
                // if dist1 == dist2 then we must return dist1, with no roundoff error
            else if (dist1.Equals(dist2)) SetTo(dist1);
            else if (double.IsPositiveInfinity(weight1))
            {
                if (double.IsPositiveInfinity(weight2))
                {
                    throw new ArgumentException("both weights are infinity");
                }
                else
                {
                    SetTo(dist1);
                }
            }
            else if (double.IsPositiveInfinity(weight2)) SetTo(dist2);
            else
            {
                double m1, mlf1, m2, mlf2;
                m1 = dist1.GetMean();
                m2 = dist2.GetMean();
                mlf1 = dist1.GetMeanLogFactorial();
                mlf2 = dist2.GetMeanLogFactorial();
                double invW = 1.0/(weight1 + weight2);
                // check for equality to avoid roundoff errors
                double mean = (m1 == m2) ? m1 : (weight1*m1 + weight2*m2)*invW;
                double meanLogFact = (mlf1 == mlf2) ? mlf1 : (weight1*mlf1 + weight2*mlf2)*invW;
                SetTo(FromMeanAndMeanLogFactorial(mean, meanLogFact));
            }
        }

        /// <summary>
        /// Gets the log of the integral of the product of this COM-Poisson with another COM-Poisson.
        /// </summary>
        /// <param name="that">The other COM-Poisson</param>
        public double GetLogAverageOf(Poisson that)
        {
            if (IsPointMass)
            {
                return that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                return GetLogProb(that.Point);
            }
            else
            {
                Poisson product = this*that;
                return product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(Poisson that, double power)
        {
            if (IsPointMass)
            {
                return power*that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                if (power < 0) throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                return this.GetLogProb(that.Point);
            }
            else
            {
                var product = this*(that ^ power);
                return product.GetLogNormalizer() - this.GetLogNormalizer() - power*that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Gets the expected logarithm of a COM-Poisson under this COM-Poisson.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(Poisson that)
        {
            if (that.IsPointMass)
            {
                if (this.IsPointMass && this.Point == that.Point) return 0.0;
                else return Double.NegativeInfinity;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                // E[x*log(that.Rate) - that.Precision*gammaln(x+1) - that.logZ]
                return GetMean()*Math.Log(that.Rate) - that.Precision*GetMeanLogFactorial() - that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Create a COM-Poisson distribution with the given sufficient statistics.
        /// </summary>
        /// <param name="mean">E[x]</param>
        /// <param name="meanLogFactorial">E[log(x!)]</param>
        /// <returns>A new COM-Poisson distribution</returns>
        /// <remarks>
        /// This routine implements maximum-likelihood estimation of a COM-Poisson distribution, as described by
        /// Thomas P. Minka, Galit Shmueli, Joseph B. Kadane, Sharad Borle, and Peter Boatwright,
        /// "Computing with the COM-Poisson distribution", CMU Tech Report 776.
        /// http://www.stat.cmu.edu/tr/tr776/tr776.html
        /// </remarks>
        public static Poisson FromMeanAndMeanLogFactorial(double mean, double meanLogFactorial)
        {
            DenseVector theta = DenseVector.Zero(2);
            theta[0] = Math.Log(mean);
            theta[1] = 1;
            DenseVector newTheta = DenseVector.Zero(2);
            DenseVector gradient2 = DenseVector.Zero(2);
            DenseVector gradient = DenseVector.Zero(2);
            PositiveDefiniteMatrix hessian = new PositiveDefiniteMatrix(2, 2);
            double stepsize = 1;
            const int numIter = 1000;
            double rate = Math.Exp(theta[0]);
            double precision = theta[1];
            double logZ = GetLogNormalizer(rate, precision);
            double meanEst = Math.Exp(GetLogPowerSum(rate, precision, 1) - logZ);
            double Z = Math.Exp(logZ);
            double meanLogFactorialEst = GetSumLogFactorial(rate, precision) / Z;
            for (int iter = 0; iter < numIter; iter++)
            {
                double varEst = Math.Exp(GetLogPowerSum(rate, precision, 2) - logZ) - meanEst * meanEst;
                double varLogFactorialEst = GetSumLogFactorial2(rate, precision) / Z - meanLogFactorialEst * meanLogFactorialEst;
                double covXLogFactorialEst = GetSumXLogFactorial(rate, precision) / Z - meanEst * meanLogFactorialEst;
                gradient[0] = mean - meanEst;
                gradient[1] = meanLogFactorialEst - meanLogFactorial;
                double sqrGrad = gradient.Inner(gradient);
                // the Hessian is guaranteed to be positive definite
                hessian[0, 0] = varEst;
                hessian[0, 1] = -covXLogFactorialEst;
                hessian[1, 0] = hessian[0, 1];
                hessian[1, 1] = varLogFactorialEst;
                gradient.PredivideBy(hessian);
                // line search to reduce sqrGrad
                // we use sqrGrad instead of the likelihood since the point of zero gradient may not exactly match the likelihood mode due to numerical inaccuracy
                while (true)
                {
                    newTheta.SetToSum(1.0, theta, stepsize, gradient);
                    double sqrGrad2 = sqrGrad;
                    if (newTheta[1] >= 0)
                    {
                        rate = Math.Exp(newTheta[0]);
                        precision = newTheta[1];
                        logZ = GetLogNormalizer(rate, precision);
                        meanEst = Math.Exp(GetLogPowerSum(rate, precision, 1) - logZ);
                        Z = Math.Exp(logZ);
                        meanLogFactorialEst = GetSumLogFactorial(rate, precision) / Z;
                        gradient2[0] = mean - meanEst;
                        gradient2[1] = meanLogFactorialEst - meanLogFactorial;
                        sqrGrad2 = gradient2.Inner(gradient2);
                    }
                    if (sqrGrad2 < sqrGrad)
                    {
                        theta.SetTo(newTheta);
                        gradient.SetTo(gradient2);
                        stepsize *= 2;
                        break;
                    }
                    else
                    {
                        stepsize /= 2;
                        if (stepsize == 0)
                            break;
                    }
                }
                //Console.WriteLine("{0}: {1}", iter, theta);
                double maxGradient = Math.Max(Math.Abs(gradient[0]), Math.Abs(gradient[1]));
                if (maxGradient < 1e-10)
                    break;
                if (stepsize == 0)
                    break;
                if (iter == numIter - 1)
                    throw new Exception("not converging");
            }
            return new Poisson(Math.Exp(theta[0]), theta[1]);
        }

        /// <summary>
        /// Computes (sum_{x=0..infinity} log(x!) Rate^x / x!^Precision) / (sum_{x=0..infinity} Rate^x / x!^Precision )
        /// </summary>
        /// <returns></returns>
        public double GetMeanLogFactorial()
        {
            if (IsPointMass) return MMath.GammaLn(Point + 1);
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else
            {
                return GetSumLogFactorial(Rate, Precision)/Math.Exp(GetLogNormalizer(Rate, Precision));
            }
        }

        /// <summary>
        /// Computes sum_{x=0..infinity} log(x!) lambda^x / x!^nu
        /// </summary>
        /// <param name="lambda">Rate.  Must be non-negative.</param>
        /// <param name="nu">Precision.  Must be non-negative.</param>
        /// <returns></returns>
        public static double GetSumLogFactorial(double lambda, double nu)
        {
            if (Double.IsPositiveInfinity(nu)) throw new ArgumentException("nu = Inf");
            if (lambda < 0) throw new ArgumentException("lambda (" + lambda + ") < 0");
            if (nu < 0) throw new ArgumentException("nu (" + nu + ") < 0");
            if (lambda == 0) return 0.0;
            double term = lambda;
            double term2 = 0;
            double result = 0; // start off with first two terms (both zero)
            int i;
            for (i = 2; i < maxiter; i++)
            {
                double oldterm2 = term2;
                term2 += Math.Log(i);
                double delta = lambda/Math.Pow(i, nu);
                term *= delta;
                result += term*term2;
                delta *= term2/oldterm2; // delta = term*term2 / (oldterm*oldterm2)
                if (i > 2 && delta < 1)
                {
                    /* truncation error */
                    // delta is decreasing, so remainder of series can be bounded by 
                    // sum_{i=0..infinity} term*term2*delta^i = term*term2/(1-delta)
                    double r = (term*term2/result)/(1 - delta);
                    if (r < tol) break;
                }
            }
            if (i == maxiter)
                throw new Exception("not converging");
            return result;
        }

        /// <summary>
        /// Computes sum_{x=0..infinity} log(x!)^2 lambda^x / x!^nu
        /// </summary>
        /// <param name="lambda">Rate.</param>
        /// <param name="nu">Precision.</param>
        /// <returns></returns>
        public static double GetSumLogFactorial2(double lambda, double nu)
        {
            if (Double.IsPositiveInfinity(nu)) throw new ArgumentException("nu = Inf");
            if (lambda < 0) throw new ArgumentException("lambda (" + lambda + ") < 0");
            if (nu < 0) throw new ArgumentException("nu (" + nu + ") < 0");
            if (lambda == 0) return 0.0;
            double term = lambda;
            double term2 = 0;
            double logfact = 0;
            double result = 0; // start off with first two terms (both zero)
            int i;
            for (i = 2; i < maxiter; i++)
            {
                double oldterm2 = term2;
                logfact += Math.Log(i);
                term2 = logfact*logfact;
                double delta = lambda/Math.Pow(i, nu);
                term *= delta;
                result += term*term2;
                delta *= term2/oldterm2; // delta = term*term2 / (oldterm*oldterm2)
                if (i > 2 && delta < 1)
                {
                    /* truncation error */
                    // delta is decreasing, so remainder of series can be bounded by 
                    // sum_{i=0..infinity} term*term2*delta^i = term*term2/(1-delta)
                    double r = (term*term2/result)/(1 - delta);
                    if (r < tol) break;
                }
            }
            if (i == maxiter)
                throw new Exception("not converging");
            return result;
        }

        /// <summary>
        /// Computes sum_{x=0..infinity} x log(x!) lambda^x / x!^nu
        /// </summary>
        /// <param name="lambda">Rate.</param>
        /// <param name="nu">Precision.</param>
        /// <returns></returns>
        public static double GetSumXLogFactorial(double lambda, double nu)
        {
            if (Double.IsPositiveInfinity(nu)) throw new ArgumentException("nu = Inf");
            if (lambda < 0) throw new ArgumentException("lambda (" + lambda + ") < 0");
            if (nu < 0) throw new ArgumentException("nu (" + nu + ") < 0");
            if (lambda == 0) return 0.0;
            double term = lambda;
            double term2 = 1;
            double logfact = 0;
            double result = 0; // start off with first two terms (both zero)
            int i;
            for (i = 2; i < maxiter; i++)
            {
                double oldterm2 = term2;
                logfact += Math.Log(i);
                term2 = i*logfact;
                double delta = lambda/Math.Pow(i, nu);
                term *= delta;
                result += term*term2;
                delta *= term2/oldterm2; // delta = term*term2 / (oldterm*oldterm2)
                if (i > 2 && delta < 1)
                {
                    /* truncation error */
                    // delta is decreasing, so remainder of series can be bounded by 
                    // sum_{i=0..infinity} term*term2*delta^i = term*term2/(1-delta)
                    double r = (term*term2/result)/(1 - delta);
                    if (r < tol) break;
                }
            }
            if (i == maxiter)
                throw new Exception("not converging");
            return result;
        }

        /// <summary>
        /// Gets the log normalizer of the distribution
        /// </summary>
        public double GetLogNormalizer()
        {
            return GetLogNormalizer(Rate, Precision);
        }

        /// <summary>
        /// Maximum number of terms to use in series expansions
        /// </summary>
        const int maxiter = 10000;
        /// <summary>
        /// Desired tolerance for evaluating the normalizer
        /// </summary>
        const double tol = 1e-10;

        /// <summary>
        /// Computes log(sum_{x=0..infinity} lambda^x / x!^nu)
        /// </summary>
        /// <param name="lambda">Rate.</param>
        /// <param name="nu">Precision.</param>
        /// <returns></returns>
        public static double GetLogNormalizer(double lambda, double nu)
        {
            if (lambda < 0)
            {
                throw new ArgumentException("lambda (" + lambda + ") < 0");
            }
            if (lambda == 0 || nu < 0 || Double.IsPositiveInfinity(nu)) return 0.0;
            if (nu == 1) return lambda;
            else if (nu == 0)
            {
                if (lambda >= 1) return 0.0;
                /* needed when lambda is close to 1 */
                return -Math.Log(1 - lambda);
            }
            double lambda_ln = Math.Log(lambda);
            double term = lambda_ln;
            double result = Math.Log(1 + lambda); // start off with first two terms
            int i;
            for (i = 2; i < maxiter; i++)
            {
                double delta = lambda_ln - nu*Math.Log(i);
                term += delta;
                result = MMath.LogSumExp(result, term);
                if (delta < 0)
                {
                    /* truncation error */
                    // the rest of the series is sum_{i=0..infinity} exp(term + delta(i)*i)
                    // because delta is decreasing, we can bound the sum of the rest of the series
                    // by sum_{i=0..infinity} exp(term + delta*i) = exp(term)/(1 - exp(delta))
                    double r = Math.Exp(term - result)/(1 - Math.Exp(delta));
                    /* truncation error in log domain */
                    /* log(z+r)-log(z) = log(1 + r/z) =approx r/z */
                    if (r < tol) break;
                }
            }
            if (i == maxiter)
            {
                double r = Math.Pow(lambda, 1/nu);
                if (r > 10)
                {
                    /* use asymptotic approximation */
                    result = nu*r - (nu - 1)/2/nu*lambda_ln - (nu - 1)*MMath.LnSqrt2PI - 0.5*Math.Log(nu);
                    i = 0;
                }
            }
            return result;
        }

        /// <summary>
        /// Computes sum_{x=0..infinity} lambda^x / x!^nu
        /// </summary>
        /// <param name="lambda">Rate.</param>
        /// <param name="nu">Precision.</param>
        /// <returns></returns>
        public static double GetNormalizer(double lambda, double nu)
        {
            if (lambda < 0)
            {
                throw new ArgumentException("lambda (" + lambda + ") < 0");
            }
            if (lambda == 0 || nu < 0 || Double.IsPositiveInfinity(nu))
                return 1.0;
            if (nu == 1)
                return Math.Exp(lambda);
            else if (nu == 0)
            {
                if (lambda >= 1)
                    return 1.0;
                /* needed when lambda is close to 1 */
                return 1/(1 - lambda);
            }
            double term = lambda;
            double result = 1 + lambda; // start off with first two terms
            int i;
            for (i = 2; i < maxiter; i++)
            {
                double delta = lambda * Math.Pow(i, -nu);
                term *= delta;
                result += term;
                if (delta < 0)
                {
                    /* truncation error */
                    // the rest of the series is sum_{i=0..infinity} term * delta(i)^i
                    // because delta is decreasing, we can bound the sum of the rest of the series
                    // by sum_{i=0..infinity} term * delta^i = term/(1 - delta)
                    double r = term / result / (1 - delta);
                    /* truncation error in log domain */
                    /* log(z+r)-log(z) = log(1 + r/z) =approx r/z */
                    if (r < tol)
                        break;
                }
            }
            if (i == maxiter)
            {
                double r = Math.Pow(lambda, 1 / nu);
                if (r > 10)
                {
                    /* use asymptotic approximation */
                    double lambda_ln = Math.Log(lambda);
                    double logZ = nu * r - (nu - 1) / 2 / nu * lambda_ln - (nu - 1) * MMath.LnSqrt2PI - 0.5 * Math.Log(nu);
                    result = Math.Exp(logZ);
                    i = 0;
                }
            }
            return result;
        }

        /// <summary>
        /// Computes log(sum_{x=0..infinity} x^p lambda^x / x!^nu)
        /// </summary>
        /// <param name="lambda">Rate.</param>
        /// <param name="nu">Precision.</param>
        /// <param name="p">Exponent.</param>
        /// <returns></returns>
        public static double GetLogPowerSum(double lambda, double nu, double p)
        {
            if (p == 0) return GetLogNormalizer(lambda, nu);
            if (lambda < 0)
            {
                throw new ArgumentException("lambda (" + lambda + ") < 0");
            }
            if (lambda == 0 || nu < 0) return 0.0;
            if (Double.IsPositiveInfinity(nu)) throw new ArgumentException("nu = Inf");
            double lambda_ln = Math.Log(lambda);
            double result = lambda_ln; // start off with first two terms (p != 0)
            double term = lambda_ln;
            if (nu == 1)
            {
                if (p == 1) return lambda + lambda_ln; // log(sum x*lambda^x/x!) = log(lambda*exp(lambda))
                else if (p == 2) return lambda + lambda_ln + Math.Log(1 + lambda); // log(sum x^2*lambda^x/x!) = log(lambda+lambda^2)+lambda
            }
            else if (nu == 0)
            {
                if (lambda >= 1) return 0.0;
                if (p == 1) return lambda_ln - 2*Math.Log(1 - lambda);
                else if (p == 2) return lambda_ln + Math.Log(1 + lambda) - 3*Math.Log(1 - lambda);
            }
            int i = 2;
            for (; i < maxiter; i++)
            {
                double delta = lambda_ln - nu*Math.Log(i) + p*Math.Log((double) i/(i - 1));
                term += delta;
                result = MMath.LogSumExp(result, term);
                if (delta < 0)
                {
                    /* truncation error */
                    double r = Math.Exp(term - result)/(1 - Math.Exp(delta));
                    /* truncation error in log domain */
                    /* log(z+r)-log(z) = log(1 + r/z) =approx r/z */
                    if (r < tol) break;
                }
            }
            if (i == maxiter)
            {
                double r = Math.Pow(lambda, 1/nu);
                if (r > 10 && p == 1)
                {
                    /* use asymptotic approximation */
                    result = nu*r - (nu - 1)/2/nu*lambda_ln - (nu - 1)*MMath.LnSqrt2PI - 0.5*Math.Log(nu);
                    result = result + Math.Log(r - (nu - 1)/2/nu);
                    i = 0;
                }
            }
            return result;
        }

        /// <summary>
        /// Samples from a Poisson distribution
        /// </summary>
        [Stochastic]
        public int Sample()
        {
            if (IsPointMass) return Point;
            else if (Precision == 1) return Sample(Rate);
            else return Sample(Rate, Precision);
        }

        /// <summary>
        /// Sample from a Poisson - use <see cref="Sample()"/> instead
        /// </summary>
        /// <param name="result"></param>
        /// <returns></returns>
        [Stochastic]
        public int Sample(int result)
        {
            return Sample();
        }

        /// <summary>
        /// Samples from a Poisson distribution with given mean
        /// </summary>
        /// <param name="mean">Must be >= 0</param>
        /// <returns>An integer in [0,infinity)</returns>
        [Stochastic]
        public static int Sample(double mean)
        {
            return Rand.Poisson(mean);
        }

        /// <summary>
        /// Samples from a COM-Poisson distribution
        /// </summary>
        /// <param name="rate">Rate.</param>
        /// <param name="precision">Precision.</param>
        /// <returns></returns>
        [Stochastic]
        public static int Sample(double rate, double precision)
        {
            if (rate == 0) return 0;
            if (precision == 1) return Sample(rate);
            double logZ = GetLogNormalizer(rate, precision);
            double p = Math.Exp(-logZ);
            double u = Rand.Double();
            double sum = p;
            int x = 0;
            while (sum < u)
            {
                x++;
                p *= rate*Math.Pow(x, -precision);
                sum += p;
            }
            return x;
        }

#if false
    /// <summary>
    /// Create a uniform Com-Poisson distribution.
    /// </summary>
        public Poisson()
        {
        }
#endif

        /// <summary>
        /// Creates a Poisson distribution with the given mean.
        /// </summary>
        /// <param name="mean"></param>
        public Poisson(double mean)
        {
            Rate = mean;
            Precision = 1;
        }

        /// <summary>
        /// Create a Com-Poisson distribution with the given rate and precision.
        /// </summary>
        /// <param name="rate"></param>
        /// <param name="precision"></param>
        [Construction("Rate", "Precision")]
        public Poisson(double rate, double precision)
        {
            Rate = rate;
            Precision = precision;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public Poisson(Poisson that)
        {
            // explicit field assignment required by C# rules
            //SetTo(that);
            Rate = that.Rate;
            Precision = that.Precision;
        }

        /// <summary>
        /// Instantiates a uniform Com-Poisson distribution
        /// </summary>
        /// <returns>A new uniform Com-Poisson distribution</returns>
        [Construction(UseWhen = "IsUniform"), Skip]
        public static Poisson Uniform()
        {
            Poisson result = new Poisson();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Creates a Com-Poisson distribution which only allows one value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static Poisson PointMass(int value)
        {
            Poisson result = new Poisson();
            result.Point = value;
            return result;
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns>String representation of the instance</returns>
        /// <exclude/>
        public override string ToString()
        {
            if (IsPointMass)
            {
                return "Poisson.PointMass(" + Point + ")";
            }
            else if (IsUniform())
            {
                return "Poisson.Uniform";
            }
            else if (Precision == 1)
            {
                return "Poisson(" + Rate + ")";
            }
            else
            {
                return "Poisson(" + Rate + "," + Precision + ")";
            }
        }
    }
}