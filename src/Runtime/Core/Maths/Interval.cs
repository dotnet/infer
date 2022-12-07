// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Utilities;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Probabilistic.Math
{
    public struct Interval : IEquatable<Interval>
    {
        public readonly static Interval ZeroPoint = Point(0);
        public readonly static Interval NaN = Point(double.NaN);

        public readonly double LowerBound, UpperBound;

        public Interval(double lowerBound, double upperBound)
        {
            this.LowerBound = lowerBound;
            this.UpperBound = upperBound;
        }

        public override string ToString()
        {
            return $"({LowerBound:g17}, {UpperBound:g17})";
        }

        public static bool operator ==(Interval c1, Interval c2)
        {
            return c1.Equals(c2);
        }

        public static bool operator !=(Interval c1, Interval c2)
        {
            return !c1.Equals(c2);
        }

        public bool Equals(Interval other)
        {
            // See the value equality implementation guidelines here:
            // https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/statements-expressions-operators/how-to-define-value-equality-for-a-type
            // Any struct has a default reflection-based implementation of value equality.
            // Although this implementation produces correct results, it is relatively slow compared to a custom implementation.
            return (LowerBound == other.LowerBound) && (UpperBound == other.UpperBound);
        }

        public override bool Equals(object obj)
        {
            if (obj is Interval that) return Equals(that);
            else return false;
        }

        public override int GetHashCode()
        {
            return Hash.Combine(LowerBound.GetHashCode(), UpperBound.GetHashCode());
        }

        public bool Contains(double value)
        {
            return (value >= LowerBound) && (value <= UpperBound);
        }

        public bool Contains(Interval interval)
        {
            return Contains(interval.LowerBound) && Contains(interval.UpperBound);
        }

        public double Width()
        {
            if (UpperBound == LowerBound) return 0.0; // catch Inf - Inf
            else return UpperBound - LowerBound;
        }

        public bool IsPoint
        {
            get
            {
                return UpperBound == LowerBound;
            }
        }

        public static Interval Point(double point)
        {
            return new Interval(point, point);
        }

        public static Interval FromRegion(Region region, int dimension)
        {
            return new Interval(region.Lower[dimension], region.Upper[dimension]);
        }

        /// <summary>
        /// Creates an interval if <paramref name="lowerBound"/> is less or equal to <paramref name="upperBound"/>.
        /// If not, an exception is thrown.
        /// </summary>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="lowerBound"/> is greater than <paramref name="upperBound"/></exception>
        public static Interval CreateOrThrow(double lowerBound, double upperBound)
        {
            if (lowerBound > upperBound)
            {
                throw new ArgumentOutOfRangeException(nameof(lowerBound), $"{lowerBound:g17} > {upperBound:g17}");
            }
            return new Interval(lowerBound, upperBound);
        }

        public static Interval operator >(Interval a, double b)
        {
            return new Interval((a.LowerBound > b) ? 1 : 0, (a.UpperBound > b) ? 1 : 0);
        }

        public static Interval operator <(Interval a, double b)
        {
            return new Interval((a.LowerBound < b) ? 1 : 0, (a.UpperBound < b) ? 1 : 0);
        }

        public static Interval operator +(Interval a, double b)
        {
            return new Interval(a.LowerBound + b, a.UpperBound + b);
        }

        public static Interval operator +(double a, Interval b)
        {
            return b + a;
        }

        public static Interval operator +(Interval a, Interval b)
        {
            return new Interval(a.LowerBound + b.LowerBound, a.UpperBound + b.UpperBound);
        }

        public static Interval operator -(Interval a, double b)
        {
            return a + (-b);
        }

        public static Interval operator -(double a, Interval b)
        {
            return new Interval(a - b.UpperBound, a - b.LowerBound);
        }

        public static Interval operator -(Interval a, Interval b)
        {
            return new Interval(a.LowerBound - b.UpperBound, a.UpperBound - b.LowerBound);
        }

        public static Interval operator -(Interval a)
        {
            return new Interval(-a.UpperBound, -a.LowerBound);
        }

        public static Interval operator *(Interval a, double b)
        {
            if (b == 0)
                return Interval.Point(b); // avoid 0*infinity
            else if (b > 0)
                return new Interval(a.LowerBound * b, a.UpperBound * b);
            else
                return new Interval(a.UpperBound * b, a.LowerBound * b);
        }

        public static Interval operator *(double a, Interval b)
        {
            return b * a;
        }

        public static Interval operator *(Interval a, Interval b)
        {
            return new Interval(System.Math.Min(System.Math.Min(Times(a.LowerBound, b.LowerBound), Times(a.LowerBound, b.UpperBound)),
                                         System.Math.Min(Times(a.UpperBound, b.LowerBound), Times(a.UpperBound, b.UpperBound))),
                                System.Math.Max(System.Math.Max(Times(a.LowerBound, b.LowerBound), Times(a.LowerBound, b.UpperBound)),
                                         System.Math.Max(Times(a.UpperBound, b.LowerBound), Times(a.UpperBound, b.UpperBound))));
        }

        private static double Times(double a, double b)
        {
            if (a == 0 || b == 0) return 0;
            else return a * b;
        }

        public static Interval operator /(Interval a, Interval b)
        {
            return a * (1.0 / b);
        }

        public static Interval operator /(Interval a, double b)
        {
            return a * (1.0 / b);
        }

        public static Interval operator /(double a, Interval b)
        {
            if (a == 0) return Interval.Point(0);
            double aLower = a / b.LowerBound;
            double aUpper = a / b.UpperBound;
            if (a > 0)
            {
                if (b.Contains(0))
                {
                    if (b.LowerBound == 0)
                        return new Interval(aUpper, double.PositiveInfinity);
                    else if (b.UpperBound == 0)
                        return new Interval(double.NegativeInfinity, aLower);
                    else
                        return new Interval(double.NegativeInfinity, double.PositiveInfinity);
                }
                else
                {
                    return new Interval(aUpper, aLower);
                }
            }
            else
            {
                if (b.Contains(0))
                {
                    if (b.LowerBound == 0)
                        return new Interval(double.NegativeInfinity, aUpper);
                    else if (b.UpperBound == 0)
                        return new Interval(aLower, double.PositiveInfinity);
                    else
                        return new Interval(double.NegativeInfinity, double.PositiveInfinity);
                }
                else
                {
                    return new Interval(aLower, aUpper);
                }
            }
        }

        public static Interval Inner(Interval[] intervals, Vector vector)
        {
            if (intervals.Length != vector.Count) throw new ArgumentException("intervals.Length != vector.Count", nameof(intervals));
            Interval sum = Interval.Point(0);
            for (int i = 0; i < vector.Count; i++)
            {
                sum += vector[i] * intervals[i];
            }
            return sum;
        }

        /// <summary>
        /// Evaluates the product x'Ax (where ' is transposition).
        /// </summary>
        /// <param name="intervals">A vector whose length equals matrix.Rows.</param>
        /// <param name="matrix">A square matrix with Rows == intervals.Length.</param>
        /// <returns>The above product.</returns>
        public static Interval QuadraticForm(Interval[] intervals, Matrix matrix)
        {
            if (matrix.Rows != matrix.Cols) throw new ArgumentException("matrix is not square", nameof(matrix));
            if (intervals.Length != matrix.Rows) throw new ArgumentException("intervals.Length != matrix.Rows", nameof(intervals));
            Interval sum = Interval.Point(0);
            for (int i = 0; i < intervals.Length; i++)
            {
                Interval rowSum = Interval.Point(0);
                for (int j = 0; j < intervals.Length; j++)
                {
                    rowSum += matrix[i, j] * intervals[j];
                }
                sum += rowSum * intervals[i];
            }
            return sum;
        }

        public static Interval QuadraticForm2(Interval[] intervals, Matrix matrix)
        {
            if (matrix.Rows != matrix.Cols) throw new ArgumentException("matrix is not square", nameof(matrix));
            if (intervals.Length != matrix.Rows) throw new ArgumentException("intervals.Length != matrix.Rows", nameof(intervals));
            Interval sum = Interval.Point(0);
            for (int i = 0; i < intervals.Length; i++)
            {
                for (int j = 0; j < intervals.Length; j++)
                {
                    if (i == j)
                        sum += matrix[i, i] * intervals[i].Square();
                    else
                        sum += matrix[i, j] * intervals[i] * intervals[j];
                }
            }
            return sum;
        }

        public Interval Min(double b)
        {
            return new Interval(System.Math.Min(LowerBound, b), System.Math.Min(UpperBound, b));
        }

        public Interval Min(Interval b)
        {
            return new Interval(System.Math.Min(LowerBound, b.LowerBound), System.Math.Min(UpperBound, b.UpperBound));
        }

        public Interval Max(double b)
        {
            return new Interval(System.Math.Max(LowerBound, b), System.Math.Max(UpperBound, b));
        }

        public Interval Max(Interval b)
        {
            return new Interval(System.Math.Max(LowerBound, b.LowerBound), System.Math.Max(UpperBound, b.UpperBound));
        }

        /// <summary>
        /// Returns the smallest interval containing the absolute value of every number in <c>this</c>.
        /// </summary>
        /// <returns></returns>
        public Interval Abs()
        {
            double absLowerBound = System.Math.Abs(LowerBound);
            double absUpperBound = System.Math.Abs(UpperBound);
            return new Interval(
                Contains(0.0) ? 0.0 : System.Math.Min(absLowerBound, absUpperBound),
                System.Math.Max(absLowerBound, absUpperBound)
                );
        }

        /// <summary>
        /// Returns the smallest interval containing the square of every number in <c>this</c>.
        /// </summary>
        /// <returns></returns>
        public Interval Square()
        {
            double squareLowerBound = LowerBound * LowerBound;
            double squareUpperBound = UpperBound * UpperBound;
            return new Interval(
                Contains(0.0) ? 0.0 : System.Math.Min(squareLowerBound, squareUpperBound),
                System.Math.Max(squareLowerBound, squareUpperBound)
                );
        }

        /// <summary>
        /// Returns the smallest interval containing every number whose square is in <c>this</c>.
        /// </summary>
        /// <param name="intersectWith">An interval to intersect with the result.  Used to improve precision.</param>
        /// <returns></returns>
        public Interval SquareInv(Interval intersectWith)
        {
            // Result is two intervals: (-sqrt(U), -sqrt(L)) union (sqrt(L),sqrt(U))
            // We intersect these with intersectWith, then project to a single interval
            double sqrtL = System.Math.Sqrt(LowerBound);
            double sqrtU = System.Math.Sqrt(UpperBound);
            double lowerBound, upperBound;
            if (intersectWith.LowerBound <= -sqrtL)
            {
                lowerBound = System.Math.Max(-sqrtU, intersectWith.LowerBound);
            }
            else
            {
                lowerBound = System.Math.Max(sqrtL, intersectWith.LowerBound);
            }
            if (intersectWith.UpperBound >= sqrtL)
            {
                upperBound = System.Math.Min(sqrtU, intersectWith.UpperBound);
            }
            else
            {
                upperBound = System.Math.Min(-sqrtL, intersectWith.UpperBound);
            }
            return new Interval(lowerBound, upperBound);
        }

        public Interval Sqrt()
        {
            return new Interval(System.Math.Sqrt(LowerBound), System.Math.Sqrt(UpperBound));
        }

        public Interval Log()
        {
            return new Interval(System.Math.Log(LowerBound), System.Math.Log(UpperBound));
        }

        public Interval Exp()
        {
            return new Interval(System.Math.Exp(LowerBound), System.Math.Exp(UpperBound));
        }

        public Interval NormalCdf()
        {
            return new Interval(MMath.NormalCdf(LowerBound), MMath.NormalCdf(UpperBound));
        }

        public Interval NormalCdfInv()
        {
            return new Interval(MMath.NormalCdfInv(LowerBound), MMath.NormalCdfInv(UpperBound));
        }

        public bool IsNaN()
        {
            return double.IsNaN(LowerBound) || double.IsNaN(UpperBound);
        }

        /// <summary>
        /// Returns the smallest interval containing sum(weights[i]*values[i])/sum(weights[i])
        /// </summary>
        /// <param name="values"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public static Interval WeightedAverage(IReadOnlyList<Interval> values, IReadOnlyList<Interval> weights)
        {
            return new Interval(
                WeightedAverage(Util.ArrayInit(values.Count, i => values[i].LowerBound), weights).LowerBound,
                WeightedAverage(Util.ArrayInit(values.Count, i => values[i].UpperBound), weights).UpperBound
                );
        }

        /// <summary>
        /// Returns the smallest interval containing sum(weights[i]*values[i])/sum(weights[i])
        /// </summary>
        /// <param name="values"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public static Interval WeightedAverage(IReadOnlyList<double> values, IReadOnlyList<Interval> weights)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));
            if (values.Count == 0) throw new ArgumentOutOfRangeException("values.Count == 0");
            if (values.Count == 1) return new Interval(values[0], values[0]);
            else if (values.Count == 2)
            {
                int maxIndex, minIndex;
                if (values[0] >= values[1])
                {
                    maxIndex = 0;
                    minIndex = 1;
                }
                else
                {
                    maxIndex = 1;
                    minIndex = 0;
                }
                double upperBound = MMath.WeightedAverage(weights[maxIndex].UpperBound, values[maxIndex], weights[minIndex].LowerBound, values[minIndex]);
                double lowerBound = MMath.WeightedAverage(weights[maxIndex].LowerBound, values[maxIndex], weights[minIndex].UpperBound, values[minIndex]);
                return new Interval(lowerBound, upperBound);
            }
            else
            {
                // Algorithm: Fixed-point iteration.  On each iteration, set weights[i] to maximize or minimize the weighted average, conditional on other weights.
                // The derivative wrt weights[i] is proportional to (values[i] - weightedAvg), so we want to maximize weights[i] if values[i] > weightedAvg, otherwise minimize weights[i].
                double lowerBound = double.PositiveInfinity;
                double upperBound = double.NegativeInfinity;
                for (int iteration = 0; iteration < 100; iteration++)
                {
                    MeanAccumulator upperAccumulator = new MeanAccumulator();
                    MeanAccumulator lowerAccumulator = new MeanAccumulator();
                    for (int i = 0; i < values.Count; i++)
                    {
                        double value = values[i];
                        // Favor the upperBound in case of equality, to avoid all-zero weights.
                        double newWeightUpper = (value >= upperBound) ? weights[i].UpperBound : weights[i].LowerBound;
                        upperAccumulator.Add(value, newWeightUpper);
                        double newWeightLower = (value <= lowerBound) ? weights[i].UpperBound : weights[i].LowerBound;
                        lowerAccumulator.Add(value, newWeightLower);
                    }
                    double oldLowerBound = lowerBound;
                    double oldUpperBound = upperBound;
                    lowerBound = lowerAccumulator.Mean;
                    upperBound = upperAccumulator.Mean;
                    if (oldLowerBound == lowerBound && oldUpperBound == upperBound) break;
                    if (double.IsNaN(lowerBound))
                        throw new Exception($"lowerBound is NaN. weights={StringUtil.CollectionToString(weights, " ")}, values={StringUtil.CollectionToString(values, " ")}");
                    if (double.IsNaN(upperBound))
                        throw new Exception($"upperBound is NaN. weights={StringUtil.CollectionToString(weights, " ")}, values={StringUtil.CollectionToString(values, " ")}");
                }
                return new Interval(lowerBound, upperBound);
            }
        }

        public static double WeightedAverage(IReadOnlyList<double> values, IReadOnlyList<double> weights)
        {
            MeanAccumulator accumulator = new MeanAccumulator();
            for (int i = 0; i < weights.Count; i++)
            {
                accumulator.Add(values[i], weights[i]);
            }
            return accumulator.Mean;
        }

        // Only used for testing
        internal static double WeightedAverage2(double weight1, double value1, double weight2, double value2)
        {
            MeanAccumulator accumulator = new MeanAccumulator();
            accumulator.Add(value1, weight1);
            accumulator.Add(value2, weight2);
            return accumulator.Mean;
        }

        public double Sample()
        {
            return Region.Uniform(LowerBound, UpperBound);
        }

        public double Midpoint()
        {
            return Region.GetMidpoint(LowerBound, UpperBound);
        }

        public double RelativeError()
        {
            return Width() / (System.Math.Abs(Midpoint()) + 1e-6);
        }

        public Interval Union(Interval that)
        {
            return new Interval(System.Math.Min(LowerBound, that.LowerBound), System.Math.Max(UpperBound, that.UpperBound));
        }

        public Interval Union(double point)
        {
            return new Interval(System.Math.Min(LowerBound, point), System.Math.Max(UpperBound, point));
        }

        public Interval Intersect(Interval that)
        {
            return new Interval(System.Math.Max(LowerBound, that.LowerBound), System.Math.Min(UpperBound, that.UpperBound));
        }

        /// <summary>
        /// Computes the minimum and maximum value of a function over the interval, up to an error tolerance.
        /// </summary>
        /// <param name="allowedRelativeError"></param>
        /// <param name="getBound"></param>
        /// <returns></returns>
        public Interval Apply(double allowedRelativeError, Func<Interval, Interval> getBound)
        {
            Interval? union = null;
            Stack<Interval> stack = new Stack<Interval>();
            stack.Push(this);
            while (stack.Count > 0)
            {
                var input = stack.Pop();
                Interval output = getBound(input);
                Interval newUnion = (union == null) ? output : union.Value.Union(output);
                if (newUnion.Equals(union))
                {
                    continue;
                }
                // compute the error threshold
                Interval reference = getBound(Interval.Point(input.LowerBound)).Union(getBound(Interval.Point(input.UpperBound)));
                if (union != null) reference = reference.Union(union.Value);
                if (newUnion.RelativeError() <= allowedRelativeError + reference.RelativeError())
                {
                    union = newUnion;
                    continue;
                }
                // subdivide 
                double midpoint = input.Midpoint();
                //Trace.WriteLine($"splitting {input} {output.RelativeError()} {newUnion.RelativeError()}");
                stack.Push(new Interval(input.LowerBound, midpoint));
                stack.Push(new Interval(midpoint, input.UpperBound));
            }

            if (union == null) throw new InvalidProgramException("The return value can never be null");
            return union.Value;
        }

        /// <summary>
        /// Compute bounds on the expected value of a function over an interval whose endpoints are uncertain.
        /// </summary>
        /// <param name="maximumError"></param>
        /// <param name="cancellationToken"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="skillDistribution"></param>
        /// <param name="preservesPoint">If true, the output interval approaches a point as the input interval does.</param>
        /// <param name="getInterval"></param>
        /// <returns></returns>
        public static Interval GetExpectation(double maximumError, CancellationToken cancellationToken, Interval left, Interval right, ITruncatableDistribution<double> skillDistribution, bool preservesPoint, Func<Interval, Interval> getInterval)
        {
            // Algorithm: divide the interval into three parts, bound the metric in each part, then work out the optimal weighting of parts.
            // The first part is the interval between left lower and upper bound.
            // The second part is the interval between left upper and right lower bound, if it exists.
            // The third part is the interval between right lower and upper bound.
            double leftLowerCdf = skillDistribution.GetProbLessThan(left.LowerBound);
            double leftUpperCdf = skillDistribution.GetProbLessThan(left.UpperBound);
            double rightLowerCdf = skillDistribution.GetProbLessThan(right.LowerBound);
            double rightUpperCdf = skillDistribution.GetProbLessThan(right.UpperBound);
            Interval boundPart1 = getInterval(left);
            Interval boundPart2;
            if (right.LowerBound > left.UpperBound)
            {
                boundPart2 = GetExpectation(maximumError, cancellationToken, skillDistribution.Truncate(left.UpperBound, right.LowerBound), preservesPoint, getInterval);
            }
            else
            {
                boundPart2 = Interval.Point(0); // doesn't matter since the weight will be zero
            }
            Interval boundPart3 = getInterval(right);
            Interval[] weights = new[] { new Interval(0.0, leftUpperCdf - leftLowerCdf), Interval.Point(System.Math.Max(0, rightLowerCdf - leftUpperCdf)), new Interval(0.0, rightUpperCdf - rightLowerCdf) };
            bool debug = false;
            if (debug)
            {
                Trace.WriteLine($"left = {left}, right = {right}");
                Trace.WriteLine($"boundPart1 = {boundPart1}, boundPart2 = {boundPart2}, boundPart3 = {boundPart3}");
                Trace.WriteLine($"weights = {StringUtil.CollectionToString(weights, " ")}");
            }
            Interval result = Interval.WeightedAverage(new[] { boundPart1, boundPart2, boundPart3 }, weights);
            if (result.IsNaN()) throw new Exception("result is NaN");
            return result;
        }

        /// <summary>
        /// Compute bounds on the expected value of a function over an interval whose endpoints are uncertain.
        /// Incorporates the quantile estimation error of the distribution.
        /// </summary>
        /// <param name="subdivisionCount"></param>
        /// <param name="cancellationToken"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="skillDistribution"></param>
        /// <param name="getInterval"></param>
        /// <returns></returns>
        public static Interval GetExpectation(int subdivisionCount, CancellationToken cancellationToken, Interval left, Interval right, IEstimatedDistribution skillDistribution, Func<Interval, Interval> getInterval)
        {
            Interval leftCdf = left.GetProbLessThan(skillDistribution);
            Interval rightCdf = right.GetProbLessThan(skillDistribution);

            Interval[] weights = new Interval[subdivisionCount];
            Interval[] bounds = new Interval[subdivisionCount];

            // To get a bound on the expectation, we divide the region into intervals of equal probability.
            // We compute the bound in each interval, and then take a weighted average.
            // the quantile ranks will be 1/count, 2/count, ..., count/count
            var middleProbabilityTotal = (rightCdf - leftCdf).Max(0);
            double increment = 1.0 / subdivisionCount;
            double start = increment;
            double error = 0;
            Interval previousInput = leftCdf.GetQuantile(skillDistribution);
            Interval previousCdf = leftCdf;
            Interval prevApproxProb = Interval.NaN;
            for (int i = 0; i < subdivisionCount; i++)
            {
                double quantileRank = System.Math.Min(1, start + i * increment);
                Interval cdf = (quantileRank * middleProbabilityTotal + leftCdf).Min(1);
                Interval input = cdf.GetQuantile(skillDistribution).Max(left).Min(right);
                //Trace.WriteLine($"input = {input}");
                bounds[i] = getInterval(new Interval(previousInput.LowerBound, input.UpperBound));
                previousInput = input;

                // The weights are uncertain due to the quantile estimation error.
                // If cdf(x) and cdf(y) are further apart than the quantile error, then they have independent errors.
                // But if they are closer than that, the errors are dependent.
                // The error model is provided by the implementation of IEstimatedDistribution.
                Interval approxProb = cdf - previousCdf;
                if (approxProb != prevApproxProb)
                {
                    error = skillDistribution.GetProbBetweenError(approxProb.UpperBound);
                    prevApproxProb = approxProb;
                }
                //Trace.WriteLine($"approxProb = {approxProb} error = {error}");

                weights[i] = (approxProb + error * new Interval(-1, 1)).Max(0);
                previousCdf = cdf;
            }

            bool debug = false;
            if (debug)
            {
                Trace.WriteLine($"left = {left}, right = {right}");
                Trace.WriteLine($"bounds = {StringUtil.CollectionToString(bounds, " ")}");
                Trace.WriteLine($"weights = {StringUtil.CollectionToString(weights, " ")}");
            }

            Interval result = WeightedAverage(bounds, weights);
            if (result.IsNaN()) throw new Exception("result is NaN");
            return result;
        }

        /// <summary>
        /// Compute bounds on the expected value of a function.
        /// </summary>
        /// <param name="maximumError"></param>
        /// <param name="cancellationToken"></param>
        /// <param name="canGetQuantile"></param>
        /// <param name="preservesPoint">If true, the output interval approaches a point as the input interval does.</param>
        /// <param name="getInterval"></param>
        /// <returns></returns>
        public static Interval GetExpectation(double maximumError, CancellationToken cancellationToken, CanGetQuantile<double> canGetQuantile, bool preservesPoint, Func<Interval, Interval> getInterval)
        {
            if (preservesPoint)
                return GetExpectation(maximumError, cancellationToken, canGetQuantile, getInterval);
            else
                return GetExpectation(30, canGetQuantile, getInterval);
        }

        /// <summary>
        /// Computes bounds on the expected value of a function over an interval.
        /// </summary>
        /// <param name="count">The number of subdivisions.  Increase for more accuracy.</param>
        /// <param name="canGetQuantile"></param>
        /// <param name="getInterval">Returns bounds on the function's output over all inputs in an interval.</param>
        private static Interval GetExpectation(int count, CanGetQuantile<double> canGetQuantile, Func<Interval, Interval> getInterval)
        {
            // To get a bound on the expectation, we divide the region into intervals of equal probability.
            // We compute the bound in each interval, and take an unweighted average.
            // the quantile ranks will be 1/count, 2/count, ..., count/count
            double increment = 1.0 / count;
            double start = increment;
            Interval sum = Interval.Point(0);
            double previousInput = canGetQuantile.GetQuantile(0);
            for (int i = 0; i < count; i++)
            {
                double quantileRank = start + i * increment;
                double input = canGetQuantile.GetQuantile(System.Math.Min(1, quantileRank));
                sum += getInterval(new Interval(previousInput, input));
                previousInput = input;
            }
            return sum / count;
        }

        public static Interval GetMean(double allowedRelativeError, CanGetQuantile<double> canGetQuantile)
        {
            return new Interval(0, 1).Integrate(allowedRelativeError, default(CancellationToken),
                probability => probability.BoundBetweenQuantiles(canGetQuantile, x => x));
        }

        public static Region GetExpectation(double allowedRelativeError, CancellationToken cancellationToken, CanGetQuantile<double> canGetQuantile, int dimension, Func<Interval, Region> getBounds)
        {
            return new Interval(0, 1).Integrate(allowedRelativeError, cancellationToken, dimension,
                probability => probability.BoundBetweenQuantiles(canGetQuantile, getBounds));
        }

        // The output interval must approach a point as the input interval does.
        public static Interval GetExpectation(double allowedRelativeError, CancellationToken cancellationToken, CanGetQuantile<double> canGetQuantile, Func<Interval, Interval> getBounds)
        {
            return new Interval(0, 1).Integrate(allowedRelativeError, cancellationToken,
                probability => probability.BoundBetweenQuantiles(canGetQuantile, getBounds));
        }

        public Interval NextDouble()
        {
            return new Interval(MMath.NextDouble(LowerBound), MMath.NextDouble(UpperBound));
        }

        public Interval GetProbLessThan(CanGetProbLessThan<double> canGetProbLessThan)
        {
            return new Interval(
                canGetProbLessThan.GetProbLessThan(LowerBound),
                canGetProbLessThan.GetProbLessThan(UpperBound)
                );
        }

        public Interval GetQuantile(CanGetQuantile<double> canGetQuantile)
        {
            return new Interval(
                   canGetQuantile.GetQuantile(LowerBound),
                   canGetQuantile.GetQuantile(UpperBound)
                   );
        }

        private T BoundBetweenQuantiles<T>(CanGetQuantile<double> canGetQuantile, Func<Interval, T> getBound)
        {
            // The integral from a to b of f(x)*p(x) is the same as
            // the integral from GetProbLessThan(a) to GetProbLessThan(b) of f(GetQuantile(probability))
            var quantile = GetQuantile(canGetQuantile);
            return getBound(quantile);
        }

        public Interval Integrate(double allowedRelativeError, Func<Interval, Interval> getBound)
        {
            return Integrate(allowedRelativeError, default(CancellationToken), getBound);
        }

        /// <summary>
        /// Returns bounds on the integral of a function over <c>this</c>.
        /// </summary>
        /// <param name="maximumError"></param>
        /// <param name="cancellationToken"></param>
        /// <param name="getBound">The output interval must be finite, and must approach a point as the input interval does.</param>
        /// <returns></returns>
        public Interval Integrate(double maximumError, CancellationToken cancellationToken, Func<Interval, Interval> getBound)
        {
            // Algorithm:
            // Divide the domain into exhaustive regions.
            // Subdivide the region with largest error estimate.
            // Bound the integral with the sum of region bounds.
            PriorityQueue<IntegrateQueueNode> queue = new PriorityQueue<IntegrateQueueNode>();
            Interval sum = Interval.Point(0);
            int nodeCount = 0;
            void addRegion(Interval input)
            {
                nodeCount++;
                Interval output = getBound(input);
                if (double.IsInfinity(output.LowerBound) || double.IsInfinity(output.UpperBound)) throw new Exception($"Infinite output interval: getBound({input}) = {output}");
                if (output.IsNaN()) throw new Exception($"getBound({input}) = {output}");
                sum += output * input.Width();
                IntegrateQueueNode node = new IntegrateQueueNode(input, output);
                queue.Add(node);
            }
            addRegion(this);
            while (sum.Width() > maximumError && !cancellationToken.IsCancellationRequested)
            {
                var node = queue.ExtractMinimum();  // gets the node with highest error
                // Remove node from the sum
                sum = sum.Remove(node.Output * node.Input.Width());
                // Split the node
                double midpoint = node.Input.Midpoint();
                addRegion(new Interval(node.Input.LowerBound, midpoint));
                addRegion(new Interval(midpoint, node.Input.UpperBound));
                if (nodeCount > 100000)
                {
                    Trace.WriteLine($"Integrate nodeCount = {nodeCount}.  This usually indicates a numerical problem.");
                    break;
                }
            }
            return sum;
        }

        public Interval Remove(Interval that)
        {
            return new Interval(LowerBound - that.LowerBound, UpperBound - that.UpperBound);
        }

        private class IntegrateQueueNode : IComparable<IntegrateQueueNode>
        {
            public readonly Interval Input;
            public readonly Interval Output;

            public IntegrateQueueNode(Interval input, Interval output)
            {
                this.Input = input;
                this.Output = output;
            }

            public int CompareTo(IntegrateQueueNode other)
            {
                return (other.Output.Width() * other.Input.Width()).CompareTo(this.Output.Width() * this.Input.Width());
            }

            public override string ToString()
            {
                return $"IntegrateQueueNode(Input={Input}, Output={Output})";
            }
        }

        public Region Integrate(double allowedRelativeError, int dimension, Func<Interval, Region> getBound)
        {
            return Integrate(allowedRelativeError, default(CancellationToken), dimension, getBound);
        }

        /// <summary>
        /// Returns bounds on the integral of a function over <c>this</c>.
        /// </summary>
        /// <param name="allowedRelativeError"></param>
        /// <param name="cancellationToken"></param>
        /// <param name="dimension"></param>
        /// <param name="getBound"></param>
        /// <returns></returns>
        public Region Integrate(double allowedRelativeError, CancellationToken cancellationToken, int dimension, Func<Interval, Region> getBound)
        {
            // Algorithm:
            // Divide the domain into exhaustive regions.
            // Subdivide the region with largest error estimate.
            // Bound the integral with the sum of region bounds.
            PriorityQueue<IntegrateQueueNode2> queue = new PriorityQueue<IntegrateQueueNode2>();
            Region sum = new Region(dimension);
            int nodeCount = 0;
            void addRegion(Interval input)
            {
                nodeCount++;
                Region output = getBound(input);
                Add(sum, Product(output, input.Width()));
                if (!IsFinite(sum)) throw new Exception("sum is not finite");
                IntegrateQueueNode2 node = new IntegrateQueueNode2(input, output);
                //Trace.WriteLine($"adding {node}");
                queue.Add(node);
            }
            addRegion(this);
            const int maxNodes = 30;
            while (RelativeError(sum) > allowedRelativeError && !cancellationToken.IsCancellationRequested)
            {
                var node = queue.ExtractMinimum();  // gets the node with highest error
                // Remove node from the sum
                Remove(sum, Product(node.Output, node.Input.Width()));
                //Trace.WriteLine($"splitting {node}, sum = {sum}");
                // Split the node
                double midpoint = node.Input.Midpoint();
                addRegion(new Interval(node.Input.LowerBound, midpoint));
                addRegion(new Interval(midpoint, node.Input.UpperBound));
                if (nodeCount > maxNodes)
                {
                    //Trace.WriteLine($"Integrate nodeCount = {nodeCount}");
                    break;
                }
                //Trace.WriteLine($"sum = {sum}, RelativeError = {RelativeError(sum)}");
            }
            return sum;
        }

        private static bool IsFinite(Region region)
        {
            return region.Lower.All(x => !double.IsNaN(x) && !double.IsInfinity(x)) &&
                region.Upper.All(x => !double.IsNaN(x) && !double.IsInfinity(x));
        }

        private static double RelativeError(Region region)
        {
            return Enumerable.Range(0, region.Dimension).Select(i =>
                (region.Upper[i] - region.Lower[i]) / (System.Math.Abs(Region.GetMidpoint(region.Upper[i], region.Lower[i])) + 1e-6)
                ).Max();
        }

        private static void Add(Region region, Region that)
        {
            region.Lower.SetToSum(region.Lower, that.Lower);
            region.Upper.SetToSum(region.Upper, that.Upper);
        }

        private static void Remove(Region region, Region that)
        {
            region.Lower.SetToSum(1, region.Lower, -1, that.Lower);
            region.Upper.SetToSum(1, region.Upper, -1, that.Upper);
        }

        private static Region Product(Region region, double scale)
        {
            if (scale < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(scale), scale, "scale < 0");
            }
            Region result = new Region(region);
            result.Lower.Scale(scale);
            result.Upper.Scale(scale);
            return result;
        }

        public static double MaxSize(Region region)
        {
            return Enumerable.Range(0, region.Dimension).Select(i => region.Upper[i] - region.Lower[i]).Max();
        }

        private class IntegrateQueueNode2 : IComparable<IntegrateQueueNode2>
        {
            public readonly Interval Input;
            public readonly Region Output;

            public IntegrateQueueNode2(Interval input, Region output)
            {
                this.Input = input;
                this.Output = output;
            }

            public int CompareTo(IntegrateQueueNode2 other)
            {
                return other.Input.Width().CompareTo(this.Input.Width());
                //return (MaxSize(other.Output) * other.Input.Size()).CompareTo(MaxSize(this.Output) * this.Input.Size());
            }

            public override string ToString()
            {
                return $"IntegrateQueueNode2(Input={Input}, Output={Output})";
            }
        }

        class UpperConfidenceBoundAccumulator
        {
            public int Count;
            private readonly MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
            public double UpperConfidenceBound
            {
                get
                {
                    return mva.Mean + System.Math.Sqrt(mva.Variance);
                }
            }

            public void Add(double x)
            {
                if (double.IsInfinity(x)) throw new ArgumentOutOfRangeException(nameof(x), x, "x is infinite");
                if (double.IsNaN(x)) throw new ArgumentOutOfRangeException(nameof(x), x, "x is NaN");
                mva.Add(x);
                Count++;
            }

            public override string ToString()
            {
                return $"UpperConfidenceBoundAccumulator(UpperConfidenceBound={UpperConfidenceBound},Count={Count})";
            }
        }

        /// <summary>
        /// Finds the input that maximizes a function.
        /// </summary>
        /// <param name="bounds">Bounds for the search.</param>
        /// <param name="getBound">Returns bounds on the function in a region.  Need not be tight, but must become tight as the region shrinks.</param>
        /// <param name="fTolerance">Allowable error in the function value.  Must be greater than zero.</param>
        /// <param name="cancellationToken">Cancels the search early.</param>
        /// <param name="reportProgress">Called periodically with best solution so far.</param>
        /// <returns>A Vector close to the global maximum of the function.</returns>
        public static (Vector, T, bool) FindMaximum1<T>(
            Region bounds,
            Func<Region, (Interval, T)> getBound,
            double fTolerance,
            CancellationToken cancellationToken = default(CancellationToken),
            Action<double, int> reportProgress = null)
        {
            if (fTolerance <= 0) throw new ArgumentOutOfRangeException($"fTolerance <= 0");
            int dim = bounds.Lower.Count;
            Vector argmax = bounds.GetMidpoint();
            // We always evaluate the midpoint, even if cancellationToken is set.
            // The caller is expected to provide cancellationToken to getBound if needed.
            var (intervalAtMidpoint0, dataAtMidpoint0) = getBound(RegionPoint(argmax));
            (Vector, T) argmaxWithData = (argmax, dataAtMidpoint0);
            // This global optimization algorithm is inspired by:
            // "Global Optimization by Multilevel Coordinate Search"
            // Waltraud Huyer and Arnold Neumaier
            // 1998
            // https://www.mat.univie.ac.at/~neum/ms/mcs.pdf
            UpperConfidenceBoundAccumulator[] slopeAccumulators = Util.ArrayInit(dim, i => new UpperConfidenceBoundAccumulator());
            HashSet<int> searchedDimensions = new HashSet<int>();
            double initialLogVolume = System.Math.Max(-1e10, bounds.GetLogVolume());
            double increment = -System.Math.Log(3);
            int bucketCount = 10 * dim;
            PriorityQueue<QueueNode1<T>>[] queues = Util.ArrayInit(bucketCount, i => new PriorityQueue<QueueNode1<T>>());
            int getBucketIndex(Region region)
            {
                return (int)System.Math.Min(bucketCount - 1, (region.GetLogVolume() - initialLogVolume) / increment);
            }
            // lowerBound is used to prune search regions.
            // lowerBound must only increase over time.
            double lowerBound = double.NegativeInfinity;
            long upperBoundCount = 0;
            if (Debug)
            {
                Func<Region, (Interval, T)> getBound1 = getBound;
                getBound = region =>
                {
                    Stopwatch watch = Stopwatch.StartNew();
                    var bound = getBound1(region);
                    watch.Stop();
                    if (timeAccumulator.Count > 10 && watch.ElapsedMilliseconds > timeAccumulator.Mean + 4 * System.Math.Sqrt(timeAccumulator.Variance))
                        Trace.WriteLine($"GetUpperBound took {watch.ElapsedMilliseconds}ms");
                    timeAccumulator.Add(watch.ElapsedMilliseconds);
                    //if (upperBoundCount % 100 == 0) Trace.WriteLine($"lowerBound = {lowerBound}");
                    return bound;
                };
            }
            const int minAccumulatorCount = 1;
            int countDimensionsSearched()
            {
                return searchedDimensions.Count;
            }
            // Modifies its input region.  Can be null.
            Func<Region, bool> reduceRegion = region =>
            {
                //return false;
                if (Debug) Trace.WriteLine($"reducing with lowerBound = {lowerBound}");
                bool wasReduced = false;
                for (int iter = 0; iter < 1000; iter++)
                {
                    bool wasReducedThisIteration = false;
                    double shaveFraction = 0.3;
                    for (int i = 0; i < dim; i++)
                    {
                        // Try to shave dimension i
                        foreach (bool bottom in new bool[] { true, false })
                        {
                            double size = region.Upper[i] - region.Lower[i];
                            if (size > double.MaxValue) continue;
                            double sliceWidth = size * shaveFraction;
                            if (bottom)
                            {
                                // Shave lower bound
                                double oldUpper = region.Upper[i];
                                region.Upper[i] = region.Lower[i] + sliceWidth;
                                double sliceUpper = getBound(region).Item1.UpperBound;
                                if (sliceUpper <= lowerBound)
                                {
                                    if (Debug) Trace.WriteLine($"shaved dimension {i}: {region.Lower[i]} to {region.Upper[i]}");
                                    region.Lower[i] = region.Upper[i];
                                    wasReducedThisIteration = true;
                                }
                                region.Upper[i] = oldUpper;
                            }
                            else
                            {
                                // Shave upper bound
                                double oldLower = region.Lower[i];
                                region.Lower[i] = region.Upper[i] - sliceWidth;
                                double sliceUpper = getBound(region).Item1.UpperBound;
                                if (sliceUpper <= lowerBound)
                                {
                                    if (Debug) Trace.WriteLine($"shaved dimension {i}: {region.Upper[i]} to {region.Lower[i]}");
                                    region.Upper[i] = region.Lower[i];
                                    wasReducedThisIteration = true;
                                }
                                region.Lower[i] = oldLower;
                            }
                        }
                    }
                    if (!wasReducedThisIteration) break;
                    wasReduced = true;
                }
                return wasReduced;
            };
            reduceRegion = null;
            void addRegion(Region region, Vector midpoint, int splitDim, double upperBound, double valueAtMidpoint, T dataAtMidpoint)
            {
                if (valueAtMidpoint > lowerBound)
                {
                    // argmaxWithData must be updated atomically
                    argmaxWithData = (midpoint, dataAtMidpoint);
                    lowerBound = valueAtMidpoint + fTolerance;
                    reportProgress?.Invoke(valueAtMidpoint, countDimensionsSearched());
                }
                if (upperBound > lowerBound)
                {
                    if (reduceRegion != null)
                    {
                        // reduce the region
                        // if it reduced, evaluate the new midpoint
                        // if it is better, use reduction again
                        for (int reductionCount = 0; reductionCount < 1000; reductionCount++)
                        {
                            bool wasReduced = reduceRegion(region);
                            if (!wasReduced) break;
                            if (Debug)
                                Trace.WriteLine($"reduced to {region}");
                            upperBound = getBound(region).Item1.UpperBound;
                            if (upperBound <= lowerBound) break;
                            midpoint = region.GetMidpoint();
                            valueAtMidpoint = getBound(RegionPoint(midpoint)).Item1.UpperBound;
                            if (valueAtMidpoint + fTolerance <= lowerBound) break;
                            // argmaxWithData must be updated atomically
                            argmaxWithData = (midpoint, dataAtMidpoint);
                            lowerBound = valueAtMidpoint + fTolerance;
                            reportProgress?.Invoke(valueAtMidpoint, countDimensionsSearched());
                        }
                    }
                }
                if (upperBound > lowerBound)
                {
                    var node = new QueueNode1<T>(region, upperBound, valueAtMidpoint, dataAtMidpoint);
                    int bucketIndex = getBucketIndex(region);
                    if (bucketIndex < 0 || bucketIndex >= queues.Length) Trace.WriteLine($"bucketIndex = {bucketIndex}, region.logVolume = {region.GetLogVolume()}, initialVolume = {initialLogVolume}");
                    queues[bucketIndex].Add(node);
                    if (Debug)
                        Trace.WriteLine($"added {node}");
                    return;
                }
                if (Debug)
                    Trace.WriteLine($"rejected region {region} with upperBound {upperBound:r} <= {lowerBound:r}");
            }
            upperBoundCount++;
            addRegion(bounds, argmax, 0, double.PositiveInfinity, intervalAtMidpoint0.UpperBound, dataAtMidpoint0);
            var task = Task.Run(() =>
            {
                while (queues.Any(queue => queue.Count > 0))
                {
                    if (cancellationToken.IsCancellationRequested) break;
                    for (int bucketIndex = 0; bucketIndex < queues.Length; bucketIndex++)
                    {
                        if (cancellationToken.IsCancellationRequested) break;
                        var queue = queues[bucketIndex];
                        if (queue.Count == 0) continue;
                        // get the node with highest upper confidence bound
                        var node = queue.ExtractMinimum();
                        if (node.UpperBound <= lowerBound)
                            continue;
                        Region region = new Region(node.Region);
                        // compute the lower bound
                        Vector midpoint = region.GetMidpoint();
                        if (Debug)
                        {
                            Trace.WriteLine($"bucket {bucketIndex}: expanding {node} valueAtMidpoint = {node.ValueAtMidpoint:r}");
                        }
                        if (node.ValueAtMidpoint > node.UpperBound) throw new Exception("node.ValueAtMidpoint > node.UpperBound");

                        int splitDim;
                        bool randomSplit = false;
                        if (randomSplit)
                        {
                            // Find a dimension to split on.
                            // Count the number of dimensions that can split
                            int splittableIndex = (System.Math.Abs(region.GetHashCode()) % dim);
                            for (splitDim = 0; splitDim < dim; splitDim++)
                            {
                                if (splittableIndex == 0)
                                {
                                    break;
                                }
                                splittableIndex--;
                            }
                            if (splittableIndex != 0)
                            {
                                throw new Exception();
                            }
                        }
                        else
                        {
                            double maxUpperConfidenceBound = double.NegativeInfinity;
                            splitDim = -1;
                            for (int i = 0; i < dim; i++)
                            {
                                double slopeUpperConfidenceBound;
                                if (slopeAccumulators[i].Count < minAccumulatorCount)
                                    slopeUpperConfidenceBound = double.MaxValue;
                                else
                                    slopeUpperConfidenceBound = slopeAccumulators[i].UpperConfidenceBound;
                                double radius = (node.Region.Upper[i] - node.Region.Lower[i]) / 2;
                                double upperConfidenceBound = slopeUpperConfidenceBound * radius;
                                if (upperConfidenceBound > maxUpperConfidenceBound)
                                {
                                    splitDim = i;
                                    maxUpperConfidenceBound = upperConfidenceBound;
                                }
                            }
                            if (maxUpperConfidenceBound == 0)
                            {
                                // split the dimension with largest width
                                for (int i = 0; i < dim; i++)
                                {
                                    double radius = (node.Region.Upper[i] - node.Region.Lower[i]) / 2;
                                    double upperConfidenceBound = radius;
                                    if (upperConfidenceBound > maxUpperConfidenceBound)
                                    {
                                        splitDim = i;
                                        maxUpperConfidenceBound = upperConfidenceBound;
                                    }
                                }
                            }
                            if (splitDim == -1)
                            {
                                if (Debug) Trace.WriteLine($"no dimension to split");
                                continue;
                            }
                        }

                        // split the node
                        if (Debug)
                            Trace.WriteLine($"splitting dimension {splitDim}");
                        searchedDimensions.Add(splitDim);
                        double midpointAtSplitDim = midpoint[splitDim];
                        if (region.Lower[splitDim] != midpointAtSplitDim)
                        {
                            Region leftRegion = new Region(region);
                            if (double.IsInfinity(region.Lower[splitDim]) || double.IsInfinity(midpointAtSplitDim)) throw new Exception();
                            // This ensures that the boundary between regions is halfway between the midpoints.
                            double leftBoundary = (region.Lower[splitDim] + 2 * midpointAtSplitDim) / 3;
                            leftRegion.Upper[splitDim] = leftBoundary;
                            region.Lower[splitDim] = leftBoundary;
                            upperBoundCount++;
                            double upperBoundLeft = getBound(leftRegion).Item1.UpperBound;
                            if (upperBoundLeft > lowerBound)
                            {
                                var leftMidpoint = leftRegion.GetMidpoint();
                                var (intervalAtMidpointLeft, dataAtMidpointLeft) = getBound(RegionPoint(leftMidpoint));
                                if (Debug) Trace.WriteLine($"intervalAtMidpointLeft = {intervalAtMidpointLeft}");
                                double valueAtMidpointLeft = intervalAtMidpointLeft.UpperBound;
                                double slope = (valueAtMidpointLeft == node.ValueAtMidpoint) // avoid inf - inf
                                    ? 0
                                    : System.Math.Min(System.Math.Abs((valueAtMidpointLeft - node.ValueAtMidpoint) / (leftMidpoint[splitDim] - midpointAtSplitDim)), double.MaxValue);
                                slopeAccumulators[splitDim].Add(slope);
                                if (cancellationToken.IsCancellationRequested) break;
                                addRegion(leftRegion, leftMidpoint, splitDim, upperBoundLeft, valueAtMidpointLeft, dataAtMidpointLeft);
                            }
                            else if (Debug) Trace.WriteLine($"rejected left region with upper bound {upperBoundLeft:r} <= {lowerBound:r}");
                        }
                        if (region.Upper[splitDim] != midpointAtSplitDim)
                        {
                            Region rightRegion = new Region(region);
                            if (double.IsInfinity(region.Upper[splitDim]) || double.IsInfinity(midpointAtSplitDim)) throw new Exception();
                            double rightBoundary = (region.Upper[splitDim] + 2 * midpointAtSplitDim) / 3;
                            rightRegion.Lower[splitDim] = rightBoundary;
                            region.Upper[splitDim] = rightBoundary;
                            upperBoundCount++;
                            double upperBoundRight = getBound(rightRegion).Item1.UpperBound;
                            if (upperBoundRight > lowerBound)
                            {
                                var rightMidpoint = rightRegion.GetMidpoint();
                                var (intervalAtMidpointRight, dataAtMidpointRight) = getBound(RegionPoint(rightMidpoint));
                                if (Debug) Trace.WriteLine($"intervalAtMidpointRight = {intervalAtMidpointRight}");
                                double valueAtMidpointRight = intervalAtMidpointRight.UpperBound;
                                double slope = (valueAtMidpointRight == node.ValueAtMidpoint) // avoid inf - inf
                                    ? 0
                                    : System.Math.Min(System.Math.Abs((valueAtMidpointRight - node.ValueAtMidpoint) / (rightMidpoint[splitDim] - midpointAtSplitDim)), double.MaxValue);
                                slopeAccumulators[splitDim].Add(slope);
                                if (cancellationToken.IsCancellationRequested) break;
                                addRegion(rightRegion, rightMidpoint, splitDim, upperBoundRight, valueAtMidpointRight, dataAtMidpointRight);
                            }
                            else if (Debug) Trace.WriteLine($"rejected right region with upper bound {upperBoundRight:r} <= {lowerBound:r}");
                        }
                        // region is now smaller
                        double newUpperBound = getBound(region).Item1.UpperBound;
                        if (cancellationToken.IsCancellationRequested) break;
                        addRegion(region, midpoint, splitDim, newUpperBound, node.ValueAtMidpoint, node.DataAtMidpoint);
                    }
                }
                if (Debug)
                    Trace.WriteLine($"Interval.FindMaximum upperBoundCount = {upperBoundCount}");
            }, cancellationToken);
            try
            {
                // If cancellationToken is cancelled, task.Wait() is cancelled and throws OperationCancelledException,
                // but note that the task can continue to execute.
                // If the task throws an exception, task.Wait() throws AggregateException with InnerExceptions collection containing
                // the thrown exception.
                task.Wait(cancellationToken);
            }
            catch (OperationCanceledException) { }
            bool searchCompleted = !queues.Any(queue => queue.Count > 0);
            reportProgress?.Invoke(lowerBound, searchCompleted ? dim : countDimensionsSearched());
            return (argmaxWithData.Item1, argmaxWithData.Item2, false);
        }

#pragma warning disable CS1570 // XML comment has badly formed XML
        /// <summary>
        /// Finds the input that maximizes a function.
        /// </summary>
        /// <param name="bounds">Bounds for the search.</param>
        /// <param name="getBound">Returns bounds on the function in a region.  Need not be tight, but must become tight as the region shrinks.
        /// Interval for a region must contain an interval for its inner region or a point.</param>
        /// <param name="fTolerance">Allowable error in the function value.  Must be greater than zero.</param>
        /// <param name="cancellationToken">Cancels the search early.</param>
        /// <param name="reportProgress">Called periodically with best solution so far.</param>
        /// <returns>A Vector close to the global maximum of the function.</returns>
        /// <remarks>
        /// This method searches through subregions of <paramref name="bounds"/> and for each subregion evaluates the bounds on the function,
        /// to find a solution which has the greatest lower bound of the interval.
        /// 
        /// For a region (R) the method gets the bounds for that region and for its midpoint (MP).
        /// A solution is a midpoint of a region, and the maximum for the solution is the function's lower bound for that midpoint (see also <see cref="Solution{T}"/>).
        /// The successors() method splits a region of a given solution into three parts (left, middle and right) and thus yields three other solutions.
        /// In order to determine if a new solution can contain new maximum and it makes sense to divide it further, 
        /// the algorithm keeps the upper bound on the function for the solution's region.
        /// 
        ///    Lower(R)    Lower(MP)  Upper(MP)    Upper(R)
        /// ------(----------[============]-------------)-------------> function domain
        ///                  |                          |
        ///                  |____solution interval_____|
        ///                  
        /// As the algorithm tolerates non-point bounds on the function given a point region as an argument (e.g. the function is noisy), it should address the following issue.
        /// If it was found that the region can contain a new maximum, the algorithm divides it into smaller regions,
        /// so the middle regions will tend to the midpoint. If <paramref name="getBound"/> for the midpoint returns a non-point interval, 
        /// the bounds for the shrinking regions will tend to the midpoint's interval, and if that interval contains the current maximum
        /// optimization will stuck in dividing this regions unless it finds a better solution outside of that interval, or it is cancelled.
        /// 
        ///    Lower(R)    Lower(MP)  Upper(MP)    Upper(R)
        /// ------(--------->[====*=======]<------------)-------------> function domain
        ///                  |   Max      |           
        ///                  |____________| 
        ///                  min solution interval as R -> MP
        /// 
        /// To address that, the algorithm takes the solution's upper bound as the upper bound for the region without width of the midpoint's interval (uncertainty of the function),
        /// so if R --> MP, then bounds(R) --> lower(MP), i.e. solution interval tends to a point.
        /// 
        ///    Lower(R)    Lower(MP)  Upper(MP)    Upper(R)
        /// ------(----------[====*=======]--o~~~~~~~~~~)-------------> function domain
        ///                  |   Max         |                 
        ///                  |_______________| 
        ///                  solution interval
        /// 
        /// The drawback is that we now think that the solution is worse than it can potentially be  
        /// (i.e. its upper bound is less than can be by amount of the function uncertainty).
        /// It means that the new solution's upper bound must be greater than the current maximum by at least the function's uncertainty at MP,
        /// otherwise the solution will be rejected.
        /// 
        /// One alternative is to use (Upper(MP), Upper(R)) as the solution's interval. In this cases we think the solution is better than it really is. 
        /// Then we can't miss a better solution, but a found solution can be worse than we thought. This violates our goal to find a solution with greatest lower bound.
        /// </remarks>
        public static (Vector, T, bool) FindMaximum<T>(
            Region bounds,
            Func<Region, (Interval, T)> getBound,
            double fTolerance,
            CancellationToken cancellationToken = default(CancellationToken),
            Action<double, int> reportProgress = null)
        {
            int dim = bounds.Lower.Count;
            // This global optimization algorithm is inspired by:
            // "Global Optimization by Multilevel Coordinate Search"
            // Waltraud Huyer and Arnold Neumaier
            // 1998
            // https://www.mat.univie.ac.at/~neum/ms/mcs.pdf
            UpperConfidenceBoundAccumulator[] slopeAccumulators = Util.ArrayInit(dim, i => new UpperConfidenceBoundAccumulator());
            HashSet<int> searchedDimensions = new HashSet<int>();
            if (Debug)
            {
                Func<Region, (Interval, T)> getBound1 = getBound;
                getBound = region =>
                {
                    Stopwatch watch = Stopwatch.StartNew();
                    var bound = getBound1(region);
                    watch.Stop();
                    if (timeAccumulator.Count > 10 && watch.ElapsedMilliseconds > timeAccumulator.Mean + 4 * System.Math.Sqrt(timeAccumulator.Variance))
                        Trace.WriteLine($"GetUpperBound took {watch.ElapsedMilliseconds}ms");
                    timeAccumulator.Add(watch.ElapsedMilliseconds);
                    //if (upperBoundCount % 100 == 0) Trace.WriteLine($"lowerBound = {lowerBound}");
                    return bound;
                };
            }
            const int minAccumulatorCount = 1;
            Vector initialMidpoint = bounds.GetMidpoint();
            Region initial = RegionPoint(initialMidpoint);
            var (initialIntervalAtMidpoint, initialData) = getBound(initial);
            if (cancellationToken.IsCancellationRequested)
            {
                if (Debug)
                    Trace.WriteLine($"FindMaximum: Cancelled after first evaluation");
                return (initialMidpoint, initialData, false);
            }
            else
            {
                var initialValue = getBound(bounds);
                var initialInterval = new Interval(initialIntervalAtMidpoint.LowerBound, initialValue.Item1.UpperBound);
                Solution<T> initialState = new Solution<T>(bounds, initialData, initialIntervalAtMidpoint.Width());
                var (solution, max, completed) = AnytimeColumnSearch(initialState, initialInterval, successors, fTolerance, cancellationToken, reportProgress2);
                // Report the number of searched dimensions.
                reportProgress2(solution, max);
                return (solution.Region.GetMidpoint(), solution.Data, completed);
            }

            IEnumerable<(Solution<T>, Interval)> successors((Solution<T>, Interval) solutionAndValue)
            {
                // copy the region since we will change it
                var (parentSolution, parentValue) = solutionAndValue;
                var region = new Region(parentSolution.Region);

                int splitDim;
                bool randomSplit = false;
                if (randomSplit)
                {
                    // Find a dimension to split on.
                    // Count the number of dimensions that can split
                    int splittableIndex = (System.Math.Abs(region.GetHashCode()) % dim);
                    for (splitDim = 0; splitDim < dim; splitDim++)
                    {
                        if (splittableIndex == 0)
                        {
                            break;
                        }
                        splittableIndex--;
                    }
                    if (splittableIndex != 0)
                    {
                        throw new Exception();
                    }
                }
                else
                {
                    double maxUpperConfidenceBound = double.NegativeInfinity;
                    splitDim = -1;
                    for (int i = 0; i < dim; i++)
                    {
                        double slopeUpperConfidenceBound;
                        if (slopeAccumulators[i].Count < minAccumulatorCount)
                            slopeUpperConfidenceBound = double.MaxValue;
                        else
                            slopeUpperConfidenceBound = slopeAccumulators[i].UpperConfidenceBound;
                        double radius = (region.Upper[i] - region.Lower[i]) / 2;
                        double upperConfidenceBound = slopeUpperConfidenceBound * radius;
                        if (upperConfidenceBound > maxUpperConfidenceBound)
                        {
                            splitDim = i;
                            maxUpperConfidenceBound = upperConfidenceBound;
                        }
                    }
                    if (maxUpperConfidenceBound == 0)
                    {
                        // split the dimension with largest width
                        for (int i = 0; i < dim; i++)
                        {
                            double radius = (region.Upper[i] - region.Lower[i]) / 2;
                            double upperConfidenceBound = radius;
                            if (upperConfidenceBound > maxUpperConfidenceBound)
                            {
                                splitDim = i;
                                maxUpperConfidenceBound = upperConfidenceBound;
                            }
                        }
                    }
                    if (splitDim == -1)
                    {
                        if (Debug) Trace.WriteLine($"no dimension to split");
                        yield break;
                    }
                }

                // split the node
                if (Debug)
                    Trace.WriteLine($"splitting dimension {splitDim}");
                searchedDimensions.Add(splitDim);
                Vector midpoint = region.GetMidpoint();
                double midpointAtSplitDim = midpoint[splitDim];
                double lowerAtSplitDim = region.Lower[splitDim];
                double upperAtSplitDim = region.Upper[splitDim];
                bool regionWasSplit = false;
                if (lowerAtSplitDim != midpointAtSplitDim)
                {
                    if (double.IsInfinity(lowerAtSplitDim) || double.IsInfinity(midpointAtSplitDim)) throw new Exception();
                    // This ensures that the boundary between regions is halfway between the midpoints.
                    double leftBoundary = (lowerAtSplitDim + 2 * midpointAtSplitDim) / 3;
                    if (!MMath.AreEqual(leftBoundary, lowerAtSplitDim) && !MMath.AreEqual(leftBoundary, upperAtSplitDim))
                    {
                        // leftRegion will be a duplicate if leftBoundary == region.Upper[splitDim]
                        Region leftRegion = new Region(region);
                        leftRegion.Upper[splitDim] = leftBoundary;
                        // region will be a duplicate if leftBoundary == region.Lower[splitDim]
                        region.Lower[splitDim] = leftBoundary;
                        yield return getBoundLeftOrRight(leftRegion, parentValue, splitDim, midpointAtSplitDim);
                        regionWasSplit = true;
                    }
                }
                if (cancellationToken.IsCancellationRequested) yield break;
                if (upperAtSplitDim != midpointAtSplitDim)
                {
                    if (double.IsInfinity(upperAtSplitDim) || double.IsInfinity(midpointAtSplitDim)) throw new Exception();
                    double rightBoundary = (upperAtSplitDim + 2 * midpointAtSplitDim) / 3;
                    if (!MMath.AreEqual(rightBoundary, upperAtSplitDim) && !MMath.AreEqual(rightBoundary, lowerAtSplitDim))
                    {
                        Region rightRegion = new Region(region);
                        rightRegion.Lower[splitDim] = rightBoundary;
                        region.Upper[splitDim] = rightBoundary;
                        yield return getBoundLeftOrRight(rightRegion, parentValue, splitDim, midpointAtSplitDim);
                        regionWasSplit = true;
                    }
                }
                if (cancellationToken.IsCancellationRequested) yield break;
                if (regionWasSplit)
                {
                    // region is now smaller
                    yield return getBoundMiddle(region, parentValue, parentSolution);
                }

                // Gets the bound when a region is shrink around the midpoint.
                (Solution<T>, Interval) getBoundMiddle(Region region2, Interval parentValue2, Solution<T> parentState)
                {
                    var (regionInterval, regiondData) = getBound(region2);
                    // In this case, there is no need to call getBound since we know the result will be the same as the parent.
                    // Note that regionInterval.UpperBound can be lower than parentValue.UpperBound
                    return (new Solution<T>(region2, parentState.Data, parentState.Uncertainty),
                        new Interval(parentValue2.LowerBound, regionInterval.UpperBound - parentState.Uncertainty));
                }

                (Solution<T>, Interval) getBoundLeftOrRight(Region region2, Interval parentValue2, int splitDim2, double parentMidpointAtSplitDim)
                {
                    var (regionInterval, regionData) = getBound(region2);
                    if (cancellationToken.IsCancellationRequested) return (new Solution<T>(region2, regionData, 0), regionInterval);
                    var midpoint2 = region2.GetMidpoint();
                    var (intervalAtMidpoint, dataAtMidpoint) = getBound(RegionPoint(midpoint2));
                    if (cancellationToken.IsCancellationRequested) return (new Solution<T>(region2, regionData, 0), regionInterval);
                    bool checkContainment = true;
                    if (checkContainment && !regionInterval.Contains(intervalAtMidpoint))
                    {
                        const double tolerance = 1e-10;
                        double up = MMath.AbsDiff(regionInterval.UpperBound, intervalAtMidpoint.UpperBound, tolerance);
                        if (up < tolerance) intervalAtMidpoint = new Interval(intervalAtMidpoint.LowerBound, regionInterval.UpperBound);
                        double low = MMath.AbsDiff(regionInterval.LowerBound, intervalAtMidpoint.LowerBound, tolerance);
                        if (low < tolerance) intervalAtMidpoint = new Interval(regionInterval.LowerBound, intervalAtMidpoint.UpperBound);
                        if (!regionInterval.Contains(intervalAtMidpoint))
                        {
                            // To debug this exception, set optimizer.GetExpectedIntervalsDebugger
                            throw new Exception($"!regionInterval.Contains(intervalAtMidpoint) regionInterval={regionInterval} intervalAtMidpoint={intervalAtMidpoint}");
                        }
                    }
                    double valueAtMidpoint = intervalAtMidpoint.LowerBound;
                    if (!cancellationToken.IsCancellationRequested)
                    {
                        double slope = (valueAtMidpoint == parentValue2.LowerBound) // avoid inf - inf
                            ? 0
                            : System.Math.Min(System.Math.Abs((valueAtMidpoint - parentValue2.LowerBound) / (midpoint2[splitDim2] - parentMidpointAtSplitDim)), double.MaxValue);
                        slopeAccumulators[splitDim2].Add(slope);
                    }
                    double midpointUncertainty = intervalAtMidpoint.Width();
                    // Here we decrease the region's upper bound by the midpoint uncertainty
                    // to satisfy the algorithm's requirements that the interval should tend to zero, as the region shrinks to a point.
                    if (Debug) Trace.WriteLine($"intervalAtMidpoint = {intervalAtMidpoint} regionInterval = {regionInterval} midpointUncertainty = {midpointUncertainty}");
                    return (new Solution<T>(region2, dataAtMidpoint, midpointUncertainty),
                        new Interval(intervalAtMidpoint.LowerBound, regionInterval.UpperBound - midpointUncertainty));
                }
            }

            void reportProgress2(Solution<T> nodeState, Interval value)
            {
                reportProgress?.Invoke(value.LowerBound, countDimensionsSearched());

                int countDimensionsSearched()
                {
                    return searchedDimensions.Count;
                }
            }
        }
#pragma warning restore CS1570 // XML comment has badly formed XML

        /// <summary>
        /// Represents a solution found during optimization.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        struct Solution<T>
        {
            public Solution(Region region, T data, double uncertainty)
            {
                Region = region;
                Data = data;
                Uncertainty = uncertainty;
            }

            /// <summary>
            /// Region of the solution (not a point in general).
            /// </summary>
            public Region Region { get; }

            /// <summary>
            /// Gets data at the midpoint of the region.
            /// </summary>
            public T Data { get; }

            /// <summary>
            /// Gets the width of the interval at the midpoint.
            /// </summary>
            public double Uncertainty { get; }

            public override string ToString()
            {
                return $"Solution (Region={Region}, Uncertainty={Uncertainty})";
            }
        }

        /// <summary>
        /// Finds the best node in a bounded-depth search tree.
        /// </summary>
        /// <param name="start">Starting state for the search.  The root of the search tree.</param>
        /// <param name="startingValue"></param>
        /// <param name="successors">The successors of a state in the search tree, with bounds on the value of each state.  Bounds need not be tight, but must become tight as the depth increases.</param>
        /// <param name="fTolerance">Allowable error in the optimal value.  Must be greater than zero.</param>
        /// <param name="cancellationToken">Cancels the search early.</param>
        /// <param name="reportProgress">Called periodically with best solution so far.</param>
        /// <returns>A State close to the global maximum of the function and whether the search completed before cancellation.</returns>
        public static (State, Interval, bool) AnytimeColumnSearch<State>(
            State start,
            Interval startingValue,
            Func<(State, Interval), IEnumerable<(State, Interval)>> successors,
            double fTolerance,
            CancellationToken cancellationToken = default(CancellationToken),
            Action<State, Interval> reportProgress = null)
        {
            // Reference:
            // "Anytime column search"
            // Satya Gautam Vadlamudi, Piyush Gaurav, Sandip Aine, Partha Pratim Chakrabarti
            // Australasian Joint Conference on Artificial Intelligence, 2012

            if (fTolerance <= 0) throw new ArgumentOutOfRangeException(nameof(fTolerance), fTolerance, "fTolerance <= 0");
            // We always evaluate the midpoint, even if cancellationToken is set.
            // The caller is expected to provide cancellationToken to getBound if needed.
            State argmax = start;
            Interval max = startingValue;
            List<PriorityQueue<QueueNode<State>>> queues = new List<PriorityQueue<QueueNode<State>>>();
            // lowerBound is used to prune search regions.
            // lowerBound must only increase over time.
            double lowerBound = double.NegativeInfinity;
            void addState(int depth, State state, Interval value)
            {
                if (value.LowerBound > value.UpperBound) throw new Exception($"value.LowerBound > value.UpperBound: {value}");
                if (value.LowerBound > lowerBound)
                {
                    // argmax must be updated atomically
                    argmax = state;
                    max = value;
                    lowerBound = value.LowerBound + fTolerance;
                    reportProgress?.Invoke(argmax, value);
                }
                if (value.UpperBound > lowerBound)
                {
                    while (queues.Count <= depth) queues.Add(new PriorityQueue<QueueNode<State>>());
                    var node = new QueueNode<State>(state, value);
                    queues[depth].Add(node);
                    if (Debug)
                        Trace.WriteLine($"AnytimeColumnSearch: added {node}");
                    return;
                }
                if (Debug)
                    Trace.WriteLine($"AnytimeColumnSearch: rejected state {state} with upperBound {value.UpperBound:r} <= {lowerBound:r}");
            }
            addState(0, start, startingValue);
            var task = Task.Run(() =>
            {
                while (queues.Any(queue => queue.Count > 0))
                {
                    if (cancellationToken.IsCancellationRequested) break;
                    for (int depth = 0; depth < queues.Count; depth++)
                    {
                        if (cancellationToken.IsCancellationRequested) break;
                        var queue = queues[depth];
                        if (queue.Count == 0) continue;
                        // get the node with highest upper bound
                        var node = queue.ExtractMinimum();
                        if (node.Value.UpperBound <= lowerBound)
                            continue;

                        // expand the node
                        if (Debug)
                            Trace.WriteLine($"AnytimeColumnSearch: expanding {node}");
                        foreach (var (successor, interval) in successors((node.state, node.Value)))
                        {
                            if (cancellationToken.IsCancellationRequested) break;
                            if (interval.UpperBound > lowerBound)
                            {
                                if (Debug) Trace.WriteLine($"AnytimeColumnSearch: successor = {successor} interval = {interval}");
                                if (interval.LowerBound > interval.UpperBound) throw new Exception($"interval.LowerBound > interval.UpperBound {interval}");
                                addState(depth + 1, successor, interval);
                            }
                            else if (Debug) Trace.WriteLine($"AnytimeColumnSearch: rejected successor {successor} with upper bound {interval.UpperBound:r} <= {lowerBound:r}");
                        }
                    }
                }
            }, cancellationToken);
            try
            {
                // If cancellationToken is cancelled, task.Wait() is cancelled and throws OperationCancelledException,
                // but note that the task can continue to execute.
                // If the task throws an exception, task.Wait() throws AggregateException with InnerExceptions collection containing
                // the thrown exception.
                task.Wait(cancellationToken);
            }
            catch (OperationCanceledException) { }

            // If the task's loop was cancelled right after the only element was dequeued, then the inner loop (by successors)
            // will also be cancelled and the execution goes here with potentially missing elements in the queue.
            // Thus if the queues are empty here but the search cancelled, we cannot be sure that it completed.
            bool searchCompleted = !queues.Any(queue => queue.Count > 0) && !cancellationToken.IsCancellationRequested;
            return (argmax, max, searchCompleted);
        }

        /// <summary>
        /// Finds the input that maximizes a function.
        /// </summary>
        /// <param name="bounds">Bounds for the search.</param>
        /// <param name="getBound">Returns bounds on the function in a region.  Need not be tight, but must become tight as the region shrinks.</param>
        /// <param name="fTolerance">Allowable error in the function value.  Must be greater than zero.</param>
        /// <param name="cancellationToken"></param>
        /// <param name="reportProgress"></param>
        /// <returns>A Vector close to the global maximum of the function.</returns>
        public static Vector FindMaximum2(
            Region bounds,
            Func<Region, Interval> getBound,
            double fTolerance,
            CancellationToken cancellationToken = default(CancellationToken),
            Action<double> reportProgress = null)
        {
            if (fTolerance <= 0) throw new ArgumentOutOfRangeException($"fTolerance <= 0");
            int dim = bounds.Lower.Count;
            if (dim == 0) return Vector.Zero(dim);
            double lowerBound = double.NegativeInfinity;
            Vector argmax = bounds.GetMidpoint();
            // We always evaluate the midpoint, even if cancellationToken is set.
            // The caller is expected to provide cancellationToken to getBound if needed.
            double valueAtMidpoint0 = getBound(RegionPoint(argmax)).LowerBound;
            long upperBoundCount = 0;
            if (Debug)
            {
                Func<Region, Interval> getBound1 = getBound;
                getBound = region =>
                {
                    Stopwatch watch = Stopwatch.StartNew();
                    Interval bound = getBound1(region);
                    watch.Stop();
                    if (timeAccumulator.Count > 10 && watch.ElapsedMilliseconds > timeAccumulator.Mean + 4 * System.Math.Sqrt(timeAccumulator.Variance))
                        Trace.WriteLine($"GetUpperBound took {watch.ElapsedMilliseconds}ms");
                    timeAccumulator.Add(watch.ElapsedMilliseconds);
                    //if (upperBoundCount % 100 == 0) Trace.WriteLine($"lowerBound = {lowerBound}");
                    return bound;
                };
            }
            Stack<Block> stack = new Stack<Block>();
            void processUnsplittable(int splitDim, Region region)
            {
                // go to next dim
                int nextDim = (splitDim + 1) % dim;
                stack.Push(new Block(nextDim, region));
            }
            stack.Push(new Block(0, bounds));
            double addTolerance(double f) => f + fTolerance;// * (Math.Abs(f) + 1e-10);
            void updateLowerBound(Region region, Interval valueAtMidpoint)
            {
                double newLowerBound = addTolerance(valueAtMidpoint.LowerBound);
                if (newLowerBound > lowerBound)
                {
                    lowerBound = newLowerBound;
                    argmax = region.GetMidpoint();
                    Trace.WriteLine($"lowerBound plus fTolerance = {lowerBound}");
                    reportProgress?.Invoke(valueAtMidpoint.LowerBound);
                }
            }
            var task = Task.Run(() =>
            {
                while (stack.Count > 0)
                {
                    Block block = stack.Pop();
                    if (RegionIsPoint(block.region))
                    {
                        var valueAtMidpoint = getBound(block.region);
                        if (Debug)
                        {
                            Trace.WriteLine($"point region {block.region}, valueAtMidpoint = {valueAtMidpoint}");
                        }
                        updateLowerBound(block.region, valueAtMidpoint);
                        continue;
                    }
                    int splitDim = block.splitDim;
                    if (block.region.Lower[splitDim] == block.region.Upper[splitDim])
                    {
                        processUnsplittable(splitDim, block.region);
                        continue;
                    }
                    PriorityQueue<QueueNode2> queue = new PriorityQueue<QueueNode2>();
                    double partialLowerBound = double.NegativeInfinity;
                    void updatePartialLowerBound(Interval valueAtMidpoint)
                    {
                        double newPartialLowerBound = addTolerance(valueAtMidpoint.LowerBound);
                        if (newPartialLowerBound > partialLowerBound)
                        {
                            partialLowerBound = newPartialLowerBound;
                        }
                    }
                    upperBoundCount++;
                    Interval blockRegionBounds = getBound(block.region);
                    if (blockRegionBounds.UpperBound <= lowerBound) continue; // TODO: this uses full objective
                    Region blockMidpoint = RegionMidpoint(block.region, splitDim);
                    Interval blockMidpointBounds = getBound(blockMidpoint);
                    updatePartialLowerBound(blockMidpointBounds);
                    updateLowerBound(block.region, blockMidpointBounds); // TODO: this uses full objective
                    QueueNode2 blockNode = new QueueNode2(block.region, blockMidpoint, blockRegionBounds, blockMidpointBounds);
                    queue.Add(blockNode);
                    while (queue.Count > 0)
                    {
                        if (cancellationToken.IsCancellationRequested) break;
                        // get the node with highest upper confidence bound
                        var node = queue.ExtractMinimum();
                        if (node.Bounds.UpperBound <= partialLowerBound)
                        {
                            if (Debug)
                            {
                                Trace.WriteLine($"unsplittable on dimension {splitDim}: {node}");
                            }
                            processUnsplittable(splitDim, node.Midpoint);
                            // all other nodes must be inferior to this one
                            queue.Clear();
                            continue;
                        }
                        // Can we gain more by splitting other dimensions?
                        double outerGap = node.Bounds.UpperBound - node.ValueAtMidpoint.UpperBound;
                        outerGap += node.ValueAtMidpoint.LowerBound - node.Bounds.LowerBound;
                        double innerGap = node.ValueAtMidpoint.Width();
                        if (outerGap < 1e-2 * innerGap) // TODO: this could lead to infinite loop
                        {
                            if (Debug)
                            {
                                Trace.WriteLine($"not splitting on {splitDim}: {node}, outerGap = {outerGap}, innerGap = {innerGap}");
                            }
                            // Push all remaining regions onto the stack, worst first
                            List<Region> regions = new List<Region>();
                            regions.Add(node.Region);
                            while (queue.Count > 0)
                            {
                                node = queue.ExtractMinimum();
                                // TODO: only keep nodes that are not pruned
                                regions.Add(node.Region);
                            }
                            regions.Reverse();
                            foreach (var region2 in regions)
                            {
                                processUnsplittable(splitDim, region2);
                            }
                            continue;
                        }
                        Region region = node.Region;
                        if (Debug)
                        {
                            Trace.WriteLine($"splitting dimension {splitDim} of {node}");
                        }
                        if (node.ValueAtMidpoint.LowerBound > node.Bounds.UpperBound) throw new Exception("node.ValueAtMidpoint > node.UpperBound");

                        // split the node
                        double midpointAtSplitDim = node.Midpoint.Lower[splitDim];
                        QueueNode2 leftNode = null, rightNode = null;
                        if (region.Upper[splitDim] != midpointAtSplitDim)
                        {
                            Region leftRegion = new Region(region);
                            if (double.IsInfinity(region.Lower[splitDim]) || double.IsInfinity(midpointAtSplitDim)) throw new Exception();
                            // This ensures that the boundary between regions is halfway between the midpoints.
                            double leftBoundary = (region.Lower[splitDim] + 2 * midpointAtSplitDim) / 3;
                            leftRegion.Upper[splitDim] = leftBoundary;
                            region.Lower[splitDim] = leftBoundary;
                            upperBoundCount++;
                            var leftBounds = getBound(leftRegion);
                            if (leftBounds.UpperBound <= partialLowerBound)
                            {
                                if (Debug)
                                    Trace.WriteLine($"rejected region {leftRegion} with partial upperBound {leftBounds.UpperBound} <= partial lowerBound {partialLowerBound}");
                            }
                            else
                            {
                                var leftMidpoint = RegionMidpoint(leftRegion, splitDim);
                                var valueAtMidpointLeft = getBound(leftMidpoint);
                                if (cancellationToken.IsCancellationRequested) break;
                                updatePartialLowerBound(valueAtMidpointLeft);
                                leftNode = new QueueNode2(leftRegion, leftMidpoint, leftBounds, valueAtMidpointLeft);
                            }
                        }
                        if (region.Lower[splitDim] != midpointAtSplitDim)
                        {
                            Region rightRegion = new Region(region);
                            if (double.IsInfinity(region.Upper[splitDim]) || double.IsInfinity(midpointAtSplitDim)) throw new Exception();
                            double rightBoundary = (region.Upper[splitDim] + 2 * midpointAtSplitDim) / 3;
                            rightRegion.Lower[splitDim] = rightBoundary;
                            region.Upper[splitDim] = rightBoundary;
                            upperBoundCount++;
                            var rightBounds = getBound(rightRegion);
                            if (rightBounds.UpperBound <= partialLowerBound)
                            {
                                if (Debug)
                                    Trace.WriteLine($"rejected region {rightRegion} with partial upperBound {rightBounds.UpperBound} <= partial lowerBound {partialLowerBound}");
                            }
                            else
                            {
                                var rightMidpoint = RegionMidpoint(rightRegion, splitDim);
                                var valueAtMidpointRight = getBound(rightMidpoint);
                                if (cancellationToken.IsCancellationRequested) break;
                                updatePartialLowerBound(valueAtMidpointRight);
                                rightNode = new QueueNode2(rightRegion, rightMidpoint, rightBounds, valueAtMidpointRight);
                                // since partialLowerBound may have changed, re-check whether left child is pruned.
                                if (leftNode != null && leftNode.Bounds.UpperBound <= partialLowerBound)
                                {
                                    if (Debug)
                                        Trace.WriteLine($"rejected region {leftNode.Region} with partial upperBound {leftNode.Bounds.UpperBound} <= partial lowerBound {partialLowerBound}");
                                    leftNode = null;
                                }
                            }
                        }
                        QueueNode2 middleNode;
                        upperBoundCount++;
                        var newBounds = getBound(region);
                        if (newBounds.UpperBound <= partialLowerBound)
                        {
                            middleNode = null;
                            if (Debug)
                                Trace.WriteLine($"rejected region {region} with partial upperBound {newBounds.UpperBound} <= partial lowerBound {partialLowerBound}");
                        }
                        else
                        {
                            middleNode = new QueueNode2(region, node.Midpoint, newBounds, node.ValueAtMidpoint);
                        }
                        // if left or right was pruned, add middle
                        // if middle was pruned, add left and right
                        // if middle midpoint prunes left midpoint or vice versa, add remaining (this implies above)
                        // otherwise flip a coin
                        if (leftNode != null) queue.Add(leftNode);
                        if (rightNode != null) queue.Add(rightNode);
                        if (middleNode != null) queue.Add(middleNode);
                    }
                }
                if (Debug)
                    Trace.WriteLine($"Interval.FindMaximum2 upperBoundCount = {upperBoundCount}");
            }, cancellationToken);
            try
            {
                // If cancellationToken is cancelled, task.Wait() is cancelled and throws OperationCancelledException,
                // but note that the task can continue to execute.
                // If the task throws an exception, task.Wait() throws AggregateException with InnerExceptions collection containing
                // the thrown exception.
                task.Wait(cancellationToken);
            }
            catch (OperationCanceledException) { }
            return argmax;
        }

        private class Block
        {
            public readonly int splitDim;
            public readonly Region region;

            public Block(int splitDim, Region region)
            {
                this.splitDim = splitDim;
                this.region = region;
            }
        }

        private class QueueNode<State> : IComparable<QueueNode<State>>
        {
            public readonly State state;
            public readonly Interval Value;

            public QueueNode(State state, Interval value)
            {
                this.state = state;
                this.Value = value;
            }

            public int CompareTo(QueueNode<State> other)
            {
                // Arguments are flipped so that queue.ExtractMinimum returns the largest value.
                bool greedy = true;
                if (greedy)
                {
                    return other.Value.LowerBound.CompareTo(Value.LowerBound);
                }
                else
                {
                    int result = other.Value.UpperBound.CompareTo(Value.UpperBound);
                    if (result == 0)
                    {
                        result = other.Value.LowerBound.CompareTo(Value.LowerBound);
                        //if (result == 0) result = state.CompareTo(other.state);
                    }
                    return result;
                }
            }

            public override string ToString()
            {
                return $"{nameof(QueueNode<State>)}({this.state}, Value={Value:r})";
            }
        }

        private class QueueNode1<T> : IComparable<QueueNode1<T>>
        {
            public readonly Region Region;
            public readonly double UpperBound;
            public double UpperConfidenceBound;
            /// <summary>
            /// Must be less than or equal to UpperBound
            /// </summary>
            public readonly double ValueAtMidpoint;
            public readonly T DataAtMidpoint;

            public QueueNode1(Region region, double upperBound, double valueAtMidpoint, T dataAtMidpoint)
            {
                this.Region = region;
                this.UpperBound = upperBound;
                this.UpperConfidenceBound = valueAtMidpoint;
                this.ValueAtMidpoint = valueAtMidpoint;
                this.DataAtMidpoint = dataAtMidpoint;
            }

            public int CompareTo(QueueNode1<T> other)
            {
                int result = other.UpperConfidenceBound.CompareTo(UpperConfidenceBound);
                if (result == 0)
                {
                    result = other.ValueAtMidpoint.CompareTo(ValueAtMidpoint);
                    if (result == 0) result = Region.CompareTo(other.Region);
                }
                return result;
            }

            public override string ToString()
            {
                return $"QueueNode({Region}, ValueAtMidpoint={ValueAtMidpoint:r}, UpperConfidenceBound={UpperConfidenceBound:r}, UpperBound={UpperBound:r})";
            }
        }

        private class QueueNode2 : IComparable<QueueNode2>
        {
            public readonly Region Region;
            public readonly Region Midpoint;
            public readonly Interval Bounds;
            public readonly Interval ValueAtMidpoint;

            public QueueNode2(Region region, Region midpoint, Interval bounds, Interval valueAtMidpoint)
            {
                this.Region = region;
                this.Midpoint = midpoint;
                this.Bounds = bounds;
                this.ValueAtMidpoint = valueAtMidpoint;
            }

            public int CompareTo(QueueNode2 other)
            {
                int result = other.Bounds.UpperBound.CompareTo(Bounds.UpperBound);
                if (result == 0)
                {
                    result = other.ValueAtMidpoint.LowerBound.CompareTo(ValueAtMidpoint.LowerBound);
                    if (result == 0) result = Region.CompareTo(other.Region);
                }
                return result;
            }

            public override string ToString()
            {
                return $"QueueNode2({Region}, ValueAtMidpoint={ValueAtMidpoint}, Bounds={Bounds})";
            }
        }

        public static bool Debug;
        static readonly MeanVarianceAccumulator timeAccumulator = new MeanVarianceAccumulator();

        public static Region RegionPoint(Vector vector)
        {
            Region region = new Region(vector.Count);
            region.Lower.SetTo(vector);
            region.Upper.SetTo(vector);
            return region;
        }

        public static Region RegionMidpoint(Region region, int dim)
        {
            Region result = new Region(region);
            double midpoint = Region.GetMidpoint(region.Lower[dim], region.Upper[dim]);
            result.Lower[dim] = midpoint;
            result.Upper[dim] = midpoint;
            return result;
        }

        public static bool RegionIsPoint(Region region)
        {
            for (int i = 0; i < region.Dimension; i++)
            {
                if (region.Lower[i] != region.Upper[i]) return false;
            }
            return true;
        }

        public Region ToRegion()
        {
            Region region = new Region(1);
            region.Lower[0] = LowerBound;
            region.Upper[0] = UpperBound;
            return region;
        }
    }

    public static class IntervalExtensions
    {
        public static Interval Sum(this IEnumerable<Interval> intervals)
        {
            Interval sum = Interval.Point(0);
            foreach (var interval in intervals)
            {
                sum += interval;
            }
            return sum;
        }

        public static Interval Average(this IEnumerable<Interval> intervals)
        {
            Interval sum = Interval.Point(0);
            int count = 0;
            foreach (var interval in intervals)
            {
                sum += interval;
                count++;
            }
            return sum / count;
        }

        public static Interval WeightedAverage(this IEnumerable<(Interval, double)> tuples)
        {
            IntervalMeanAccumulator accumulator = new IntervalMeanAccumulator();
            foreach (var (interval, weight) in tuples)
            {
                accumulator.Add(interval, weight);
            }
            return accumulator.Mean;
        }
    }

    public class IntervalMeanAccumulator
    {
        readonly MeanAccumulator lowerBoundAccumulator = new MeanAccumulator();
        readonly MeanAccumulator upperBoundAccumulator = new MeanAccumulator();

        public void Add(Interval x, double weight)
        {
            lowerBoundAccumulator.Add(x.LowerBound, weight);
            upperBoundAccumulator.Add(x.UpperBound, weight);
        }

        public Interval Mean => new Interval(lowerBoundAccumulator.Mean, upperBoundAccumulator.Mean);
    }
}