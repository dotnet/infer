// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
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
    /// <summary>
    /// Represents a linear enclosure of a function over an input interval.  The input interval is not represented.
    /// </summary>
    public class Beam
    {
        public readonly Vector Slope;
        public readonly Interval Offset;

        public Beam(Vector slope, Interval offset)
        {
            this.Slope = slope;
            this.Offset = offset;
        }

        public Beam(Vector slope, double offset) : this(slope, Interval.Point(offset))
        {
        }

        public Beam(double slope, Interval offset) : this(Vector.FromArray(slope), offset)
        {
        }

        public Beam(double slope, double offset) : this(slope, Interval.Point(offset))
        {
        }

        public Beam(Interval offset) : this(Vector.FromArray(), offset)
        {
        }

        public Beam(double offset) : this(Interval.Point(offset))
        {
        }

        public override string ToString()
        {
            return $"Beam(Slope={Slope}, Offset={Offset})";
        }

        public Interval GetOutputInterval(Vector input)
        {
            return Slope.Inner(input) + Offset;
        }

        public Interval GetOutputInterval(double input)
        {
            if (Slope.Count != 1) throw new ArgumentException();
            return Slope[0] * input + Offset;
        }

        public Interval GetOutputInterval(params Interval[] inputs)
        {
            if (Slope.Count != inputs.Length) throw new ArgumentException($"inputs.Length ({inputs.Length}) != Slope.Count ({Slope.Count})", nameof(inputs));
            Interval output = Offset;
            for (int i = 0; i < inputs.Length; i++)
            {
                output += Slope[i] * inputs[i];
            }
            return output;
        }

        public Beam Integrate(Interval input)
        {
            // integral is 
            // Slope*(U^2 - L^2)/2 + Offset*(U - L) 
            // = (Slope*(U+L)/2 + Offset)*(U-L)
            double width = input.Width();
            return new Beam(
                slope: Vector.FromArray(Slope.Skip(1).Select(x => x * width).ToArray()),
                offset: (Slope[0] * input.Midpoint() + Offset) * width
            );
        }

        public static Beam operator +(Beam beam, double value)
        {
            return new Beam(beam.Slope, beam.Offset + value);
        }

        public static Beam operator -(Beam beam, double value)
        {
            return new Beam(beam.Slope, beam.Offset - value);
        }

        public static Beam operator -(Beam beam)
        {
            return new Beam(-beam.Slope, -beam.Offset);
        }

        public static Beam operator +(Beam beam, Beam beam2)
        {
            return new Beam(beam.Slope + beam2.Slope, beam.Offset + beam2.Offset);
        }

        public static Beam operator -(Beam beam, Beam beam2)
        {
            return new Beam(beam.Slope - beam2.Slope, beam.Offset - beam2.Offset);
        }

        public static Beam operator *(Beam beam, double value)
        {
            return new Beam(beam.Slope * value, beam.Offset * value);
        }

        public static Beam operator /(Beam beam, double value)
        {
            return new Beam(beam.Slope / value, beam.Offset / value);
        }

        public bool IsNaN()
        {
            return Offset.IsNaN() || Slope.Any(double.IsNaN);
        }

        public Beam Compose(params Beam[] inputs)
        {
            if (inputs.Length != this.Slope.Count) throw new ArgumentException($"inputs.Length ({inputs.Length}) != Slope.Count ({Slope.Count})", nameof(inputs));
            if (inputs.Length == 0) return this;
            double weight0 = Slope[0];
            Vector slope = weight0 * inputs[0].Slope;
            Interval offset = this.Offset + weight0 * inputs[0].Offset;
            for (int i = 1; i < inputs.Length; i++)
            {
                double weight = Slope[i];
                slope.SetToSum(1, slope, weight, inputs[i].Slope);
                offset += weight * inputs[i].Offset;
            }
            return new Beam(slope, offset);
        }

        public Beam ComposePartial(Beam input)
        {
            if (this.Slope.Count == 0) throw new InvalidOperationException($"Slope.Count == 0");
            double weight0 = Slope[0];
            Vector slope = Vector.FromArray(Util.ArrayInit(checked(input.Slope.Count + this.Slope.Count - 1), i =>
            {
                if (i < input.Slope.Count)
                {
                    return weight0 * input.Slope[i];
                }
                else
                {
                    return this.Slope[checked(i - input.Slope.Count + 1)];
                }
            }));
            Interval offset = this.Offset + weight0 * input.Offset;
            return new Beam(slope, offset);
        }

        public static Beam Exp(Interval input)
        {
            double fLower = System.Math.Exp(input.LowerBound);
            double width = input.Width();
            if (width == 0)
            {
                // Want continuity as width->0, and to allow Beams to generalize autodiff
                double derivative = fLower;
                // Offset is exp(x) - exp(x)*x
                return new Beam(derivative, fLower - fLower * input.LowerBound);
            }
            double fUpper = System.Math.Exp(input.UpperBound);
            double slope = (fUpper - fLower) / width;
            // Offset comes from extrema of (exp(x) - slope*x)
            // Stationary point satisfies (exp(x) - slope = 0), x = log(slope)
            Interval offset = new Interval(slope * (1 - System.Math.Log(slope)), System.Math.Max(fUpper - slope * input.UpperBound, fLower - slope * input.LowerBound));
            return new Beam(slope, offset);
        }

        public static Beam Reciprocal(Interval input)
        {
            if (input.LowerBound <= 0)
            {
                return new Beam(0, 1 / input);
            }
            double fLower = 1 / input.LowerBound;
            double width = input.Width();
            if (width == 0)
            {
                double derivative = -fLower * fLower;
                // Offset is 1/x - (-1/x^2)*x = 2/x
                return new Beam(derivative, 2 * fLower);
            }
            double fUpper = 1 / input.UpperBound;
            double slope = (fUpper - fLower) / width;
            // Offset comes from extrema of (1/x - slope*x)
            // Stationary point satisfies (-1/x^2 - slope = 0), x = 1/sqrt(-slope)
            // At this point, 1/x - slope*x = sqrt(-slope) + sqrt(-slope) = 2*sqrt(-slope)
            Interval offset = new Interval(2 * System.Math.Sqrt(-slope), System.Math.Max(fUpper - slope * input.UpperBound, fLower - slope * input.LowerBound));
            return new Beam(slope, offset);
        }

        public static Beam Square(Interval input)
        {
            double fLower = input.LowerBound * input.LowerBound;
            double width = input.Width();
            if (width == 0)
            {
                // Want continuity as width->0, and to allow Beams to generalize autodiff
                double derivative = 2 * input.LowerBound;
                // Offset is x^2 - (2*x)*x = -x^2
                return new Beam(derivative, -fLower);
            }
            double fUpper = input.UpperBound * input.UpperBound;
            double slope = (fUpper - fLower) / width;
            // Offset comes from extrema of (x^2 - slope*x)
            // Stationary point satisfies (2x - slope = 0), x = slope/2
            // At this point, x^2 - slope*x = slope^2/4 - slope^2/2 = -slope^2/4
            Interval offset = new Interval(-slope * slope / 4, System.Math.Max(fUpper - slope * input.UpperBound, fLower - slope * input.LowerBound));
            return new Beam(slope, offset);
        }

        public static Beam Abs(Interval input)
        {
            if (input.LowerBound >= 0) return new Beam(1, 0);
            if (input.UpperBound <= 0) return new Beam(-1, 0);
            bool ensureNonnegative = false;
            if (ensureNonnegative)
            {
                // Returning slope=0 is bad because slopes are multiplied during composition so the overall slope will be zero, reducing the Beam to an interval.
                return new Beam(0, input.Abs());
            }
            else
            {
                double fLower = System.Math.Abs(input.LowerBound);
                double fUpper = System.Math.Abs(input.UpperBound);
                // width cannot be 0
                if (input.LowerBound < double.MinValue)
                {
                    if (input.UpperBound > double.MaxValue)
                    {
                        return new Beam(0, input.Abs());
                    }
                    else
                    {
                        // input.UpperBound > 0
                        return new Beam(-1, new Interval(0, fUpper + input.UpperBound));
                    }
                }
                else if (input.UpperBound > double.MaxValue)
                {
                    // input.LowerBound < 0
                    return new Beam(1, new Interval(0, fLower - input.LowerBound));
                }
                else
                {
                    double slope = (fUpper - fLower) / input.Width();
                    Interval offset = new Interval(0, System.Math.Max(fUpper - slope * input.UpperBound, fLower - slope * input.LowerBound));
                    return new Beam(slope, offset);
                }
            }
        }

        public static Beam Min(Interval input, double maximum)
        {
            if (input.LowerBound >= maximum) return new Beam(0, maximum);
            if (input.UpperBound <= maximum) return new Beam(1, 0);
            double fLower = input.LowerBound;
            double fUpper = maximum;
            // width cannot be 0
            double slope = (fUpper - fLower) / input.Width();
            Interval offset = new Interval(System.Math.Min(fLower - slope * input.LowerBound, fUpper - slope * input.UpperBound), maximum - slope * maximum);
            return new Beam(slope, offset);
        }

        public static Beam Max(Interval input, double minimum)
        {
            if (input.LowerBound >= minimum) return new Beam(1, 0);
            if (input.UpperBound <= minimum) return new Beam(0, minimum);
            double fLower = minimum;
            double fUpper = input.UpperBound;
            // width cannot be 0
            double slope = (fUpper - fLower) / input.Width();
            Interval offset = new Interval(minimum - slope * minimum, System.Math.Max(fLower - slope * input.LowerBound, fUpper - slope * input.UpperBound));
            return new Beam(slope, offset);
        }

        public static Beam NormalCdf(Interval input)
        {
            double fLower = MMath.NormalCdf(input.LowerBound);
            double width = input.Width();
            if (width == 0)
            {
                double derivative = System.Math.Exp(-0.5 * input.LowerBound * input.LowerBound) * MMath.InvSqrt2PI;
                if (MMath.AreEqual(derivative, 0)) return new Beam(0, fLower);
                else return new Beam(derivative, fLower - derivative * input.LowerBound);
            }
            double fUpper = MMath.NormalCdf(input.UpperBound);
            double slope = (fUpper - fLower) / width;
            if (double.IsInfinity(slope)) slope = 0;
            double offsetAtUpper, offsetAtLower;
            if (MMath.AreEqual(slope, 0))
            {
                // avoid 0 * infinity
                offsetAtUpper = fUpper;
                offsetAtLower = fLower;
            }
            else
            {
                offsetAtUpper = fUpper - slope * input.UpperBound;
                offsetAtLower = fLower - slope * input.LowerBound;
            }
            double offsetUpper = System.Math.Max(offsetAtLower, offsetAtUpper);
            double offsetLower = System.Math.Min(offsetAtLower, offsetAtUpper);
            // Offset comes from extrema of (NormalCdf(x) - slope*x)
            // Stationary point satisfies (N(x;0,1) - slope = 0), x = sqrt(-2*log(Sqrt2PI*slope))
            double x = System.Math.Sqrt(-2 * System.Math.Log(MMath.Sqrt2PI * slope));
            if (x > input.LowerBound && x < input.UpperBound)
            {
                double offsetAtStationaryPoint = MMath.NormalCdf(x) - slope * x;
                offsetUpper = System.Math.Max(offsetUpper, offsetAtStationaryPoint);
            }
            x = -x;
            if (x > input.LowerBound && x < input.UpperBound)
            {
                double offsetAtStationaryPoint = MMath.NormalCdf(x) - slope * x;
                offsetLower = System.Math.Min(offsetLower, offsetAtStationaryPoint);
            }
            Interval offset = new Interval(offsetLower, offsetUpper);
            return new Beam(slope, offset);
        }

        public static Beam GetProbLessThan(CanGetProbLessThan<double> canGetProbLessThan, int count, Interval input, out Interval output)
        {
            return NondecreasingFunction(canGetProbLessThan.GetProbLessThan, count, input, out output);
        }

        public static Beam GetQuantile(CanGetQuantile<double> canGetQuantile, int count, Interval input, out Interval output)
        {
            return NondecreasingFunction(p => canGetQuantile.GetQuantile(System.Math.Min(1, System.Math.Max(0, p))), count, input, out output);
        }

        public static Beam NondecreasingFunction(Func<double, double> func, int count, Interval input, out Interval output)
        {
            double fLower = func(input.LowerBound);
            double fUpper = func(input.UpperBound);
            output = new Interval(fLower, fUpper);
            if (input.LowerBound < double.MinValue || input.UpperBound > double.MaxValue || fLower < double.MinValue || fUpper > double.MaxValue)
            {
                return new Beam(output);
            }
            double width = input.Width();
            if (width == 0)
            {
                double newUpperBound = input.LowerBound + 10000 * MMath.Ulp(input.LowerBound / 2);
                if (newUpperBound > double.MaxValue)
                {
                    input = new Interval(input.UpperBound - 10000 * MMath.Ulp(input.UpperBound / 2), input.UpperBound);
                }
                else
                {
                    input = new Interval(input.LowerBound, newUpperBound);
                }
                width = input.Width();
                count = 2;
            }
            double[] outputs = linspace(input.LowerBound, input.UpperBound, count).Select(func).ToArray();
            fLower = outputs[0];
            fUpper = outputs[count - 1];
            double slope = (fUpper - fLower) / width;
            if (double.IsInfinity(slope)) slope = 0;
            double offsetAtUpper, offsetAtLower;
            if (MMath.AreEqual(slope, 0))
            {
                offsetAtUpper = fUpper;
                offsetAtLower = fLower;
            }
            else
            {
                offsetAtUpper = fUpper - slope * input.UpperBound;
                offsetAtLower = fLower - slope * input.LowerBound;
            }
            double offsetUpper = System.Math.Max(offsetAtLower, offsetAtUpper);
            double offsetLower = System.Math.Min(offsetAtLower, offsetAtUpper);
            if (!MMath.AreEqual(slope, 0))
            {
                // Test every intermediate point for an offset extremum
                double increment = width / (count - 1);
                for (int i = 1; i < count; i++)
                {
                    // Between left and right is a box of possible probabilities.
                    // Since slope is non-negative, the highest offset must be at left, and lowest at right.
                    double left = input.LowerBound + increment * (i - 1);
                    double right = input.LowerBound + increment * i;
                    double offsetAtLeft = outputs[i] - slope * left;
                    double offsetAtRight = outputs[i - 1] - slope * right;
                    offsetUpper = System.Math.Max(offsetUpper, offsetAtLeft);
                    offsetLower = System.Math.Min(offsetLower, offsetAtRight);
                }
            }
            Interval offset = new Interval(offsetLower, offsetUpper);
            return new Beam(slope, offset);
        }

        private static double[] linspace(double min, double max, int count)
        {
            if (count < 2)
                throw new ArgumentException("count < 2");
            if (min == max) return new double[] { min };
            double inc = (max - min) / (count - 1);
            return Util.ArrayInit(count, i => (min + i * inc));
        }

        /// <summary>
        /// Computes a beam for the function a*b.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Beam Product(Interval a, Interval b)
        {
            // Want to enclose x1*x2 with c1*x1 + c2*x2 + b
            // Let c1 = midpt(x2), c2 = midpt(x1)
            // The deriv of difference is [x2 - c1, x1 - c2] so midpt is a stationary point.
            // But this stationary point cannot be an extremum.
            double aMidpoint = a.Midpoint();
            double bMidpoint = b.Midpoint();
            Vector slope = Vector.FromArray(bMidpoint, aMidpoint);
            double offsetMidpoint = -aMidpoint * bMidpoint;
            double bHalfWidth = b.Width() / 2;
            double aLowerBHalfWidth = a.LowerBound * bHalfWidth; // a.LowerBound * b.UpperBound - bMidpoint * a.LowerBound
            double aUpperBHalfWidth = a.UpperBound * bHalfWidth;
            double bLowerAMidpoint = aMidpoint * b.LowerBound;
            double bUpperAMidpoint = aMidpoint * b.UpperBound;
            double offsetLowerLower = -aLowerBHalfWidth - bLowerAMidpoint;
            double offsetLowerUpper = aLowerBHalfWidth - bUpperAMidpoint;
            double offsetUpperLower = -aUpperBHalfWidth - bLowerAMidpoint;
            double offsetUpperUpper = aUpperBHalfWidth - bUpperAMidpoint;
            double offsetUpper = System.Math.Max(System.Math.Max(offsetLowerLower,
                                                   offsetLowerUpper),
                                          System.Math.Max(offsetUpperLower,
                                                   offsetUpperUpper));
            double offsetLower = System.Math.Min(System.Math.Min(offsetLowerLower,
                                                   offsetLowerUpper),
                                          System.Math.Min(offsetUpperLower,
                                                   offsetUpperUpper));
            return new Beam(slope, new Interval(offsetLower, offsetUpper));
        }

        /// <summary>
        /// Returns a linear enclosure over (weights[0], values[0], ..., weights[n-1], values[n-1])
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        public static Beam WeightedAverage(Interval[] weights, Interval[] values)
        {
            int count = values.Length;
            Interval previousWeightSum = weights[0];
            Interval previousWeightedAverage = values[0];
            Beam result = null;
            Beam previousWeightSumBeam = null;
            for (int i = 1; i < count; i++)
            {
                Beam beam = WeightedAverage(previousWeightSum, previousWeightedAverage, weights[i], values[i]);
                previousWeightedAverage = beam.GetOutputInterval(previousWeightSum, previousWeightedAverage, weights[i], values[i]);
                Beam weightBeam = new Beam(Vector.FromArray(1, 1), 0);
                previousWeightSum += weights[i];
                if (result == null)
                {
                    result = beam;
                    previousWeightSumBeam = weightBeam;
                }
                else
                {
                    // result is a Beam over (weights[0], values[0], ..., weights[i-1], values[i-1])
                    // previousWeightSumBeam is a Beam over (weights[0], ..., weights[i-1])
                    int resultCount = result.Slope.Count;
                    result = beam.Compose(previousWeightSumBeam.PadRight(checked(resultCount + 2 - i)), result.PadRight(2), Unit(resultCount, resultCount + 2), Unit(resultCount + 1, resultCount + 2));
                    previousWeightSumBeam = weightBeam.Compose(previousWeightSumBeam.PadRight(1), Unit(i, i + 1));
                }
            }
            return result;
        }

        public Beam Permute(params int[] permutation)
        {
            return new Beam(Vector.FromArray(Util.ArrayInit(Slope.Count, i => Slope[permutation[i]])), Offset);
        }

        public static Beam Unit(int index, int count)
        {
            return new Beam(UnitVector(index, count), 0);
        }

        private static Vector UnitVector(int index, int count)
        {
            return Vector.FromArray(Util.ArrayInit(count, i => (i == index) ? 1.0 : 0.0));
        }

        /// <summary>
        /// Computes a beam for the function (weight1*value1 + weight2*value2)/(weight1 + weight2)
        /// </summary>
        /// <param name="weight1"></param>
        /// <param name="value1"></param>
        /// <param name="weight2"></param>
        /// <param name="value2"></param>
        /// <returns></returns>
        public static Beam WeightedAverage(Interval weight1, Interval value1, Interval weight2, Interval value2)
        {
            Interval weight1z = weight1.Max(0);
            Interval weight2z = weight2.Max(0);
            double weight1Midpoint = weight1z.Midpoint();
            double value1Midpoint = value1.Midpoint();
            double weight2Midpoint = weight2z.Midpoint();
            double value2Midpoint = value2.Midpoint();
            // These checks avoid denom == 0
            if (weight1Midpoint == 0)
            {
                if (weight2Midpoint == 0)
                    return new Beam(Vector.FromArray(0, 0.5, 0, 0.5), 0);
                else
                    return new Beam(Vector.FromArray(0, 0, 0, 1), 0);
            }
            else if (weight2Midpoint == 0)
                return new Beam(Vector.FromArray(0, 1, 0, 0), 0);
            double numer = weight1Midpoint * value1Midpoint + weight2Midpoint * value2Midpoint;
            double denom = weight1Midpoint + weight2Midpoint;
            double weightedAverageMidpoint = numer / denom;
            // slope is the vector of partial derivatives at midpoint
            Vector slope = Vector.FromArray((value1Midpoint - weightedAverageMidpoint) / denom, weight1Midpoint / denom, (value2Midpoint - weightedAverageMidpoint) / denom, weight2Midpoint / denom);
            Interval offsetLowerLower = GetOffset(value1.LowerBound, value2.LowerBound);
            Interval offsetLowerUpper = GetOffset(value1.LowerBound, value2.UpperBound);
            Interval offsetUpperLower = GetOffset(value1.UpperBound, value2.LowerBound);
            Interval offsetUpperUpper = GetOffset(value1.UpperBound, value2.UpperBound);
            double offsetUpper = System.Math.Max(System.Math.Max(offsetLowerLower.UpperBound,
                                                   offsetLowerUpper.UpperBound),
                                          System.Math.Max(offsetUpperLower.UpperBound,
                                                   offsetUpperUpper.UpperBound));
            double offsetLower = System.Math.Min(System.Math.Min(offsetLowerLower.LowerBound,
                                                   offsetLowerUpper.LowerBound),
                                          System.Math.Min(offsetUpperLower.LowerBound,
                                                   offsetUpperUpper.LowerBound));
            return new Beam(slope, new Interval(offsetLower, offsetUpper));

            // Computes an interval over WeightedAverage(weight1, v1, weight2, v2) - slope*(weight1,v1,weight2,v2)
            Interval GetOffset(double v1, double v2)
            {
                Interval offsetW1Lower = GetCornerBounds(weight2.LowerBound);
                Interval offsetW1Upper = GetCornerBounds(weight2.UpperBound);
                double lowerBound = System.Math.Min(offsetW1Lower.LowerBound, offsetW1Upper.LowerBound);
                double upperBound = System.Math.Max(offsetW1Lower.UpperBound, offsetW1Upper.UpperBound);
                ProcessWeight1StationaryPoint(weight2.LowerBound);
                ProcessWeight1StationaryPoint(weight2.UpperBound);
                ProcessWeight2StationaryPoint(weight1.LowerBound);
                ProcessWeight2StationaryPoint(weight1.UpperBound);
                ProcessJointStationaryPoint();
                return new Interval(lowerBound, upperBound);

                void ProcessJointStationaryPoint()
                {
                    if (slope[0] != slope[2])
                    {
                        // Besides endpoints, we must check stationary points.
                        // Derivative of difference wrt weight1 gives
                        // v1/(w1+w2) - (w1*v1+w2*v2)/(w1+w2)^2 - slope[0] = 0
                        // v1*(w1+w2) = (w1*v1+w2*v2) + slope[0]*(w1+w2)^2
                        // Derivative of difference wrt weight2 gives
                        // v2*(w1+w2) = (w1*v1+w2*v2) + slope[2]*(w1+w2)^2
                        // If both derivs are zero, then
                        // (v1-v2)*(w1+w2) = (slope[0]-slope[2])*(w1+w2)^2
                        // w1+w2 = (v1-v2)/(slope[0]-slope[2])
                        // w1*v1+w2*v2 = w1*v1 + (s - w1)*v2 = w1*(v1-v2) + s*v2
                        // w1*(v1-v2) = s*(v1 - v2 - slope[0]*s)
                        // w1 = s*(1 - slope[0]*s/(v1-v2))
                        // w2 = s - w1 = s*s*slope[0]/(v1-v2)
                        double slopeDiff = slope[0] - slope[2];
                        double sum = (v1 - v2) / slopeDiff;
                        double root1 = -sum * slope[2] / slopeDiff;
                        double root2 = sum * slope[0] / slopeDiff;
                        if (weight1z.Contains(root1) && weight2z.Contains(root2))
                        {
                            //Trace.WriteLine($"joint stationary point at ({root1},{root2})");
                            double stationaryOffset = MMath.WeightedAverage(root1, v1, root2, v2) - (slope[0] * root1 + slope[1] * v1 + slope[2] * root2 + slope[3] * v2);
                            lowerBound = System.Math.Min(lowerBound, stationaryOffset);
                            upperBound = System.Math.Max(upperBound, stationaryOffset);
                        }
                    }
                }

                // Computes an interval over WeightedAverage(weight1, v1, w2, v2) - slope*(weight1,v1,w2,v2)
                Interval GetCornerBounds(double w2)
                {
                    double offsetWeight1Lower = MMath.WeightedAverage(weight1z.LowerBound, v1, w2, v2) - (slope[0] * weight1.LowerBound + slope[1] * v1 + slope[2] * w2 + slope[3] * v2);
                    double offsetWeight1Upper = MMath.WeightedAverage(weight1z.UpperBound, v1, w2, v2) - (slope[0] * weight1.UpperBound + slope[1] * v1 + slope[2] * w2 + slope[3] * v2);
                    double lowerBound1, upperBound1;
                    if (offsetWeight1Lower < offsetWeight1Upper)
                    {
                        lowerBound1 = offsetWeight1Lower;
                        upperBound1 = offsetWeight1Upper;
                    }
                    else
                    {
                        lowerBound1 = offsetWeight1Upper;
                        upperBound1 = offsetWeight1Lower;
                    }
                    return new Interval(lowerBound1, upperBound1);
                }

                void ProcessWeight1StationaryPoint(double w2)
                {
                    if (w2 == 0 && weight1.Contains(0))
                    {
                        // Take the limit as weight1 -> 0
                        double offset0WeightLower = v1 - (slope[0] * weight1.LowerBound + slope[1] * v1 + slope[2] * w2 + slope[3] * v2);
                        double offset0WeightUpper = v1 - (slope[0] * weight1.UpperBound + slope[1] * v1 + slope[2] * w2 + slope[3] * v2);
                        if (offset0WeightLower < offset0WeightUpper)
                        {
                            lowerBound = System.Math.Min(lowerBound, offset0WeightLower);
                            upperBound = System.Math.Max(upperBound, offset0WeightUpper);
                        }
                        else
                        {
                            lowerBound = System.Math.Min(lowerBound, offset0WeightUpper);
                            upperBound = System.Math.Max(upperBound, offset0WeightLower);
                        }
                    }
                    if (slope[0] != 0)
                    {
                        // Derivative of difference wrt weight1 gives
                        // v1/(w1+w2) - (w1*v1+w2*v2)/(w1+w2)^2 - slope[0] = 0
                        // v1*(w1+w2) = (w1*v1+w2*v2) + slope[0]*(w1+w2)^2
                        GaussianOp_Slow.GetRealRoots(new[] { slope[0], 2 * slope[0] * w2, w2 * (slope[0] * w2 + v2 - v1) }, out List<double> roots);
                        foreach (var root in roots)
                        {
                            if (weight1z.Contains(root))
                            {
                                //Trace.WriteLine($"root1 at ({root},{w2})");
                                double value = MMath.WeightedAverage(root, v1, System.Math.Max(0, w2), v2) - (slope[0] * root + slope[1] * v1 + slope[2] * w2 + slope[3] * v2);
                                lowerBound = System.Math.Min(lowerBound, value);
                                upperBound = System.Math.Max(upperBound, value);
                            }
                        }
                    }
                }

                // Computes an interval over WeightedAverage(w1, v1, weight2, v2) - slope*(w1,v1,weight2,v2)
                void ProcessWeight2StationaryPoint(double w1)
                {
                    if (w1 == 0 && weight2.Contains(0))
                    {
                        // Take the limit as weight2 -> 0
                        double offset0WeightLower = v2 - (slope[0] * w1 + slope[1] * v1 + slope[2] * weight2.LowerBound + slope[3] * v2);
                        double offset0WeightUpper = v2 - (slope[0] * w1 + slope[1] * v1 + slope[2] * weight2.UpperBound + slope[3] * v2);
                        if (offset0WeightLower < offset0WeightUpper)
                        {
                            lowerBound = System.Math.Min(lowerBound, offset0WeightLower);
                            upperBound = System.Math.Max(upperBound, offset0WeightUpper);
                        }
                        else
                        {
                            lowerBound = System.Math.Min(lowerBound, offset0WeightUpper);
                            upperBound = System.Math.Max(upperBound, offset0WeightLower);
                        }
                    }
                    if (slope[2] != 0)
                    {
                        // Derivative of difference wrt weight2 gives
                        // v2*(w1+w2) = (w1*v1+w2*v2) + slope[2]*(w1+w2)^2
                        GaussianOp_Slow.GetRealRoots(new[] { slope[2], 2 * slope[2] * w1, w1 * (slope[2] * w1 + v1 - v2) }, out List<double> roots);
                        foreach (var root in roots)
                        {
                            if (weight2z.Contains(root))
                            {
                                //Trace.WriteLine($"root2 at ({w1},{root})");
                                double value = MMath.WeightedAverage(System.Math.Max(0, w1), v1, root, v2) - (slope[0] * w1 + slope[1] * v1 + slope[2] * root + slope[3] * v2);
                                lowerBound = System.Math.Min(lowerBound, value);
                                upperBound = System.Math.Max(upperBound, value);
                            }
                        }
                        bool check = false;
                        if (check)
                        {
                            foreach (var w2 in linspace(weight2.LowerBound, weight2.UpperBound, 100))
                            {
                                double value = MMath.WeightedAverage(System.Math.Max(0, w1), v1, System.Math.Max(0, w2), v2) - (slope[0] * w1 + slope[1] * v1 + slope[2] * w2 + slope[3] * v2);
                                if (value < lowerBound) Trace.WriteLine($"exceeded lowerBound at ({w1},{v1},{w2},{v2})");
                                if (value > upperBound) Trace.WriteLine($"exceeded upperBound at ({w1},{v1},{w2},{v2})");
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Computes a Beam from (left,right) to the expected value of a function over the interval [left,right].
        /// </summary>
        /// <param name="maximumError"></param>
        /// <param name="cancellationToken"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="skillDistribution"></param>
        /// <param name="preservesPoint">If true, the output interval approaches a point as the input interval does.</param>
        /// <param name="getBeam"></param>
        /// <returns></returns>
        public static Beam GetExpectation(double maximumError, CancellationToken cancellationToken, Interval left, Interval right, ITruncatableDistribution<double> skillDistribution, bool preservesPoint, Func<Interval, Beam> getBeam)
        {
            // Algorithm: divide the interval into three parts, bound the metric in each part, then work out the optimal weighting of parts.
            // The first part is the interval between left lower and upper bound.
            // The second part is the interval between left upper and right lower bound, if it exists.
            // The third part is the interval between right lower and upper bound.
            //double leftLowerCdf = skillDistribution.GetProbLessThan(left.LowerBound);
            //double leftUpperCdf = skillDistribution.GetProbLessThan(left.UpperBound);
            //double rightLowerCdf = skillDistribution.GetProbLessThan(right.LowerBound);
            //double rightUpperCdf = skillDistribution.GetProbLessThan(right.UpperBound);
            Interval truncatedLeft = new Interval(left.LowerBound, System.Math.Min(left.UpperBound, System.Math.Max(left.LowerBound, right.LowerBound)));
            // Beam from left to value
            Beam leftBeam = getBeam(truncatedLeft);
            // int(f(x)*p(x), x=left..leftUpper)/int(p(x), x=left..leftUpper) = 
            // int(f(GetQuantileTruncated(p)), p=0..1) = 
            // int(f(GetQuantile(p*(probLessThanRight - probLessThanLeft) + probLessThanLeft)), p=0..1)
            // If f(GetQuantile(p)) is linear, then s*int(p*probBetween + probLessThanLeft, p=0..1) = 
            // s * (0.5*(probLessThanRight - probLessThanLeft) + probLessThanLeft) = 
            // s * 0.5*(probLessThanRight + probLessThanLeft) 
            Beam probLessThanLeftBeam = GetProbLessThan(skillDistribution, 100, truncatedLeft, out Interval probLessThanLeft);
            Beam leftQuantileBeam = GetQuantile(skillDistribution, 100, probLessThanLeft, out Interval ignoreLeft);
            Beam leftAverageBeam = new Beam(0.5, 0.5 * probLessThanLeft.UpperBound).Compose(probLessThanLeftBeam);
            Beam boundPart1 = leftBeam.Compose(leftQuantileBeam).Compose(leftAverageBeam);

            Interval truncatedRight = new Interval(System.Math.Max(right.LowerBound, System.Math.Min(right.UpperBound, left.UpperBound)), right.UpperBound);
            // Beam from right to value
            Beam rightBeam = getBeam(truncatedRight);
            Beam probLessThanRightBeam = GetProbLessThan(skillDistribution, 100, truncatedRight, out Interval probLessThanRight);
            Beam rightQuantileBeam = GetQuantile(skillDistribution, 100, probLessThanRight, out Interval ignoreRight);
            Beam rightAverageBeam = new Beam(0.5, 0.5 * probLessThanRight.LowerBound).Compose(probLessThanRightBeam);
            Beam boundPart3 = rightBeam.Compose(rightQuantileBeam).Compose(rightAverageBeam);

            // Beam from () to value
            Beam boundPart2;
            if (left.UpperBound < right.LowerBound)
            {
                boundPart2 = GetExpectation(maximumError, cancellationToken, skillDistribution.Truncate(left.UpperBound, right.LowerBound), preservesPoint, getBeam);
                if (boundPart2.IsNaN()) throw new Exception("boundPart2 is NaN");
            }
            else
            {
                // Any function value in the center interval is a possible value for the expectation.
                Interval center = new Interval(truncatedLeft.UpperBound, truncatedRight.LowerBound);
                boundPart2 = new Beam(getBeam(center).GetOutputInterval(center));
            }
            // Beam from left to weight1
            var weight1Beam = GetProbBetween(skillDistribution, truncatedLeft, truncatedLeft.UpperBound, out Interval weight1);
            var weight2 = new Beam(Interval.Point(System.Math.Max(0, probLessThanRight.LowerBound - probLessThanLeft.UpperBound)));
            // Beam from right to weight3
            var weight3Beam = GetProbBetween(skillDistribution, truncatedRight.LowerBound, truncatedRight, out Interval weight3);
            Interval[] weights = new[] { weight1, weight2.GetOutputInterval(), weight3 };
            bool debug = false;
            if (debug)
            {
                Trace.WriteLine($"left = {left}, right = {right}");
                Trace.WriteLine($"boundPart1 = {boundPart1}, boundPart2 = {boundPart2}, boundPart3 = {boundPart3}");
                Trace.WriteLine($"weights = {StringUtil.CollectionToString(weights, " ")}");
            }
            var values = new[] { boundPart1.GetOutputInterval(left), boundPart2.GetOutputInterval(), boundPart3.GetOutputInterval(right) };
            Beam result = WeightedAverage(weights, values);
            //if (result.IsNaN()) throw new Exception("result is NaN");
            return result.Compose(weight1Beam.PadRight(), boundPart1.PadRight(), weight2.PadRight(2), boundPart2.PadRight(2), weight3Beam.PadLeft(), boundPart3.PadLeft());
        }

        /// <summary>
        /// Compute bounds on the expected value of a function.
        /// </summary>
        /// <param name="maximumError"></param>
        /// <param name="cancellationToken"></param>
        /// <param name="canGetQuantile"></param>
        /// <param name="preservesPoint">If true, the output interval approaches a point as the input interval does.</param>
        /// <param name="getBeam"></param>
        /// <returns></returns>
        public static Beam GetExpectation(double maximumError, CancellationToken cancellationToken, CanGetQuantile<double> canGetQuantile, bool preservesPoint, Func<Interval, Beam> getBeam)
        {
            if (preservesPoint)
                return GetExpectation(maximumError, cancellationToken, canGetQuantile, getBeam);
            else
                return GetExpectation(30, canGetQuantile, getBeam);
        }

        /// <summary>
        /// Computes bounds on the expected value of a function over an interval.
        /// </summary>
        /// <param name="count">The number of subdivisions.  Increase for more accuracy.</param>
        /// <param name="canGetQuantile"></param>
        /// <param name="getInterval">Returns bounds on the function's output over all inputs in an interval.</param>
        private static Beam GetExpectation(int count, CanGetQuantile<double> canGetQuantile, Func<Interval, Beam> getInterval)
        {
            // To get a bound on the expectation, we divide the region into intervals of equal probability.
            // We compute the bound in each interval, and take an unweighted average.
            // the quantile ranks will be 1/count, 2/count, ..., count/count
            double increment = 1.0 / count;
            double start = increment;
            Beam sum = null;
            double previousInput = canGetQuantile.GetQuantile(0);
            for (int i = 0; i < count; i++)
            {
                double quantileRank = start + i * increment;
                double input = canGetQuantile.GetQuantile(System.Math.Min(1, quantileRank));
                Beam beam = getInterval(new Interval(previousInput, input));
                if (sum == null) sum = beam;
                else sum += beam;
                previousInput = input;
            }
            return sum / count;
        }

        /// <summary>
        /// Returns Beam from right to (GetProbLessThan(right) - GetProbLessThan(left))
        /// </summary>
        /// <param name="canGetProbLessThan"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="probBetween"></param>
        /// <returns></returns>
        public static Beam GetProbBetween(CanGetProbLessThan<double> canGetProbLessThan, double left, Interval right, out Interval probBetween)
        {
            int count = 100;
            double probLessThanLeft = canGetProbLessThan.GetProbLessThan(left);
            Beam difference = Beam.GetProbLessThan(canGetProbLessThan, count, right, out Interval probLessThanRight) - probLessThanLeft;
            probBetween = probLessThanRight - probLessThanLeft;
            return difference;
        }

        /// <summary>
        /// Returns Beam from left to (GetProbLessThan(right) - GetProbLessThan(left))
        /// </summary>
        /// <param name="canGetProbLessThan"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="probBetween"></param>
        /// <returns></returns>
        public static Beam GetProbBetween(CanGetProbLessThan<double> canGetProbLessThan, Interval left, double right, out Interval probBetween)
        {
            int count = 100;
            double probLessThanRight = canGetProbLessThan.GetProbLessThan(right);
            Beam difference = new Beam(0, probLessThanRight) - Beam.GetProbLessThan(canGetProbLessThan, count, left, out Interval probLessThanLeft);
            probBetween = probLessThanRight - probLessThanLeft;
            return difference;
        }

        public static Beam GetProbBetween(CanGetProbLessThan<double> canGetProbLessThan, Interval left, Interval right, out Interval probBetween)
        {
            int count = 100;
            Beam leftBeam = Beam.GetProbLessThan(canGetProbLessThan, count, left, out Interval probLessThanLeft);
            leftBeam = leftBeam.PadRight();
            Beam rightBeam = Beam.GetProbLessThan(canGetProbLessThan, count, right, out Interval probLessThanRight);
            rightBeam = rightBeam.PadLeft();
            probBetween = probLessThanRight - probLessThanLeft;
            Beam subtract = new Beam(Vector.FromArray(-1, 1), 0);
            return subtract.Compose(leftBeam, rightBeam);
        }

        /// <summary>
        /// Returns a linear enclosure of the expectation as a function of (left,right,...)
        /// </summary>
        /// <param name="allowedRelativeError"></param>
        /// <param name="cancellationToken"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="skillDistribution"></param>
        /// <param name="getBeam">The output interval must approach a point as the input interval does.</param>
        /// <returns></returns>
        public static Beam GetExpectation2(double allowedRelativeError, CancellationToken cancellationToken, Interval left, Interval right, ITruncatableDistribution<double> skillDistribution, Func<Interval, Beam> getBeam)
        {
            int count = 100;
            Beam probLessThanLeftBeam = Beam.GetProbLessThan(skillDistribution, count, left, out Interval probLessThanLeft);
            probLessThanLeftBeam = probLessThanLeftBeam.PadRight();
            Beam probLessThanRightBeam = Beam.GetProbLessThan(skillDistribution, count, right, out Interval probLessThanRight);
            probLessThanRightBeam = probLessThanRightBeam.PadLeft();
            Beam subtract = new Beam(Vector.FromArray(-1, 1), 0);
            Interval probBetween = (probLessThanRight - probLessThanLeft).Abs();
            // beam from (left,right) to probBetween
            Beam probBetweenBeam = subtract.Compose(probLessThanLeftBeam, probLessThanRightBeam);
            return Integrate(new Interval(0, 1), allowedRelativeError, cancellationToken, probability =>
            {
                // beam from (truncated prob, probBetween)
                Beam product = Beam.Product(probability, probBetween);
                // beam from (truncated prob,probLessThanLeft,probLessThanRight) to untruncated prob
                Beam probabilityBeam = product.Compose(new Beam(Vector.FromArray(1, 0, 0), 0), subtract.PadLeft()) + new Beam(Vector.FromArray(0, 1, 0), 0);
                //Interval untruncatedProbability1 = probability * (probLessThanRight - probLessThanLeft) + probLessThanLeft;
                Interval untruncatedProbability = probabilityBeam.GetOutputInterval(probability, probLessThanLeft, probLessThanRight);
                //Interval untruncatedProbability = probability * probLessThanRight + (1 - probability) * probLessThanLeft;
                untruncatedProbability = untruncatedProbability.Min(probLessThanRight.UpperBound).Max(probLessThanLeft.LowerBound);
                // beam from prob to quantile
                Beam quantileBeam = GetQuantile(skillDistribution, 100, untruncatedProbability, out Interval quantile);
                // beam from (truncated prob,left,right) to quantile
                Beam quantileBeam2 = quantileBeam.Compose(probabilityBeam).Compose(new Beam(Vector.FromArray(1,0,0),0), probLessThanLeftBeam.PadLeft(), probLessThanRightBeam.PadLeft());
                // getBeam returns beam over (quantile,...)
                // ComposePartial returns beam over (truncated prob,left,right,...)
                return getBeam(quantile).ComposePartial(quantileBeam2);
            });
        }

        public Beam PadLeft()
        {
            return new Beam(Vector.FromArray(Util.ArrayInit(this.Slope.Count + 1, i => (i == 0) ? 0 : this.Slope[i - 1])), Offset);
        }

        public Beam PadRight(int padding = 1)
        {
            int count = this.Slope.Count;
            return new Beam(Vector.FromArray(Util.ArrayInit(checked(this.Slope.Count + padding), i => (i >= count) ? 0 : this.Slope[i])), Offset);
        }

        // The output interval must approach a point as the input interval does.
        public static Beam GetExpectation(double maximumError, CancellationToken cancellationToken, CanGetQuantile<double> canGetQuantile, Func<Interval, Beam> getBeam)
        {
            return Integrate(new Interval(0, 1), maximumError, cancellationToken, probability =>
            {
                Beam quantileBeam = GetQuantile(canGetQuantile, 100, probability, out Interval quantile);
                return getBeam(quantile).ComposePartial(quantileBeam);
            });
        }

        /// <summary>
        /// Returns bounds on the integral of a function over <c>this</c>.
        /// </summary>
        /// <param name="interval"></param>
        /// <param name="maximumError"></param>
        /// <param name="cancellationToken"></param>
        /// <param name="getBeam">The output interval must approach a point as the input interval does.</param>
        /// <returns></returns>
        public static Beam Integrate(Interval interval, double maximumError, CancellationToken cancellationToken, Func<Interval, Beam> getBeam)
        {
            // Algorithm:
            // Divide the domain into exhaustive regions.
            // Subdivide the region with largest error estimate.
            // Bound the integral with the sum of region bounds.
            PriorityQueue<IntegrateQueueNode> queue = new PriorityQueue<IntegrateQueueNode>();
            Beam sum = null;
            int nodeCount = 0;
            void addRegion(Interval input)
            {
                nodeCount++;
                Beam output = getBeam(input).Integrate(input);
                if (double.IsInfinity(output.Offset.LowerBound) || double.IsInfinity(output.Offset.UpperBound))
                    throw new Exception($"output.Offset is infinite: {output.Offset}");
                if (sum == null) sum = output;
                else sum += output;
                IntegrateQueueNode node = new IntegrateQueueNode(input, output);
                queue.Add(node);
            }
            addRegion(interval);
            while (sum.Offset.Width() > maximumError && !cancellationToken.IsCancellationRequested)
            {
                var node = queue.ExtractMinimum();  // gets the node with highest error
                // Remove node from the sum
                sum = sum.Remove(node.Output);
                // Split the node
                double midpoint = node.Input.Midpoint();
                addRegion(new Interval(node.Input.LowerBound, midpoint));
                addRegion(new Interval(midpoint, node.Input.UpperBound));
                if (nodeCount > 10000)
                {
                    Trace.WriteLine($"Integrate nodeCount = {nodeCount}.  This usually indicates a numerical problem.");
                    break;
                }
                if (sum.IsNaN()) throw new Exception("sum is NaN");
            }
            return sum;
        }

        public Beam Remove(Beam that)
        {
            return new Beam(Slope - that.Slope, Offset.Remove(that.Offset));
        }

        private class IntegrateQueueNode : IComparable<IntegrateQueueNode>
        {
            public readonly Interval Input;
            public readonly Beam Output;

            public IntegrateQueueNode(Interval input, Beam output)
            {
                this.Input = input;
                this.Output = output;
            }

            public int CompareTo(IntegrateQueueNode other)
            {
                return (other.Output.Offset.Width() * other.Input.Width()).CompareTo(this.Output.Offset.Width() * this.Input.Width());
            }

            public override string ToString()
            {
                return $"IntegrateQueueNode(Input={Input}, Output={Output})";
            }
        }

    }
}
