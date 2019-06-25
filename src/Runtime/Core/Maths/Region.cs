// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Math
{
    using Microsoft.ML.Probabilistic.Utilities;
    using System;

    /// <summary>
    /// Represents a hyper-rectangle in arbitrary dimensions.
    /// </summary>
    public class Region
    {
        public readonly Vector Lower, Upper;

        public int Dimension
        {
            get
            {
                return Lower.Count;
            }
        }

        /// <summary>
        /// Creates a new Region containing only the all-zero vector.
        /// </summary>
        /// <param name="dimension"></param>
        public Region(int dimension)
        {
            Lower = Vector.Zero(dimension);
            Upper = Vector.Zero(dimension);
        }

        public Region(Region that)
        {
            Lower = Vector.Copy(that.Lower);
            Upper = Vector.Copy(that.Upper);
        }

        public double GetLogVolume()
        {
            double logVolume = 0;
            for (int i = 0; i < Dimension; i++)
            {
                logVolume += Math.Log(Math.Max(1e-10, Upper[i] - Lower[i]));
            }
            return logVolume;
        }

        public bool Contains(Vector x)
        {
            for (int i = 0; i < Dimension; i++)
            {
                if (x[i] < Lower[i] || x[i] > Upper[i]) return false;
            }
            return true;
        }

        public Vector GetMidpoint()
        {
            Vector x = Vector.Zero(Dimension);
            for (int i = 0; i < Dimension; i++)
            {
                x[i] = GetMidpoint(Lower[i], Upper[i]);
            }
            return x;
        }

        public static double GetMidpoint(double lower, double upper)
        {
            double midpoint;
            if (double.IsNegativeInfinity(lower))
            {
                if (double.IsPositiveInfinity(upper)) midpoint = 0.0;
                else if (upper > 0) midpoint = -upper;
                else if (upper < 0) midpoint = 2 * upper;
                else midpoint = -1;
            }
            else if (double.IsPositiveInfinity(upper))
            {
                if (lower > 0) midpoint = 2 * lower;
                else if (lower < 0) midpoint = -lower;
                else midpoint = 1;
            }
            else
            {
                midpoint = MMath.Average(lower, upper);
            }
            return midpoint;
        }

        public Vector Sample()
        {
            Vector x = Vector.Zero(Dimension);
            for (int i = 0; i < Dimension; i++)
            {
                x[i] = Uniform(Lower[i], Upper[i]);
            }
            return x;
        }

        public static double Uniform(double lowerBound, double upperBound)
        {
            return Rand.Double() * (upperBound - lowerBound) + lowerBound;
        }

        public override string ToString()
        {
            return ToString("g4");
        }

        public string ToString(string format)
        {
            return string.Format("[{0},{1}]", Lower.ToString(format), Upper.ToString(format));
        }

        public override bool Equals(object obj)
        {
            Region that = obj as Region;
            if (that == null) return false;
            return (that.Lower == this.Lower) && (that.Upper == this.Upper);
        }

        public override int GetHashCode()
        {
            return Hash.Combine(Lower.GetHashCode(), Upper.GetHashCode());
        }

        public int CompareTo(Region other)
        {
            int result = CompareTo(Lower, other.Lower);
            if (result == 0) result = CompareTo(Upper, other.Upper);
            return result;
        }

        public int CompareTo(Vector a, Vector b)
        {
            int result = 0;
            for (int i = 0; i < a.Count; i++)
            {
                result = a[i].CompareTo(b[i]);
                if (result != 0) return result;
            }
            return result;
        }
    }
}
