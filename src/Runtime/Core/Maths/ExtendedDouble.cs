using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Represents a number as Mantissa * exp(Exponent).
    /// </summary>
    public class ExtendedDouble
    {
        public readonly double Mantissa, Exponent;

        public ExtendedDouble(double mantissa, double exponent)
        {
            this.Mantissa = mantissa;
            this.Exponent = exponent;
        }

        public override bool Equals(object obj)
        {
            if (obj is ExtendedDouble that)
            {
                return (this.Mantissa == that.Mantissa) && (this.Exponent == that.Exponent);
            }
            else return false;
        }

        public override int GetHashCode()
        {
            return Hash.Combine(Mantissa.GetHashCode(), Exponent);
        }

        public override string ToString()
        {
            return $"{Mantissa:g17}*exp({Exponent:g17})";
        }

        public static ExtendedDouble Zero()
        {
            return new ExtendedDouble(0, 0);
        }

        public static ExtendedDouble PositiveInfinity()
        {
            return new ExtendedDouble(double.PositiveInfinity, 0);
        }

        public static ExtendedDouble NaN()
        {
            return new ExtendedDouble(double.NaN, 0);
        }

        public static ExtendedDouble FromDouble(double value)
        {
            return new ExtendedDouble(value, 0);
        }

        public double ToDouble()
        {
            if (System.Math.Abs(Exponent) >= 700)
            {
                // avoid overflow/underflow when computing Exp(Exponent)
                // Abs(Exponent)/2 <= 1e-16 causes loss of precision here.
                double expHalf = System.Math.Exp(Exponent / 2);
                return Mantissa * expHalf * expHalf;
            }
            else
            {
                return Mantissa * System.Math.Exp(Exponent);
            }
        }

        public double Log()
        {
            return Exponent + System.Math.Log(Mantissa);
        }

        public ExtendedDouble Max(double minimum)
        {
            return new ExtendedDouble(System.Math.Max(minimum, Mantissa), Exponent);
        }

        public ExtendedDouble MultiplyExp(double logarithm)
        {
            return new ExtendedDouble(Mantissa, Exponent + logarithm);
        }

        public static ExtendedDouble operator *(ExtendedDouble x, ExtendedDouble y)
        {
            return new ExtendedDouble(x.Mantissa * y.Mantissa, x.Exponent + y.Exponent);
        }

        public static ExtendedDouble operator /(ExtendedDouble x, ExtendedDouble y)
        {
            return new ExtendedDouble(x.Mantissa / y.Mantissa, x.Exponent - y.Exponent);
        }

        public static ExtendedDouble operator *(ExtendedDouble x, double y)
        {
            return new ExtendedDouble(x.Mantissa * y, x.Exponent);
        }

        public static ExtendedDouble operator /(ExtendedDouble x, double y)
        {
            return new ExtendedDouble(x.Mantissa / y, x.Exponent);
        }

        public static ExtendedDouble operator +(ExtendedDouble x, ExtendedDouble y)
        {
            if (x.Mantissa == 0)
            {
                return y;
            }
            else if (y.Mantissa == 0)
            {
                return x;
            }
            else if (y.Exponent > x.Exponent)
            {
                if (double.IsInfinity(x.Mantissa)) throw new ArgumentOutOfRangeException(nameof(x), x, "x is infinite");
                return new ExtendedDouble(x.Mantissa * System.Math.Exp(x.Exponent - y.Exponent) + y.Mantissa, y.Exponent);
            }
            else
            {
                if (double.IsInfinity(y.Mantissa)) throw new ArgumentOutOfRangeException(nameof(y), y, "y is infinite");
                return new ExtendedDouble(x.Mantissa + y.Mantissa * System.Math.Exp(y.Exponent - x.Exponent), x.Exponent);
            }
        }

        public static ExtendedDouble operator -(ExtendedDouble x)
        {
            return new ExtendedDouble(-x.Mantissa, x.Exponent);
        }

        public static ExtendedDouble operator -(ExtendedDouble x, ExtendedDouble y)
        {
            if (x.Mantissa == 0)
            {
                return -y;
            }
            else if (y.Mantissa == 0)
            {
                return x;
            }
            else if (y.Exponent > x.Exponent)
            {
                if (double.IsInfinity(x.Mantissa)) throw new ArgumentOutOfRangeException(nameof(x), x, "x is infinite");
                return new ExtendedDouble(x.Mantissa * System.Math.Exp(x.Exponent - y.Exponent) - y.Mantissa, y.Exponent);
            }
            else
            {
                if (double.IsInfinity(y.Mantissa)) throw new ArgumentOutOfRangeException(nameof(y), y, "y is infinite");
                return new ExtendedDouble(x.Mantissa - y.Mantissa * System.Math.Exp(y.Exponent - x.Exponent), x.Exponent);
            }
        }
    }
}
