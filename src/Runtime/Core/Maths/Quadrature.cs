// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Microsoft.ML.Probabilistic.Math
{
    // These routines use tables of nodes and weights.  To compute nodes and weights from scratch, see the routines at:
    // http://www.csit.fsu.edu/~burkardt/m_src/quadrule/quadrule.html
    /// <summary>
    /// Quadrature nodes and weights
    /// </summary>
    public static class Quadrature
    {
        private const int AdaptiveQuadratureMaxNodes = 10000;

        /// <summary>
        /// Integrate the function f from -Infinity to Infinity
        /// </summary>
        /// <param name="f">The function to integrate</param>
        /// <param name="scale">A positive tuning parameter.  f is assumed to be negligible outside of [-scale,scale]</param>
        /// <param name="nodeCount">The initial number of nodes.  Should be at least 2 and a power of 2.</param>
        /// <param name="relTol">A threshold to stop subdividing</param>
        /// <returns></returns>
        public static double AdaptiveClenshawCurtis(Converter<double, double> f, double scale, int nodeCount, double relTol)
        {
            // To get fast convergence, the transformation function should grow like 1/x (or faster) near the endpoints of the integration.
            // In this case, we use 1/tan(x) which is approximately 1/x when x is near 0 and approximately -1/x when x is near pi.
            // The transformation function atanh(x) for x in [-1,1] would be a bad choice since it only grows like log(1/x) near the endpoints.
            double fInvTan(double x)
            {
                if (x == 0 || x == System.Math.PI)
                    return 0;
                double sinX = System.Math.Sin(x);
                return f(scale / System.Math.Tan(x)) / (sinX * sinX);
            }
            return scale * AdaptiveTrapeziumRule(fInvTan, nodeCount, 0, System.Math.PI, relTol, AdaptiveQuadratureMaxNodes);
        }

        /// <summary>
        /// Integrate the function f from 0 to +Infinity using exp-sinh quadrature.
        /// </summary>
        /// <param name="f">The function to integrate</param>
        /// <param name="scale">A positive tuning parameter.
        /// <paramref name="f"/> is assumed to be continuous and at least one of <paramref name="f"/>(0) and
        /// cosh(arsinh(2 * ln(scale)/pi)) * scale * <paramref name="f"/>(<paramref name="scale"/>) to be non-zero.</param>
        /// <param name="relTol">A threshold to stop subdividing</param>
        /// <returns></returns>
        public static double AdaptiveExpSinh(Converter<double, double> f, double scale, double relTol)
        {
            const double halfPi = System.Math.PI / 2;
            const double ln2ByHalfPi = MMath.Ln2 / halfPi;
            // Let g(x) = f(exp(pi/2 * sinh(x))) * cosh(x) * exp(pi/2 * sinh(x)),
            // fExpSinh(x) = g(x) + g(-x), when x != 0, fExpSinh(0) = g(0),
            // then int_-inf^+inf g(x) dx = int_0^+inf fExpSinh(x) dx
            double fExpSinh(double x)
            {
                if (x == 0)
                    return f(1);
                double halfPiSinhX = halfPi * System.Math.Sinh(x);
                double expHalfPiSinhX = System.Math.Exp(halfPiSinhX);
                double expHalfPiSinhNegX = System.Math.Exp(-halfPiSinhX);
                return System.Math.Cosh(x) * (expHalfPiSinhX * f(expHalfPiSinhX) + expHalfPiSinhNegX * f(expHalfPiSinhNegX));
            }
            double invHalfPiLnScale = System.Math.Log(scale) / halfPi;
            double rescale = System.Math.Abs(System.Math.Log(invHalfPiLnScale + System.Math.Sqrt(invHalfPiLnScale * invHalfPiLnScale + 1))); // abs . arsinh
            while (fExpSinh(rescale) == 0)
            {
                invHalfPiLnScale -= ln2ByHalfPi;
                rescale = System.Math.Abs(System.Math.Log(invHalfPiLnScale + System.Math.Sqrt(invHalfPiLnScale * invHalfPiLnScale + 1)));
            }
            return halfPi * AdaptivePositiveHalfAxisTrapeziumRule(fExpSinh, rescale, relTol, AdaptiveQuadratureMaxNodes / 2 + 1);
        }

        /// <summary>
        /// Integrate the function f from 0 to +Infinity
        /// </summary>
        /// <param name="f">The function to integrate. Must have at most one extremum.</param>
        /// <param name="scale">A positive tuning parameter.
        /// <paramref name="f"/> is assumed to be continuous on (0, + Infinity) and non-negligible somewhere on (0, <paramref name="scale"/>]</param>
        /// <param name="relTol">A threshold to stop subdividing</param>
        /// <param name="maxNodes">Another threshold to stop subdividing</param>
        /// <returns></returns>
        public static double AdaptivePositiveHalfAxisTrapeziumRule(Converter<double, double> f, double scale, double relTol, int maxNodes)
        {
            if (double.IsNaN(scale)) throw new ArgumentException($"scale is NaN", nameof(scale));
            double intervalWidth = 1.0 / 8;
            double sumf1 = f(0);
            if (double.IsNaN(sumf1)) throw new ArgumentException($"f(0) is NaN", nameof(f));
            double x = 0;
            double oldSum;
            int usedNodes = 1;
            do
            {
                ++usedNodes;
                x += intervalWidth;
                oldSum = sumf1;
                sumf1 += f(x);
            }
            while (!MMath.AreEqual(oldSum, sumf1) || x < scale);

            if (x > scale)
                scale = x;

            while (usedNodes < maxNodes)
            {
                intervalWidth /= 2;
                double sumf2 = sumf1;
                x = intervalWidth;
                sumf2 += f(x);
                do
                {
                    ++usedNodes;
                    if (x < scale)
                        x += 2 * intervalWidth;
                    else
                        x += intervalWidth;
                    oldSum = sumf2;
                    sumf2 += f(x);
                }
                while (!MMath.AreEqual(oldSum, sumf2) || x < scale);
                if (x > scale)
                    scale = x;
                double i1 = sumf1 * intervalWidth * 2;
                double i2 = sumf2 * intervalWidth;
                double err_est = System.Math.Abs((i1 - i2) / i2);
                if (err_est < relTol)
                {
                    return i2;
                }
                sumf1 = sumf2;
            }

            return sumf1 * intervalWidth;
        }

        /// <summary>
        /// Integrate the function f from a to b
        /// </summary>
        /// <param name="f">The function to integrate.  Must have f(a)=f(b)=0.</param>
        /// <param name="nodeCount">The initial number of nodes.  Should be at least 2 and a power of 2.</param>
        /// <param name="a">The lower bound</param>
        /// <param name="b">The upper bound</param>
        /// <param name="relTol">A threshold to stop subdividing</param>
        /// <param name="maxNodes">Another threshold to stop subdividing</param>
        /// <returns></returns>
        internal static double AdaptiveTrapeziumRule(Converter<double, double> f, int nodeCount, double a, double b, double relTol, int maxNodes)
        {
            double intervalWidth = (b - a) / nodeCount;
            double sumf1 = 0, sumf2 = 0;
            for (double x = intervalWidth + a; x < b; x += intervalWidth)
            {
                sumf1 += f(x);
            }
            while (nodeCount < maxNodes)
            {
                nodeCount *= 2;
                intervalWidth /= 2;
                sumf2 = sumf1;
                for (double x = intervalWidth + a; x < b; x += intervalWidth * 2)
                {
                    // skip already included nodes
                    double fx = f(x);
                    if (Double.IsNaN(fx)) throw new Exception("f(x) is NaN, x = " + x);
                    if (Double.IsInfinity(fx)) throw new Exception("f(x) is infinite, x = " + x);
                    sumf2 += fx;
                }
                double i1 = sumf1 * intervalWidth * 2;
                double i2 = sumf2 * intervalWidth;
                double err_est = System.Math.Abs((i1 - i2) / i2);
                if (err_est < relTol)
                {
                    //Trace.WriteLine($"Reached tolerance ({err_est} < {relTol})");
                    return i2;
                }
                sumf1 = sumf2;
            }
            //Trace.WriteLine($"Reached maxNodes ({nodeCount} >= {maxNodes})");
            return sumf2 * intervalWidth;
        }

        /// <summary>
        /// Quadrature nodes for Gaussian expectations.
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        /// <param name="nodes">A list in which to store the nodes.</param>
        /// <param name="weights">A list in which to store the weights.</param>
        /// <remarks>
        /// The nodes and weights lists are modified to have the property that for any function f with a fast-converging Taylor series,
        /// <c>sum_i weights[i] f(nodes[i]) =approx int_{-inf..inf} f(x) N(x; m, v) dx</c>.
        /// If f is a polynomial of order 2*n-1, then the result is exact.
        /// For example, to compute E[x*x] where x ~ N(2,3):
        /// <code>
        /// Vector nodes = new Vector(2);
        /// Vector weights = new Vector(2);
        /// Quadrature.GaussianNodesAndWeights(2,3,nodes,weights);
        /// double result = (weights*nodes*nodes).Sum();
        /// </code>
        /// The result is mean^2 + variance = 7. 
        /// </remarks>
        public static void GaussianNodesAndWeights(double mean, double variance, IList<double> nodes, IList<double> weights)
        {
            int n = nodes.Count;
            if (n < 2 || n > HermiteNodesAndWeights.Length + 1)
                throw new Exception("The requested number of nodes is outside [2," + (HermiteNodesAndWeights.Length + 1) + "]");
            // these scale factors convert from the Hermite weight function to a Gaussian weight function.
            double scale = System.Math.Sqrt(2 * variance);
            double weightScale = 1.0 / System.Math.Sqrt(System.Math.PI);
            int firstHalf = n / 2;
            int secondHalf = firstHalf;
            if (n % 2 == 1) secondHalf++;
            // first half: apply symmetry transformation
            int index = 0;
            for (int i = 0; i < firstHalf; i++)
            {
                nodes[index] = -HermiteNodesAndWeights[n - 2][secondHalf - 1 - i, 0] * scale + mean;
                weights[index] = HermiteNodesAndWeights[n - 2][secondHalf - 1 - i, 1] * weightScale;
                index++;
            }
            // second half
            for (int i = 0; i < secondHalf; i++)
            {
                nodes[index] = HermiteNodesAndWeights[n - 2][i, 0] * scale + mean;
                weights[index] = HermiteNodesAndWeights[n - 2][i, 1] * weightScale;
                index++;
            }
        }

        /// <summary>
        /// Quadrature nodes for Gamma expectations.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="nodes">A list in which to store the nodes.</param>
        /// <param name="logWeights">A list in which to store the weights.</param>
        /// <remarks>
        /// The nodes and weights lists are modified to have the property that for any function f with a fast-converging Taylor series,
        /// <c>sum_i weights[i] f(nodes[i]) =approx int_{0..inf} f(x) Ga(x; a, b) dx</c> where
        /// <c>Ga(x; a, b) = x^a*exp(-x*b)*b^(a+1)/Gamma(a+1)</c>.
        /// For example, to approximate E[x*x] where x ~ Ga(2,3):
        /// <code>
        /// Vector nodes = new Vector(3);
        /// Vector logWeights = new Vector(3);
        /// Quadrature.GammaNodesAndWeights(2,3,nodes,logWeights);
        /// double result = (exp(logWeights)*nodes*nodes).Sum();
        /// </code>
        /// The result is mean^2 + variance = ((a+1)^2 + (a+1))/b^2 = 4/3.
        /// </remarks>
        public static void GammaNodesAndWeights(double a, double b, IList<double> nodes, IList<double> logWeights)
        {
            int n = nodes.Count;
            if (a + 1 < 0) throw new ArgumentOutOfRangeException("a is too small (" + a + ")");
            if (b <= 0) throw new ArgumentOutOfRangeException("b is too small (" + b + ")");
            bool useLaguerre = false;
            if (useLaguerre)
            {
                if (n < 2 || n > LaguerreNodesAndWeights.Length + 1)
                    throw new Exception("The requested number of nodes is outside [2," + (LaguerreNodesAndWeights.Length + 1) + "]");
                // these scale factors convert from the Laguerre weight function to a Gamma weight function.
                double scale = 1.0 / b;
                double logWeightScale = -MMath.GammaLn(a + 1);
                for (int i = 0; i < n; i++)
                {
                    nodes[i] = LaguerreNodesAndWeights[n - 2][i, 0];
                    logWeights[i] = System.Math.Log(LaguerreNodesAndWeights[n - 2][i, 1]) + a * System.Math.Log(nodes[i]) + logWeightScale;
                    nodes[i] *= scale;
                }
            }
            else
            {
                // get nodes from exp-Normal and adjust to mimic a Gamma.
                double v = MMath.Log1Plus(1 / (a + 1));
                if (v == 0) throw new Exception("v == 0");
                double m = System.Math.Log(a + 1) - System.Math.Log(b) + v * 0.5;
                // pass 0 instead of m here for better numerical accuracy below
                GaussianNodesAndWeights(0, v, nodes, logWeights);
                //double z = (a + 1)*Math.Log(b) - MMath.GammaLn(a + 1) + MMath.LnSqrt2PI + 0.5*Math.Log(v);
                // z2 = z + (a+1)*m - b*exp(m)
                // where b*Math.Exp(m) = (a+1)*Math.Exp(v*0.5)
                // could use MMath.GammaLnSeries here
                double z2 = ((a + 1) * System.Math.Log(a + 1) + MMath.LnSqrt2PI - (a + 1) - MMath.GammaLn(a + 1)) + 0.5 * System.Math.Log(v)
                    + (a + 1) * (v * 0.5 - MMath.ExpMinus1(v * 0.5));
                double s = 0.5 / v;
                for (int i = 0; i < n; i++)
                {
                    double diff = nodes[i];
                    double expnode = System.Math.Exp(nodes[i] + m);
                    // diffexpnode = b/(a+1)*(exp(m+diff)-exp(m))
                    double diffexpnode = System.Math.Exp(v * 0.5) * MMath.ExpMinus1(diff);
                    // original version:
                    //double newLogWeight2 = Math.Log(logWeights[i]) + s*diff*diff + ((a + 1)*nodes[i] - b*expnode + z);
                    double newLogWeight = System.Math.Log(logWeights[i]) + s * diff * diff + (a + 1) * (diff - diffexpnode) + z2;
                    //Console.WriteLine("{0} {1} {2}", nodes[i], newLogWeight, newLogWeight2);
                    logWeights[i] = newLogWeight;
                    nodes[i] = expnode;
                }
            }
        }

        public static void LaguerreGammaNodesAndWeights(double a, double b, IList<double> nodes, IList<double> weights)
        {
            int n = nodes.Count;
            if (a < 0) throw new ArgumentOutOfRangeException("a is too small (" + a + ")");
            if (b <= 0) throw new ArgumentOutOfRangeException("b is too small (" + b + ")");
            if (n < 2 || n > LaguerreNodesAndWeights.Length + 1)
                throw new Exception("The requested number of nodes is outside [2," + (LaguerreNodesAndWeights.Length + 1) + "]");
            // these scale factors convert from the Laguerre weight function to a Gamma weight function.
            double scale = 1.0 / b;
            double logweightScale = -MMath.GammaLn(a);
            for (int i = 0; i < n; i++)
            {
                nodes[i] = LaguerreNodesAndWeights[n - 2][i, 0];
                weights[i] = System.Math.Exp(System.Math.Log(LaguerreNodesAndWeights[n - 2][i, 1]) + (a - 1.0) * System.Math.Log(nodes[i]) + logweightScale);
                nodes[i] *= scale;
            }
        }

        /// <summary>
        /// Quadrature nodes for integrals on [low,high].
        /// </summary>
        /// <param name="low">Lower limit of integration.  Must be finite.</param>
        /// <param name="high">Upper limit of integration.  Must be finite.</param>
        /// <param name="nodes">A list in which to store the nodes.</param>
        /// <param name="weights">A list in which to store the weights.</param>
        /// <remarks>
        /// The nodes and weights lists are modified to have the property that for any function f with a fast-converging Taylor series,
        /// <c>sum_i weights[i] f(nodes[i]) =approx int_{low..high} f(x) dx</c>.
        /// If f is a polynomial of order 2*n-1, then the result is exact.
        /// For example, to compute <c>int_{0..1} x^3 dx</c>:
        /// <code>
        /// Vector nodes = new Vector(2);
        /// Vector weights = new Vector(2);
        /// Quadrature.UniformNodesAndWeights(0,1,nodes,weights);
        /// double result = (weights*nodes*nodes*nodes).Sum();
        /// </code>
        /// The result is 1/4.
        /// </remarks>
        public static void UniformNodesAndWeights(double low, double high, IList<double> nodes, IList<double> weights)
        {
            int n = nodes.Count;
            if (n < 2 || n > LegendreNodesAndWeights.Length + 1)
                throw new Exception("The requested number of nodes is outside [2," + (LegendreNodesAndWeights.Length + 1) + "]");
            double scale = 0.5 * (high - low), offset = low + scale;
            double weightScale = scale;
            int firstHalf = n / 2;
            int secondHalf = firstHalf;
            if (n % 2 == 1) secondHalf++;
            // first half: apply symmetry transformation
            int index = 0;
            for (int i = 0; i < firstHalf; i++)
            {
                nodes[index] = -LegendreNodesAndWeights[n - 2][i, 0] * scale + offset;
                weights[index] = LegendreNodesAndWeights[n - 2][i, 1] * weightScale;
                index++;
            }
            // second half
            for (int i = 0; i < secondHalf; i++)
            {
                nodes[index] = LegendreNodesAndWeights[n - 2][secondHalf - 1 - i, 0] * scale + offset;
                weights[index] = LegendreNodesAndWeights[n - 2][secondHalf - 1 - i, 1] * weightScale;
                index++;
            }
        }

        /// <summary>
        /// Legendre nodes and weights
        /// </summary>
        public static double[][,] LegendreNodesAndWeights =
            {
                // Origin: http://www.math.ntnu.no/num/nnm/Program/Numlibc/gauss_co.c
                // For each n, we only store floor((n+1)/2) nodes and weights since the others come from symmetry.
                new double[,]
                    {
                        // n=2
                        {0.577350269189626, 1.000000000000000},
                    },
                new double[,]
                    {
                        // n=3
                        {0.774596669241483, 0.555555555555556},
                        {0.000000000000000, 0.888888888888889},
                    },
                new double[,]
                    {
                        // n=4
                        {0.861136311594053, 0.347854845137454},
                        {0.339981043584856, 0.652145154862546},
                    },
                new double[,]
                    {
                        // n=5
                        {0.906179845938664, 0.236926885056189},
                        {0.538469310105683, 0.478628670499366},
                        {0.000000000000000, 0.568888888888889},
                    },
                new double[,]
                    {
                        // n=6
                        {0.932469514203152, 0.171324492379170},
                        {0.661209386466265, 0.360761573048139},
                        {0.238619186083197, 0.467913934572691},
                    },
                new double[,]
                    {
                        // n=7
                        {0.949107912342759, 0.129484966168870},
                        {0.741531185599394, 0.279705391489277},
                        {0.405845151377397, 0.381830050505119},
                        {0.000000000000000, 0.417959183673469},
                    },
                new double[,]
                    {
                        // n=8
                        {0.960289856497536, 0.101228536290376},
                        {0.796666477413627, 0.222381034453374},
                        {0.525532409916329, 0.313706645877887},
                        {0.183434642495650, 0.362683783378362},
                    },
                new double[,]
                    {
                        // n=9
                        {0.968160239507626, 0.081274388361574},
                        {0.836031107326636, 0.180648160694857},
                        {0.613371432700590, 0.260610696402935},
                        {0.324253423403809, 0.312347077040003},
                        {0.000000000000000, 0.330239355001260},
                    },
                new double[,]
                    {
                        // n=10
                        {0.973906528517172, 0.066671344308688},
                        {0.865063366688985, 0.149451349150581},
                        {0.679409568299024, 0.219086362515982},
                        {0.433395394129247, 0.269266719309996},
                        {0.148874338981631, 0.295524224714753},
                    },
                new double[,]
                    {
                        // n=11
                        {0.978228658146057, 0.055668567116174},
                        {0.887062599768095, 0.125580369464905},
                        {0.730152005574049, 0.186290210927734},
                        {0.519096129206812, 0.233193764591990},
                        {0.269543155952345, 0.262804544510247},
                        {0.000000000000000, 0.272925086777901},
                    },
                new double[,]
                    {
                        // n=12
                        {0.981560634246719, 0.047175336386512},
                        {0.904117256370475, 0.106939325995318},
                        {0.769902674194305, 0.160078328543346},
                        {0.587317954286617, 0.203167426723066},
                        {0.367831498998180, 0.233492536538355},
                        {0.125233408511469, 0.249147045813403},
                    },
                new double[,]
                    {
                        // n=13
                        {0.984183054718588, 0.040484004765316},
                        {0.917598399222978, 0.092121499837728},
                        {0.801578090733310, 0.138873510219787},
                        {0.642349339440340, 0.178145980761946},
                        {0.448492751036447, 0.207816047536889},
                        {0.230458315955135, 0.226283180262897},
                        {0.000000000000000, 0.232551553230874},
                    },
                new double[,]
                    {
                        // n=14
                        {0.986283808696812, 0.035119460331752},
                        {0.928434883663574, 0.080158087159760},
                        {0.827201315069765, 0.121518570687903},
                        {0.687292904811685, 0.157203167158194},
                        {0.515248636358154, 0.185538397477938},
                        {0.319112368927890, 0.205198463721296},
                        {0.108054948707344, 0.215263853463158},
                    },
                new double[,]
                    {
                        // n=15
                        {0.987992518020485, 0.030753241996117},
                        {0.937273392400706, 0.070366047488108},
                        {0.848206583410427, 0.107159220467172},
                        {0.724417731360170, 0.139570677926154},
                        {0.570972172608539, 0.166269205816994},
                        {0.394151347077563, 0.186161000015562},
                        {0.201194093997435, 0.198431485327111},
                        {0.000000000000000, 0.202578241925561},
                    },
                new double[,]
                    {
                        // n=16
                        {0.989400934991650, 0.027152459411754},
                        {0.944575023073233, 0.062253523938648},
                        {0.865631202387832, 0.095158511682493},
                        {0.755404408355003, 0.124628971255534},
                        {0.617876244402644, 0.149595988816577},
                        {0.458016777657227, 0.169156519395003},
                        {0.281603550779259, 0.182603415044924},
                        {0.095012509837637, 0.189450610455069},
                    },
                new double[,]
                    {
                        // n=17
                        {0.990575475314417, 0.024148302868548},
                        {0.950675521768768, 0.055459529373987},
                        {0.880239153726986, 0.085036148317179},
                        {0.781514003896801, 0.111883847193404},
                        {0.657671159216691, 0.135136368468526},
                        {0.512690537086477, 0.154045761076810},
                        {0.351231763453876, 0.168004102156450},
                        {0.178484181495848, 0.176562705366993},
                        {0.000000000000000, 0.179446470356207},
                    },
                new double[,]
                    {
                        // n=18
                        {0.991565168420931, 0.021616013526483},
                        {0.955823949571398, 0.049714548894970},
                        {0.892602466497556, 0.076425730254889},
                        {0.803704958972523, 0.100942044106287},
                        {0.691687043060353, 0.122555206711478},
                        {0.559770831073948, 0.140642914670651},
                        {0.411751161462843, 0.154684675126265},
                        {0.251886225691506, 0.164276483745833},
                        {0.084775013041735, 0.169142382963144},
                    },
                new double[,]
                    {
                        // n=19
                        {0.992406843843584, 0.019461788229726},
                        {0.960208152134830, 0.044814226765700},
                        {0.903155903614818, 0.069044542737641},
                        {0.822714656537143, 0.091490021622450},
                        {0.720966177335229, 0.111566645547334},
                        {0.600545304661681, 0.128753962539336},
                        {0.464570741375961, 0.142606702173607},
                        {0.316564099963630, 0.152766042065860},
                        {0.160358645640225, 0.158968843393954},
                        {0.000000000000000, 0.161054449848784},
                    },
                new double[,]
                    {
                        // n=20
                        {0.993128599185094, 0.017614007139152},
                        {0.963971927277913, 0.040601429800386},
                        {0.912234428251325, 0.062672048334109},
                        {0.839116971822218, 0.083276741576704},
                        {0.746331906460150, 0.101930119817240},
                        {0.636053680726515, 0.118194531961518},
                        {0.510867001950827, 0.131688638449176},
                        {0.373706088715419, 0.142096109318382},
                        {0.227785851141645, 0.149172986472603},
                        {0.076526521133497, 0.152753387130725},
                    },
            };

        /// <summary>
        /// Laguerre nodes and weights
        /// </summary>
        public static double[][,] LaguerreNodesAndWeights =
            {
                // Origin: http://www.math.ntnu.no/num/nnm/Program/Numlibc/laguer_c.c
                new double[,]
                    {
                        // n=2
                        {0.585786437627, 0.853553390593},
                        {3.414213562373, 0.146446609407},
                    },
                new double[,]
                    {
                        // n=3
                        {0.415774556783, 0.711093009929},
                        {2.294280360279, 0.278517733569},
                        {6.289945082937, 0.0103892565016},
                    },
                new double[,]
                    {
                        // n=4
                        {0.322547689619, 0.603154104342},
                        {1.745761101158, 0.357418692438},
                        {4.536620296921, 0.0388879085150},
                        {9.395070912301, 0.000539294705561},
                    },
                new double[,]
                    {
                        // n=5
                        {0.263560319718, 0.521755610583},
                        {1.413403059107, 0.398666811083},
                        {3.596425771041, 0.0759424496817},
                        {7.085810005859, 0.00361175867992},
                        {12.640800844276, 0.0000233699723858},
                    },
                new double[,]
                    {
                        // n=6
                        {0.222846604179, 0.458964673950},
                        {1.188932101673, 0.417000830772},
                        {2.992736326059, 0.113373382074},
                        {5.775143569105, 0.0103991974531},
                        {9.837467418383, 0.000261017202815},
                        {15.982873980602, 0.000000898547906430},
                    },
                new double[,]
                    {
                        // n=7
                        {0.193043676560, 0.409318951701},
                        {1.026664895339, 0.421831277862},
                        {2.567876744951, 0.147126348658},
                        {4.900353084526, 0.0206335144687},
                        {8.182153444563, 0.00107401014328},
                        {12.734180291798, 0.0000158654643486},
                        {19.395727862263, 0.0000000317031547900},
                    },
                new double[,]
                    {
                        // n=8
                        {0.170279632305, 0.369188589342},
                        {0.903701776799, 0.418786780814},
                        {2.251086629866, 0.175794986637},
                        {4.266700170288, 0.0333434922612},
                        {7.045905402393, 0.00279453623523},
                        {10.758516010181, 0.0000907650877336},
                        {15.740678641278, 0.000000848574671627},
                        {22.863131736889, 0.00000000104800117487},
                    },
                new double[,]
                    {
                        // n=9
                        {0.152322227732, 0.336126421798},
                        {0.807220022742, 0.411213980424},
                        {2.005135155619, 0.199287525371},
                        {3.783473973331, 0.0474605627657},
                        {6.204956777877, 0.00559962661079},
                        {9.372985251688, 0.000305249767093},
                        {13.466236911092, 0.00000659212302608},
                        {18.833597788992, 0.0000000411076933035},
                        {26.374071890927, 0.0000000000329087403035},
                    },
                new double[,]
                    {
                        // n=10
                        {0.137793470540, 0.308441115765},
                        {0.729454549503, 0.401119929155},
                        {1.808342902740, 0.218068287612},
                        {3.401433697855, 0.0620874560987},
                        {5.552496140064, 0.00950151697518},
                        {8.330152746764, 0.000753008388588},
                        {11.843785837900, 0.0000282592334960},
                        {16.279257831378, 0.000000424931398496},
                        {21.996585811981, 0.00000000183956482398},
                        {29.920697012274, 0.000000000000991182721961},
                    },
                new double[,]
                    {
                        // n=11
                        {0.125796442188, 0.284933212894},
                        {0.665418255839, 0.389720889528},
                        {1.647150545872, 0.232781831849},
                        {3.091138143035, 0.0765644535462},
                        {5.029284401580, 0.0143932827674},
                        {7.509887863807, 0.00151888084648},
                        {10.605950999547, 0.0000851312243547},
                        {14.431613758064, 0.00000229240387957},
                        {19.178857403215, 0.0000000248635370277},
                        {25.217709339678, 0.0000000000771262693369},
                        {33.497192847176, 0.0000000000000288377586832},
                    },
                new double[,]
                    {
                        // n=12
                        {0.115722117358, 0.264731371055},
                        {0.611757484515, 0.377759275873},
                        {1.512610269776, 0.244082011320},
                        {2.833751337744, 0.0904492222117},
                        {4.599227639418, 0.0201023811546},
                        {6.844525453115, 0.00266397354187},
                        {9.621316842457, 0.000203231592663},
                        {13.006054993306, 0.00000836505585682},
                        {17.116855187462, 0.000000166849387654},
                        {22.151090379397, 0.00000000134239103052},
                        {28.487967250984, 0.00000000000306160163504},
                        {37.099121044467, 0.000000000000000814807746743},
                    },
                new double[,]
                    {
                        // n=13
                        {0.107142388472, 0.247188708430},
                        {0.566131899040, 0.365688822901},
                        {1.398564336451, 0.252562420058},
                        {2.616597108406, 0.103470758024},
                        {4.238845929017, 0.0264327544156},
                        {6.292256271140, 0.00422039604027},
                        {8.815001941187, 0.000411881770473},
                        {11.861403588811, 0.0000235154739815},
                        {15.510762037704, 0.000000731731162025},
                        {19.884635663880, 0.0000000110884162570},
                        {25.185263864678, 0.0000000000677082669221},
                        {31.800386301947, 0.000000000000115997995991},
                        {40.723008669266, 0.0000000000000000224509320389},
                    },
                new double[,]
                    {
                        // n=14
                        {0.099747507033, 0.231815577145},
                        {0.526857648852, 0.353784691598},
                        {1.300629121251, 0.258734610245},
                        {2.430801078731, 0.115482893557},
                        {3.932102822293, 0.0331920921593},
                        {5.825536218302, 0.00619286943701},
                        {8.140240141565, 0.000739890377867},
                        {10.916499507366, 0.0000549071946684},
                        {14.210805011161, 0.00000240958576409},
                        {18.104892220218, 0.0000000580154398168},
                        {22.723381628269, 0.000000000681931469249},
                        {28.272981723248, 0.00000000000322120775189},
                        {35.149443660592, 0.00000000000000422135244052},
                        {44.366081711117, 0.000000000000000000605237502229},
                    },
                new double[,]
                    {
                        // n=15
                        {0.093307812017, 0.218234885940},
                        {0.492691740302, 0.342210177923},
                        {1.215595412071, 0.263027577942},
                        {2.269949526204, 0.126425818106},
                        {3.667622721751, 0.0402068649210},
                        {5.425336627414, 0.00856387780361},
                        {7.565916226613, 0.00121243614721},
                        {10.120228568019, 0.000111674392344},
                        {13.130282482176, 0.00000645992676202},
                        {16.654407708330, 0.000000222631690710},
                        {20.776478899449, 0.00000000422743038498},
                        {25.623894226729, 0.0000000000392189726704},
                        {31.407519169754, 0.000000000000145651526407},
                        {38.530683306486, 0.000000000000000148302705111},
                        {48.026085572686, 0.0000000000000000000160059490621}
                    }
            };

        /// <summary>
        /// Hermite nodes and weights
        /// </summary>
        public static double[][,] HermiteNodesAndWeights =
            {
                // This table comes from MNT/tests/test_gauss_hermite.m
                // A smaller table can be found at http://www.math.ntnu.no/num/nnm/Program/Numlibc/herm_cof.c
                // For each n, we only store floor((n+1)/2) nodes and weights since the others come from symmetry.
                new double[,]
                    {
                        // n=2
                        {0.7071067811865475, 0.8862269254527497},
                    },
                new double[,]
                    {
                        // n=3
                        {0, 1.18163590060368},
                        {1.224744871391589, 0.2954089751509169},
                    },
                new double[,]
                    {
                        // n=4
                        {0.5246476232752904, 0.8049140900055056},
                        {1.650680123885784, 0.08131283544724451},
                    },
                new double[,]
                    {
                        // n=5
                        {0, 0.945308720482944},
                        {0.9585724646138185, 0.3936193231522377},
                        {2.020182870456086, 0.01995324205904574},
                    },
                new double[,]
                    {
                        // n=6
                        {0.4360774119276165, 0.724629595224386},
                        {1.335849074013697, 0.1570673203228551},
                        {2.350604973674492, 0.0045300099055088},
                    },
                new double[,]
                    {
                        // n=7
                        {0, 0.8102646175568091},
                        {0.8162878828589647, 0.4256072526101238},
                        {1.673551628767471, 0.05451558281912654},
                        {2.651961356835233, 0.0009717812450995022},
                    },
                new double[,]
                    {
                        // n=8
                        {0.3811869902073222, 0.6611470125582353},
                        {1.15719371244678, 0.20780232581489},
                        {1.981656756695843, 0.01707798300741328},
                        {2.930637420257244, 0.0001996040722113658},
                    },
                new double[,]
                    {
                        // n=9
                        {0, 0.7202352156060525},
                        {0.7235510187528376, 0.4326515590025515},
                        {1.468553289216668, 0.08847452739437579},
                        {2.266580584531843, 0.004943624275536841},
                        {3.190993201781527, 3.960697726326392e-005},
                    },
                new double[,]
                    {
                        // n=10
                        {0.3429013272237046, 0.6108626337353201},
                        {1.036610829789514, 0.2401386110823126},
                        {1.756683649299882, 0.03387439445548068},
                        {2.53273167423279, 0.001343645746781223},
                        {3.436159118837738, 7.640432855232573e-006},
                    },
                new double[,]
                    {
                        // n=11
                        {0, 0.6547592869145933},
                        {0.6568095668820998, 0.4293597523561212},
                        {1.326557084494933, 0.1172278751677075},
                        {2.025948015825756, 0.01191139544491136},
                        {2.783290099781652, 0.0003468194663233418},
                        {3.668470846559583, 1.439560393714247e-006},
                    },
                new double[,]
                    {
                        // n=12
                        {0.3142403762543591, 0.5701352362624745},
                        {0.9477883912401637, 0.2604923102641586},
                        {1.597682635152605, 0.05160798561588345},
                        {2.27950708050106, 0.003905390584628997},
                        {3.02063702512089, 8.573687043587757e-005},
                        {3.889724897869782, 2.658551684356266e-007},
                    },
                new double[,]
                    {
                        // n=13
                        {0, 0.604393187921163},
                        {0.6057638791710601, 0.4216162968985392},
                        {1.220055036590748, 0.1403233206870223},
                        {1.853107651601512, 0.02086277529616975},
                        {2.519735685678238, 0.001207459992719362},
                        {3.24660897837241, 2.043036040270691e-005},
                        {4.10133759617864, 4.825731850073071e-008},
                    },
                new double[,]
                    {
                        // n=14
                        {0.2917455106725621, 0.5364059097120856},
                        {0.8787137873293994, 0.2731056090642441},
                        {1.476682731141141, 0.06850553422346459},
                        {2.095183258507717, 0.007850054726457894},
                        {2.748470724985403, 0.0003550926135519168},
                        {3.462656933602271, 4.716484355018873e-006},
                        {4.304448570473632, 8.628591168125088e-009},
                    },
                new double[,]
                    {
                        // n=15
                        {0, 0.5641003087264188},
                        {0.5650695832555758, 0.4120286874988949},
                        {1.136115585210921, 0.1584889157959343},
                        {1.719992575186489, 0.03078003387254578},
                        {2.325732486173858, 0.002778068842912713},
                        {2.967166927905603, 0.0001000044412324979},
                        {3.669950373404453, 1.05911554771106e-006},
                        {4.499990707309392, 1.522475804253497e-009},
                    },
                new double[,]
                    {
                        // n=16
                        {0.2734810461381524, 0.507929479016609},
                        {0.8229514491446559, 0.2806474585285311},
                        {1.380258539198881, 0.08381004139898501},
                        {1.951787990916254, 0.01288031153550986},
                        {2.546202157847481, 0.000932284008624164},
                        {3.176999161979956, 2.711860092537841e-005},
                        {3.869447904860123, 2.320980844865183e-007},
                        {4.688738939305819, 2.65480747401114e-010},
                    },
                new double[,]
                    {
                        // n=17
                        {0, 0.5309179376248648},
                        {0.5316330013426547, 0.4018264694704083},
                        {1.067648725743451, 0.1726482976700954},
                        {1.612924314221231, 0.04092003414975592},
                        {2.173502826666621, 0.00506734995762748},
                        {2.757762915703889, 0.0002986432866977488},
                        {3.378932091141494, 7.112289140021187e-006},
                        {4.061946675875475, 4.977078981630734e-008},
                        {4.871345193674403, 4.580578930798559e-011},
                    },
                new double[,]
                    {
                        // n=18
                        {0.2582677505190967, 0.4834956947254513},
                        {0.7766829192674116, 0.2848072856699769},
                        {1.300920858389618, 0.09730174764131445},
                        {1.835531604261629, 0.01864004238754449},
                        {2.386299089166686, 0.001888522630268404},
                        {2.961377505531607, 9.181126867929312e-005},
                        {3.573769068486266, 1.810654481093406e-006},
                        {4.248117873568127, 1.046720579579191e-008},
                        {5.048364008874467, 7.828199772115814e-012},
                    },
                new double[,]
                    {
                        // n=19
                        {0, 0.5029748882761876},
                        {0.5035201634238882, 0.3916089886130269},
                        {1.010368387134311, 0.1836327013069954},
                        {1.524170619393533, 0.05081038690905166},
                        {2.049231709850619, 0.007988866777722915},
                        {2.591133789794543, 0.0006708775214071705},
                        {3.157848818347602, 2.720919776316125e-005},
                        {3.76218735196402, 4.488243147223087e-007},
                        {4.428532806603779, 2.163051009863515e-009},
                        {5.220271690537483, 1.326297094498518e-012},
                    },
                new double[,]
                    {
                        // n=20
                        {0.2453407083009013, 0.4622436696006055},
                        {0.7374737285453944, 0.2866755053628314},
                        {1.234076215395323, 0.1090172060200222},
                        {1.738537712116586, 0.0248105208874634},
                        {2.254974002089276, 0.003243773342237826},
                        {2.788806058428131, 0.000228338636016351},
                        {3.347854567383216, 7.80255647853199e-006},
                        {3.944764040115625, 1.086069370769273e-007},
                        {4.603682449550744, 4.399340992273143e-010},
                        {5.387480890011233, 2.229393645534139e-013},
                    },
                new double[,]
                    {
                        // n=21
                        {0, 0.4790237031201787},
                        {0.4794507070791076, 0.3816690736134984},
                        {0.9614996344183691, 0.192120324066996},
                        {1.448934250650732, 0.06017964665891174},
                        {1.944962949186254, 0.01141406583743428},
                        {2.453552124512838, 0.001254982041726399},
                        {2.979991207704598, 7.478398867309986e-005},
                        {3.531972877137678, 2.17188489805665e-006},
                        {4.12199554749184, 2.571230180059275e-008},
                        {4.773992343411219, 8.818611242049894e-011},
                        {5.550351873264678, 3.720365070136e-014},
                    },
                new double[,]
                    {
                        // n=22
                        {0.2341791399309907, 0.4435452264349553},
                        {0.703686097170007, 0.2869714332469045},
                        {1.176713958481245, 0.1191023609587813},
                        {1.655874373286423, 0.03114037088442356},
                        {2.144233592798535, 0.004978399335051601},
                        {2.645637441058172, 0.0004648850508842428},
                        {3.165265909202137, 2.365512855251028e-005},
                        {3.710701532877805, 5.88428756330096e-007},
                        {4.294312480593161, 5.966990986059605e-009},
                        {4.939834131060175, 1.744339007547958e-011},
                        {5.709201353205264, 6.167183424403655e-015},
                    },
                new double[,]
                    {
                        // n=23
                        {0, 0.4581965855932192},
                        {0.4585383500681048, 0.3721438248775655},
                        {0.9191514654425638, 0.1986448985780228},
                        {1.384039585682495, 0.06889028942908741},
                        {1.855677037671371, 0.01520708400448413},
                        {2.337016211474456, 0.002069567874960639},
                        {2.831803787126157, 0.0001655616991418744},
                        {3.345127159941224, 7.249295918002279e-006},
                        {3.884472708106102, 1.555339329145767e-007},
                        {4.462091173740006, 1.359629650402896e-009},
                        {5.101534610476677, 3.408314098030542e-012},
                        {5.864309498984572, 1.016038462063679e-015},
                    },
                new double[,]
                    {
                        // n=24
                        {0.2244145474725156, 0.4269311638686952},
                        {0.6741711070372123, 0.28617953534644},
                        {1.126760817611245, 0.1277396217845579},
                        {1.584250010961694, 0.03744547050323036},
                        {2.049003573661699, 0.007048355810072596},
                        {2.523881017011427, 0.0008236924826884114},
                        {3.012546137565565, 5.688691636404337e-005},
                        {3.520006813034525, 2.158245704902317e-006},
                        {4.05366440244815, 4.018971174941368e-008},
                        {4.625662756423788, 3.04625426998756e-010},
                        {5.259382927668044, 6.584620243078132e-013},
                        {6.01592556142574, 1.664368496489088e-016},
                    },
                new double[,]
                    {
                        // n=25
                        {0, 0.4398687221694849},
                        {0.4401472986453083, 0.3630889892758866},
                        {0.8819827562138214, 0.2036211366781217},
                        {1.327280702073084, 0.07688899517580794},
                        {1.778001124337147, 0.01924309896540868},
                        {2.236420130267281, 0.003115708720125599},
                        {2.705320237173026, 0.0003150836387454755},
                        {3.188294924425105, 1.891597295734027e-005},
                        {3.690282876998356, 6.257032499691042e-007},
                        {4.218609444386561, 1.017038250301837e-008},
                        {4.785320367352224, 6.719638417706176e-011},
                        {5.413636355280034, 1.258814987746545e-013},
                        {6.164272434052451, 2.711923514038347e-017},
                    },
                new double[,]
                    {
                        // n=26
                        {0.2157778562434635, 0.4120436505903654},
                        {0.6480952139934484, 0.284632241176782},
                        {1.082733011077883, 0.1351133279117866},
                        {1.521361516651921, 0.04359822721725036},
                        {1.965854785641137, 0.00939790129115948},
                        {2.418415764773779, 0.001319064722323846},
                        {2.881762219543087, 0.0001162297016031083},
                        {3.35942718235083, 6.103291717395944e-006},
                        {3.856288419909149, 1.770106337397344e-007},
                        {4.379602662983305, 2.52449403449051e-009},
                        {4.941324957241379, 1.460999933981593e-011},
                        {5.564524981950103, 2.383148659372136e-014},
                        {6.309550385625694, 4.396916094753826e-018},
                    },
                new double[,]
                    {
                        // n=27
                        {0, 0.4235772880150572},
                        {0.423807900543853, 0.354517304099748},
                        {0.8490113420601031, 0.2073704807510058},
                        {1.277066817339858, 0.08417308108405049},
                        {1.709560739260337, 0.02341593362534149},
                        {2.148296645361628, 0.004381279835792464},
                        {2.595416338910818, 0.0005367696156881029},
                        {3.053582419822255, 4.146758004384075e-005},
                        {3.526275340134353, 1.915280900595265e-006},
                        {4.018318670408739, 4.89540040969946e-008},
                        {4.536906663372442, 6.155031578231709e-010},
                        {5.093910003113184, 3.134117613622995e-012},
                        {5.712255552816536, 4.470772457393076e-015},
                        {6.451940140753472, 7.09577929705112e-019},
                    },
                new double[,]
                    {
                        // n=28
                        {0.2080673826907369, 0.3986047178264497},
                        {0.6248367195052093, 0.2825613912593877},
                        {1.043535273754208, 0.1413946097869542},
                        {1.465537263457409, 0.04951488928989797},
                        {1.892360496837685, 0.01196842321435478},
                        {2.325749842656441, 0.001957331294408981},
                        {2.767795352913594, 0.0002106181000240288},
                        {3.221112076561456, 1.434550422971432e-005},
                        {3.689134238461679, 5.857719720992951e-007},
                        {4.176636742129269, 1.3256825015417e-008},
                        {4.690756523943118, 1.475853168277689e-010},
                        {5.243285373202936, 6.639436714909604e-013},
                        {5.85701464138285, 8.315937951206847e-016},
                        {6.591605442367743, 1.14013934790366e-019},
                    },
                new double[,]
                    {
                        // n=29
                        {0, 0.4089711746352331},
                        {0.4091646363949288, 0.346418939071669},
                        {0.8194986812709116, 0.2101426944492099},
                        {1.232215755084753, 0.09076884221557775},
                        {1.648622913892317, 0.02763965559202363},
                        {2.070181076053428, 0.00584550354527149},
                        {2.498585691019404, 0.0008407925061402603},
                        {2.935882504290126, 7.990920354521791e-005},
                        {3.384645141092214, 4.823073497647741e-006},
                        {3.84826679221362, 1.749229129949939e-007},
                        {4.33147829381915, 3.520312327600651e-009},
                        {4.841363651059164, 3.484130161308409e-011},
                        {5.389640521966752, 1.390107271449596e-013},
                        {5.99897128946382, 1.534500444605328e-016},
                        {6.72869519860885, 1.824460852767223e-020},
                    },
                new double[,]
                    {
                        // n=30
                        {0.2011285765488715, 0.3863948895418133},
                        {0.6039210586255523, 0.2801309308392124},
                        {1.008338271046724, 0.14673584754089},
                        {1.415527800198189, 0.05514417687023416},
                        {1.826741143603688, 0.01470382970482669},
                        {2.243391467761504, 0.002737922473067656},
                        {2.667132124535617, 0.0003483101243186857},
                        {3.099970529586442, 2.938725228922981e-005},
                        {3.54444387315535, 1.579094887324706e-006},
                        {4.003908603861229, 5.108522450775957e-008},
                        {4.483055357092519, 9.178580424378511e-010},
                        {4.988918968589944, 8.106186297462978e-012},
                        {5.533147151567496, 2.87860708054869e-014},
                        {6.138279220123935, 2.810333602750941e-017},
                        {6.863345293529892, 2.908254700131222e-021},
                    },
                new double[,]
                    {
                        // n=31
                        {0, 0.395778556098609},
                        {0.3959427364714231, 0.3387726578941033},
                        {0.7928769769153089, 0.2121327886687622},
                        {1.191826998350047, 0.09671794816086889},
                        {1.59388586047214, 0.03184723073129998},
                        {2.000258548935639, 0.007482799914035114},
                        {2.41231770548042, 0.001233683307306873},
                        {2.831680453390205, 0.0001395209039504671},
                        {3.260320732313541, 1.049860275767544e-005},
                        {3.700743403231469, 5.043712558939715e-007},
                        {4.156271755818145, 1.461198834491036e-008},
                        {4.63155950631286, 2.352492003208596e-010},
                        {5.133595577112381, 1.860373521452123e-012},
                        {5.673961444618588, 5.899556498753787e-015},
                        {6.27507870494286, 5.110609007927079e-018},
                        {6.99568012371854, 4.618968394464114e-022},
                    },
                new double[,]
                    {
                        // n=32
                        {0.1948407415693993, 0.375238352592801},
                        {0.5849787654359324, 0.2774581423025291},
                        {0.9765004635896829, 0.1512697340766419},
                        {1.370376410952872, 0.06045813095591234},
                        {1.767654109463202, 0.01755342883157336},
                        {2.169499183606112, 0.003654890326654416},
                        {2.577249537732318, 0.00053626836552797},
                        {2.992490825002374, 5.416584061819945e-005},
                        {3.417167492818571, 3.650585129562358e-006},
                        {3.853755485471444, 1.574167792545593e-007},
                        {4.305547953351199, 4.09883216477086e-009},
                        {4.777164503502597, 5.933291463396669e-011},
                        {5.275550986515881, 4.215010211326394e-013},
                        {5.812225949515914, 1.197344017092855e-015},
                        {6.40949814926966, 9.231736536518371e-019},
                        {7.125813909830727, 7.310676427384045e-023},
                    },
                new double[,]
                    {
                        // n=33
                        {0, 0.3837852665198695},
                        {0.3839260145084091, 0.3315520007507425},
                        {0.7687013797588687, 0.2134939311502926},
                        {1.15520020412679, 0.1020690799955418},
                        {1.544348261243122, 0.03598798231857711},
                        {1.937154581822207, 0.009265689970685262},
                        {2.334751151529515, 0.001718454637760934},
                        {2.738445824351355, 0.0002254427705963286},
                        {3.149796681703825, 2.04236840514238e-005},
                        {3.570721980232718, 1.237693367201214e-006},
                        {4.003671609956932, 4.807745676323175e-008},
                        {4.451911148832827, 1.128922247108341e-009},
                        {4.920028520595008, 1.473980937092492e-011},
                        {5.414929002614192, 9.434814159015075e-014},
                        {5.948071182087144, 2.407785679558019e-016},
                        {6.541655445738077, 1.6570947415337e-019},
                        {7.253851822015201, 1.153316218545882e-023},
                    },
                new double[,]
                    {
                        // n=34
                        {0.1891080605271425, 0.3649924469966374},
                        {0.5677172685548746, 0.2746277156351316},
                        {0.9475164580334473, 0.1551104166233062},
                        {1.329335551884786, 0.06544513410875125},
                        {1.714062553387338, 0.02047315172701905},
                        {2.102673690467332, 0.004698463629266493},
                        {2.496271940816548, 0.0007798175996231825},
                        {2.896138943174432, 9.186118982872119e-005},
                        {3.303808431564416, 7.493448783302125e-006},
                        {3.721175232476153, 4.097974035224509e-007},
                        {4.150665602970781, 1.43877329125397e-008},
                        {4.59551974810817, 3.056252041915439e-010},
                        {5.060296018605762, 3.60988174748582e-012},
                        {5.551861330988777, 2.087840373115996e-014},
                        {6.081616993936317, 4.799901997894774e-017},
                        {6.67165913607017, 2.95670892236047e-020},
                        {7.379890950481246, 1.81380011195974e-024},
                    },
                new double[,]
                    {
                        // n=35
                        {0, 0.3728199731907259},
                        {0.3729417170496169, 0.3247287215745667},
                        {0.7466176398798671, 0.2143471905960716},
                        {1.121780990720303, 0.1068729069554615},
                        {1.49922448861173, 0.04002477513309988},
                        {1.879803988730917, 0.01116680659026563},
                        {2.264467501042569, 0.002295028329584077},
                        {2.654292781197172, 0.0003423400931996045},
                        {3.050538420430446, 3.635276917415717e-005},
                        {3.454716495751991, 2.679815654613502e-006},
                        {3.868700730969155, 1.326940517306656e-007},
                        {4.294895814492763, 4.223976130009355e-009},
                        {4.736518477413211, 8.14069568667739e-011},
                        {5.198099346197752, 8.722526012536775e-013},
                        {5.686468948090441, 4.570444190417737e-015},
                        {6.212973747633717, 9.489884879472967e-018},
                        {6.79960941328413, 5.245652729174515e-021},
                        {7.504021146448936, 2.844113465725289e-025},
                    },
                new double[,]
                    {
                        // n=36
                        {0.1838533671058128, 0.3555400742737054},
                        {0.5519014332904229, 0.2717012470095249},
                        {0.9209818015707532, 0.1583554537511649},
                        {1.291810958820924, 0.07010475010517703},
                        {1.665150001843414, 0.0234257675097315},
                        {2.04182718355442, 0.005856425986559502},
                        {2.422766042053562, 0.001082534756686791},
                        {2.809022235131104, 0.0001456961596263549},
                        {3.201833945788159, 1.399702922032882e-005},
                        {3.602693857148476, 9.355570089773166e-007},
                        {4.01345656774947, 4.207491323109036e-008},
                        {4.436506970192857, 1.217867820955714e-009},
                        {4.875039972467084, 2.135354181784526e-011},
                        {5.333560107113064, 2.080912674286861e-013},
                        {5.818863279505577, 9.902901686905969e-016},
                        {6.342243330994412, 1.861597987807975e-018},
                        {6.925598990259942, 9.256403083558336e-022},
                        {7.626325754003895, 4.447153417575513e-026},
                    },
                new double[,]
                    {
                        // n=37
                        {0, 0.3627437576990796},
                        {0.362849905050658, 0.3182746797544245},
                        {0.7263396166051201, 0.2147888759647792},
                        {1.091123764975933, 0.1111791651071928},
                        {1.457887646874209, 0.04393136229953821},
                        {1.827365248763605, 0.01316017997721124},
                        {2.200360934009252, 0.002960800184602846},
                        {2.577776858113272, 0.0004941715012146832},
                        {2.960649181303289, 6.02269516526726e-005},
                        {3.350197894972536, 5.252143577119942e-006},
                        {3.74789820647548, 3.19288548100315e-007},
                        {4.155587281126479, 1.307969882180196e-008},
                        {4.575631748667359, 3.451906227211979e-010},
                        {5.011206138573073, 5.520397264637374e-012},
                        {5.46679033596856, 4.904739364185124e-014},
                        {5.949147217461971, 2.124885194131399e-016},
                        {6.469520036524031, 3.62472695124433e-019},
                        {7.049713855778229, 1.62498532719164e-022},
                        {7.746882249649456, 6.935083550583646e-027},
                    },
                new double[,]
                    {
                        // n=38
                        {0.1790137232958775, 0.3467841178687811},
                        {0.5373398108709835, 0.2687237600289282},
                        {0.896568346193136, 0.1610879714954814},
                        {1.257323131700713, 0.07444398336823892},
                        {1.620262755633014, 0.02638053352048689},
                        {1.986097778039066, 0.007115236826134025},
                        {2.355611733035508, 0.001446307121562705},
                        {2.729687962888326, 0.0002187575953393909},
                        {3.109345311717942, 2.42234151209353e-005},
                        {3.495787454835627, 1.923465686732391e-006},
                        {3.890473760963341, 1.066586570440924e-007},
                        {4.295225419749605, 3.990652885688934e-009},
                        {4.712392132084887, 9.627029292605497e-011},
                        {5.145129320740823, 1.407633343845247e-012},
                        {5.597893514184678, 1.142861675025671e-014},
                        {6.077416003537561, 4.517371223604762e-017},
                        {6.594891327265494, 7.007843330558469e-020},
                        {7.172033935320031, 2.838738753209435e-023},
                        {7.865762803380041, 1.078718882074102e-027},
                    },
                new double[,]
                    {
                        // n=39
                        {0, 0.353442635706801},
                        {0.3535358469963293, 0.3121628488671676},
                        {0.7076332733485723, 0.2148960767774629},
                        {1.062865567281179, 0.1150350457997687},
                        {1.419830157685736, 0.04769007133163493},
                        {1.779162582854313, 0.01522198083728929},
                        {2.14155301198688, 0.00371123023171413},
                        {2.507766693891319, 0.000684057595218552},
                        {2.878670311374955, 9.407727946766989e-005},
                        {3.25526723599223, 9.494299536141832e-006},
                        {3.638746424874536, 6.884420409340073e-007},
                        {4.030552814602468, 3.491508843910792e-008},
                        {4.432492882593038, 1.196166423766726e-009},
                        {4.846900568743526, 2.644004876992974e-011},
                        {5.276913315230426, 3.542650801979128e-013},
                        {5.726965451782105, 2.634096110722432e-015},
                        {6.203757997728109, 9.519320482116129e-018},
                        {6.718438506444092, 1.345726820134313e-020},
                        {7.292633670865722, 4.935907337590388e-024},
                        {7.983034772719782, 1.67378924574982e-028},
                    },
                new double[,]
                    {
                        // n=40
                        {0.1745372145975824, 0.3386432774255846},
                        {0.5238747138322772, 0.2657282518773736},
                        {0.8740066123570881, 0.1633787327132693},
                        {1.225480109046289, 0.07847460586540328},
                        {1.578869894931614, 0.02931256553617195},
                        {1.934791472282296, 0.008460888008258017},
                        {2.293917141875083, 0.00187149682959793},
                        {2.656995998442896, 0.0003138535945413275},
                        {3.024879883901285, 3.936933981092435e-005},
                        {3.398558265859629, 3.631576150692968e-006},
                        {3.779206753435223, 2.411144163670497e-007},
                        {4.1682570668325, 1.121236083227569e-008},
                        {4.567502072844395, 3.525620791365371e-010},
                        {4.979260978545256, 7.156528052690257e-012},
                        {5.406654247970128, 8.805707645216005e-014},
                        {5.8540950560304, 6.008358789490764e-016},
                        {6.328255351220082, 1.989181012116482e-018},
                        {6.840237305249356, 2.567593365411615e-021},
                        {7.411582531485469, 8.544056963775337e-025},
                        {8.098761139250851, 2.59104371384699e-029},
                    },
                new double[,]
                    {
                        // n=41
                        {0, 0.3448220836163917},
                        {0.3449044630154328, 0.3063678169378494},
                        {0.6903050523302081, 0.2147308644900834},
                        {1.036707252924206, 0.1184843912468433},
                        {1.384635789160033, 0.05128987239709441},
                        {1.734645608822029, 0.01733088136211674},
                        {2.087334681918724, 0.004540392987303245},
                        {2.443359553123411, 0.0009142342563287043},
                        {2.803454961484319, 0.0001399302565967409},
                        {3.168459453941986, 1.605596573667714e-005},
                        {3.539349937363712, 1.357378140487069e-006},
                        {3.917289854837782, 8.272658418749785e-008},
                        {4.303698767154651, 3.535681648380595e-009},
                        {4.700356896304117, 1.022679892778216e-010},
                        {5.109569626533134, 1.9103833646809e-012},
                        {5.534441340613446, 2.162987247175006e-014},
                        {5.979365004165134, 1.356987529586108e-016},
                        {6.450984597174753, 4.123408537531483e-019},
                        {6.96035840063675, 4.868737936012951e-022},
                        {7.528945464539621, 1.472653728652042e-025},
                        {8.213000895598281, 4.001959664666403e-030},
                    },
                new double[,]
                    {
                        // n=42
                        {0.170380585561817, 0.3310489138908529},
                        {0.5113749183154693, 0.2627389067822919},
                        {0.8530729091605537, 0.1652880012746656},
                        {1.19595779437781, 0.08221126930329287},
                        {1.540534800915546, 0.03220210128890721},
                        {1.887341620543485, 0.009879524053188407},
                        {2.236960787054318, 0.002357161394596291},
                        {2.590034870617127, 0.00043341227172125},
                        {2.947285782305479, 6.071962107788217e-005},
                        {3.309540096510922, 6.39024596773534e-006},
                        {3.677763316388557, 4.963659393579784e-007},
                        {4.053107744424767, 2.783471526549048e-008},
                        {4.436981705881031, 1.095805228807824e-009},
                        {4.831153629128276, 2.92172883723332e-011},
                        {5.23791588501765, 5.032705582184011e-013},
                        {5.660357581283058, 5.253337715568542e-015},
                        {6.102852334381526, 3.0358903478107e-017},
                        {6.572017171387476, 8.482152080086104e-020},
                        {7.078867873049109, 9.177890695692418e-023},
                        {7.644783295704742, 2.527869864053534e-026},
                        {8.325809389566931, 6.167858925810693e-031},
                    },
                new double[,]
                    {
                        // n=43
                        {0, 0.3368029653927552},
                        {0.3368761966255331, 0.3008659938676586},
                        {0.6741932767423139, 0.2143434970018828},
                        {1.01239968456332, 0.1215673798863875},
                        {1.35195936867087, 0.05472480908187454},
                        {1.69336053093991, 0.01946816015110158},
                        {2.037125688864194, 0.005441455293241784},
                        {2.38382370728436, 0.001186070092767416},
                        {2.734084694537905, 0.0001997243567132279},
                        {3.088619039600419, 2.568057307705646e-005},
                        {3.448242482220003, 2.485137742411515e-006},
                        {3.813910124065428, 1.777783044547971e-007},
                        {4.186764021366003, 9.193196067539776e-009},
                        {4.568202075544155, 3.34071791531755e-010},
                        {4.959981675194955, 8.227106120816511e-012},
                        {5.364382901151557, 1.309210416658464e-013},
                        {5.784480314077557, 1.262203220975816e-015},
                        {6.224628966894221, 6.730811703242703e-018},
                        {6.691419872712108, 1.732075379091327e-020},
                        {7.195827612346434, 1.720336826028185e-023},
                        {7.759153084732534, 4.322146059933881e-027},
                        {8.437238631083377, 9.486306341256315e-032},
                    },
                new double[,]
                    {
                        // n=44
                        {0.1665074670736324, 0.3239426288419897},
                        {0.4997302083006743, 0.2597733961158831},
                        {0.8335797610020734, 0.1668671658256427},
                        {1.168485463190206, 0.08567018730024452},
                        {1.504894434482872, 0.03503375710968631},
                        {1.84327992061761, 0.01135786693906971},
                        {2.184151019566526, 0.002901293775296569},
                        {2.528064427147665, 0.0005794204327156525},
                        {2.875639082170442, 8.959797364061479e-005},
                        {3.227574974277427, 1.060017177251744e-005},
                        {3.584677993105228, 9.45453854228538e-007},
                        {3.947893711524655, 6.242724506281358e-008},
                        {4.318354723442113, 2.983121019039561e-009},
                        {4.697449223001439, 1.002600077890905e-010},
                        {5.086924229758253, 2.284807897121994e-012},
                        {5.489048183880743, 3.365035612792188e-014},
                        {5.906881759027546, 3.001521402980774e-016},
                        {6.344762164896892, 1.479409975817153e-018},
                        {6.809255271535833, 3.512144955836433e-021},
                        {7.311295678916332, 3.207221691904894e-024},
                        {7.872108442774851, 7.362126104296998e-028},
                        {8.547337566735539, 1.456115308176205e-032},
                    },
                new double[,]
                    {
                        // n=45
                        {0, 0.3293184550506927},
                        {0.3293838996966692, 0.2956356519666262},
                        {0.6591616888741352, 0.2137748785020163},
                        {0.9897334248657113, 0.1243205019289351},
                        {1.321511785639231, 0.0579927466316283},
                        {1.654929119154792, 0.02161764443376538},
                        {1.990445458646639, 0.006407070656203967},
                        {2.328557971453189, 0.001500127882929748},
                        {2.669812465251346, 0.0002752424270995285},
                        {3.014817819749226, 3.918329544881909e-005},
                        {3.364264595856541, 4.275455886397217e-006},
                        {3.718949689341563, 3.522519241506496e-007},
                        {4.079809907925394, 2.151283907419426e-008},
                        {4.447969073601931, 9.518266941233628e-010},
                        {4.824806308703258, 2.964204989315723e-011},
                        {5.212058863162008, 6.262038315835287e-013},
                        {5.611984121839665, 8.55008746637021e-015},
                        {6.027629472253695, 7.067461915754193e-017},
                        {6.463314943664254, 3.224829870732622e-019},
                        {6.925582073277595, 7.073724650269579e-022},
                        {7.425326625856195, 5.948174715923177e-025},
                        {7.983699816222003, 1.249482912055957e-028},
                        {8.656152325990327, 2.230812662026306e-033},
                    },
                new double[,]
                    {
                        // n=46
                        {0.1628870279552622, 0.3172743849181584},
                        {0.4888472358511748, 0.2568445435065128},
                        {0.8153686793446816, 0.1681601273621182},
                        {1.142834903126431, 0.08886822756600293},
                        {1.471643971732072, 0.03779583099224597},
                        {1.802215195057495, 0.01288348755977684},
                        {2.134996872970726, 0.003501057850522687},
                        {2.470475513228143, 0.0007533967345431184},
                        {2.809187135477267, 0.0001273198375510654},
                        {3.151731519638471, 1.672593592065347e-005},
                        {3.498790641246365, 1.686946839735463e-006},
                        {3.851153150502552, 1.286532413056298e-007},
                        {4.209747761326976, 7.281634756563617e-009},
                        {4.575690134570433, 2.988545587992269e-010},
                        {4.950350886739886, 8.639140291448051e-012},
                        {5.33545803605723, 1.694707616774956e-013},
                        {5.733258441849944, 2.148647282367566e-015},
                        {6.146786754505402, 1.648444738950635e-017},
                        {6.580346434737409, 6.973799596694638e-020},
                        {7.040455442805218, 1.415504201675454e-022},
                        {7.537971787225827, 1.097655801996636e-025},
                        {8.093974741267118, 2.113201496269658e-029},
                        {8.763726442576393, 3.41137922990956e-034},
                    },
                new double[,]
                    {
                        // n=47
                        {0, 0.3223116794113205},
                        {0.3223704491561506, 0.2906568738512459},
                        {0.6450945230276884, 0.2130584602651455},
                        {0.9685309227378149, 0.126776700155003},
                        {1.293048717010953, 0.06109438622180215},
                        {1.619033081359051, 0.02376555576240628},
                        {1.946891757999186, 0.007429691537890198},
                        {2.2770625758595, 0.001856252778873262},
                        {2.610022474406152, 0.0003680616958748643},
                        {2.946298635192171, 5.742593823832077e-005},
                        {3.286482571324668, 6.977107961790638e-006},
                        {3.631248409292617, 6.518105523040183e-007},
                        {3.981377210706469, 4.610436485663544e-008},
                        {4.337790188492421, 2.422813493724372e-009},
                        {4.701595384967936, 9.240297888364304e-011},
                        {5.074155416985066, 2.483613543124752e-012},
                        {5.457189555681651, 4.531267646747932e-014},
                        {5.852934616125422, 5.342842396148613e-016},
                        {6.264413014970475, 3.810146347922942e-018},
                        {6.695912211063251, 1.496629916625583e-020},
                        {7.153927294775717, 2.814954742391994e-023},
                        {7.649279536628988, 2.015855816654601e-026},
                        {8.202978072797794, 3.561981780781785e-030},
                        {8.870101054023253, 5.207455090640105e-035},
                    },
                new double[,]
                    {
                        // n=48
                        {0.1594929358488625, 0.3110010303779577},
                        {0.4786463375944961, 0.2539615426647547},
                        {0.7983046277785623, 0.1692044719456381},
                        {1.118812152402157, 0.09182229707928347},
                        {1.440525220137565, 0.0404796769846031},
                        {1.7638175798953, 0.0144449615749808},
                        {2.089086660944276, 0.00415300491197747},
                        {2.416760904873216, 0.0009563923198193995},
                        {2.747308624822384, 0.0001751504318011702},
                        {3.081248988645106, 2.528599027748451e-005},
                        {3.419165969363885, 2.847258691734789e-006},
                        {3.761726490228358, 2.468658993669707e-007},
                        {4.109704603560591, 1.622514135895724e-008},
                        {4.464014546934459, 7.930467495165189e-010},
                        {4.82575722813321, 2.815296537838114e-011},
                        {5.196287718792364, 7.04693258154582e-013},
                        {5.577316981223729, 1.197589865479132e-014},
                        {5.971072225013545, 1.315159622658395e-016},
                        {6.380564096186411, 8.730159601186553e-019},
                        {6.810064578074141, 3.18838732350503e-021},
                        {7.26604655416435, 5.564577468902225e-024},
                        {7.759295519765774, 3.685036080150535e-027},
                        {8.310752190704784, 5.984612693313733e-031},
                        {8.975315081931687, 7.935551460774023e-036},
                    },
                new double[,]
                    {
                        // n=49
                        {0, 0.3157338900355686},
                        {0.3157869005237549, 0.2859114516946562},
                        {0.6318926995109617, 0.2122217192581611},
                        {0.9486405076102575, 0.1289655983084849},
                        {1.266362023687358, 0.0640324962122083},
                        {1.585402251455047, 0.02590030443834197},
                        {1.906124760988361, 0.008501807562954518},
                        {2.228917745385947, 0.002253673775338702},
                        {2.554201193951963, 0.0004795208860047624},
                        {2.8824356106456, 8.129104645315132e-005},
                        {3.214132868456856, 1.088180548567383e-005},
                        {3.549870037212095, 1.137841199677367e-006},
                        {3.890307405766162, 9.172961321170071e-008},
                        {4.236212530068787, 5.611842873387344e-009},
                        {4.588493140274482, 2.555470243998716e-010},
                        {4.948243443004891, 8.45752417143091e-012},
                        {5.316811374183517, 1.974484936063132e-013},
                        {5.695899985035209, 3.130139144124183e-015},
                        {6.087727281054756, 3.205958952400266e-017},
                        {6.495292565007659, 1.983641660279331e-019},
                        {6.922852834959762, 6.744674345658709e-022},
                        {7.376859390631928, 1.093680943612757e-024},
                        {7.868062864081665, 6.706356335345984e-028},
                        {8.417337186267979, 1.002361342743562e-031},
                        {9.079405395199434, 1.207287288128326e-036},
                    },
                new double[,]
                    {
                        // n=50
                        {0.1563025468894687, 0.3050851292043925},
                        {0.4690590566782361, 0.2511308563319972},
                        {0.7822717295546069, 0.1700324556771604},
                        {1.096251128957682, 0.09454893547708418},
                        {1.4113177548983, 0.04307915915676466},
                        {1.727806547515899, 0.01603194106841172},
                        {2.046071968686409, 0.004853263826171843},
                        {2.366493904298664, 0.001189011781749621},
                        {2.689484702267745, 0.000234269892109251},
                        {3.015497769574522, 3.684019053780618e-005},
                        {3.345038313937891, 4.581682707955437e-006},
                        {3.678677062515269, 4.457029966817737e-007},
                        {4.017068172858134, 3.346793404021373e-008},
                        {4.360973160454579, 1.909040543811849e-009},
                        {4.711293666169043, 8.111877364930015e-011},
                        {5.069117584917235, 2.506655523899624e-012},
                        {5.435786087224948, 5.465944031815467e-014},
                        {5.812994675420406, 8.094261893464909e-016},
                        {6.202952519274671, 7.742382957043171e-018},
                        {6.608647973855359, 4.470984365407736e-020},
                        {7.03432350977061, 1.417093599573388e-022},
                        {7.486409429864194, 2.13765830836005e-025},
                        {7.975622368205636, 1.215244123404456e-028},
                        {8.522771030917804, 1.673801667907778e-032},
                        {9.182406958129317, 1.833794048573436e-037},
                    },
            };
    }
}