// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Line search algorithm 
// Reference: Nocedal and Wright (Second edition, 2006)

using System;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Delegate type for function evaluation
    /// </summary>
    /// <param name="step">Step value</param>
    /// <param name="calcDeriv">Calculate the derivative</param>
    /// <param name="deriv">Where to put the derivative</param>
    /// <returns>Function evaluation</returns>
    public delegate double LineSearchEval(double step, bool calcDeriv, out double deriv);

    /// <summary>
    /// This line search algorithm is algorithm 3.5/3.6 from
    /// Nocedal and Wright (Second edition, 2006). It provides
    /// a step length that satisfies the strong Wolfe conditions.
    /// </summary>
    public class LineSearch
    {
#pragma warning disable 1591
        internal bool debug = false;
#pragma warning restore 1591
        private double c1 = 0.0001;
        private double c2 = 0.9;

        /// <summary>
        /// Set Wolfe conditions
        /// </summary>
        /// <param name="c1">Sufficient decrease condition</param>
        /// <param name="c2">Curvature condition</param>
        public void SetWolfeConstants(double c1, double c2)
        {
            if (c1 <= 0 || c1 >= 1.0 || c2 <= 0 || c2 >= 1.0 || c2 <= c1)
                throw new ArgumentOutOfRangeException("Wolfe condition constants must satisfy 0 < c1 < c2 < 1");

            this.c1 = c1;
            this.c2 = c2;
        }

        private LineSearchEval phi;

        /// <summary>
        /// Line search evaluation function
        /// </summary>
        public LineSearchEval Phi
        {
            get { return phi; }
            set { phi = value; }
        }

        private double phi0 = 0.0;
        private double c1_dPhi0;
        private double c2_dPhi0;
        private double fractInterpTol = 0.05;
        private int maxZoomIters = 100;
        private double extrapMult = 2.0;

        /// <summary>
        /// When performing interpolation this is the minimum
        /// fraction of the range by which the interpolant
        /// must be from one of the two end points
        /// </summary>
        public double InterpolationTolerance
        {
            get { return fractInterpTol; }
            set
            {
                if (value <= 0.0 || value >= 0.5)
                    throw new ArgumentOutOfRangeException();
                fractInterpTol = value;
            }
        }

        /// <summary>
        /// Maximum number of iterations for the zoom method
        /// </summary>
        public int MaximumZoomIterations
        {
            get { return maxZoomIters; }
            set
            {
                if (value < 1)
                    throw new ArgumentOutOfRangeException();
                maxZoomIters = value;
            }
        }

        /// <summary>
        /// Multiplier for the extrapolation
        /// </summary>
        public double ExtrapolationMultiplier
        {
            get { return extrapMult; }
            set
            {
                // Value must be somewhat greater than 1 to make
                // reasonable progress
                if (value < 1.1)
                    throw new ArgumentOutOfRangeException();
                extrapMult = value;
            }
        }

        /// <summary>
        /// Zoom in once we have bracketed the desired step length
        /// </summary>
        /// <param name="lo">Low point of step interval</param>
        /// <param name="phiLo">Evaluation at lo</param>
        /// <param name="dPhiLo">Derivative at lo</param>
        /// <param name="hi">High point of step interval</param>
        /// <param name="phiHi">Evaluation at lo</param>
        /// <param name="dPhiHi">Derivative at lo</param>
        /// <returns>The best step length</returns>
        private double Zoom(double lo, double phiLo, double dPhiLo, double hi, double phiHi, double dPhiHi)
        {
            double mid = lo;
            double phiMid, dPhiMid;
            double d1, d2, delta;
            double tol;

            for (int i = 0;; i++)
            {
                if (hi == lo || i == maxZoomIters)
                {
                    break;
                    //throw new Exception("cannot satisfy strong Wolfe conditions; check your derivative calculations");
                }
                // Cubic interpolation
                delta = hi - lo;
                d1 = dPhiLo + dPhiHi - 3.0*((phiHi - phiLo)/delta);
                d2 = d1*d1 - dPhiLo*dPhiHi;
                if (d2 <= 0.0)
                {
                    // Just use bisection
                    mid = 0.5*(lo + hi);
                }
                else
                {
                    d2 = System.Math.Sqrt(d1 * d1 - dPhiLo* dPhiHi);
                    if (delta < 0.0)
                        d2 = -d2;
                    mid = hi - delta*((dPhiHi + d2 - d1)/(dPhiHi - dPhiLo + 2*d2));

                    // If too close to the bracketing points, just use the bisection
                    tol = System.Math.Abs(fractInterpTol * delta);
                    if (System.Math.Abs(mid - lo) < tol || System.Math.Abs(mid - hi) < tol)
                        mid = 0.5*(lo + hi);
                }

                //------------------------------------------------------------------------
                // Make sure that the interpolant is not too close to the end points
                //------------------------------------------------------------------------

                phiMid = phi(mid, true, out dPhiMid);
                if (Double.IsNaN(phiMid))
                    throw new Exception("function returned NaN");

                if ((phiMid > phi0 + mid*c1_dPhi0) || phiMid >= phiLo)
                {
                    hi = mid;
                    phiHi = phiMid;
                    dPhiHi = dPhiMid;
                }
                else
                {
                    if (System.Math.Abs(dPhiMid) <= -c2_dPhi0)
                    {
                        // Strong Wolfe conditions satisfied - converged
                        break;
                    }
                    else
                    {
                        if (dPhiMid * delta >= 0)
                        {
                            hi = lo;
                            phiHi = phiLo;
                            dPhiHi = dPhiLo;
                        }
                        lo = mid;
                        phiLo = phiMid;
                        dPhiLo = dPhiMid;
                    }
                }
            }

            return mid;
        }

        /// <summary>
        /// Main execution method for line search
        /// </summary>
        /// <param name="initialStep"></param>
        /// <param name="maxStep"></param>
        /// <param name="phi0"></param>
        /// <param name="dPhi0"></param>
        /// <returns>Step satisfying Wolfe conditions</returns>
        public double Run(double initialStep, double maxStep, double phi0, double dPhi0)
        {
            if (phi == null)
                throw new InferRuntimeException("Evaluation function has not been set");

            // Following are used to test for Wolfe conditions
            //double dPhi0;
            //phi0 = phi(0.0, true, out dPhi0);
            //Console.WriteLine("phi0 {0}", phi0); 
            if (dPhi0 >= 0)
                return 0.0;
            //throw new InferRuntimeException("Derivative at current point must be negative");

            c1_dPhi0 = c1*dPhi0;
            c2_dPhi0 = c2*dPhi0;

            double result = 0.0;
            double prev, curr, phiPrev, dPhiPrev, phiCurr, dPhiCurr;

            prev = 0.0;
            phiPrev = phi0;
            dPhiPrev = dPhi0;
            curr = initialStep;
            if (curr > maxStep)
                curr = maxStep;
            for (int i = 0;; i++)
            {
                // no point NOT computing the gradient: we always need it in the next step anyway
                phiCurr = phi(curr, true, out dPhiCurr);
                //Console.WriteLine("phi {0}", phiCurr); 
                if ((phiCurr > phi0 + curr*c1_dPhi0) || (phiCurr >= phiPrev && i != 0))
                {
                    // Sufficient decrease condition is not met - so guaranteed to be met
                    // in the interval somewhere
                    result = Zoom(prev, phiPrev, dPhiPrev, curr, phiCurr, dPhiCurr);
                    if (debug)
                        Console.WriteLine("Sufficient decrease not met: zoom!");
                    break;
                }
                // We need to look at curvature
                if (System.Math.Abs(dPhiCurr) <= -c2_dPhi0)
                {
                    // Both strong conditions are met
                    result = curr;
                    if (debug)
                        Console.WriteLine("Strong conditions met after {0} line search its", i);
                    break;
                }
                if (dPhiCurr >= 0)
                {
                    // Curvature condition is not met, but positive
                    // gradient at current
                    result = Zoom(curr, phiCurr, dPhiCurr, prev, phiPrev, dPhiPrev);
                    if (debug)
                        Console.WriteLine("Curvature condition not met, zoom!");
                    break;
                }
                if (curr >= maxStep)
                {
                    if (debug)
                        Console.WriteLine("Max step reached.");
                    break; // We've run out of room
                }
                // Extrapolate
                prev = curr;
                phiPrev = phiCurr;
                dPhiPrev = dPhiCurr;
                curr *= extrapMult;
                if (curr > maxStep)
                    curr = maxStep;
            }
            return result;
        }
    }
}