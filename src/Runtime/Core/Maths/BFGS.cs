// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// BFGS unconstrained optimiser 
// Reference: Nocedal and Wright (1999)

using System;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Delegate type for function evaluation
    /// </summary>
    /// <param name="x">Independent value</param>
    /// <param name="dX">If reference is not null, calculate the deriv here</param>
    /// <returns>Function evaluation</returns>
    public delegate double FunctionEval(Vector x, ref Vector dX);

    /// <summary>
    /// Event delegate for handling iteration event
    /// </summary>
    public delegate void IterationEventHandler(object sender, OptimiserIterationEventArgs oie);

    /// <summary>
    /// Optimiser iteration event
    /// </summary>
    public class OptimiserIterationEventArgs : EventArgs
    {
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="iteration"></param>
        /// <param name="objVal"></param>
        /// <param name="rmsDeriv"></param>
        public OptimiserIterationEventArgs(int iteration, double objVal, double rmsDeriv)
        {
            this.iteration = iteration;
            this.objVal = objVal;
            this.rmsDeriv = rmsDeriv;
        }

        /// <summary>
        /// Iteration
        /// </summary>
        public int iteration;

        /// <summary>
        /// Objective value
        /// </summary>
        public double objVal;

        /// <summary>
        /// Root mean square derivative
        /// </summary>
        public double rmsDeriv;
    }

    /// <summary>
    /// Class used to check analytic derivatives using finite difference approximation
    /// </summary>
    public static class DerivativeChecker
    {
        /// <summary>
        /// Check whether "func" correctly calculates derivatives
        /// </summary>
        /// <param name="func">Return the objective and calculates gradients</param>
        /// <param name="x0">The point to perform the check around</param>
        /// <returns>Whether the derivatives are correctly calculated. </returns>
        public static bool CheckDerivatives(FunctionEval func, Vector x0)
        {
            double eps = 1e-4;
            int K = x0.Count;
            var grad = Vector.Zero(K);
            Vector dummy = Vector.Zero(K);
            double f0 = func(x0, ref grad);
            bool allGood = true;
            for (int i = 0; i < K; i++)
            {
                var x = x0.Clone();
                double eps2 = eps* System.Math.Max(System.Math.Abs(x[i]), 1e-8);
                x[i] += eps2;
                double f = func(x, ref dummy);
                double fd = (f - f0)/eps2;
                Console.WriteLine("{2} Analytic gradient: {0} finite difference: {1}", grad[i], fd, i);
            }
            return allGood;
        }
    }

    /// <summary>
    /// This implementation of BFGS is based on Algorithm 6.1 from
    /// Nocedal and Wright (Second edition, 2006).
    /// </summary>
    public class BFGS
    {
#pragma warning disable 1591
        public LineSearch lineSearch = null;
        protected internal double epsilon = 0.000001;
        protected internal int iter = 0;
        protected internal double stepMax = 20.0;
        protected internal int maxIterations = 50;
        protected double initialStep = 1.0;
        protected internal bool cancel = false;

        /// <summary>
        /// Following fields are set up by the main run
        /// routine, and are used to provide the 1-D function that
        /// the line search needs
        /// </summary>
        protected internal int dimension;

        protected internal Vector currentX;
        protected internal Vector searchDir;
        protected internal Vector currentDeriv;
        protected internal FunctionEval feval;
        protected internal double currentObj;
#pragma warning restore 1591

        /// <summary>
        /// Whether to output debug info
        /// </summary>
        public bool debug = false;

        /// <summary>
        /// Whether to debug the line search
        /// </summary>
        public bool linesearchDebug
        {
            set { lineSearch.debug = value; }
        }

        /// <summary>
        ///  Convergence criteria:
        /// - Gradient: |grad F|/sqrt(dimensions) &lt;= eps
        /// - Objective: |F(k+1)-F(k)|&lt;=eps*max{|F(k)|,|F(k+1)|,1}
        /// </summary>
        public enum ConvergenceCriteria
        {
            /// <summary>
            /// |grad F|/sqrt(dimensions) &lt;= eps
            /// </summary>
            Gradient,

            /// <summary>
            /// |F(k+1)-F(k)|&lt;=eps*max{|F(k)|,|F(k+1)|,1}
            /// </summary>
            Objective
        };

        /// <summary>
        ///  Convergence criteria:
        /// </summary>
        public ConvergenceCriteria convergenceCriteria = ConvergenceCriteria.Gradient;

        /// <summary>
        ///  Number of iterations performed to reach convergence (or failure)
        /// </summary>
        public int IterationsPerformed
        {
            get { return iter; }
        }

        /// <summary>
        /// Convergence tolerance
        /// </summary>
        public double Epsilon
        {
            get { return epsilon; }
            set
            {
                if (epsilon <= 0.0)
                    throw new ArgumentOutOfRangeException("Epsilon must be positive");
                epsilon = value;
            }
        }

        /// <summary>
        /// Maximum step
        /// </summary>
        public double MaximumStep
        {
            get { return stepMax; }
            set
            {
                if (value <= 1.0)
                    throw new ArgumentOutOfRangeException("Maximum step length must be >= 1");
                stepMax = value;
            }
        }

        /// <summary>
        /// Initial step
        /// </summary>
        public double InitialStep
        {
            get { return initialStep; }
            set
            {
                if (value <= 0.0)
                    throw new ArgumentOutOfRangeException("Initial step length must be > 0");
                initialStep = value;
            }
        }

        /// <summary>
        /// Maximum number of iterations
        /// </summary>
        public int MaximumIterations
        {
            get { return maxIterations; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("Maximum iterations must be > 0");
                maxIterations = value;
            }
        }

        /// <summary>
        /// Cancel
        /// </summary>
        public bool Cancel
        {
            get { return cancel; }
            set { cancel = value; }
        }

        /// <summary>
        /// BFGS constructor
        /// </summary>
        public BFGS()
        {
            // Line search
            lineSearch = new LineSearch();
            lineSearch.SetWolfeConstants(0.0001, 0.9); // Standard for BFGS
            lineSearch.Phi = new LineSearchEval(LSEval); // 1-D function evaluator
            epsilon = 0.000001;
            stepMax = 20.0;
            initialStep = 1.0;
            maxIterations = 50;
            cancel = false;
        }

        /// <summary>
        /// Set Wolfe conditions
        /// </summary>
        /// <param name="c1">Sufficient decrease condition</param>
        /// <param name="c2">Curvature condition</param>
        public void SetWolfeConstants(double c1, double c2)
        {
            lineSearch.SetWolfeConstants(c1, c2);
        }

        /// <summary>
        /// This is the 1-D function that the line search sees.
        /// The Run function will set up the necessary fields
        /// </summary>
        /// <param name="step">step</param>
        /// <param name="calcDeriv">flag to calculate derivative</param>
        /// <param name="deriv">the derivative</param>
        /// <returns></returns>
        protected internal virtual double LSEval(double step, bool calcDeriv, out double deriv)
        {
            var trialX = Vector.Zero(dimension, searchDir.Sparsity);
            trialX.SetToProduct(searchDir, step);
            trialX.SetToSum(currentX, trialX);
            if (calcDeriv)
            {
                currentObj = feval(trialX, ref currentDeriv);
                deriv = currentDeriv.Inner(searchDir);
            }
            else
            {
                Vector nullDeriv = null;
                currentObj = feval(trialX, ref nullDeriv);
                deriv = 0.0;
            }
            return currentObj;
        }

        /// <summary>
        /// Line search
        /// </summary>
        /// <param name="s0">Initial step</param>
        /// <param name="smax">Maximum step</param>
        /// <param name="currentGrad"></param>
        /// <returns></returns>
        protected double TryLineSearch(double s0, double smax, double currentGrad)
        {
            double step = 0.0;
            //try
            {
                // Run the line search
                step = lineSearch.Run(s0, smax, currentObj, currentGrad);
            }
            //catch
            //{
            //    // An exception has been thrown. This means that the
            //    // function evaluation at one or other end of the interval is blowing
            //    // up. Try to recover. First, check if the exception is at the near end,
            //    // and back it off. Then check the far end, and back it off
            //    if (debug)
            //        Console.WriteLine("Line search threw exception, trying to recover"); 
            //    double drv;
            //    step = 0.0;
            //    int xx;
            //    int xcnt = 10;
            //    for (xx = 0; xx < xcnt; xx++)
            //    {
            //        try
            //        {
            //            LSEval(s0, false, out drv);
            //            break;
            //        }
            //        catch
            //        {
            //            smax = s0;
            //            s0 *= 0.25;
            //        }
            //    }
            //    if (xx != xcnt)
            //    {
            //        // We're in with a chance of recovering
            //        // Try dealing with the far end
            //        double span = smax - s0;
            //        for (xx = 0; xx < xcnt; xx++)
            //        {
            //            try
            //            {
            //                LSEval(smax, false, out drv);
            //                break;
            //            }
            //            catch
            //            {
            //                span *= 0.25;
            //                smax = s0 + span;
            //            }
            //        }

            //        if (xx != xcnt)
            //        {
            //            // Run the line search again
            //            step = lineSearch.Run(s0, smax);
            //        }
            //    }
            //}  
            return step;
        }

        /// <summary>
        /// Run an unconstrained minimization using BFGS
        /// </summary>
        /// <param name="x0">Starting step</param>
        /// <param name="normOfFirstStep">Norm of first step</param>
        /// <param name="func">Multivariate function and derivative evaluator</param>
        /// <returns>The local minimum point</returns>
        public virtual Vector Run(Vector x0, double normOfFirstStep, FunctionEval func)
        {
            dimension = x0.Count;
            currentX = Vector.Copy(x0);

            if (func == null)
                throw new ArgumentException("Null function");

            cancel = false;

            // Initialize all the fields needed for optimisation
            feval = new FunctionEval(func);
            searchDir = Vector.Zero(dimension);
            currentDeriv = Vector.Zero(dimension);

            // Following vectors are the components of the
            // inverse Hessian update
            Vector S = Vector.Zero(dimension);
            Vector Y = Vector.Zero(dimension);
            Vector minusRhoS = Vector.Zero(dimension);
            Matrix EyeMinusSYt = new Matrix(dimension, dimension);
            Vector prevDeriv = Vector.Zero(dimension);
            PositiveDefiniteMatrix H = new PositiveDefiniteMatrix(dimension, dimension);
            H.SetToIdentity();
            Matrix HWork = new Matrix(dimension, dimension);

            // Get the derivatives
            currentObj = feval(currentX, ref currentDeriv);
            double prevObj = currentObj;
            double rho;
            double step = 0.0;
            double step0 = initialStep;
            double convergenceCheck = double.MaxValue;
            double invDim = 1.0/dimension;
            iter = 0;
            while (convergenceCheck > this.epsilon && !cancel)
            {
                prevDeriv.SetTo(currentDeriv);

                // Compute search direction
                searchDir.SetToProduct(H, currentDeriv);

                // Negate
                for (int i = 0; i < dimension; i++)
                    searchDir[i] = -searchDir[i];

                if (iter == 0)
                {
                    double sdNorm = System.Math.Sqrt(searchDir.Inner(searchDir));
                    double sdMult = normOfFirstStep/sdNorm;
                    searchDir.SetToProduct(searchDir, sdMult);
                    for (int i = 0; i < dimension; i++)
                        H[i, i] = sdMult;
                }

                //------------------------------------------------------
                // Line search enforces strong Wolfe conditions
                // so curvature condition is guaranteed
                //------------------------------------------------------
                step = TryLineSearch(step0, stepMax, currentDeriv.Inner(searchDir));

                // Get the delta between the old and new point
                S.SetToProduct(searchDir, step);

                // Update the current point
                currentX.SetToSum(currentX, S);

                // Calculate the objective and the derivative at the new point
                //double objVal = feval(currentX, ref currentDeriv);

                // If the step is 0, break now, once we have re-evaluated the function and derivs.
                if (step == 0.0)
                    break;

                // Difference of derivs
                Y.SetToDifference(currentDeriv, prevDeriv);

                // BFGS update
                rho = 1.0/(S.Inner(Y));
                if (iter == 0)
                {
                    // Modify the Hessian to yTs/yTy times Identity
                    double beta = S.Inner(S)*rho;
                    for (int i = 0; i < dimension; i++)
                        H[i, i] = beta;
                }


                EyeMinusSYt.SetToIdentity();
                EyeMinusSYt.SetToSumWithOuter(EyeMinusSYt, -rho, S, Y);
                HWork.SetToProduct(EyeMinusSYt, H);
                H.SetToProduct(HWork, EyeMinusSYt.Transpose());
                H.SetToSumWithOuter(H, rho, S, S);

                switch (convergenceCriteria)
                {
                    case ConvergenceCriteria.Gradient:
                        convergenceCheck = System.Math.Sqrt(invDim * currentDeriv.Inner(currentDeriv));
                        break;
                    case ConvergenceCriteria.Objective:
                        convergenceCheck = MMath.AbsDiff(prevObj, currentObj, 1);
                        break;
                }
                prevObj = currentObj;
                // Let anyone who's interested know that we have completed an iteration
                RaiseIterationEvent(iter, currentObj, convergenceCheck);

                if (++iter > maxIterations)
                    break;
            }
            return currentX;
        }

        /// <summary>
        /// Safely invoke the iteration event
        /// </summary>
        /// <param name="iteration"></param>
        /// <param name="objVal"></param>
        /// <param name="rmsDeriv"></param>
        protected internal void RaiseIterationEvent(int iteration, double objVal, double rmsDeriv)
        {
            OnIteration?.Invoke(this, new OptimiserIterationEventArgs(iteration, objVal, rmsDeriv));
        }

        /// <summary>
        /// Event triggered at each iteration
        /// </summary>
        public event IterationEventHandler OnIteration;
    }
}