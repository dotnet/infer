// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// L-BFGS unconstrained optimiser 
// Reference: Nocedal and Wright (1999)

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Implements the LBFGS compact Quasi-Newton solver. 
    /// </summary>
    public class LBFGS : BFGS
    {
#pragma warning disable 1591
        protected internal List<Vector> ess = null;
        protected internal List<Vector> wye = null;
        protected internal List<double> rho = null;
#pragma warning restore 1591
        protected internal int approxDim;

        /// <summary>
        /// Number of dimesions to use in the approximation to the inverse Hessian. 
        /// </summary>
        public int ApproximationDimension
        {
            get { return approxDim; }
            set
            {
                if (value < 1)
                    throw new ArgumentException("Approximation dimension must be at least 1");
                approxDim = value;
            }
        }

        /// <summary>
        /// Limited-memory BFGS constructor
        /// </summary>
        /// <param name="approxDim">Approximation dimension</param>
        public LBFGS(int approxDim)
            : base()
        {
            ApproximationDimension = approxDim;
            Init();

            MaximumIterations = 200; // Need more than BFGS
        }

        /// <summary>
        /// Clear various arrays
        /// </summary>
        private void Init()
        {
            ess = new List<Vector>();
            wye = new List<Vector>();
            rho = new List<double>();
        }

        /// <summary>
        /// Run an unconstrained minimization using BFGS
        /// </summary>
        /// <param name="x0">Starting step</param>
        /// <param name="normOfFirstStep">Norm of first step</param>
        /// <param name="func">Multivariate function and derivative evaluator</param>
        /// <returns>The local minimum point</returns>
        public override Vector Run(Vector x0, double normOfFirstStep, FunctionEval func)
        {
            Init();
            dimension = x0.Count;
            if (approxDim >= dimension)
                return base.Run(x0, normOfFirstStep, func);

            if (func == null)
                throw new ArgumentException("Null function");

            cancel = false;

            // Initialize all the fields needed for optimisation
            feval = new FunctionEval(func);
            searchDir = Vector.Zero(dimension, sparsity: x0.Sparsity);
            currentX = Vector.Copy(x0);
            currentDeriv = Vector.Zero(dimension, sparsity: x0.Sparsity);

            // Following vectors are the components of the
            // inverse Hessian update
            Vector S = Vector.Zero(dimension, sparsity: x0.Sparsity);
            Vector Y = Vector.Zero(dimension, sparsity: x0.Sparsity);
            Vector q = Vector.Zero(dimension, sparsity: x0.Sparsity);
            Vector alpha = Vector.Zero(approxDim);
            Vector work = Vector.Zero(dimension, sparsity: x0.Sparsity);
            Vector prevDeriv = Vector.Zero(dimension, sparsity: x0.Sparsity);
            Vector H0 = Vector.Zero(dimension, sparsity: x0.Sparsity);

            // Get the derivatives
            currentObj = feval(currentX, ref currentDeriv);

            double beta;
            double prevObj = currentObj;
            double step = 0.0;
            double step0 = initialStep;
            double convergenceCheck = double.MaxValue;
            double invDim = 1.0/dimension;
            iter = 0;
            while (convergenceCheck > this.epsilon && !cancel)
            {
                prevDeriv.SetTo(currentDeriv);

                // H0 = gamma x Identity
                if (iter == 0)
                {
                    double sdNorm = System.Math.Sqrt(currentDeriv.Inner(currentDeriv));
                    H0.SetAllElementsTo(normOfFirstStep/sdNorm);
                }
                else
                {
                    H0.SetAllElementsTo(ess[0].Inner(wye[0])/wye[0].Inner(wye[0]));
                }

                //---------------------------------------------
                // L-BFGS two loop recursion. Algorithm 7.4 in
                // Nocedal and Wright 2006
                //---------------------------------------------
                q.SetTo(currentDeriv);
                int cnt = ess.Count;
                for (int i = 0; i < cnt; i++)
                {
                    alpha[i] = rho[i]*ess[i].Inner(q);
                    work.SetToProduct(wye[i], alpha[i]);
                    q.SetToDifference(q, work);
                }

                searchDir.SetToProduct(q, H0);

                for (int i = cnt - 1; i >= 0; i--)
                {
                    beta = rho[i]*wye[i].Inner(searchDir);
                    work.SetToProduct(ess[i], alpha[i] - beta);
                    searchDir.SetToSum(searchDir, work);
                }

                // Negate
                searchDir.Scale(-1.0);

                //------------------------------------------------------
                // Line search enforces strong Wolfe conditions
                // so curvature condition is guaranteed
                //------------------------------------------------------
                step = TryLineSearch(step0, stepMax, currentDeriv.Inner(searchDir));

                // Discard items falling off the bottom of the list
                if (cnt >= approxDim)
                {
                    ess.RemoveAt(cnt - 1);
                    wye.RemoveAt(cnt - 1);
                    rho.RemoveAt(cnt - 1);
                }

                // we have done this calculation already...
                // Get the delta between the old and new point
                S.SetToProduct(searchDir, step);
                ess.Insert(0, Vector.Copy(S));

                // Update the current point
                currentX.SetToSum(currentX, S);

                // Find the derivative at the new point
                // we should already have this
                //if (objVal != currentObj)
                //    throw new InferRuntimeException("error"); 

                // If the step is 0, break now, once we have re-evaluated the function and derivs.
                if (step == 0.0)
                {
                    if (debug)
                        Console.WriteLine("Step size is 0.0, stopping LBFGS.");
                    break;
                }

                // Difference of derivs
                Y.SetToDifference(currentDeriv, prevDeriv);
                wye.Insert(0, Vector.Copy(Y));

                // Rho
                rho.Insert(0, 1.0/(S.Inner(Y)));

                // Value used for convergence
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
                {
                    if (debug)
                        Console.WriteLine("Max iterations reached, stopping LBFGS.");
                    break;
                }
            }
            return currentX;
        }
    }

    /// <summary>
    /// Implements the LBFGS compact Quasi-Newton solver on an array of Vectors (which may be sparse)
    /// </summary>
    public class LBFGSArray : LBFGS
    {
        protected internal new Vector[] currentX;
        protected internal new Vector[] searchDir;
        protected internal new Vector[] currentDeriv;
        protected internal new FunctionEvalArray feval;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="approxDim">Rank of the hessian approximation</param>
        public LBFGSArray(int approxDim)
            : base(approxDim)
        {
            Init();
        }

        private Func<Vector, Vector, double> inner = (p, o) => p.Inner(o);

        protected internal override double LSEval(double step, bool calcDeriv, out double deriv)
        {
            var trialX = currentX.Select(o => Vector.Zero(o.Count, o.Sparsity)).ToArray();
            int segments = currentX.Length;
            for (int s = 0; s < segments; s++)
            {
                trialX[s].SetToProduct(searchDir[s], step);
                trialX[s].SetToSum(currentX[s], trialX[s]);
            }
            if (calcDeriv)
            {
                currentObj = feval(trialX, ref currentDeriv);
                deriv = inner.Map(currentDeriv, searchDir).Sum();
            }
            else
            {
                Vector[] nullDeriv = null;
                currentObj = feval(trialX, ref nullDeriv);
                deriv = 0.0;
            }
            return currentObj;
        }


        protected internal new List<Vector[]> ess = null;
        protected internal new List<Vector[]> wye = null;

        private void Init()
        {
            ess = new List<Vector[]>();
            wye = new List<Vector[]>();
            rho = new List<double>();
        }

        /// <summary>
        /// Delegate type for function evaluation
        /// </summary>
        /// <param name="x">Independent value</param>
        /// <param name="dX">If reference is not null, calculate the deriv here</param>
        /// <returns>Function evaluation</returns>
        public delegate double FunctionEvalArray(Vector[] x, ref Vector[] dX);

        /// <summary>
        /// Run an unconstrained minimization using LBFGS
        /// </summary>
        /// <param name="x0">Starting value</param>
        /// <param name="normOfFirstStep">Norm of first step</param>
        /// <param name="func">Multivariate function and derivative evaluator</param>
        /// <returns>The local minimum point</returns>
        public Vector[] Run(Vector[] x0, double normOfFirstStep, FunctionEvalArray func)
        {
            int segments = x0.Length;
            dimension = x0[0].Count;

            if (func == null)
                throw new ArgumentException("Null function");

            cancel = false;

            // Initialize all the fields needed for optimisation
            feval = new FunctionEvalArray(func);
            searchDir = x0.Select(o => Vector.Zero(o.Count, o.Sparsity)).ToArray();
            currentX = x0.Select(o => Vector.Copy(o)).ToArray();
            currentDeriv = x0.Select(o => Vector.Zero(o.Count, o.Sparsity)).ToArray();

            // Following vectors are the components of the
            // inverse Hessian update
            var S = x0.Select(o => Vector.Zero(o.Count, o.Sparsity)).ToArray();
            var Y = x0.Select(o => Vector.Zero(o.Count, o.Sparsity)).ToArray();
            var q = x0.Select(o => Vector.Zero(o.Count, o.Sparsity)).ToArray();
            Vector alpha = Vector.Zero(approxDim);
            var work = x0.Select(o => Vector.Zero(o.Count, o.Sparsity)).ToArray();
            var prevDeriv = x0.Select(o => Vector.Zero(o.Count, o.Sparsity)).ToArray();
            var H0 = x0.Select(o => Vector.Zero(o.Count, o.Sparsity)).ToArray();

            // Get the derivatives
            currentObj = feval(currentX, ref currentDeriv);

            double beta;
            double prevObj = currentObj;
            double step = 0.0;
            double step0 = initialStep;
            double convergenceCheck = double.MaxValue;
            double invDim = 1.0/dimension;
            iter = 0;

            while (convergenceCheck > this.epsilon && !cancel)
            {
                for (int i = 0; i < segments; i++)
                {
                    prevDeriv[i].SetTo(currentDeriv[i]);
                }
                // H0 = gamma x Identity
                if (iter == 0)
                {
                    double sdNorm = System.Math.Sqrt(currentDeriv.Select(o => o.Inner(o)).Sum());
                    for (int i = 0; i < segments; i++)
                        H0[i].SetAllElementsTo(normOfFirstStep/sdNorm);
                }
                else
                {
                    double common = inner.Map(ess[0], wye[0]).Sum()/wye[0].Select(o => o.Inner(o)).Sum();
                    for (int i = 0; i < segments; i++)
                        H0[i].SetAllElementsTo(common);
                }

                //---------------------------------------------
                // L-BFGS two loop recursion. Algorithm 7.4 in
                // Nocedal and Wright 2006
                //---------------------------------------------
                for (int i = 0; i < segments; i++)
                    q[i].SetTo(currentDeriv[i]);

                int cnt = ess.Count;

                for (int i = 0; i < cnt; i++)
                {
                    alpha[i] = rho[i]*inner.Map(q, ess[i]).Sum();
                    for (int s = 0; s < segments; s++)
                    {
                        work[s].SetToProduct(wye[i][s], alpha[i]);
                        q[s].SetToDifference(q[s], work[s]);
                    }
                }

                for (int s = 0; s < segments; s++)
                    searchDir[s].SetToProduct(q[s], H0[s]);

                for (int i = cnt - 1; i >= 0; i--)
                {
                    beta = rho[i]*inner.Map(wye[i], searchDir).Sum();
                    for (int s = 0; s < segments; s++)
                    {
                        work[s].SetToProduct(ess[i][s], alpha[i] - beta);
                        searchDir[s].SetToSum(searchDir[s], work[s]);
                    }
                }

                // Negate
                for (int i = 0; i < segments; i++)
                    searchDir[i].Scale(-1.0);

                //------------------------------------------------------
                // Line search enforces strong Wolfe conditions
                // so curvature condition is guaranteed
                //------------------------------------------------------
                step = TryLineSearch(step0, stepMax, inner.Map(currentDeriv, searchDir).Sum());

                // Discard items falling off the bottom of the list
                if (cnt >= approxDim)
                {
                    ess.RemoveAt(cnt - 1);
                    wye.RemoveAt(cnt - 1);
                    rho.RemoveAt(cnt - 1);
                }

                // we have done this calculation already...
                // Get the delta between the old and new point
                for (int s = 0; s < segments; s++)
                {
                    S[s].SetToProduct(searchDir[s], step);
                    // Update the current point
                    currentX[s].SetToSum(currentX[s], S[s]);
                    // Difference of derivs
                    Y[s].SetToDifference(currentDeriv[s], prevDeriv[s]);
                }
                ess.Insert(0, S.Select(o => Vector.Copy(o)).ToArray());

                // Find the derivative at the new point
                // we should already have this
                //if (objVal != currentObj)
                //    throw new InferRuntimeException("error"); 

                // If the step is 0, break now, once we have re-evaluated the function and derivs.
                if (step == 0.0)
                {
                    if (debug)
                        Console.WriteLine("Step size is 0.0, stopping LBFGS.");
                    break;
                }


                wye.Insert(0, Y.Select(o => Vector.Copy(o)).ToArray());

                // Rho
                rho.Insert(0, 1.0/inner.Map(S, Y).Sum());

                // Value used for convergence
                switch (convergenceCriteria)
                {
                    case ConvergenceCriteria.Gradient:
                        convergenceCheck = System.Math.Sqrt(invDim * currentDeriv.Select(o => o.Inner(o)).Sum());
                        break;
                    case ConvergenceCriteria.Objective:
                        convergenceCheck = MMath.AbsDiff(prevObj, currentObj, 1);
                        break;
                }
                prevObj = currentObj;
                // Let anyone who's interested know that we have completed an iteration
                RaiseIterationEvent(iter, currentObj, convergenceCheck);

                if (++iter > maxIterations)
                {
                    if (debug)
                        Console.WriteLine("Max iterations reached, stopping LBFGS.");
                    break;
                }
            }
            return currentX;
        }
    }
}