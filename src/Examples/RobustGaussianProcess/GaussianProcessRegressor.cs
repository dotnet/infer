using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using System;

namespace RobustGaussianProcess
{
    public class GaussianProcessRegressor
    {
        public Variable<SparseGP> Prior { get; }
        public Variable<bool> Evidence { get; }
        public Variable<IFunction> F { get; }
        public IfBlock Block;
        public VariableArray<Vector> X { get; }
        public Range J;
        public Variable<double> Score { get; }

        public GaussianProcessRegressor(Vector[] trainingInputs)
        {
            // Modelling code
            Evidence = Variable.Bernoulli(0.5).Named("evidence");
            Block = Variable.If(Evidence);
            Prior = Variable.New<SparseGP>().Named("prior");
            F = Variable<IFunction>.Random(Prior).Named("f");
            X = Variable.Observed(trainingInputs).Named("x");
            J = X.Range.Named("j");
        }

        public GaussianProcessRegressor(Vector[] trainingInputs, bool useStudentTLikelihood, double[] trainingOutputs) : this(trainingInputs)
        {
            VariableArray<double> y = Variable.Observed(trainingOutputs, J).Named("y");

            if (!useStudentTLikelihood)
            {
                // Standard Gaussian Process
                Console.WriteLine("Training a Gaussian Process regressor");
                var score = GetScore(X, F, J);
                y[J] = Variable.GaussianFromMeanAndVariance(score, 0.8);
            }
            else
            {
                // Gaussian Process with Student-t likelihood
                Console.WriteLine("Training a Gaussian Process regressor with Student-t likelihood");
                var noisyScore = GetNoisyScore(X, F, J, trainingOutputs);
                y[J] = Variable.GaussianFromMeanAndVariance(noisyScore[J], 0.8);
            }
        }

        private static Variable<double> GetScore(VariableArray<Vector> x, Variable<IFunction> f, Range j)
        {
            return Variable.FunctionEvaluate(f, x[j]);
        }

        private static VariableArray<double> GetNoisyScore(VariableArray<Vector> x, Variable<IFunction> f, Range j, double[] trainingOutputs)
        {
            // The student-t distribution arises as the mean of a normal distribution once an unknown precision is marginalised out
            Variable<double> score = GetScore(x, f, j);
            VariableArray<double> noisyScore = Variable.Observed(trainingOutputs, j).Named("noisyScore");
            using (Variable.ForEach(j))
            {
                // The precision of the Gaussian is modelled with a Gamma distribution
                var precision = Variable.GammaFromShapeAndRate(4, 1).Named("precision");
                noisyScore[j] = Variable.GaussianFromMeanAndPrecision(score, precision);
            }
            return noisyScore;
        }
    }
}
