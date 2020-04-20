// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Math;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using GaussianArrayArray = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using GaussianArray2D = DistributionStructArray2D<Gaussian, double>;
    using GaussianArrayArray2D = DistributionRefArray<DistributionStructArray2D<Gaussian, double>, double[,]>;
    using GaussianArray2DArray = DistributionRefArray2D<DistributionStructArray<Gaussian, double>, double[]>;
    using GaussianArrayArrayArray = DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>, double[][]>;
    using DirichletArray = DistributionRefArray<Dirichlet, Vector>;
    using DirichletArrayArray = DistributionRefArray<DistributionRefArray<Dirichlet, Vector>, Vector[]>;
    using DirichletArray2D = DistributionRefArray<Dirichlet, Vector>;
    using DirichletArrayArrayArray = DistributionRefArray<DistributionRefArray<DistributionRefArray<Dirichlet, Vector>, Vector[]>, Vector[][]>;
    using GaussianArrayEstimator = ArrayEstimator<GaussianEstimator, DistributionStructArray<Gaussian, double>, Gaussian, double>;

    /// <summary>
    /// Summary description for EstimatorTest
    /// </summary>
    public class EstimatorTest
    {
        private int dim1;
        private int dim2;
        private GaussianArray2DArray ga2aDistArray;
        private Gaussian[,][] ga2aArrayOfDist;

        public EstimatorTest()
        {
            // Create distribution jagged 2D array, and create
            // the parallel jagged 2D array of distributions
            dim1 = 2;
            dim2 = 3;
            ga2aDistArray = new GaussianArray2DArray(dim1, dim2);
            ga2aArrayOfDist = new Gaussian[dim1,dim2][];
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    ga2aDistArray[i, j] = new GaussianArray(i + j + 1);
                    ga2aArrayOfDist[i, j] = new Gaussian[i + j + 1];
                    for (int k = 0; k < ga2aDistArray[i, j].Count; k++)
                    {
                        ga2aDistArray[i, j][k] = Gaussian.FromMeanAndPrecision((double) k, (double) ((k + 1)*(k + 1)));
                        ga2aArrayOfDist[i, j][k] = new Gaussian(ga2aDistArray[i, j][k]);
                    }
                }
            }
        }

        [Fact]
        public void ArrayEstimatorTest()
        {
            int length = 3;
            GaussianArray garray = new GaussianArray(length, i => new Gaussian(i, i + 1));
            ArrayEstimator<GaussianEstimator, GaussianArray, Gaussian, double> est =
                new ArrayEstimator<GaussianEstimator, GaussianArray, Gaussian, double>(
                    Utilities.Util.ArrayInit(length, i => new GaussianEstimator()));
            double[] sum = new double[length];
            double[] sum2 = new double[length];
            int count = 5;
            for (int nSamp = 0; nSamp < count; nSamp++)
            {
                double[] sample = garray.Sample();
                est.Add(sample);
                for (int i = 0; i < length; i++)
                {
                    sum[i] += sample[i];
                    sum2[i] += sample[i]*sample[i];
                }
            }
            GaussianArray expected = new GaussianArray(length, delegate(int i)
                {
                    double m = sum[i]/count;
                    double v = sum2[i]/count - m*m;
                    return new Gaussian(m, v);
                });
            GaussianArray actual = new GaussianArray(length);
            actual = est.GetDistribution(actual);
            Console.WriteLine(Utilities.StringUtil.JoinColumns("result = ", actual, " should be ", expected));
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void Array2DEstimatorTest()
        {
            int length1 = 3;
            int length2 = 2;
            GaussianArray2D garray = new GaussianArray2D(length1, length2, (i, j) => new Gaussian(i, j + 1));
            Array2DEstimator<GaussianEstimator, GaussianArray2D, Gaussian, double> est =
                new Array2DEstimator<GaussianEstimator, GaussianArray2D, Gaussian, double>(
                    Utilities.Util.ArrayInit(length1, length2, (i, j) => new GaussianEstimator()));
            double[,] sum = new double[length1,length2];
            double[,] sum2 = new double[length1,length2];
            int count = 5;
            for (int nSamp = 0; nSamp < count; nSamp++)
            {
                double[,] sample = garray.Sample();
                est.Add(sample);
                for (int i = 0; i < length1; i++)
                {
                    for (int j = 0; j < length2; j++)
                    {
                        sum[i, j] += sample[i, j];
                        sum2[i, j] += sample[i, j]*sample[i, j];
                    }
                }
            }
            GaussianArray2D expected = new GaussianArray2D(length1, length2, delegate(int i, int j)
                {
                    double m = sum[i, j]/count;
                    double v = sum2[i, j]/count - m*m;
                    return new Gaussian(m, v);
                });
            GaussianArray2D actual = new GaussianArray2D(length1, length2);
            actual = est.GetDistribution(actual);
            Console.WriteLine(Utilities.StringUtil.JoinColumns("result = ", actual, " should be ", expected));
            Assert.True(expected.MaxDiff(actual) < 1e-9);
        }

        [Fact]
        public void Array2DArrayEstimatorTest()
        {
            // Create the estimator
            // element type is double[]
            //Estimator<GaussianArray2DArray> est = null;
            int length1 = ga2aArrayOfDist.GetLength(0);
            int length2 = ga2aArrayOfDist.GetLength(1);
            Array2DEstimator<GaussianArrayEstimator, GaussianArray2DArray, GaussianArray, double[]> est =
                new Array2DEstimator<GaussianArrayEstimator, GaussianArray2DArray, GaussianArray, double[]>(
                    Utilities.Util.ArrayInit(length1, length2, (i, j) =>
                                                           new GaussianArrayEstimator(Utilities.Util.ArrayInit(ga2aArrayOfDist[i, j].Length, k => new GaussianEstimator()))
                        ));

            // Add some samples to the estimator
            int nSamples = 5;

            // Create a sample an mean variable of the right structure
            double[,][] mean = (double[,][]) JaggedArray.ConvertToNew(
                ga2aArrayOfDist, typeof (double), delegate(object elt) { return 0.0; });
            double[,][] sample = (double[,][]) JaggedArray.ConvertToNew(
                ga2aArrayOfDist, typeof (double), delegate(object elt) { return 0.0; });

            // Create samples and add them to the estimator. Accumulate sum of samples at the same time
            for (int nSamp = 0; nSamp < nSamples; nSamp++)
            {
                JaggedArray.ConvertElements2(
                    sample, ga2aArrayOfDist, delegate(object smp, object dst) { return ((Gaussian) dst).Sample(); });
                est.Add(sample);

                JaggedArray.ConvertElements2(
                    mean, sample, delegate(object mn, object smp) { return (double) mn + (double) smp; });
            }

            // Hand calculate the sample mean
            JaggedArray.ConvertElements(
                mean, delegate(object mn) { return ((double) mn)/((double) nSamples); });

            // Let the estimator do the work
            GaussianArray2DArray result = new GaussianArray2DArray(length1, length2, (i, j) => new GaussianArray(ga2aArrayOfDist[i, j].Length));
            result = est.GetDistribution(result);

            // The results should be identical to a very high precision
            for (int i = 0; i < dim1; i++)
                for (int j = 0; j < dim2; j++)
                    for (int k = 0; k < result[i, j].Count; k++)
                        Assert.True(System.Math.Abs(result[i, j][k].GetMean() - mean[i, j][k]) < 1e-9);
        }

        [Fact]
        // Tests that visitor works when one jagged is a distribution array and the
        // other is a jagged array
        public void Array2DArrayVisitor()
        {
            int count = JaggedArray.GetLength(ga2aArrayOfDist);
            // We must use the expanded form of VisitElements2 here because the element type of
            // ga2aDistArray will be wrongly determined by the shorter version
            JaggedArray.VisitElements2(
                ga2aDistArray, ga2aArrayOfDist, typeof (Gaussian), typeof (Gaussian), delegate(object a, object b)
                    {
                        Assert.True(((Gaussian) a).Equals((Gaussian) b));
                        count--;
                    });
            Assert.Equal(0, count);
        }

        private void GaussianEstimator(double mean, double precision)
        {
            Rand.Restart(12347);
            Gaussian g = Gaussian.FromMeanAndPrecision(mean, precision);
            Estimator<Gaussian> ge = EstimatorFactory.Instance.CreateEstimator<Gaussian, double>(Gaussian.Uniform());
            Accumulator<double> gea = ge as Accumulator<double>;

            for (int i = 0; i < 10000; i++)
            {
                gea.Add(g.Sample());
            }

            Gaussian gest = Gaussian.Uniform();
            gest = ge.GetDistribution(gest);

            double expectedMean = g.GetMean();
            double expectedSDev = System.Math.Sqrt(g.GetVariance());
            double estimatedMean = gest.GetMean();
            double estimatedSDev = System.Math.Sqrt(gest.GetVariance());
            Assert.True(System.Math.Abs(expectedMean - estimatedMean) < 0.02);
            Assert.True(System.Math.Abs(expectedSDev - estimatedSDev) < 0.02);
        }

        [Fact]
        public void GaussianEstimatorTest()
        {
            double mean = 1.0;
            double precision = 1.0;
            GaussianEstimator(mean, precision);
        }

        private void GammaEstimator(double shape, double rate)
        {
            Rand.Restart(12347);
            Gamma g = Gamma.FromShapeAndRate(shape, rate);
            Estimator<Gamma> ge = EstimatorFactory.Instance.CreateEstimator<Gamma, double>(Gamma.Uniform());
            Accumulator<double> gea = ge as Accumulator<double>;

            for (int i = 0; i < 10000; i++)
            {
                double d = g.Sample();
                gea.Add(d);
            }

            Gamma gest = Gamma.Uniform();
            gest = ge.GetDistribution(gest);

            double expectedMean = g.GetMean();
            double expectedSDev = System.Math.Sqrt(g.GetVariance());
            double estimatedMean = gest.GetMean();
            double estimatedSDev = System.Math.Sqrt(gest.GetVariance());
            Assert.True(System.Math.Abs(expectedMean - estimatedMean) < 0.02);
            Assert.True(System.Math.Abs(expectedSDev - estimatedSDev) < 0.02);
        }

        [Fact]
        public void GammaEstimatorTest()
        {
            double shape = 1.0;
            double rate = 1.0;
            GammaEstimator(shape, rate);
        }

        [Fact]
        public void GaussianEstimatorInfinityTest()
        {
            foreach (var infinity in new[] { double.PositiveInfinity, double.NegativeInfinity })
            {
                var infiniteMeanGaussian = Gaussian.PointMass(infinity);
                var estimator = new GaussianEstimator();

                estimator.Add(infiniteMeanGaussian);
                estimator.Add(infiniteMeanGaussian);

                var estimation = estimator.GetDistribution(new Gaussian());
                Assert.Equal(infiniteMeanGaussian, estimation);
            }
        }
    }
}