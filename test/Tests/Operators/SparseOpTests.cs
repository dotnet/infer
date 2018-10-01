// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using TAssert = Xunit.Assert;
using System;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class SparseOpTests
    {
        [Fact]
        public void SparseBernoulliFromBetaFactor()
        {
            var calcSuffix = ": calculation differs between sparse and dense";
            var sparsitySuffix = ": result is not sparse as expected";
            var calcErrMsg = "";
            var sparsityErrMsg = "";
            var tolerance = 1e-10;
            Rand.Restart(12347);
            int listSize = 50;
            var sparseProbTrueDist = SparseBetaList.Constant(listSize, new Beta(1, 2));
            sparseProbTrueDist[3] = new Beta(4, 5);
            sparseProbTrueDist[6] = new Beta(7, 8);
            var probTrueDist = sparseProbTrueDist.ToArray();
            var sparseProbTruePoint = SparseList<double>.Constant(listSize, 0.1);
            sparseProbTruePoint[3] = 0.7;
            sparseProbTruePoint[6] = 0.8;
            var probTruePoint = sparseProbTruePoint.ToArray();

            var sparseSampleDist = SparseBernoulliList.Constant(listSize, new Bernoulli(0.1));
            sparseSampleDist[3] = new Bernoulli(0.8);
            sparseSampleDist[9] = new Bernoulli(0.9);
            var sampleDist = sparseSampleDist.ToArray();
            var sparseSamplePoint = SparseList<bool>.Constant(listSize, false);
            sparseSamplePoint[3] = true;
            sparseSamplePoint[9] = true;
            var samplePoint = sparseSamplePoint.ToArray();

            var toSparseSampleDist = SparseBernoulliList.Constant(listSize, new Bernoulli(0.1));
            toSparseSampleDist[3] = new Bernoulli(0.4);
            toSparseSampleDist[4] = new Bernoulli(0.8);
            var toSampleDist = toSparseSampleDist.ToArray();

            // ---------------------------
            // Check average log factor
            // ---------------------------
            calcErrMsg = "Average log factor" + calcSuffix;
            // Dist, dist
            var sparseAvgLog = SparseBernoulliFromBetaOp.AverageLogFactor(sparseSampleDist, sparseProbTrueDist);
            var avgLog = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.AverageLogFactor(sampleDist[i], probTrueDist[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);
            // Dist, point
            sparseAvgLog = SparseBernoulliFromBetaOp.AverageLogFactor(sparseSampleDist, sparseProbTruePoint);
            avgLog = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.AverageLogFactor(sampleDist[i], probTruePoint[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);
            // Point, dist
            sparseAvgLog = SparseBernoulliFromBetaOp.AverageLogFactor(sparseSamplePoint, sparseProbTrueDist);
            avgLog = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.AverageLogFactor(samplePoint[i], probTrueDist[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);
            // Point, point
            sparseAvgLog = SparseBernoulliFromBetaOp.AverageLogFactor(sparseSamplePoint, sparseProbTruePoint);
            avgLog = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.AverageLogFactor(samplePoint[i], probTruePoint[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);

            // ---------------------------
            // Check log average factor
            // ---------------------------
            calcErrMsg = "Log average factor" + calcSuffix;
            var sparseLogAvg = SparseBernoulliFromBetaOp.LogAverageFactor(sparseSampleDist, toSparseSampleDist);
            var logAvg = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.LogAverageFactor(sampleDist[i], toSampleDist[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);
            sparseLogAvg = SparseBernoulliFromBetaOp.LogAverageFactor(sparseSamplePoint, sparseProbTrueDist);
            logAvg = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.LogAverageFactor(samplePoint[i], probTrueDist[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);
            sparseLogAvg = SparseBernoulliFromBetaOp.LogAverageFactor(sparseSamplePoint, sparseProbTruePoint);
            logAvg = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.LogAverageFactor(samplePoint[i], probTruePoint[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);

            // ---------------------------
            // Check log evidence ratio
            // ---------------------------
            calcErrMsg = "Log evidence ratio" + calcSuffix;
            // Dist, dist
            var sparseEvidRat = SparseBernoulliFromBetaOp.LogEvidenceRatio(sparseSampleDist, sparseProbTrueDist);
            var evidRat = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.LogEvidenceRatio(sampleDist[i], probTrueDist[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);
            // Dist, point
            sparseEvidRat = SparseBernoulliFromBetaOp.LogEvidenceRatio(sparseSampleDist, sparseProbTruePoint);
            evidRat = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.LogEvidenceRatio(sampleDist[i], probTruePoint[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);
            // Point, dist
            sparseEvidRat = SparseBernoulliFromBetaOp.LogEvidenceRatio(sparseSamplePoint, sparseProbTrueDist);
            evidRat = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.LogEvidenceRatio(samplePoint[i], probTrueDist[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);
            // Point, point
            sparseEvidRat = SparseBernoulliFromBetaOp.LogEvidenceRatio(sparseSamplePoint, sparseProbTruePoint);
            evidRat = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.LogEvidenceRatio(samplePoint[i], probTruePoint[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);

            // ---------------------------
            // Check SampleConditional
            // ---------------------------
            calcErrMsg = "SampleConditional" + calcSuffix;
            sparsityErrMsg = "SampleConditional" + sparsitySuffix;
            // Use different common value to ensure this gets properly set
            var sparseSampleConditional = SparseBernoulliList.Constant(listSize, new Bernoulli(0.5));
            sparseSampleConditional = SparseBernoulliFromBetaOp.SampleConditional(sparseProbTruePoint, sparseSampleConditional);
            var sampleConditional = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.SampleConditional(probTruePoint[i]));
            TAssert.True(2 == sparseSampleConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseSampleConditional.MaxDiff(sampleConditional) < tolerance, calcErrMsg);

            // ---------------------------
            // Check SampleAverageConditional
            // ---------------------------
            calcErrMsg = "SampleAverageConditional" + calcSuffix;
            sparsityErrMsg = "SampleAverageConditional" + sparsitySuffix;
            // Use different common value to ensure this gets properly set
            var sparseSampleAvgConditional = SparseBernoulliList.Constant(listSize, new Bernoulli(0.5));
            sparseSampleAvgConditional = SparseBernoulliFromBetaOp.SampleAverageConditional(sparseProbTrueDist, sparseSampleAvgConditional);
            var sampleAvgConditional = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.SampleAverageConditional(probTrueDist[i]));
            TAssert.True(2 == sparseSampleAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseSampleAvgConditional.MaxDiff(sampleAvgConditional) < tolerance, calcErrMsg);
            sparseSampleAvgConditional = SparseBernoulliList.Constant(listSize, new Bernoulli(0.5));
            sparseSampleAvgConditional = SparseBernoulliFromBetaOp.SampleAverageConditional(sparseProbTruePoint, sparseSampleAvgConditional);
            sampleAvgConditional = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.SampleAverageConditional(probTruePoint[i]));
            TAssert.True(2 == sparseSampleAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseSampleAvgConditional.MaxDiff(sampleAvgConditional) < tolerance, calcErrMsg);

            // ---------------------------
            // Check ProbTrueConditional
            // ---------------------------
            calcErrMsg = "ProbTrueConditional" + calcSuffix;
            sparsityErrMsg = "ProbTrueConditional" + sparsitySuffix;
            // Use different common value to ensure this gets properly set
            var sparseProbTrueConditional = SparseBetaList.Constant(listSize, new Beta(1.1, 2.2));
            sparseProbTrueConditional = SparseBernoulliFromBetaOp.ProbTrueConditional(sparseSamplePoint, sparseProbTrueConditional);
            var probTrueConditional = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.ProbTrueConditional(samplePoint[i]));
            TAssert.True(2 == sparseProbTrueConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseProbTrueConditional.MaxDiff(probTrueConditional) < tolerance, calcErrMsg);

            // ---------------------------
            // Check ProbTrueAverageConditional
            // ---------------------------
            calcErrMsg = "ProbTrueAverageConditional" + calcSuffix;
            sparsityErrMsg = "ProbTrueAverageConditional" + sparsitySuffix;
            // Use different common value to ensure this gets properly set
            var sparseProbTrueAvgConditional = SparseBetaList.Constant(listSize, new Beta(1.1, 2.2));
            sparseProbTrueAvgConditional = SparseBernoulliFromBetaOp.ProbTrueAverageConditional(sparseSampleDist, sparseProbTrueDist, sparseProbTrueAvgConditional);
            var probTrueAvgConditional = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.ProbTrueAverageConditional(sampleDist[i], probTrueDist[i]));
            TAssert.True(2 == sparseProbTrueAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseProbTrueAvgConditional.MaxDiff(probTrueAvgConditional) < tolerance, calcErrMsg);
            sparseProbTrueAvgConditional = SparseBetaList.Constant(listSize, new Beta(1.1, 2.2));
            sparseProbTrueAvgConditional = SparseBernoulliFromBetaOp.ProbTrueAverageConditional(sparseSamplePoint, sparseProbTrueAvgConditional);
            probTrueAvgConditional = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.ProbTrueAverageConditional(samplePoint[i]));
            TAssert.True(2 == sparseProbTrueAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseProbTrueAvgConditional.MaxDiff(probTrueAvgConditional) < tolerance, calcErrMsg);

            // ---------------------------
            // Check SampleAverageLogarithm
            // ---------------------------
            calcErrMsg = "SampleAverageLogarithm" + calcSuffix;
            sparsityErrMsg = "SampleAverageLogarithm" + sparsitySuffix;
            // Use different common value to ensure this gets properly set
            var sparseSampleAvgLogarithm = SparseBernoulliList.Constant(listSize, new Bernoulli(0.5));
            sparseSampleAvgLogarithm = SparseBernoulliFromBetaOp.SampleAverageLogarithm(sparseProbTrueDist, sparseSampleAvgLogarithm);
            var sampleAvgLogarithm = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.SampleAverageLogarithm(probTrueDist[i]));
            TAssert.True(2 == sparseSampleAvgLogarithm.SparseCount, sparsityErrMsg);
            TAssert.True(sparseSampleAvgLogarithm.MaxDiff(sampleAvgLogarithm) < tolerance, calcErrMsg);
            sparseSampleAvgLogarithm = SparseBernoulliList.Constant(listSize, new Bernoulli(0.5));
            sparseSampleAvgLogarithm = SparseBernoulliFromBetaOp.SampleAverageLogarithm(sparseProbTruePoint, sparseSampleAvgLogarithm);
            sampleAvgLogarithm = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.SampleAverageLogarithm(probTruePoint[i]));
            TAssert.True(2 == sparseSampleAvgLogarithm.SparseCount, sparsityErrMsg);
            TAssert.True(sparseSampleAvgLogarithm.MaxDiff(sampleAvgLogarithm) < tolerance, calcErrMsg);

            // ---------------------------
            // Check ProbTrueAverageLogarithm
            // ---------------------------
            calcErrMsg = "ProbTrueAverageLogarithm" + calcSuffix;
            sparsityErrMsg = "ProbTrueAverageLogarithm" + sparsitySuffix;
            // Use different common value to ensure this gets properly set
            var sparseProbTrueAvgLogarithm = SparseBetaList.Constant(listSize, new Beta(1.1, 2.2));
            sparseProbTrueAvgLogarithm = SparseBernoulliFromBetaOp.ProbTrueAverageLogarithm(sparseSampleDist, sparseProbTrueAvgLogarithm);
            var probTrueAvgLogarithm = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.ProbTrueAverageLogarithm(sampleDist[i]));
            TAssert.True(2 == sparseProbTrueAvgLogarithm.SparseCount, sparsityErrMsg);
            TAssert.True(sparseProbTrueAvgLogarithm.MaxDiff(probTrueAvgLogarithm) < tolerance, calcErrMsg);
            sparseProbTrueAvgLogarithm = SparseBetaList.Constant(listSize, new Beta(1.1, 2.2));
            sparseProbTrueAvgLogarithm = SparseBernoulliFromBetaOp.ProbTrueAverageLogarithm(sparseSamplePoint, sparseProbTrueAvgLogarithm);
            probTrueAvgLogarithm = Util.ArrayInit(listSize, i => BernoulliFromBetaOp.ProbTrueAverageLogarithm(samplePoint[i]));
            TAssert.True(2 == sparseProbTrueAvgLogarithm.SparseCount, sparsityErrMsg);
            TAssert.True(sparseProbTrueAvgLogarithm.MaxDiff(probTrueAvgLogarithm) < tolerance, calcErrMsg);
        }

        [Fact]
        public void SparseGaussianListFactor()
        {
            SparseGaussianList.DefaultTolerance = 1e-10;
            var calcSuffix = ": calculation differs between sparse and dense";
            var sparsitySuffix = ": result is not sparse as expected";
            var calcErrMsg = "";
            var sparsityErrMsg = "";
            var tolerance = 1e-10;
            Rand.Restart(12347);
            int listSize = 50;

            // True distribution for the means
            var sparseMeanDist = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(1, 2), tolerance);
            sparseMeanDist[3] = Gaussian.FromMeanAndPrecision(4, 5);
            sparseMeanDist[6] = Gaussian.FromMeanAndPrecision(7, 8);
            var meanDist = sparseMeanDist.ToArray();
            var sparseMeanPoint = SparseList<double>.Constant(listSize, 0.1);
            sparseMeanPoint[3] = 0.7;
            sparseMeanPoint[6] = 0.8;
            var meanPoint = sparseMeanPoint.ToArray();

            // True distribution for the precisions
            var sparsePrecDist = SparseGammaList.Constant(listSize, Gamma.FromShapeAndRate(1.1, 1.2), tolerance);
            sparsePrecDist[3] = Gamma.FromShapeAndRate(2.3, 2.4);
            sparsePrecDist[6] = Gamma.FromShapeAndRate(3.4, 4.5);
            var precDist = sparsePrecDist.ToArray();
            var sparsePrecPoint = SparseList<double>.Constant(listSize, 0.1);
            sparsePrecPoint[3] = 5.6;
            sparsePrecPoint[6] = 0.5;
            var precPoint = sparsePrecPoint.ToArray();

            var sparseSampleDist = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(0.5, 1.5), tolerance);
            sparseSampleDist[3] = Gaussian.FromMeanAndPrecision(-0.5, 2.0);
            sparseSampleDist[9] = Gaussian.FromMeanAndPrecision(1.6, 0.4);
            var sampleDist = sparseSampleDist.ToArray();
            var sparseSamplePoint = SparseList<double>.Constant(listSize, 0.5);
            sparseSamplePoint[3] = 0.1;
            sparseSamplePoint[9] = 2.3;
            var samplePoint = sparseSamplePoint.ToArray();

            var toSparseSampleDist = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(-0.2, 0.3), tolerance);
            toSparseSampleDist[3] = Gaussian.FromMeanAndPrecision(2.1, 3.2);
            toSparseSampleDist[4] = Gaussian.FromMeanAndPrecision(1.3, 0.7);
            var toSampleDist = toSparseSampleDist.ToArray();

            var toSparsePrecDist = SparseGammaList.Constant(listSize, Gamma.FromShapeAndRate(2.3, 3.4), tolerance);
            toSparsePrecDist[3] = Gamma.FromShapeAndRate(3.4, 4.5);
            toSparsePrecDist[4] = Gamma.FromShapeAndRate(5.6, 6.7);
            var toPrecDist = toSparsePrecDist.ToArray();

            // ---------------------------
            // Check average log factor
            // ---------------------------
            calcErrMsg = "Average log factor" + calcSuffix;
            // Dist, dist, dist
            var sparseAvgLog = SparseGaussianListOp.AverageLogFactor(sparseSampleDist, sparseMeanDist, sparsePrecDist);
            var avgLog = Util.ArrayInit(listSize, i => GaussianOp.AverageLogFactor(sampleDist[i], meanDist[i], precDist[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);
            // Dist, dist, point
            sparseAvgLog = SparseGaussianListOp.AverageLogFactor(sparseSampleDist, sparseMeanDist, sparsePrecPoint);
            avgLog = Util.ArrayInit(listSize, i => GaussianOp.AverageLogFactor(sampleDist[i], meanDist[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);
            // Dist, point, dist
            sparseAvgLog = SparseGaussianListOp.AverageLogFactor(sparseSampleDist, sparseMeanPoint, sparsePrecDist);
            avgLog = Util.ArrayInit(listSize, i => GaussianOp.AverageLogFactor(sampleDist[i], meanPoint[i], precDist[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);
            // Dist, point, point
            sparseAvgLog = SparseGaussianListOp.AverageLogFactor(sparseSampleDist, sparseMeanPoint, sparsePrecPoint);
            avgLog = Util.ArrayInit(listSize, i => GaussianOp.AverageLogFactor(sampleDist[i], meanPoint[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);
            // Point, dist, dist
            sparseAvgLog = SparseGaussianListOp.AverageLogFactor(sparseSamplePoint, sparseMeanDist, sparsePrecDist);
            avgLog = Util.ArrayInit(listSize, i => GaussianOp.AverageLogFactor(samplePoint[i], meanDist[i], precDist[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);
            // Point, dist, point
            sparseAvgLog = SparseGaussianListOp.AverageLogFactor(sparseSamplePoint, sparseMeanDist, sparsePrecPoint);
            avgLog = Util.ArrayInit(listSize, i => GaussianOp.AverageLogFactor(samplePoint[i], meanDist[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);
            // Point, point, dist
            sparseAvgLog = SparseGaussianListOp.AverageLogFactor(sparseSamplePoint, sparseMeanPoint, sparsePrecDist);
            avgLog = Util.ArrayInit(listSize, i => GaussianOp.AverageLogFactor(samplePoint[i], meanPoint[i], precDist[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);
            // Point, point, point
            sparseAvgLog = SparseGaussianListOp.AverageLogFactor(sparseSamplePoint, sparseMeanPoint, sparsePrecPoint);
            avgLog = Util.ArrayInit(listSize, i => GaussianOp.AverageLogFactor(samplePoint[i], meanPoint[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(avgLog - sparseAvgLog) < tolerance, calcErrMsg);

            // ---------------------------
            // Check log average factor
            // ---------------------------
            calcErrMsg = "Log average factor" + calcSuffix;
            var sparseLogAvg = SparseGaussianListOp.LogAverageFactor(sparseSampleDist, sparseMeanDist, sparsePrecDist, toSparsePrecDist);
            var logAvg = Util.ArrayInit(listSize, i => GaussianOp.LogAverageFactor(sampleDist[i], meanDist[i], precDist[i], toPrecDist[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);
            // Dist, dist, point
            sparseLogAvg = SparseGaussianListOp.LogAverageFactor(sparseSampleDist, sparseMeanDist, sparsePrecPoint);
            logAvg = Util.ArrayInit(listSize, i => GaussianOp.LogAverageFactor(sampleDist[i], meanDist[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);
            // Dist, point, dist
            sparseLogAvg = SparseGaussianListOp.LogAverageFactor(sparseSampleDist, sparseMeanPoint, sparsePrecDist, toSparsePrecDist);
            logAvg = Util.ArrayInit(listSize, i => GaussianOp.LogAverageFactor(sampleDist[i], meanPoint[i], precDist[i], toPrecDist[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);
            // Dist, point, point
            sparseLogAvg = SparseGaussianListOp.LogAverageFactor(sparseSampleDist, sparseMeanPoint, sparsePrecPoint);
            logAvg = Util.ArrayInit(listSize, i => GaussianOp.LogAverageFactor(sampleDist[i], meanPoint[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);
            // Point, dist, dist
            sparseLogAvg = SparseGaussianListOp.LogAverageFactor(sparseSamplePoint, sparseMeanDist, sparsePrecDist, toSparsePrecDist);
            logAvg = Util.ArrayInit(listSize, i => GaussianOp.LogAverageFactor(samplePoint[i], meanDist[i], precDist[i], toPrecDist[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);
            // Point, dist, point
            sparseLogAvg = SparseGaussianListOp.LogAverageFactor(sparseSamplePoint, sparseMeanDist, sparsePrecPoint);
            logAvg = Util.ArrayInit(listSize, i => GaussianOp.LogAverageFactor(samplePoint[i], meanDist[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);
            // Point, point, dist
            sparseLogAvg = SparseGaussianListOp.LogAverageFactor(sparseSamplePoint, sparseMeanPoint, sparsePrecDist);
            logAvg = Util.ArrayInit(listSize, i => GaussianOp.LogAverageFactor(samplePoint[i], meanPoint[i], precDist[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);
            // Point, point, point
            sparseLogAvg = SparseGaussianListOp.LogAverageFactor(sparseSamplePoint, sparseMeanPoint, sparsePrecPoint);
            logAvg = Util.ArrayInit(listSize, i => GaussianOp.LogAverageFactor(samplePoint[i], meanPoint[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(logAvg - sparseLogAvg) < tolerance, calcErrMsg);

            // ---------------------------
            // Check log evidence ratio
            // ---------------------------
            calcErrMsg = "Log evidence ratio" + calcSuffix;
            var sparseEvidRat = SparseGaussianListOp.LogEvidenceRatio(sparseSampleDist, sparseMeanDist, sparsePrecDist, toSparseSampleDist, toSparsePrecDist);
            var evidRat = Util.ArrayInit(listSize, i => GaussianOp.LogEvidenceRatio(sampleDist[i], meanDist[i], precDist[i], toSampleDist[i], toPrecDist[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);
            // Dist, dist, point
            sparseEvidRat = SparseGaussianListOp.LogEvidenceRatio(sparseSampleDist, sparseMeanDist, sparsePrecPoint);
            evidRat = Util.ArrayInit(listSize, i => GaussianOp.LogEvidenceRatio(sampleDist[i], meanDist[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);
            // Dist, point, dist
            sparseEvidRat = SparseGaussianListOp.LogEvidenceRatio(sparseSampleDist, sparseMeanPoint, sparsePrecDist, toSparseSampleDist, toSparsePrecDist);
            evidRat = Util.ArrayInit(listSize, i => GaussianOp.LogEvidenceRatio(sampleDist[i], meanPoint[i], precDist[i], toSampleDist[i], toPrecDist[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);
            // Dist, point, point
            sparseEvidRat = SparseGaussianListOp.LogEvidenceRatio(sparseSampleDist, sparseMeanPoint, sparsePrecPoint);
            evidRat = Util.ArrayInit(listSize, i => GaussianOp.LogEvidenceRatio(sampleDist[i], meanPoint[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);
            // Point, dist, dist
            sparseEvidRat = SparseGaussianListOp.LogEvidenceRatio(sparseSamplePoint, sparseMeanDist, sparsePrecDist, toSparsePrecDist);
            evidRat = Util.ArrayInit(listSize, i => GaussianOp.LogEvidenceRatio(samplePoint[i], meanDist[i], precDist[i], toPrecDist[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);
            // Point, dist, point
            sparseEvidRat = SparseGaussianListOp.LogEvidenceRatio(sparseSamplePoint, sparseMeanDist, sparsePrecPoint);
            evidRat = Util.ArrayInit(listSize, i => GaussianOp.LogEvidenceRatio(samplePoint[i], meanDist[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);
            // Point, point, dist
            sparseEvidRat = SparseGaussianListOp.LogEvidenceRatio(sparseSamplePoint, sparseMeanPoint, sparsePrecDist);
            evidRat = Util.ArrayInit(listSize, i => GaussianOp.LogEvidenceRatio(samplePoint[i], meanPoint[i], precDist[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);
            // Point, point, point
            sparseEvidRat = SparseGaussianListOp.LogEvidenceRatio(sparseSamplePoint, sparseMeanPoint, sparsePrecPoint);
            evidRat = Util.ArrayInit(listSize, i => GaussianOp.LogEvidenceRatio(samplePoint[i], meanPoint[i], precPoint[i])).Sum();
            TAssert.True(System.Math.Abs(evidRat - sparseEvidRat) < tolerance, calcErrMsg);

            // ---------------------------
            // Check SampleAverageConditional
            // ---------------------------
            calcErrMsg = "SampleAverageConditional" + calcSuffix;
            sparsityErrMsg = "SampleAverageConditional" + sparsitySuffix;
            // Use different common value to ensure this gets properly set
            var sparseSampleAvgConditional = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(0.5, 0.6), tolerance);
            sparseSampleAvgConditional = SparseGaussianListOp.SampleAverageConditional(sparseSampleDist, sparseMeanDist, sparsePrecDist, toSparsePrecDist, sparseSampleAvgConditional);
            var sampleAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.SampleAverageConditional(sampleDist[i], meanDist[i], precDist[i], toPrecDist[i]));
            TAssert.True(3 == sparseSampleAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseSampleAvgConditional.MaxDiff(sampleAvgConditional) < tolerance, calcErrMsg);

            sparseSampleAvgConditional = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(0.5, 0.6), tolerance);
            sparseSampleAvgConditional = SparseGaussianListOp.SampleAverageConditional(sparseSampleDist, sparseMeanPoint, sparsePrecDist, toSparsePrecDist, sparseSampleAvgConditional);
            sampleAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.SampleAverageConditional(sampleDist[i], meanPoint[i], precDist[i], toPrecDist[i]));
            TAssert.True(3 == sparseSampleAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseSampleAvgConditional.MaxDiff(sampleAvgConditional) < tolerance, calcErrMsg);

            sparseSampleAvgConditional = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(0.5, 0.6), tolerance);
            sparseSampleAvgConditional = SparseGaussianListOp.SampleAverageConditional(sparseMeanDist, sparsePrecPoint, sparseSampleAvgConditional);
            sampleAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.SampleAverageConditional(meanDist[i], precPoint[i]));
            TAssert.True(2 == sparseSampleAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseSampleAvgConditional.MaxDiff(sampleAvgConditional) < tolerance, calcErrMsg);

            sparseSampleAvgConditional = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(0.5, 0.6), tolerance);
            sparseSampleAvgConditional = SparseGaussianListOp.SampleAverageConditional(sparseMeanPoint, sparsePrecPoint, sparseSampleAvgConditional);
            sampleAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.SampleAverageConditional(meanPoint[i], precPoint[i]));
            TAssert.True(2 == sparseSampleAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseSampleAvgConditional.MaxDiff(sampleAvgConditional) < tolerance, calcErrMsg);

            // ---------------------------
            // Check MeanAverageConditional
            // ---------------------------
            calcErrMsg = "MeanAverageConditional" + calcSuffix;
            sparsityErrMsg = "MeanAverageConditional" + sparsitySuffix;
            // Use different common value to ensure this gets properly set
            var sparseMeanAvgConditional = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(0.5, 0.6), tolerance);
            sparseMeanAvgConditional = SparseGaussianListOp.MeanAverageConditional(sparseSampleDist, sparseMeanDist, sparsePrecDist, toSparsePrecDist, sparseMeanAvgConditional);
            var meanAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.MeanAverageConditional(sampleDist[i], meanDist[i], precDist[i], toPrecDist[i]));
            TAssert.True(3 == sparseMeanAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseMeanAvgConditional.MaxDiff(meanAvgConditional) < tolerance, calcErrMsg);

            sparseMeanAvgConditional = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(0.5, 0.6), tolerance);
            sparseMeanAvgConditional = SparseGaussianListOp.MeanAverageConditional(sparseSamplePoint, sparseMeanDist, sparsePrecDist, toSparsePrecDist, sparseMeanAvgConditional);
            meanAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.MeanAverageConditional(samplePoint[i], meanDist[i], precDist[i], toPrecDist[i]));
            TAssert.True(3 == sparseMeanAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseMeanAvgConditional.MaxDiff(meanAvgConditional) < tolerance, calcErrMsg);

            sparseMeanAvgConditional = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(0.5, 0.6), tolerance);
            sparseMeanAvgConditional = SparseGaussianListOp.MeanAverageConditional(sparseSampleDist, sparsePrecPoint, sparseMeanAvgConditional);
            meanAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.MeanAverageConditional(sampleDist[i], precPoint[i]));
            TAssert.True(3 == sparseMeanAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseMeanAvgConditional.MaxDiff(meanAvgConditional) < tolerance, calcErrMsg);

            sparseMeanAvgConditional = SparseGaussianList.Constant(listSize, Gaussian.FromMeanAndPrecision(0.5, 0.6), tolerance);
            sparseMeanAvgConditional = SparseGaussianListOp.MeanAverageConditional(sparseSamplePoint, sparsePrecPoint, sparseMeanAvgConditional);
            meanAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.MeanAverageConditional(samplePoint[i], precPoint[i]));
            TAssert.True(3 == sparseMeanAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparseMeanAvgConditional.MaxDiff(meanAvgConditional) < tolerance, calcErrMsg);

            // ---------------------------
            // Check PrecisionAverageConditional
            // ---------------------------
            calcErrMsg = "PrecisionAverageConditional" + calcSuffix;
            sparsityErrMsg = "PrecisionAverageConditional" + sparsitySuffix;
            // Use different common value to ensure this gets properly set
            var sparsePrecAvgConditional = SparseGammaList.Constant(listSize, Gamma.FromShapeAndRate(2.1, 3.2), tolerance);
            sparsePrecAvgConditional = SparseGaussianListOp.PrecisionAverageConditional(sparseSampleDist, sparseMeanDist, sparsePrecDist, sparsePrecAvgConditional);
            var precAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.PrecisionAverageConditional(sampleDist[i], meanDist[i], precDist[i]));
            TAssert.True(3 == sparsePrecAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparsePrecAvgConditional.MaxDiff(precAvgConditional) < tolerance, calcErrMsg);

            sparsePrecAvgConditional = SparseGammaList.Constant(listSize, Gamma.FromShapeAndRate(2.1, 3.2), tolerance);
            sparsePrecAvgConditional = SparseGaussianListOp.PrecisionAverageConditional(sparseSamplePoint, sparseMeanDist, sparsePrecDist, sparsePrecAvgConditional);
            precAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.PrecisionAverageConditional(Gaussian.PointMass(samplePoint[i]), meanDist[i], precDist[i]));
            TAssert.True(3 == sparsePrecAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparsePrecAvgConditional.MaxDiff(precAvgConditional) < tolerance, calcErrMsg);

            sparsePrecAvgConditional = SparseGammaList.Constant(listSize, Gamma.FromShapeAndRate(2.1, 3.2), tolerance);
            sparsePrecAvgConditional = SparseGaussianListOp.PrecisionAverageConditional(sparseSampleDist, sparseMeanPoint, sparsePrecDist, sparsePrecAvgConditional);
            precAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.PrecisionAverageConditional(sampleDist[i], Gaussian.PointMass(meanPoint[i]), precDist[i]));
            TAssert.True(3 == sparsePrecAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparsePrecAvgConditional.MaxDiff(precAvgConditional) < tolerance, calcErrMsg);

            sparsePrecAvgConditional = SparseGammaList.Constant(listSize, Gamma.FromShapeAndRate(2.1, 3.2), tolerance);
            sparsePrecAvgConditional = SparseGaussianListOp.PrecisionAverageConditional(sparseSamplePoint, sparseMeanPoint, sparsePrecAvgConditional);
            precAvgConditional = Util.ArrayInit(listSize, i => GaussianOp.PrecisionAverageConditional(samplePoint[i], meanPoint[i]));
            TAssert.True(3 == sparsePrecAvgConditional.SparseCount, sparsityErrMsg);
            TAssert.True(sparsePrecAvgConditional.MaxDiff(precAvgConditional) < tolerance, calcErrMsg);
        }
    }
}
