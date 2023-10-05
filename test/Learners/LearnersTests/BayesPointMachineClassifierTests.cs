// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.IO.Compression;
    using System.Linq;
    using System.Runtime.Serialization;

    using Xunit;
    using Assert = AssertHelper;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Collections;

    using StandardPredictiveDistribution = System.Collections.Generic.Dictionary<string, double>;
    using BoolPredictiveDistribution = System.Collections.Generic.Dictionary<bool, double>;
    using IntPredictiveDistribution = System.Collections.Generic.Dictionary<int, double>;
    using Microsoft.ML.Probabilistic.Models;

    /// <summary>
    /// Tests for the Bayes point machine classifier.
    /// </summary>
    //[DeploymentItem(@"CustomSerializedLearners\", "CustomSerializedLearners")]
    public class BayesPointMachineClassifierTests
    {
        [Fact]
        public void DenseBinaryVectorTrainingTest()
        {
            var trainer = new GaussianDenseBinaryVectorBpmTraining_EP();
            trainer.InstanceCount = denseSimpleTrainingData.Count;
            trainer.FeatureValues = denseSimpleTrainingData.ToArray();
            trainer.Labels = denseSimpleTrainingLabels.ToArray();
            int featureDim = denseSimpleTrainingData[0].Count;
            trainer.WeightPriors = new VectorGaussian(Vector.Zero(featureDim), PositiveDefiniteMatrix.Identity(featureDim));
            trainer.Execute(100);

            var predictor = new GaussianDenseBinaryVectorBpmPrediction_EP();
            predictor.InstanceCount = denseSimplePredictionData.Count;
            predictor.FeatureValues = denseSimplePredictionData.ToArray();
            predictor.WeightPriors = trainer.WeightsMarginal();
            predictor.Execute(100);
            var prediction = predictor.LabelsMarginal();
            foreach (var bernoulli in prediction)
            {
                Assert.True(bernoulli.LogOdds > 0);
            }
        }

        [Fact]
        //[DeploymentItem(@"Data\W5ANormalized.csv.gz")]
        public void W5Test()
        {
            var bpm = BayesPointMachineClassifier.CreateBinaryClassifier(new CsvGzipMapping());
            bpm.Train(Path.Combine(
#if NETCOREAPP
                Path.GetDirectoryName(typeof(BayesPointMachineClassifierTests).Assembly.Location), // work dir is not the one with Microsoft.ML.Probabilistic.Learners.Tests.dll on netcore and neither is .Location on netfull
#endif
                "Data", "W5ANormalized.csv.gz"));
        }

        private class CsvGzipMapping : ClassifierMapping<string, string, string, string, Vector>
        {
            public static IEnumerable<string> ReadLinesGzip(string fileName)
            {
                using (Stream stream = File.Open(fileName, FileMode.Open))
                {
                    var gz = new GZipStream(stream, CompressionMode.Decompress);
                    using (var streamReader = new StreamReader(gz))
                    {
                        while(true)
                        {
                            string line = streamReader.ReadLine();
                            if (line == null)
                                break;
                            yield return line;
                        }
                    }
                }
            }

            public override IEnumerable<string> GetInstances(string instanceSource)
            {
                return ReadLinesGzip(instanceSource);
            }

            public override Vector GetFeatures(string instance, string instanceSource = null)
            {
                var array = instance.Split(",".ToCharArray()).Skip(1).Select(Convert.ToDouble).ToArray();
                return Vector.FromArray(array);
            }

            public override string GetLabel(string instance, string instanceSource = null, string labelSource = null)
            {
                return instance.Split(",".ToCharArray()).First();
            }

            public override IEnumerable<string> GetClassLabels(string instanceSource, string labelSource = null)
            {
                return ReadLinesGzip(instanceSource).Select(l => l.Split(",".ToCharArray()).First()).Distinct();
            }
        }

        #region Test initialization

        /// <summary>
        /// Tolerance for comparisons.
        /// </summary>
        private const double Tolerance = 5e-8;

        /// <summary>
        /// The number of iterations of the training algorithm. 
        /// </summary>
        private const int IterationCount = 50;

        #region Expected evidence

        /// <summary>
        /// The expected log evidence of the binary Bayes point machine classifier.
        /// </summary>
        private const double BinaryExpectedLogEvidence = -9.20977846948718;
                                                         
        /// <summary>
        /// The expected log evidence of the multi-class Bayes point machine classifier.
        /// </summary>
        private const double MulticlassExpectedLogEvidence = -17.1975660095695;

        /// <summary>
        /// The expected log evidence of the binary Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private const double GaussianPriorBinaryExpectedLogEvidence = -2.0794415416798357;

        /// <summary>
        /// The expected log evidence of the multi-class Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private const double GaussianPriorMulticlassExpectedLogEvidence = -9.32961091801356;

        #endregion

        #region Expected probabilities of modes

        /// <summary>
        /// The expected probabilities of modes after training of a binary Bayes point machine classifier.
        /// </summary>
        private readonly double[] expectedBernoulliModeProbabilities = { 0.78540833642160135, 0.79734315965419889 };

        /// <summary>
        /// The expected probabilities of modes after incremental training of a binary Bayes point machine classifier.
        /// </summary>
        private readonly double[] expectedIncrementalBernoulliModeProbabilities = { 0.89300722257603715, 0.90690579686340345 };

        /// <summary>
        /// The expected probabilities of modes after training of a binary Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private readonly double[] gaussianPriorExpectedBernoulliModeProbabilities = { 0.72056019196590348, 0.74344670077888442 };

        /// <summary>
        /// The expected probabilities of modes after incremental training of a binary Bayes point machine classifiers with Gaussian prior distributions over weights.
        /// </summary>
        private readonly double[] gaussianPriorExpectedIncrementalBernoulliModeProbabilities = { 0.83123739083799109, 0.86418552521719771 };

        /// <summary>
        /// The expected probabilities of modes after training of a multi-class Bayes point machine classifier.
        /// </summary>
        private readonly double[] expectedDiscreteModeProbabilities = { 0.589726490927633, 0.63111774059567671 };

        /// <summary>
        /// The expected probabilities of modes after incremental training of a multi-class Bayes point machine classifier.
        /// </summary>
        private readonly double[] expectedIncrementalDiscreteModeProbabilities = { 0.8068953354896502, 0.84795019931424553 };

        /// <summary>
        /// The expected probabilities of modes after training of a multi-class Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private readonly double[] gaussianPriorExpectedDiscreteModeProbabilities = { 0.62587241253159176, 0.6664151330140784 };

        /// <summary>
        /// The expected probabilities of modes after incremental training of a multi-class Bayes point machine classifiers with Gaussian prior distributions over weights.
        /// </summary>
        private readonly double[] gaussianPriorExpectedIncrementalDiscreteModeProbabilities = { 0.789392917209095, 0.84315547113779932 };

        #endregion

        #region Expected predictive distributions

        /// <summary>
        /// The expected prediction after training of a binary Bayes point machine classifier.
        /// </summary>
        private Bernoulli[] expectedPredictiveBernoulliDistributions;

        /// <summary>
        /// The expected prediction after incremental training of a binary Bayes point machine classifier.
        /// </summary>
        private Bernoulli[] expectedIncrementalPredictiveBernoulliDistributions;

        /// <summary>
        /// The expected prediction after training of a binary Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private Bernoulli[] gaussianPriorExpectedPredictiveBernoulliDistributions;

        /// <summary>
        /// The expected prediction after incremental training of a binary Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private Bernoulli[] gaussianPriorExpectedIncrementalPredictiveBernoulliDistributions;

        /// <summary>
        /// The expected prediction after training of a multi-class Bayes point machine classifier.
        /// </summary>
        private Discrete[] expectedPredictiveDiscreteDistributions;

        /// <summary>
        /// The expected prediction after incremental training of a multi-class Bayes point machine classifier.
        /// </summary>
        private Discrete[] expectedIncrementalPredictiveDiscreteDistributions;

        /// <summary>
        /// The expected prediction after training of a multi-class Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private Discrete[] gaussianPriorExpectedPredictiveDiscreteDistributions;

        /// <summary>
        /// The expected prediction after incremental training of a multi-class Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private Discrete[] gaussianPriorExpectedIncrementalPredictiveDiscreteDistributions;

        /// <summary>
        /// The expected prediction after training of a binary Bayes point machine classifier.
        /// </summary>
        private BoolPredictiveDistribution[] expectedPredictiveBernoulliSimpleDistributions;
        private IntPredictiveDistribution[] expectedPredictiveBernoulliSimpleIntDistributions;
        private IntPredictiveDistribution[] expectedPredictiveDiscreteIntDistributions;

        /// <summary>
        /// The expected prediction after incremental training of a binary Bayes point machine classifier.
        /// </summary>
        private BoolPredictiveDistribution[] expectedIncrementalPredictiveBernoulliSimpleDistributions;
        private IntPredictiveDistribution[] expectedIncrementalPredictiveBernoulliSimpleIntDistributions;
        private IntPredictiveDistribution[] expectedIncrementalPredictiveDiscreteIntDistributions;

        /// <summary>
        /// The expected prediction after training of a binary Bayes point machine classifier.
        /// </summary>
        private StandardPredictiveDistribution[] expectedPredictiveBernoulliStandardDistributions;

        /// <summary>
        /// The expected prediction after incremental training of a binary Bayes point machine classifier.
        /// </summary>
        private StandardPredictiveDistribution[] expectedIncrementalPredictiveBernoulliStandardDistributions;

        /// <summary>
        /// The expected prediction after training of a binary Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private StandardPredictiveDistribution[] gaussianPriorExpectedPredictiveBernoulliStandardDistributions;

        /// <summary>
        /// The expected prediction after incremental training of a binary Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private StandardPredictiveDistribution[] gaussianPriorExpectedIncrementalPredictiveBernoulliStandardDistributions;

        /// <summary>
        /// The expected prediction after training of a multi-class Bayes point machine classifier.
        /// </summary>
        private StandardPredictiveDistribution[] expectedPredictiveDiscreteStandardDistributions;

        /// <summary>
        /// The expected prediction after incremental training of a multi-class Bayes point machine classifier.
        /// </summary>
        private StandardPredictiveDistribution[] expectedIncrementalPredictiveDiscreteStandardDistributions;

        /// <summary>
        /// The expected prediction after training of a multi-class Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private StandardPredictiveDistribution[] gaussianPriorExpectedPredictiveDiscreteStandardDistributions;

        /// <summary>
        /// The expected prediction after incremental training of a multi-class Bayes point machine classifier with Gaussian prior distributions over weights.
        /// </summary>
        private StandardPredictiveDistribution[] gaussianPriorExpectedIncrementalPredictiveDiscreteStandardDistributions;

        #endregion

        #region Datasets

        /// <summary>
        /// The training dataset in dense native format.
        /// </summary>
        private NativeDataset denseNativeTrainingData;

        /// <summary>
        /// The training dataset in sparse native format.
        /// </summary>
        private NativeDataset sparseNativeTrainingData;

        /// <summary>
        /// The testing dataset in dense native format.
        /// </summary>
        private NativeDataset denseNativePredictionData;

        /// <summary>
        /// The testing dataset in sparse native format.
        /// </summary>
        private NativeDataset sparseNativePredictionData;

        /// <summary>
        /// The training dataset in dense simple format.
        /// </summary>
        private IReadOnlyList<Vector> denseSimpleTrainingData;
        private IReadOnlyList<bool> denseSimpleTrainingLabels;
        private IReadOnlyList<int> denseSimpleIntTrainingLabels;
        private IReadOnlyList<int> denseSimpleMulticlassTrainingLabels;

        /// <summary>
        /// The testing dataset in dense simple format.
        /// </summary>
        private IReadOnlyList<Vector> denseSimplePredictionData;

        /// <summary>
        /// The training dataset in dense standard format.
        /// </summary>
        private StandardDataset denseStandardTrainingData;

        /// <summary>
        /// The testing dataset in dense standard format.
        /// </summary>
        private StandardDataset denseStandardPredictionData;

        /// <summary>
        /// The training dataset in sparse standard format.
        /// </summary>
        private StandardDataset sparseStandardTrainingData;

        /// <summary>
        /// The testing dataset in sparse standard format.
        /// </summary>
        private StandardDataset sparseStandardPredictionData;

        #endregion

        #region Mappings

        /// <summary>
        /// The mapping to the native data format of the binary Bayes point machine classifier.
        /// </summary>
        private BinaryNativeBayesPointMachineClassifierTestMapping binaryNativeMapping;

        /// <summary>
        /// The mapping to the native data format of the multi-class Bayes point machine classifier.
        /// </summary>
        private MulticlassNativeBayesPointMachineClassifierTestMapping multiclassNativeMapping;

        /// <summary>
        /// The mapping to the native data format of the binary Bayes point machine classifier.
        /// </summary>
        private BinaryStandardBayesPointMachineClassifierTestMapping binaryStandardMapping;

        /// <summary>
        /// The mapping to the native data format of the multi-class Bayes point machine classifier.
        /// </summary>
        private MulticlassStandardBayesPointMachineClassifierTestMapping multiclassStandardMapping;

        #endregion

        /// <summary>
        /// Interface to a dataset.
        /// </summary>
        private interface IDataset
        {
            /// <summary>
            /// Gets the number of instances of the dataset.
            /// </summary>
            int InstanceCount { get; }
        }

        /// <summary>
        /// Prepares environment (datasets and expected predictive distributions) before each test.
        /// </summary>
        public BayesPointMachineClassifierTests()
        {
            this.InitializePredictiveDistributions();
            this.InitializeNativeData();
            this.InitializeStandardData();
        }

        #endregion

        #region Tests for Bayes point machine classifiers with compound prior distributions over weights

        #region Tests for a binary BPM on native data and dense features

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping), this.denseNativeTrainingData, this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data with a rare feature in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeRareFeatureTest()
        {
            const int InstanceCount = 1000;
            var trainingData = CreateConstantZeroFeatureDenseNativeDataset(InstanceCount, classCount: 2, featureCount: 2);
            trainingData.Labels = Util.ArrayInit(InstanceCount, i => 0);
            trainingData.Labels[0] = 1;
            trainingData.FeatureValues[0][0] = 1.0;

            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data with a constant zero-feature in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeConstantZeroFeatureTest()
        {
            var trainingData = CreateConstantZeroFeatureDenseNativeDataset();

            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping),
                this.denseNativeTrainingData,
                OnBinaryIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeIncrementalTrainingTest()
        {
            this.TestDenseNativeIncrementalTraining(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests prediction of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativePredictionRegressionTest()
        {
            this.TestRegressionBinaryNativePrediction(this.denseNativeTrainingData, this.denseNativePredictionData);
        }

        /// <summary>
        /// Tests evidence computation of an untrained binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests evidence computation of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence(
                    BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping),
                    this.denseNativeTrainingData,
                    BinaryExpectedLogEvidence,
                    batchCount,
                    this.binaryNativeMapping);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping),
                this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeCustomSerializationRegressionTest()
        {
            TestBinaryNativeClassifierCustomSerializationRegression(
                this.denseNativeTrainingData, 
                this.denseNativePredictionData,
                this.expectedPredictiveBernoulliDistributions,
                this.expectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeSerializationRegressionTest()
        {
            TestRegressionSerialization<NativeDataset, int, NativeDataset, bool, Bernoulli, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping),
                this.denseNativeTrainingData, 
                this.denseNativePredictionData,
                this.expectedPredictiveBernoulliDistributions,
                this.expectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(filename, this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests correctness of training of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeTrainingTest()
        {
            this.TestDenseBinaryNativeTraining();
        }

        /// <summary>
        /// Tests correctness of prediction of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativePredictionTest()
        {
            this.TestDenseNativePrediction(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, this.denseNativePredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, this.denseNativePredictionData, 0, true);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, null, 0, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a binary Bayes point machine classifier for data in native format and
        /// empty features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeEmptyFeaturesPredictionTest()
        {
            TestDenseNativePredictionEmptyFeatures(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeSettingsTest()
        {
            CheckBinaryClassifierSettings(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping), this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests batched training of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryNativeBatchingRegressionTest()
        {
            TestRegressionBatching(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping),
                this.denseNativeTrainingData,
                this.denseNativePredictionData,
                this.expectedPredictiveBernoulliDistributions,
                this.expectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset,
                this.binaryNativeMapping);
        }

        /// <summary>
        /// Tests the support for capabilities of the binary Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void BinaryNativeCapabilitiesTest()
        {
            CheckClassifierCapabilities(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests whether the binary Bayes point machine classifier can correctly detect problems with null mappings.
        /// </summary>
        [Fact]
        public void BinaryNativeNullMappingTest()
        {
            Assert.Throws<ArgumentNullException>(() => BayesPointMachineClassifier.CreateBinaryClassifier<NativeDataset, int, NativeDataset>(null));
        }

        #endregion

        #region Tests for a binary BPM on native data and sparse features

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping), this.sparseNativeTrainingData, this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data with a rare feature in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeRareFeatureTest()
        {
            const int InstanceCount = 1000;
            var trainingData = CreateConstantZeroFeatureSparseNativeDataset(InstanceCount, classCount: 2, featureCount: 2);
            trainingData.Labels = Util.ArrayInit(InstanceCount, i => 0);
            trainingData.Labels[0] = 1;

            var featureIndexes = new List<int>(trainingData.FeatureIndexes[0]) { 0 };
            var featureValues = new List<double>(trainingData.FeatureValues[0]) { 1.0 };
            trainingData.FeatureIndexes[0] = featureIndexes.ToArray();
            trainingData.FeatureValues[0] = featureValues.ToArray();

            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for highly sparse data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeHighSparsityFeatureTest()
        {
            // Testing for numerical inaccuracies in Gaussian operator
            var trainingData = CreateConstantZeroFeatureSparseNativeDataset(50000, classCount: 2, featureCount: 10000, sparsity: 0.0002, addBias: true);

            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for highly sparse data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeHighSparsityFeatureTest2()
        {
            // Testing for division by zero in Gaussian operator
            var trainingData = CreateConstantZeroFeatureSparseNativeDataset(800000, classCount: 2, featureCount: 40000, sparsity: 0.00005, addBias: true);

            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data with a constant zero-feature in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeConstantZeroFeatureTest()
        {
            var trainingData = CreateConstantZeroFeatureSparseNativeDataset();

            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping),
                this.sparseNativeTrainingData,
                OnBinaryIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeIncrementalTrainingTest()
        {
            this.TestSparseNativeIncrementalTraining(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests prediction of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativePredictionRegressionTest()
        {
            this.TestRegressionBinaryNativePrediction(this.sparseNativeTrainingData, this.sparseNativePredictionData);
        }

        /// <summary>
        /// Tests evidence computation of an untrained binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests evidence computation of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence(
                    BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping),
                    this.sparseNativeTrainingData,
                    BinaryExpectedLogEvidence,
                    batchCount,
                    this.binaryNativeMapping);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping),
                this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeCustomSerializationRegressionTest()
        {
            TestBinaryNativeClassifierCustomSerializationRegression(
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.expectedPredictiveBernoulliDistributions,
                this.expectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeSerializationRegressionTest()
        {
            TestRegressionSerialization<NativeDataset, int, NativeDataset, bool, Bernoulli, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping),
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.expectedPredictiveBernoulliDistributions,
                this.expectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(filename, this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests correctness of training of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeTrainingTest()
        {
            this.TestSparseBinaryNativeTraining();
        }

        /// <summary>
        /// Tests correctness of prediction of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativePredictionTest()
        {
            this.TestSparseNativePrediction(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping);
            
            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, this.sparseNativePredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, this.sparseNativePredictionData, 0, true);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, null, 0, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a binary Bayes point machine classifier for data in native format and
        /// empty features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeEmptyFeaturesPredictionTest()
        {
            TestSparseNativePredictionEmptyFeatures(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeSettingsTest()
        {
            CheckBinaryClassifierSettings(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping), this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests batched training of the binary Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryNativeBatchingRegressionTest()
        {
            TestRegressionBatching(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping),
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.expectedPredictiveBernoulliDistributions,
                this.expectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset,
                this.binaryNativeMapping);
        }

        #endregion

        #region Tests for a multi-class BPM on native data and dense features

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping), this.denseNativeTrainingData, this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data with a rare feature in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeRareFeatureTest()
        {
            const int InstanceCount = 1000;
            var trainingData = CreateConstantZeroFeatureDenseNativeDataset(InstanceCount, classCount: 3, featureCount: 2);
            trainingData.Labels = Util.ArrayInit(InstanceCount, i => 0);
            trainingData.Labels[0] = 1;
            trainingData.FeatureValues[0][0] = 1.0;

            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data with a constant zero-feature in native format and
        /// features in a dense representation.
        /// </summary>
        /// <remarks>Fails only if run within a 64-bit process.</remarks>
        [Fact]
        public void DenseMulticlassNativeConstantZeroFeatureTest()
        {
            var trainingData = CreateConstantZeroFeatureDenseNativeDataset();

            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping),
                this.denseNativeTrainingData,
                OnMulticlassIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeIncrementalTrainingTest()
        {
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping);
            this.TestDenseNativeIncrementalTraining(classifier);

            // Check different labels: Throw
            var nativeData = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 0.0, 1.0 } },
                FeatureIndexes = null,
                ClassCount = 4,
                Labels = new[] { 3 }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(nativeData));
        }

        /// <summary>
        /// Tests prediction of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativePredictionRegressionTest()
        {
            this.TestRegressionMulticlassNativePrediction(this.denseNativeTrainingData, this.denseNativePredictionData);
        }

        /// <summary>
        /// Tests evidence computation of an untrained multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests evidence computation of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence(
                    BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping),
                    this.denseNativeTrainingData,
                    MulticlassExpectedLogEvidence,
                    batchCount,
                    this.multiclassNativeMapping);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping),
                this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeCustomSerializationRegressionTest()
        {
            TestMulticlassNativeClassifierCustomSerializationRegression(
                this.denseNativeTrainingData,
                this.denseNativePredictionData,
                this.expectedPredictiveDiscreteDistributions,
                this.expectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeSerializationRegressionTest()
        {
            TestRegressionSerialization<NativeDataset, int, NativeDataset, int, Discrete, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping),
                this.denseNativeTrainingData,
                this.denseNativePredictionData,
                this.expectedPredictiveDiscreteDistributions,
                this.expectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(filename, this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests correctness of training of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeTrainingTest()
        {
            this.TestDenseMulticlassNativeTraining();
        }

        /// <summary>
        /// Tests correctness of prediction of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativePredictionTest()
        {
            this.TestDenseNativePrediction(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, this.denseNativePredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, this.denseNativePredictionData, 0, true);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, null, 0, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a multi-class Bayes point machine classifier for data in native format and
        /// empty features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeEmptyFeaturesPredictionTest()
        {
            TestDenseNativePredictionEmptyFeatures(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeSettingsTest()
        {
            CheckMulticlassClassifierSettings(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping), this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests batched training of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassNativeBatchingRegressionTest()
        {
            TestRegressionBatching(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping),
                this.denseNativeTrainingData,
                this.denseNativePredictionData,
                this.expectedPredictiveDiscreteDistributions,
                this.expectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset,
                this.multiclassNativeMapping);
        }

        /// <summary>
        /// Tests the support for capabilities of the multi-class Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void MulticlassNativeCapabilitiesTest()
        {
            CheckClassifierCapabilities(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests whether the multi-class Bayes point machine classifier can correctly detect problems with null mappings.
        /// </summary>
        [Fact]
        public void MulticlassNativeNullMappingTest()
        {
            Assert.Throws<ArgumentNullException>(() => BayesPointMachineClassifier.CreateMulticlassClassifier<NativeDataset, int, NativeDataset>(null));
        }

        #endregion

        #region Tests for a multi-class BPM on native data and sparse features

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping), this.sparseNativeTrainingData, this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data with a rare feature in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeRareFeatureTest()
        {
            const int InstanceCount = 1000;
            var trainingData = CreateConstantZeroFeatureSparseNativeDataset(InstanceCount, classCount: 3, featureCount: 2);
            trainingData.Labels = Util.ArrayInit(InstanceCount, i => 0);
            trainingData.Labels[0] = 1;

            var featureIndexes = new List<int>(trainingData.FeatureIndexes[0]) { 0 };
            var featureValues = new List<double>(trainingData.FeatureValues[0]) { 1.0 };
            trainingData.FeatureIndexes[0] = featureIndexes.ToArray();
            trainingData.FeatureValues[0] = featureValues.ToArray();

            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data with a constant zero-feature in native format and
        /// features in a sparse representation.
        /// </summary>
        /// <remarks>Fails only if run within a 64-bit process.</remarks>
        [Fact]
        public void SparseMulticlassNativeConstantZeroFeatureTest()
        {
            var trainingData = CreateConstantZeroFeatureSparseNativeDataset();

            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping),
                this.sparseNativeTrainingData,
                OnMulticlassIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeIncrementalTrainingTest()
        {
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping);
            this.TestSparseNativeIncrementalTraining(classifier);

            // Check different labels: Throw
            var nativeData = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 1.0 } },
                FeatureIndexes = new[] { new[] { 1 } },
                ClassCount = 4,
                Labels = new[] { 3 }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(nativeData));
        }

        /// <summary>
        /// Tests prediction of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativePredictionRegressionTest()
        {
            this.TestRegressionMulticlassNativePrediction(this.sparseNativeTrainingData, this.sparseNativePredictionData);
        }

        /// <summary>
        /// Tests evidence computation of an untrained multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests evidence computation of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence(
                    BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping),
                    this.sparseNativeTrainingData,
                    MulticlassExpectedLogEvidence,
                    1,
                    this.multiclassNativeMapping);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping),
                this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeCustomSerializationRegressionTest()
        {
            TestMulticlassNativeClassifierCustomSerializationRegression(
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.expectedPredictiveDiscreteDistributions,
                this.expectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeSerializationRegressionTest()
        {
            TestRegressionSerialization<NativeDataset, int, NativeDataset, int, Discrete, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping),
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.expectedPredictiveDiscreteDistributions,
                this.expectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(filename, this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests correctness of training of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeTrainingTest()
        {
            this.TestSparseMulticlassNativeTraining();
        }

        /// <summary>
        /// Tests correctness of prediction of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativePredictionTest()
        {
            this.TestSparseNativePrediction(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping);
            
            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, this.sparseNativePredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, this.sparseNativePredictionData, 0, true);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, null, 0, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a multi-class Bayes point machine classifier for data in native format and
        /// empty features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeEmptyFeaturesPredictionTest()
        {
            TestSparseNativePredictionEmptyFeatures(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeSettingsTest()
        {
            CheckMulticlassClassifierSettings(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping), this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests batched training of the multi-class Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassNativeBatchingRegressionTest()
        {
            TestRegressionBatching(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping),
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.expectedPredictiveDiscreteDistributions,
                this.expectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset,
                this.multiclassNativeMapping);
        }

        #endregion

        #region Tests for a binary BPM on standard data and dense features

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping), this.denseStandardTrainingData, this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping),
                this.denseStandardTrainingData,
                OnBinaryIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardIncrementalTrainingTest()
        {
            this.TestDenseStandardIncrementalTraining(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests prediction of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardPredictionRegressionTest()
        {
            this.TestRegressionBinaryStandardPrediction(this.denseStandardTrainingData, this.denseStandardPredictionData);
        }

        /// <summary>
        /// Tests evidence computation of an untrained binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests evidence computation of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence<StandardDataset, string, StandardDataset, bool, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                    BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping),
                    this.denseStandardTrainingData,
                    BinaryExpectedLogEvidence,
                    batchCount);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping),
                this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardCustomSerializationRegressionTest()
        {
            TestBinaryStandardClassifierCustomSerializationRegression(
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.expectedPredictiveBernoulliStandardDistributions,
                this.expectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier for data in simple format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinarySimpleCustomSerializationRegressionTest()
        {
            TestBinarySimpleClassifierCustomSerializationRegression(
                this.denseSimpleTrainingData,
                this.denseSimpleTrainingLabels,
                this.denseSimplePredictionData,
                this.expectedPredictiveBernoulliSimpleDistributions,
                this.expectedIncrementalPredictiveBernoulliSimpleDistributions,
                CheckPredictedBernoulliDistributionSimpleTestingDataset);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier for data in simple format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinarySimpleIntCustomSerializationRegressionTest()
        {
            TestBinarySimpleClassifierCustomSerializationRegression(
                this.denseSimpleTrainingData,
                this.denseSimpleIntTrainingLabels,
                this.denseSimplePredictionData,
                this.expectedPredictiveBernoulliSimpleIntDistributions,
                this.expectedIncrementalPredictiveBernoulliSimpleIntDistributions,
                CheckPredictedBernoulliDistributionSimpleTestingDataset);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier for data in simple format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassSimpleCustomSerializationRegressionTest()
        {
            TestMulticlassSimpleClassifierCustomSerializationRegression(
                this.denseSimpleTrainingData,
                this.denseSimpleMulticlassTrainingLabels,
                this.denseSimplePredictionData,
                this.expectedPredictiveDiscreteIntDistributions,
                this.expectedIncrementalPredictiveDiscreteIntDistributions,
                CheckPredictedBernoulliDistributionSimpleTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardSerializationRegressionTest()
        {
            TestRegressionSerialization<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping),
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.expectedPredictiveBernoulliStandardDistributions,
                this.expectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(filename, this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests correctness of training of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardTrainingTest()
        {
            this.TestDenseBinaryStandardTraining();
        }

        /// <summary>
        /// Tests correctness of prediction of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardPredictionTest()
        {
            this.TestDenseStandardPrediction(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, "First", true);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, "First", true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, null, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a binary Bayes point machine classifier for data in standard format and
        /// empty features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardEmptyFeaturesPredictionTest()
        {
            TestDenseStandardPredictionEmptyFeatures(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardSettingsTest()
        {
            CheckBinaryClassifierSettings(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping), this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests the support for capabilities of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void BinaryStandardCapabilitiesTest()
        {
            CheckClassifierCapabilities(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests whether the binary Bayes point machine classifier can correctly detect problems with null mappings.
        /// </summary>
        [Fact]
        public void BinaryStandardNullMappingTest()
        {
            Assert.Throws<ArgumentNullException>(() => BayesPointMachineClassifier.CreateBinaryClassifier<StandardDataset, int, StandardDataset>(null));
        }

        /// <summary>
        /// Tests batched training of the binary Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseBinaryStandardBatchingRegressionTest()
        {
            TestRegressionBatching<StandardDataset, string, StandardDataset, bool, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping),
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.expectedPredictiveBernoulliStandardDistributions,
                this.expectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset);
        }

        #endregion

        #region Tests for a binary BPM on standard data and sparse features

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping), this.sparseStandardTrainingData, this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping),
                this.sparseStandardTrainingData,
                OnBinaryIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardIncrementalTrainingTest()
        {
            this.TestSparseStandardIncrementalTraining(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests prediction of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardPredictionRegressionTest()
        {
            this.TestRegressionBinaryStandardPrediction(this.sparseStandardTrainingData, this.sparseStandardPredictionData);
        }

        /// <summary>
        /// Tests evidence computation of an untrained binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests evidence computation of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence<StandardDataset, string, StandardDataset, bool, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                    BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping),
                    this.sparseStandardTrainingData,
                    BinaryExpectedLogEvidence,
                    batchCount);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping),
                this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardCustomSerializationRegressionTest()
        {
            TestBinaryStandardClassifierCustomSerializationRegression(
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.expectedPredictiveBernoulliStandardDistributions,
                this.expectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardSerializationRegressionTest()
        {
            TestRegressionSerialization<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping),
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.expectedPredictiveBernoulliStandardDistributions,
                this.expectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(filename, this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests correctness of training of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardTrainingTest()
        {
            this.TestSparseBinaryStandardTraining();
        }

        /// <summary>
        /// Tests correctness of prediction of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardPredictionTest()
        {
            this.TestSparseStandardPrediction(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, "First", true);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, "First", true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, null, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a binary Bayes point machine classifier for data in standard format and
        /// empty features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardEmptyFeaturesPredictionTest()
        {
            TestSparseStandardPredictionEmptyFeatures(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardSettingsTest()
        {
            CheckBinaryClassifierSettings(BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping), this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests batched training of the binary Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseBinaryStandardBatchingRegressionTest()
        {
            TestRegressionBatching<StandardDataset, string, StandardDataset, bool, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping),
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.expectedPredictiveBernoulliStandardDistributions,
                this.expectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset);
        }

        #endregion

        #region Tests for a multi-class BPM on standard data and dense features

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping), this.denseStandardTrainingData, this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping),
                this.denseStandardTrainingData,
                OnMulticlassIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardIncrementalTrainingTest()
        {
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping);
            this.TestDenseStandardIncrementalTraining(classifier);

            // Check different labels: Throw
            var standardData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 1) } },
                Labels = new Dictionary<string, string> { { "First", "D" } },
                ClassLabels = new[] { "A", "B", "C", "D" }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(standardData));
        }

        /// <summary>
        /// Tests prediction of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardPredictionRegressionTest()
        {
            this.TestRegressionMulticlassStandardPrediction(this.denseStandardTrainingData, this.denseStandardPredictionData);
        }

        /// <summary>
        /// Tests evidence computation of an untrained multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests evidence computation of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence<StandardDataset, string, StandardDataset, int, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                    BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping),
                    this.denseStandardTrainingData,
                    MulticlassExpectedLogEvidence,
                    batchCount);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping),
                this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardCustomSerializationRegressionTest()
        {
            TestMulticlassStandardClassifierCustomSerializationRegression(
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.expectedPredictiveDiscreteStandardDistributions,
                this.expectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardSerializationRegressionTest()
        {
            TestRegressionSerialization<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping),
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.expectedPredictiveDiscreteStandardDistributions,
                this.expectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(filename, this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests correctness of training of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardTrainingTest()
        {
            this.TestDenseMulticlassStandardTraining();
        }

        /// <summary>
        /// Tests correctness of prediction of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardPredictionTest()
        {
            this.TestDenseStandardPrediction(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, "First", true);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, "First", true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, null, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a multi-class Bayes point machine classifier for data in standard format and
        /// empty features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardEmptyFeaturesPredictionTest()
        {
            TestDenseStandardPredictionEmptyFeatures(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardSettingsTest()
        {
            CheckMulticlassClassifierSettings(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping), this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests batched training of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void DenseMulticlassStandardBatchingRegressionTest()
        {
            TestRegressionBatching<StandardDataset, string, StandardDataset, int, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping),
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.expectedPredictiveDiscreteStandardDistributions,
                this.expectedIncrementalPredictiveDiscreteStandardDistributions, 
                CheckPredictedDiscreteDistributionStandardTestingDataset);
        }

        /// <summary>
        /// Tests the support for capabilities of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void MulticlassStandardCapabilitiesTest()
        {
            CheckClassifierCapabilities(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests whether the multi-class Bayes point machine classifier can correctly detect problems with null mappings.
        /// </summary>
        [Fact]
        public void MulticlassStandardNullMappingTest()
        {
            Assert.Throws<ArgumentNullException>(() => BayesPointMachineClassifier.CreateMulticlassClassifier<StandardDataset, int, StandardDataset>(null));
        }

        #endregion

        #region Tests for a multi-class BPM on standard data and sparse features

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping), this.sparseStandardTrainingData, this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping),
                this.sparseStandardTrainingData,
                OnMulticlassIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardIncrementalTrainingTest()
        {
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping);
            this.TestSparseStandardIncrementalTraining(classifier);

            // Check different labels: Throw
            var standardData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 1) } },
                Labels = new Dictionary<string, string> { { "First", "D" } },
                ClassLabels = new[] { "A", "B", "C", "D" }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(standardData));
        }

        /// <summary>
        /// Tests prediction of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardPredictionRegressionTest()
        {
            this.TestRegressionMulticlassStandardPrediction(this.sparseStandardTrainingData, this.sparseStandardPredictionData);
        }

        /// <summary>
        /// Tests evidence computation of an untrained multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests evidence computation of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence<StandardDataset, string, StandardDataset, int, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                    BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping),
                    this.sparseStandardTrainingData,
                    MulticlassExpectedLogEvidence,
                    batchCount);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping),
                this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardCustomSerializationRegressionTest()
        {
            TestMulticlassStandardClassifierCustomSerializationRegression(
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.expectedPredictiveDiscreteStandardDistributions,
                this.expectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardSerializationRegressionTest()
        {
            TestRegressionSerialization<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping),
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.expectedPredictiveDiscreteStandardDistributions,
                this.expectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(filename, this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests correctness of training of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardTrainingTest()
        {
            this.TestSparseMulticlassStandardTraining();
        }

        /// <summary>
        /// Tests correctness of prediction of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardPredictionTest()
        {
            this.TestSparseStandardPrediction(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, "First", true);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, "First", true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, null, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a multi-class Bayes point machine classifier for data in standard format and
        /// empty features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardEmptyFeaturesPredictionTest()
        {
            TestSparseStandardPredictionEmptyFeatures(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardSettingsTest()
        {
            CheckMulticlassClassifierSettings(BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping), this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests batched training of the multi-class Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void SparseMulticlassStandardBatchingRegressionTest()
        {
            TestRegressionBatching<StandardDataset, string, StandardDataset, int, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping),
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.expectedPredictiveDiscreteStandardDistributions,
                this.expectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset);
        }

        #endregion

        #endregion

        #region Tests for Bayes point machine classifiers with Gaussian prior distributions over weights

        #region Tests for a binary BPM on native data and dense features

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping), this.denseNativeTrainingData, this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data with a rare feature in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeRareFeatureTest()
        {
            const int InstanceCount = 1000;
            var trainingData = CreateConstantZeroFeatureDenseNativeDataset(InstanceCount, classCount: 2, featureCount: 2);
            trainingData.Labels = Util.ArrayInit(InstanceCount, i => 0);
            trainingData.Labels[0] = 1;
            trainingData.FeatureValues[0][0] = 1.0;

            var classifier = BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data with a constant zero-feature in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeConstantZeroFeatureTest()
        {
            var trainingData = CreateConstantZeroFeatureDenseNativeDataset();

            var classifier = BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping),
                this.denseNativeTrainingData,
                OnGaussianPriorBinaryIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeIncrementalTrainingTest()
        {
            this.TestDenseNativeIncrementalTraining(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests prediction of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativePredictionRegressionTest()
        {
            this.TestRegressionBinaryNativePrediction(this.denseNativeTrainingData, this.denseNativePredictionData, false);
        }

        /// <summary>
        /// Tests evidence computation of an untrained binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests evidence computation of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence(
                    BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping),
                    this.denseNativeTrainingData,
                    GaussianPriorBinaryExpectedLogEvidence,
                    batchCount,
                    this.binaryNativeMapping);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping),
                this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeCustomSerializationRegressionTest()
        {
            TestGaussianPriorBinaryNativeClassifierCustomSerializationRegression(
                this.denseNativeTrainingData,
                this.denseNativePredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeSerializationRegressionTest()
        {
            TestRegressionSerialization<NativeDataset, int, NativeDataset, bool, Bernoulli, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping),
                this.denseNativeTrainingData,
                this.denseNativePredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(filename, this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests correctness of training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeTrainingTest()
        {
            this.TestDenseBinaryNativeTraining(false);
        }

        /// <summary>
        /// Tests correctness of prediction of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativePredictionTest()
        {
            this.TestDenseNativePrediction(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, this.denseNativePredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, this.denseNativePredictionData, 0, true);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, null, 0, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and empty features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeEmptyFeaturesPredictionTest()
        {
            TestDenseNativePredictionEmptyFeatures(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeSettingsTest()
        {
            CheckGaussianPriorBinaryClassifierSettings(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping), this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests batched training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryNativeBatchingRegressionTest()
        {
            TestRegressionBatching(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping),
                this.denseNativeTrainingData,
                this.denseNativePredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset,
                this.binaryNativeMapping);
        }

        /// <summary>
        /// Tests the support for capabilities of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianBinaryNativeCapabilitiesTest()
        {
            CheckClassifierCapabilities(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests whether the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights can correctly detect problems with null mappings.
        /// </summary>
        [Fact]
        public void GaussianBinaryNativeNullMappingTest()
        {
            Assert.Throws<ArgumentNullException>(() => BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier<NativeDataset, int, NativeDataset>(null));
        }

        #endregion

        #region Tests for a binary BPM on native data and sparse features

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping), this.sparseNativeTrainingData, this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data with a rare feature in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeRareFeatureTest()
        {
            const int InstanceCount = 1000;
            var trainingData = CreateConstantZeroFeatureSparseNativeDataset(InstanceCount, classCount: 2, featureCount: 2);
            trainingData.Labels = Util.ArrayInit(InstanceCount, i => 0);
            trainingData.Labels[0] = 1;

            var featureIndexes = new List<int>(trainingData.FeatureIndexes[0]) { 0 };
            var featureValues = new List<double>(trainingData.FeatureValues[0]) { 1.0 };
            trainingData.FeatureIndexes[0] = featureIndexes.ToArray();
            trainingData.FeatureValues[0] = featureValues.ToArray();

            var classifier = BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data with a constant zero-feature in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeConstantZeroFeatureTest()
        {
            var trainingData = CreateConstantZeroFeatureSparseNativeDataset();

            var classifier = BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping),
                this.sparseNativeTrainingData,
                OnGaussianPriorBinaryIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeIncrementalTrainingTest()
        {
            this.TestSparseNativeIncrementalTraining(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests prediction of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativePredictionRegressionTest()
        {
            this.TestRegressionBinaryNativePrediction(this.sparseNativeTrainingData, this.sparseNativePredictionData, false);
        }

        /// <summary>
        /// Tests evidence computation of an untrained binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests evidence computation of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence(
                    BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping),
                    this.sparseNativeTrainingData,
                    GaussianPriorBinaryExpectedLogEvidence,
                    batchCount,
                    this.binaryNativeMapping);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping),
                this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeCustomSerializationRegressionTest()
        {
            TestGaussianPriorBinaryNativeClassifierCustomSerializationRegression(
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeSerializationRegressionTest()
        {
            TestRegressionSerialization<NativeDataset, int, NativeDataset, bool, Bernoulli, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping),
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(filename, this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests correctness of training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeTrainingTest()
        {
            this.TestSparseBinaryNativeTraining(false);
        }

        /// <summary>
        /// Tests correctness of prediction of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativePredictionTest()
        {
            this.TestSparseNativePrediction(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, this.sparseNativePredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, this.sparseNativePredictionData, 0, true);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, bool, Bernoulli>(
                classifier, null, 0, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and empty features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeEmptyFeaturesPredictionTest()
        {
            TestSparseNativePredictionEmptyFeatures(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeSettingsTest()
        {
            CheckGaussianPriorBinaryClassifierSettings(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping), this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests batched training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryNativeBatchingRegressionTest()
        {
            TestRegressionBatching(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping),
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliDistributions,
                CheckPredictedBernoulliDistributionNativeTestingDataset,
                this.binaryNativeMapping);
        }

        #endregion

        #region Tests for a multi-class BPM on native data and dense features

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping), this.denseNativeTrainingData, this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data with a rare feature in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeRareFeatureTest()
        {
            const int InstanceCount = 1000;
            var trainingData = CreateConstantZeroFeatureDenseNativeDataset(InstanceCount, classCount: 3, featureCount: 2);
            trainingData.Labels = Util.ArrayInit(InstanceCount, i => 0);
            trainingData.Labels[0] = 1;
            trainingData.FeatureValues[0][0] = 1.0;

            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data with a constant zero-feature in native format and features in a dense representation.
        /// </summary>
        /// <remarks>Fails only if run within a 64-bit process.</remarks>
        [Fact]
        public void GaussianDenseMulticlassNativeConstantZeroFeatureTest()
        {
            var trainingData = CreateConstantZeroFeatureDenseNativeDataset();

            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping),
                this.denseNativeTrainingData,
                OnGaussianPriorMulticlassIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeIncrementalTrainingTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping);
            this.TestDenseNativeIncrementalTraining(classifier);

            // Check different labels: Throw
            var nativeData = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 0.0, 1.0 } },
                FeatureIndexes = null,
                ClassCount = 4,
                Labels = new[] { 3 }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(nativeData));
        }

        /// <summary>
        /// Tests prediction of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativePredictionRegressionTest()
        {
            this.TestRegressionMulticlassNativePrediction(this.denseNativeTrainingData, this.denseNativePredictionData, false);
        }

        /// <summary>
        /// Tests evidence computation of an untrained multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests evidence computation of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence(
                    BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping),
                    this.denseNativeTrainingData,
                    GaussianPriorMulticlassExpectedLogEvidence,
                    batchCount,
                    this.multiclassNativeMapping);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping),
                this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights for data in native format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeCustomSerializationRegressionTest()
        {
            TestGaussianPriorMulticlassNativeClassifierCustomSerializationRegression(
                this.denseNativeTrainingData,
                this.denseNativePredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeSerializationRegressionTest()
        {
            TestRegressionSerialization<NativeDataset, int, NativeDataset, int, Discrete, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping),
                this.denseNativeTrainingData,
                this.denseNativePredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(filename, this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests correctness of training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeTrainingTest()
        {
            this.TestDenseMulticlassNativeTraining(false);
        }

        /// <summary>
        /// Tests correctness of prediction of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativePredictionTest()
        {
            this.TestDenseNativePrediction(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, this.denseNativePredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, this.denseNativePredictionData, 0, true);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, null, 0, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and empty features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeEmptyFeaturesPredictionTest()
        {
            TestDenseNativePredictionEmptyFeatures(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeSettingsTest()
        {
            CheckGaussianPriorMulticlassClassifierSettings(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping), this.denseNativeTrainingData);
        }

        /// <summary>
        /// Tests batched training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassNativeBatchingRegressionTest()
        {
            TestRegressionBatching(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping),
                this.denseNativeTrainingData,
                this.denseNativePredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset,
                this.multiclassNativeMapping);
        }

        /// <summary>
        /// Tests the support for capabilities of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianMulticlassNativeCapabilitiesTest()
        {
            CheckClassifierCapabilities(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests whether the multi-class Bayes point machine classifier can correctly detect problems with null mappings.
        /// </summary>
        [Fact]
        public void GaussianMulticlassNativeNullMappingTest()
        {
            Assert.Throws<ArgumentNullException>(() => BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier<NativeDataset, int, NativeDataset>(null));
        }

        #endregion

        #region Tests for a multi-class BPM on native data and sparse features

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping), this.sparseNativeTrainingData, this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data with a rare feature in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeRareFeatureTest()
        {
            const int InstanceCount = 1000;
            var trainingData = CreateConstantZeroFeatureSparseNativeDataset(InstanceCount, classCount: 3, featureCount: 2);
            trainingData.Labels = Util.ArrayInit(InstanceCount, i => 0);
            trainingData.Labels[0] = 1;

            var featureIndexes = new List<int>(trainingData.FeatureIndexes[0]) { 0 };
            var featureValues = new List<double>(trainingData.FeatureValues[0]) { 1.0 };
            trainingData.FeatureIndexes[0] = featureIndexes.ToArray();
            trainingData.FeatureValues[0] = featureValues.ToArray();

            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data with a constant zero-feature in native format and features in a sparse representation.
        /// </summary>
        /// <remarks>Fails only if run within a 64-bit process.</remarks>
        [Fact]
        public void GaussianSparseMulticlassNativeConstantZeroFeatureTest()
        {
            var trainingData = CreateConstantZeroFeatureSparseNativeDataset();

            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping);
            classifier.Settings.Training.IterationCount = IterationCount;

            TestTraining(classifier, trainingData, trainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping),
                this.sparseNativeTrainingData,
                OnGaussianPriorMulticlassIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeIncrementalTrainingTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping);
            this.TestSparseNativeIncrementalTraining(classifier);

            // Check different labels: Throw
            var nativeData = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 1.0 } },
                FeatureIndexes = new[] { new[] { 1 } },
                ClassCount = 4,
                Labels = new[] { 3 }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(nativeData));
        }

        /// <summary>
        /// Tests prediction of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativePredictionRegressionTest()
        {
            this.TestRegressionMulticlassNativePrediction(this.sparseNativeTrainingData, this.sparseNativePredictionData, false);
        }

        /// <summary>
        /// Tests evidence computation of an untrained multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests evidence computation of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence(
                    BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping),
                    this.sparseNativeTrainingData,
                    GaussianPriorMulticlassExpectedLogEvidence,
                    1,
                    this.multiclassNativeMapping);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping),
                this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights for data in native format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeCustomSerializationRegressionTest()
        {
            TestGaussianPriorMulticlassNativeClassifierCustomSerializationRegression(
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeSerializationRegressionTest()
        {
            TestRegressionSerialization<NativeDataset, int, NativeDataset, int, Discrete, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping),
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(filename, this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests correctness of training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeTrainingTest()
        {
            this.TestSparseMulticlassNativeTraining(false);
        }

        /// <summary>
        /// Tests correctness of prediction of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativePredictionTest()
        {
            this.TestSparseNativePrediction(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, this.sparseNativePredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, this.sparseNativePredictionData, 0, true);

            CheckPredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, NativeDataset, int, NativeDataset, int, Discrete>(
                classifier, null, 0, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and empty features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeEmptyFeaturesPredictionTest()
        {
            TestSparseNativePredictionEmptyFeatures(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeSettingsTest()
        {
            CheckGaussianPriorMulticlassClassifierSettings(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping), this.sparseNativeTrainingData);
        }

        /// <summary>
        /// Tests batched training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in native format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassNativeBatchingRegressionTest()
        {
            TestRegressionBatching(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping),
                this.sparseNativeTrainingData,
                this.sparseNativePredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteDistributions,
                CheckPredictedDiscreteDistributionNativeTestingDataset,
                this.multiclassNativeMapping);
        }

        #endregion

        #region Tests for a binary BPM on standard data and dense features

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping), this.denseStandardTrainingData, this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping),
                this.denseStandardTrainingData,
                OnGaussianPriorBinaryIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardIncrementalTrainingTest()
        {
            this.TestDenseStandardIncrementalTraining(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests prediction of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardPredictionRegressionTest()
        {
            this.TestRegressionBinaryStandardPrediction(this.denseStandardTrainingData, this.denseStandardPredictionData, false);
        }

        /// <summary>
        /// Tests evidence computation of an untrained binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests evidence computation of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence<StandardDataset, string, StandardDataset, bool, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                    BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping),
                    this.denseStandardTrainingData,
                    GaussianPriorBinaryExpectedLogEvidence,
                    batchCount);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping),
                this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardCustomSerializationRegressionTest()
        {
            TestGaussianPriorBinaryStandardClassifierCustomSerializationRegression(
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardSerializationRegressionTest()
        {
            TestRegressionSerialization<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping),
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(filename, this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests correctness of training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardTrainingTest()
        {
            this.TestDenseBinaryStandardTraining(false);
        }

        /// <summary>
        /// Tests correctness of prediction of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardPredictionTest()
        {
            this.TestDenseStandardPrediction(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, "First", true);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, "First", true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, null, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and empty features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardEmptyFeaturesPredictionTest()
        {
            TestDenseStandardPredictionEmptyFeatures(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardSettingsTest()
        {
            CheckGaussianPriorBinaryClassifierSettings(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping), this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests the support for capabilities of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianBinaryStandardCapabilitiesTest()
        {
            CheckClassifierCapabilities(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests whether the binary Bayes point machine classifier can correctly detect problems with null mappings.
        /// </summary>
        [Fact]
        public void GaussianBinaryStandardNullMappingTest()
        {
            Assert.Throws<ArgumentNullException>(() => BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier<StandardDataset, int, StandardDataset>(null));
        }

        /// <summary>
        /// Tests batched training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseBinaryStandardBatchingRegressionTest()
        {
            TestRegressionBatching<StandardDataset, string, StandardDataset, bool, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping),
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset);
        }

        #endregion

        #region Tests for a binary BPM on standard data and sparse features

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping), this.sparseStandardTrainingData, this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping),
                this.sparseStandardTrainingData,
                OnGaussianPriorBinaryIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardIncrementalTrainingTest()
        {
            this.TestSparseStandardIncrementalTraining(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests prediction of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardPredictionRegressionTest()
        {
            this.TestRegressionBinaryStandardPrediction(this.sparseStandardTrainingData, this.sparseStandardPredictionData, false);
        }

        /// <summary>
        /// Tests evidence computation of an untrained binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests evidence computation of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence<StandardDataset, string, StandardDataset, bool, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                    BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping),
                    this.sparseStandardTrainingData,
                    GaussianPriorBinaryExpectedLogEvidence,
                    batchCount);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping),
                this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardCustomSerializationRegressionTest()
        {
            TestGaussianPriorBinaryStandardClassifierCustomSerializationRegression(
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardSerializationRegressionTest()
        {
            TestRegressionSerialization<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping),
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(filename, this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests correctness of training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardTrainingTest()
        {
            this.TestSparseBinaryStandardTraining(false);
        }

        /// <summary>
        /// Tests correctness of prediction of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardPredictionTest()
        {
            this.TestSparseStandardPrediction(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, "First", true);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, "First", true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, null, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and empty features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardEmptyFeaturesPredictionTest()
        {
            TestSparseStandardPredictionEmptyFeatures(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardSettingsTest()
        {
            CheckGaussianPriorBinaryClassifierSettings(BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping), this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests batched training of the binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseBinaryStandardBatchingRegressionTest()
        {
            TestRegressionBatching<StandardDataset, string, StandardDataset, bool, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping),
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveBernoulliStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveBernoulliStandardDistributions,
                CheckPredictedBernoulliDistributionStandardTestingDataset);
        }

        #endregion

        #region Tests for a multi-class BPM on standard data and dense features

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping), this.denseStandardTrainingData, this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping),
                this.denseStandardTrainingData,
                OnGaussianPriorMulticlassIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardIncrementalTrainingTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping);
            this.TestDenseStandardIncrementalTraining(classifier);

            // Check different labels: Throw
            var standardData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 1) } },
                Labels = new Dictionary<string, string> { { "First", "D" } },
                ClassLabels = new[] { "A", "B", "C", "D" }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(standardData));
        }

        /// <summary>
        /// Tests prediction of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardPredictionRegressionTest()
        {
            this.TestRegressionMulticlassStandardPrediction(this.denseStandardTrainingData, this.denseStandardPredictionData, false);
        }

        /// <summary>
        /// Tests evidence computation of an untrained multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests evidence computation of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence<StandardDataset, string, StandardDataset, int, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                    BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping),
                    this.denseStandardTrainingData,
                    GaussianPriorMulticlassExpectedLogEvidence,
                    batchCount);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping),
                this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights for data in standard format and
        /// features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardCustomSerializationRegressionTest()
        {
            TestGaussianPriorMulticlassStandardClassifierCustomSerializationRegression(
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardSerializationRegressionTest()
        {
            TestRegressionSerialization<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping),
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(filename, this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests correctness of training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardTrainingTest()
        {
            this.TestDenseMulticlassStandardTraining(false);
        }

        /// <summary>
        /// Tests correctness of prediction of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardPredictionTest()
        {
            this.TestDenseStandardPrediction(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, "First", true);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, "First", true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.denseStandardPredictionData, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, null, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and empty features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardEmptyFeaturesPredictionTest()
        {
            TestDenseStandardPredictionEmptyFeatures(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardSettingsTest()
        {
            CheckGaussianPriorMulticlassClassifierSettings(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping), this.denseStandardTrainingData);
        }

        /// <summary>
        /// Tests batched training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianDenseMulticlassStandardBatchingRegressionTest()
        {
            TestRegressionBatching<StandardDataset, string, StandardDataset, int, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping),
                this.denseStandardTrainingData,
                this.denseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset);
        }

        /// <summary>
        /// Tests the support for capabilities of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a dense representation.
        /// </summary>
        [Fact]
        public void GaussianMulticlassStandardCapabilitiesTest()
        {
            CheckClassifierCapabilities(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests whether the multi-class Bayes point machine classifier can correctly detect problems with null mappings.
        /// </summary>
        [Fact]
        public void GaussianMulticlassStandardNullMappingTest()
        {
            Assert.Throws<ArgumentNullException>(() => BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier<StandardDataset, int, StandardDataset>(null));
        }

        #endregion

        #region Tests for a multi-class BPM on standard data and sparse features

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardTrainingRegressionTest()
        {
            TestTraining(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping), this.sparseStandardTrainingData, this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardTrainingIterationChangedRegressionTest()
        {
            TestRegressionTrainingIterationChanged(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping),
                this.sparseStandardTrainingData,
                OnGaussianPriorMulticlassIterationChanged);
        }

        /// <summary>
        /// Tests incremental training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardIncrementalTrainingTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping);
            this.TestSparseStandardIncrementalTraining(classifier);

            // Check different labels: Throw
            var standardData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 1) } },
                Labels = new Dictionary<string, string> { { "First", "D" } },
                ClassLabels = new[] { "A", "B", "C", "D" }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(standardData));
        }

        /// <summary>
        /// Tests prediction of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardPredictionRegressionTest()
        {
            this.TestRegressionMulticlassStandardPrediction(this.sparseStandardTrainingData, this.sparseStandardPredictionData, false);
        }

        /// <summary>
        /// Tests evidence computation of an untrained multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardUntrainedEvidenceTest()
        {
            TestUntrainedEvidence(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests evidence computation of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardEvidenceRegressionTest()
        {
            for (int batchCount = 1; batchCount < 4; batchCount++)
            {
                TestRegressionEvidence<StandardDataset, string, StandardDataset, int, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                    BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping),
                    this.sparseStandardTrainingData,
                    GaussianPriorMulticlassExpectedLogEvidence,
                    batchCount);
            }
        }

        /// <summary>
        /// Tests support of evidence computation of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardUnsupportedEvidenceTest()
        {
            TestUnsupportedEvidence(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping),
                this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardCustomSerializationRegressionTest()
        {
            TestGaussianPriorMulticlassStandardClassifierCustomSerializationRegression(
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset);
        }

        /// <summary>
        /// Tests serialization and deserialization of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardSerializationRegressionTest()
        {
            TestRegressionSerialization<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping),
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset,
                filename => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(filename, this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests correctness of training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardTrainingTest()
        {
            this.TestSparseMulticlassStandardTraining(false);
        }

        /// <summary>
        /// Tests correctness of prediction of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardPredictionTest()
        {
            this.TestSparseStandardPrediction(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests correctness of prediction on an untrained multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardUntrainedPredictionTest()
        {
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, "First", true);

            CheckPredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, "First", true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, this.sparseStandardPredictionData, null, true);
            CheckSinglePredictionException<InvalidOperationException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(
                classifier, null, null, true);
        }

        /// <summary>
        /// Tests correctness of prediction on a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and empty features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardEmptyFeaturesPredictionTest()
        {
            TestSparseStandardPredictionEmptyFeatures(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping));
        }

        /// <summary>
        /// Tests the support for training and prediction settings of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardSettingsTest()
        {
            CheckGaussianPriorMulticlassClassifierSettings(BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping), this.sparseStandardTrainingData);
        }

        /// <summary>
        /// Tests batched training of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions 
        /// over weights for data in standard format and features in a sparse representation.
        /// </summary>
        [Fact]
        public void GaussianSparseMulticlassStandardBatchingRegressionTest()
        {
            TestRegressionBatching<StandardDataset, string, StandardDataset, int, string, IDictionary<string, double>, GaussianBayesPointMachineClassifierTrainingSettings>(
                BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping),
                this.sparseStandardTrainingData,
                this.sparseStandardPredictionData,
                this.gaussianPriorExpectedPredictiveDiscreteStandardDistributions,
                this.gaussianPriorExpectedIncrementalPredictiveDiscreteStandardDistributions,
                CheckPredictedDiscreteDistributionStandardTestingDataset);
        }

        #endregion

        #endregion

        #region Helpers

        /// <summary>
        /// Creates an array of <see cref="Bernoulli"/> distributions from a specified array of mode probabilities.
        /// </summary>
        /// <param name="modeProbabilities">The probabilities of the modes.</param>
        /// <returns>The newly created <see cref="Bernoulli"/> distributions.</returns>
        private static Bernoulli[] CreateBernoulliDistributionsFromProbabilities(IEnumerable<double> modeProbabilities)
        {
            return modeProbabilities.Select(mode => new Bernoulli(mode)).ToArray();
        }

        /// <summary>
        /// Creates an array of <see cref="Discrete"/> distributions from a specified array of mode probabilities.
        /// </summary>
        /// <param name="modeProbabilities">The probabilities of the modes.</param>
        /// <returns>The newly created <see cref="Discrete"/> distributions.</returns>
        private static Discrete[] CreateDiscreteDistributionsFromProbabilities(double[] modeProbabilities)
        {
            Assert.Equal(2, modeProbabilities.Length);
            return new[]
                {
                    new Discrete((1 - modeProbabilities[0]) / 2.0, (1 - modeProbabilities[0]) / 2.0, modeProbabilities[0]),
                    new Discrete((1 - modeProbabilities[1]) / 2.0, modeProbabilities[1], (1 - modeProbabilities[1]) / 2.0)
                };
        }

        /// <summary>
        /// Creates an array of standard distributions from a specified array of <see cref="Bernoulli"/> distributions.
        /// </summary>
        /// <param name="bernoulliDistributions">The <see cref="Bernoulli"/> distributions to wrap.</param>
        /// <returns>The newly created standard distributions.</returns>
        private static BoolPredictiveDistribution[] CreateSimpleDistributionsFromBernoulliDistributions(Bernoulli[] bernoulliDistributions)
        {
            var standardDistributions = new BoolPredictiveDistribution[bernoulliDistributions.Length];
            for (int i = 0; i < bernoulliDistributions.Length; i++)
            {
                standardDistributions[i] = new BoolPredictiveDistribution
                {
                    { false, bernoulliDistributions[i].GetProbFalse() }, 
                    { true, bernoulliDistributions[i].GetProbTrue() }
                };
            }

            return standardDistributions;
        }

        /// <summary>
        /// Creates an array of standard distributions from a specified array of <see cref="Bernoulli"/> distributions.
        /// </summary>
        /// <param name="bernoulliDistributions">The <see cref="Bernoulli"/> distributions to wrap.</param>
        /// <returns>The newly created standard distributions.</returns>
        private static IntPredictiveDistribution[] CreateIntDistributionsFromBernoulliDistributions(Bernoulli[] bernoulliDistributions)
        {
            var standardDistributions = new IntPredictiveDistribution[bernoulliDistributions.Length];
            for (int i = 0; i < bernoulliDistributions.Length; i++)
            {
                standardDistributions[i] = new IntPredictiveDistribution
                {
                    { 0, bernoulliDistributions[i].GetProbFalse() },
                    { 1, bernoulliDistributions[i].GetProbTrue() }
                };
            }

            return standardDistributions;
        }

        /// <summary>
        /// Creates an array of standard distributions from a specified array of <see cref="Discrete"/> distributions.
        /// </summary>
        /// <param name="discreteDistributions">The <see cref="Discrete"/> distributions to wrap.</param>
        /// <returns>The newly created standard distributions.</returns>
        private static IntPredictiveDistribution[] CreateIntDistributionsFromDiscreteDistributions(Discrete[] discreteDistributions)
        {
            var standardDistributions = new IntPredictiveDistribution[discreteDistributions.Length];
            for (int i = 0; i < discreteDistributions.Length; i++)
            {
                Assert.Equal(3, discreteDistributions[i].Dimension);
                standardDistributions[i] = new IntPredictiveDistribution
                {
                    { 0, discreteDistributions[i][0] },
                    { 1, discreteDistributions[i][1] },
                    { 2, discreteDistributions[i][2] }
                };
            }

            return standardDistributions;
        }

        /// <summary>
        /// Creates an array of standard distributions from a specified array of <see cref="Bernoulli"/> distributions.
        /// </summary>
        /// <param name="bernoulliDistributions">The <see cref="Bernoulli"/> distributions to wrap.</param>
        /// <returns>The newly created standard distributions.</returns>
        private static StandardPredictiveDistribution[] CreateStandardDistributionsFromBernoulliDistributions(Bernoulli[] bernoulliDistributions)
        {
            var standardDistributions = new StandardPredictiveDistribution[bernoulliDistributions.Length];
            for (int i = 0; i < bernoulliDistributions.Length; i++)
            {
                standardDistributions[i] = new StandardPredictiveDistribution
                {
                    { "False", bernoulliDistributions[i].GetProbFalse() },
                    { "True", bernoulliDistributions[i].GetProbTrue() }
                };
            }

            return standardDistributions;
        }

        /// <summary>
        /// Creates an array of standard distributions from a specified array of <see cref="Discrete"/> distributions.
        /// </summary>
        /// <param name="discreteDistributions">The <see cref="Discrete"/> distributions to wrap.</param>
        /// <returns>The newly created standard distributions.</returns>
        private static StandardPredictiveDistribution[] CreateStandardDistributionsFromDiscreteDistributions(Discrete[] discreteDistributions)
        {
            var standardDistributions = new StandardPredictiveDistribution[discreteDistributions.Length];
            for (int i = 0; i < discreteDistributions.Length; i++)
            {
                Assert.Equal(3, discreteDistributions[i].Dimension);
                standardDistributions[i] = new StandardPredictiveDistribution
                                               {
                                                   { "A", discreteDistributions[i][0] }, 
                                                   { "B", discreteDistributions[i][1] }, 
                                                   { "C", discreteDistributions[i][2] }
                                               };
            }

            return standardDistributions;
        }

        /// <summary>
        /// Creates a native data format dataset with features in a dense representation 
        /// where one of the features is zero for all instances.
        /// </summary>
        /// <param name="instanceCount">An optional number of instances. Defaults to 200.</param>
        /// <param name="classCount">An optional number of classes. Defaults to 3.</param>
        /// <param name="featureCount">An optional number of features. Defaults to 10.</param>
        /// <returns>The created dataset.</returns>
        private static NativeDataset CreateConstantZeroFeatureDenseNativeDataset(int instanceCount = 200, int classCount = 3, int featureCount = 10)
        {
            Rand.Restart(728);

            var featureValues = new double[instanceCount][];
            for (int instance = 0; instance < instanceCount; instance++)
            {
                featureValues[instance] = new double[featureCount];
                for (int index = 0; index < Math.Round(featureCount / 3.0); index++)
                {
                    featureValues[instance][Rand.Int(1, featureCount)] = 1.0;
                }
            }

            return new NativeDataset
            {
                IsSparse = false,
                FeatureCount = featureCount,
                FeatureValues = featureValues,
                FeatureIndexes = null,
                ClassCount = classCount,
                Labels = Util.ArrayInit(instanceCount, instance => Rand.Int(classCount))
            };
        }

        /// <summary>
        /// Creates a native data format dataset with features in a sparse representation 
        /// where one of the features is zero for all instances.
        /// </summary>
        /// <param name="instanceCount">An optional number of instances. Defaults to 200.</param>
        /// <param name="classCount">An optional number of classes. Defaults to 3.</param>
        /// <param name="featureCount">An optional number of features. Defaults to 10.</param>
        /// <param name="sparsity">
        /// A value between 0 and 1 which determines how many feature values will be present 
        /// per instance, 0 meaning none and 1 meaning all. Defaults to 0.333.
        /// </param>
        /// <param name="addBias">If true, a constant feature is added for each instance. Defaults to false.</param>
        /// <returns>The created dataset.</returns>
        private static NativeDataset CreateConstantZeroFeatureSparseNativeDataset(int instanceCount = 200, int classCount = 3, int featureCount = 10, double sparsity = 0.333, bool addBias = false)
        {
            Rand.Restart(728);

            var featureValues = new double[instanceCount][];
            var featureIndexes = new int[instanceCount][];
            for (int instance = 0; instance < instanceCount; instance++)
            {
                var nonZeroFeatureIndexes = new HashSet<int>();
                for (int i = 0; i < Math.Round(sparsity * featureCount); i++)
                {
                    nonZeroFeatureIndexes.Add(Rand.Int(1, featureCount));
                }

                if (addBias)
                {
                    nonZeroFeatureIndexes.Add(featureCount - 1);
                }

                featureValues[instance] = new double[nonZeroFeatureIndexes.Count];
                featureIndexes[instance] = new int[nonZeroFeatureIndexes.Count];

                int index = 0;
                foreach (int nonZeroFeatureIndex in nonZeroFeatureIndexes)
                {
                    featureIndexes[instance][index] = nonZeroFeatureIndex;
                    featureValues[instance][index] = 1.0;
                    index++;
                }
            }

            return new NativeDataset
            {
                IsSparse = true,
                FeatureCount = featureCount,
                FeatureValues = featureValues,
                FeatureIndexes = featureIndexes,
                ClassCount = classCount,
                Labels = Util.ArrayInit(instanceCount, instance => Rand.Int(classCount))
            };
        }

        /// <summary>
        /// Checks the posterior distributions of the weights.
        /// </summary>
        /// <param name="expectedDistributions">The expected posterior distributions of the weights.</param>
        /// <param name="inferredDistributions">The inferred posterior distributions of the weights.</param>
        private static void CheckWeightsPosteriorDistributions(Gaussian[][] expectedDistributions, IReadOnlyList<IReadOnlyList<Gaussian>> inferredDistributions)
        {
            Assert.Equal(expectedDistributions.Length, inferredDistributions.Count);
            for (var c = 0; c < inferredDistributions.Count; ++c)
            {
                Assert.Equal(expectedDistributions[c].Length, inferredDistributions[c].Count);
                Assert.Equal(inferredDistributions[0].Count, inferredDistributions[c].Count);
                for (int f = 0; f < inferredDistributions[c].Count; f++)
                {
                    Assert.Equal(0, expectedDistributions[c][f].MaxDiff(inferredDistributions[c][f]), Tolerance);
                }
            }
        }

        /// <summary>
        /// Checks the predicted <see cref="Bernoulli"/> distributions over the test set labels.
        /// </summary>
        /// <param name="expectedDistributions">The expected distributions for the native test dataset.</param>
        /// <param name="predictedDistributions">The predicted distributions for the native test dataset.</param>
        /// <param name="tolerance">The maximum tolerated absolute difference between the expected and predicted distributions.</param>
        private static void CheckPredictedBernoulliDistributionNativeTestingDataset(
            IEnumerable<Bernoulli> expectedDistributions, 
            IEnumerable<Bernoulli> predictedDistributions, 
            double tolerance = Tolerance)
        {
            var expectedDistributionsList = expectedDistributions.ToList();
            var predictedDistributionsList = predictedDistributions.ToList();
            Assert.Equal(expectedDistributionsList.Count, predictedDistributionsList.Count);
            var distributions = expectedDistributionsList.Zip(predictedDistributionsList, ValueTuple.Create);
            foreach (var pair in distributions)
            {
                Assert.Equal(pair.Item1.GetProbTrue(), pair.Item2.GetProbTrue(), tolerance);
            }
        }

        /// <summary>
        /// Checks the predicted <see cref="Discrete"/> distributions over the test set labels.
        /// </summary>
        /// <param name="expectedDistributions">The expected distributions for the native test dataset.</param>
        /// <param name="predictedDistributions">The predicted distributions for the native test dataset.</param>
        /// <param name="tolerance">The maximum tolerated absolute difference between the expected and predicted distributions.</param>
        private static void CheckPredictedDiscreteDistributionNativeTestingDataset(
            IEnumerable<Discrete> expectedDistributions,
            IEnumerable<Discrete> predictedDistributions,
            double tolerance = Tolerance)
        {
            var expectedDistributionsList = expectedDistributions.ToList();
            var predictedDistributionsList = predictedDistributions.ToList();
            Assert.Equal(expectedDistributionsList.Count, predictedDistributionsList.Count);
            var distributions = expectedDistributionsList.Zip(predictedDistributionsList, ValueTuple.Create);
            foreach (var pair in distributions)
            {
                Assert.Equal(pair.Item1.Dimension, pair.Item2.Dimension);
                for (var i = 0; i < pair.Item1.Dimension; ++i)
                {
                    Assert.Equal(pair.Item1[i], pair.Item2[i], tolerance);
                }
            }
        }

        /// <summary>
        /// Checks the predicted <see cref="Bernoulli"/> distributions over the test set labels.
        /// </summary>
        /// <param name="expectedDistributions">The expected distributions for the simple test dataset.</param>
        /// <param name="predictedDistributions">The predicted distributions for the simple test dataset.</param>
        /// <param name="tolerance">The maximum tolerated absolute difference between the expected and predicted distributions.</param>
        private static void CheckPredictedBernoulliDistributionSimpleTestingDataset<TLabel>(
            IEnumerable<IDictionary<TLabel, double>> expectedDistributions,
            IEnumerable<IDictionary<TLabel, double>> predictedDistributions,
            double tolerance = Tolerance)
        {
            var expectedDistributionsList = expectedDistributions.ToList();
            var predictedDistributionsList = predictedDistributions.ToList();
            Assert.Equal(expectedDistributionsList.Count, predictedDistributionsList.Count);
            var distributions = expectedDistributionsList.Zip(predictedDistributionsList, ValueTuple.Create);
            foreach (var pair in distributions)
            {
                Assert.Equal(pair.Item1.Count, pair.Item2.Count);
                if (true is TLabel t && false is TLabel f)
                {
                    Assert.Equal(pair.Item1[f], pair.Item2[f], tolerance);
                    Assert.Equal(pair.Item1[t], pair.Item2[t], tolerance);
                }
                else if (0 is TLabel zero && 1 is TLabel one)
                {
                    Assert.Equal(pair.Item1[zero], pair.Item2[zero], tolerance);
                    Assert.Equal(pair.Item1[one], pair.Item2[one], tolerance);
                }
                else throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Checks the predicted <see cref="Bernoulli"/> distributions over the test set labels.
        /// </summary>
        /// <param name="expectedDistributions">The expected distributions for the standard test dataset.</param>
        /// <param name="predictedDistributions">The predicted distributions for the standard test dataset.</param>
        /// <param name="tolerance">The maximum tolerated absolute difference between the expected and predicted distributions.</param>
        private static void CheckPredictedBernoulliDistributionStandardTestingDataset(
            IEnumerable<IDictionary<string, double>> expectedDistributions, 
            IEnumerable<IDictionary<string, double>> predictedDistributions, 
            double tolerance = Tolerance)
        {
            var expectedDistributionsList = expectedDistributions.ToList();
            var predictedDistributionsList = predictedDistributions.ToList();
            Assert.Equal(expectedDistributionsList.Count, predictedDistributionsList.Count);
            var distributions = expectedDistributionsList.Zip(predictedDistributionsList, ValueTuple.Create);
            foreach (var pair in distributions)
            {
                Assert.Equal(pair.Item1.Count, pair.Item2.Count);
                Assert.Equal(pair.Item1["False"], pair.Item2["False"], tolerance);
                Assert.Equal(pair.Item1["True"], pair.Item2["True"], tolerance);
            }
        }

        /// <summary>
        /// Checks the predicted <see cref="Discrete"/> distributions over the test set labels.
        /// </summary>
        /// <param name="expectedDistributions">The expected distributions for the standard test dataset.</param>
        /// <param name="predictedDistributions">The predicted distributions for the standard test dataset.</param>
        /// <param name="tolerance">The maximum tolerated absolute difference between the expected and predicted distributions.</param>
        private static void CheckPredictedDiscreteDistributionStandardTestingDataset(
            IEnumerable<IDictionary<string, double>> expectedDistributions, 
            IEnumerable<IDictionary<string, double>> predictedDistributions, 
            double tolerance = Tolerance)
        {
            var expectedDistributionsList = expectedDistributions.ToList();
            var predictedDistributionsList = predictedDistributions.ToList();
            Assert.Equal(expectedDistributionsList.Count, predictedDistributionsList.Count);
            var distributions = expectedDistributionsList.Zip(predictedDistributionsList, ValueTuple.Create);
            foreach (var pair in distributions)
            {
                Assert.Equal(pair.Item1.Count, pair.Item2.Count);
                Assert.Equal(pair.Item1["A"], pair.Item2["A"], tolerance);
                Assert.Equal(pair.Item1["B"], pair.Item2["B"], tolerance);
                Assert.Equal(pair.Item1["C"], pair.Item2["C"], tolerance);
            }
        }

        /// <summary>
        /// Checks the posterior weights at the final iteration of training of the binary Bayes point machine classifier.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="iterationChangedEventArgs">The information describing the change in iterations.</param>
        private static void OnBinaryIterationChanged(object sender, BayesPointMachineClassifierIterationChangedEventArgs iterationChangedEventArgs)
        {
            int iterationCount = iterationChangedEventArgs.CompletedIterationCount;
            var inferredWeightsDistributions = iterationChangedEventArgs.WeightPosteriorDistributions;

            if (iterationCount == IterationCount)
            {
                var expectedDistribution = new Gaussian(1.0967308978755794, 1.6744100008217033);
                var expectedWeightsDistributions = new[]
                {
                    new[] { expectedDistribution, expectedDistribution }
                };

                CheckWeightsPosteriorDistributions(expectedWeightsDistributions, inferredWeightsDistributions);
            }
        }

        /// <summary>
        /// Checks the posterior weights at the final iteration of training of the binary Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over weights.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="iterationChangedEventArgs">The information describing the change in iterations.</param>
        private static void OnGaussianPriorBinaryIterationChanged(object sender, BayesPointMachineClassifierIterationChangedEventArgs iterationChangedEventArgs)
        {
            int iterationCount = iterationChangedEventArgs.CompletedIterationCount;
            var inferredWeightsDistributions = iterationChangedEventArgs.WeightPosteriorDistributions;

            if (iterationCount == IterationCount)
            {
                var expectedDistribution = new Gaussian(0.5641895835477565, 0.68169011381620925); 
                var expectedWeightsDistributions = new[]
                {
                    new[] { expectedDistribution, expectedDistribution }
                };

                CheckWeightsPosteriorDistributions(expectedWeightsDistributions, inferredWeightsDistributions);
            }
        }

        /// <summary>
        /// Checks the posterior weights at the final iteration of training of the multi-class Bayes point machine classifier.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="iterationChangedEventArgs">The information describing the change in iterations.</param>
        private static void OnMulticlassIterationChanged(object sender, BayesPointMachineClassifierIterationChangedEventArgs iterationChangedEventArgs)
        {
            int iterationCount = iterationChangedEventArgs.CompletedIterationCount;
            var inferredWeightsDistributions = iterationChangedEventArgs.WeightPosteriorDistributions;

            if (iterationCount == IterationCount)
            {
                var expectedWeightsDistributions = new[]
                {
                    new[] { new Gaussian(-0.23956437302835448, 0.46241556332680323), new Gaussian(-0.23956437302835448, 0.46241556332680323) },
                    new[] { new Gaussian(0.47912874605670908, 0.46304603108829834), new Gaussian(-0.23956437302835448, 0.46241556332680323) },
                    new[] { new Gaussian(-0.23956437302835448, 0.46241556332680323), new Gaussian(0.47912874605670908, 0.46304603108829834) }
                };

                CheckWeightsPosteriorDistributions(expectedWeightsDistributions, inferredWeightsDistributions);
            }
        }

        /// <summary>
        /// Checks the posterior weights at the final iteration of training of the multi-class Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over weights.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="iterationChangedEventArgs">The information describing the change in iterations.</param>
        private static void OnGaussianPriorMulticlassIterationChanged(object sender, BayesPointMachineClassifierIterationChangedEventArgs iterationChangedEventArgs)
        {
            int iterationCount = iterationChangedEventArgs.CompletedIterationCount;
            var inferredWeightsDistributions = iterationChangedEventArgs.WeightPosteriorDistributions;

            if (iterationCount == IterationCount)
            {
                var expectedWeightsDistributions = new[]
                {
                    new[] { new Gaussian(-0.28862827663989832, 0.5473392857009084), new Gaussian(-0.28862827663989832, 0.5473392857009084) },
                    new[] { new Gaussian(0.57725655327979652, 0.52309407592243773), new Gaussian(-0.28862827663989832, 0.5473392857009084) },
                    new[] { new Gaussian(-0.28862827663989832, 0.5473392857009084), new Gaussian(0.57725655327979652, 0.52309407592243773) }
                };

                CheckWeightsPosteriorDistributions(expectedWeightsDistributions, inferredWeightsDistributions);
            }
        }

        /// <summary>
        /// Tests each iteration of training of the Bayes point machine classifier.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="iterationChangedEventHandler">The handler for the IterationChanged event.</param>
        private static void TestRegressionTrainingIterationChanged<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> classifier,
            TInstanceSource instanceSource,
            EventHandler<BayesPointMachineClassifierIterationChangedEventArgs> iterationChangedEventHandler)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.IterationChanged += iterationChangedEventHandler;

            classifier.Train(instanceSource);

            classifier.IterationChanged -= iterationChangedEventHandler;
        }

        /// <summary>
        /// Tests training of the Bayes point machine classifier.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        private static void TestTraining<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(
            IPredictor<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution> classifier, 
            TInstanceSource instanceSource,
            TLabelSource labelSource)
        {
            classifier.Train(instanceSource, labelSource);

            // Check no incremental training
            Assert.Throws<InvalidOperationException>(() => classifier.Train(instanceSource));
        }

        /// <summary>
        /// Tests evidence computation of an untrained Bayes point machine classifier.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private static void TestUntrainedEvidence<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Classifier has not been trained yet: Throw
            Assert.Throws<InvalidOperationException>(
                () =>
                {
                    classifier.Settings.Training.ComputeModelEvidence = true;
                    Assert.True(classifier.LogModelEvidence <= 0.0);
                });
        }

        /// <summary>
        /// Tests evidence computation of the Bayes point machine classifier for data in native format.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TNativeLabel">The type of a label in native data format.</typeparam>
        /// <typeparam name="TStandardLabel">The type of a label in standard data format.</typeparam>
        /// <typeparam name="TStandardLabelDistribution">The type of a distribution over labels in standard data format.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="expectedLogEvidence">The expected logarithm of the evidence.</param>
        /// <param name="batchCount">The number of batches to use. Defaults to 1.</param>
        /// <param name="mapping">An optional mapping to the native data format.</param>
        private static void TestRegressionEvidence<TInstanceSource, TInstance, TLabelSource, TNativeLabel, TStandardLabel, TStandardLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TStandardLabel, TStandardLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TStandardLabel>> classifier,
            TInstanceSource instanceSource,
            double expectedLogEvidence,
            int batchCount = 1,
            NativeBayesPointMachineClassifierTestMapping<TNativeLabel> mapping = null)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            classifier.Settings.Training.ComputeModelEvidence = true;
            classifier.Settings.Training.BatchCount = batchCount;
            if (mapping != null)
            {
                mapping.BatchCount = classifier.Settings.Training.BatchCount;
            }
            classifier.Settings.Training.IterationCount = IterationCount;

            // Check model evidence
            classifier.Train(instanceSource);
            Assert.Equal(expectedLogEvidence, classifier.LogModelEvidence, Tolerance);

            // Incremental training: no evidence
            classifier.TrainIncremental(instanceSource);
            Assert.Throws<InvalidOperationException>(() => { double logEvidence = classifier.LogModelEvidence; });

            if (mapping != null)
            {
                mapping.BatchCount = 1;
            }
        }

        /// <summary>
        /// Tests evidence computation of the Bayes point machine classifier for data in native format.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        /// <param name="instanceSource">The instance source.</param>
        private static void TestUnsupportedEvidence<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> classifier,
            TInstanceSource instanceSource)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Check for unsupported evidence computation: Throw
            Assert.Throws<InvalidOperationException>(() =>
            {
                classifier.Train(instanceSource);
                Assert.True(classifier.LogModelEvidence <= 0.0);
            });
        }

        /// <summary>
        /// Tests .NET serialization and deserialization of the Bayes point machine classifier.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestRegressionSerialization<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BayesPointMachineClassifierPredictionSettings<TLabel>> classifier,
            TInstanceSource trainingData,
            TInstanceSource testData,
            IEnumerable<TLabelDistribution> expectedLabelDistributions,
            IEnumerable<TLabelDistribution> expectedIncrementalLabelDistributions,
            Action<IEnumerable<TLabelDistribution>, IEnumerable<TLabelDistribution>, double> checkPrediction,
            Func<string, IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, BayesPointMachineClassifierTrainingSettings, BayesPointMachineClassifierPredictionSettings<TLabel>>> load)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            const string TrainedFileName = "trainedClassifier.bin";
            const string UntrainedFileName = "untrainedClassifier.bin";

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            classifier.SaveForwardCompatible(UntrainedFileName);
            classifier.Train(trainingData);
            classifier.SaveForwardCompatible(TrainedFileName);

            // Deserialize and test
            var trainedClassifier = load(TrainedFileName);
            var untrainedClassifier = load(UntrainedFileName);

            untrainedClassifier.Train(trainingData);

            checkPrediction(expectedLabelDistributions, trainedClassifier.PredictDistribution(testData), Tolerance);
            checkPrediction(expectedLabelDistributions, untrainedClassifier.PredictDistribution(testData), Tolerance);

            // Incremental training
            trainedClassifier.TrainIncremental(trainingData);
            untrainedClassifier.TrainIncremental(trainingData);

            checkPrediction(expectedIncrementalLabelDistributions, trainedClassifier.PredictDistribution(testData), Tolerance);
            checkPrediction(expectedIncrementalLabelDistributions, untrainedClassifier.PredictDistribution(testData), Tolerance);
        }

        #region Custom binary serialization

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier and data in native format.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestBinaryNativeClassifierCustomSerializationRegression(
            NativeDataset trainingData,
            NativeDataset testData,
            IEnumerable<Bernoulli> expectedLabelDistributions,
            IEnumerable<Bernoulli> expectedIncrementalLabelDistributions,
            Action<IEnumerable<Bernoulli>, IEnumerable<Bernoulli>, double> checkPrediction)
        {
            var mapping = new BinaryNativeBayesPointMachineClassifierTestMapping();
            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(mapping);

            const string TrainedFileName = "trainedBinaryNativeClassifier" + extension;
            const string UntrainedFileName = "untrainedBinaryNativeClassifier" + extension;

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            // Save untrained classifier together with its mapping
            mapping.SaveForwardCompatible(UntrainedFileName); 
            classifier.SaveForwardCompatible(UntrainedFileName, FileMode.Append);

            classifier.Train(trainingData);

            // Save trained classifier without its mapping
            classifier.SaveForwardCompatible(TrainedFileName);

            // Check wrong versions throw a serialization exception
            CheckCustomSerializationVersionException(reader => BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(reader, mapping));

            // Deserialize classifier alone
            var trainedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(TrainedFileName, mapping);

            // Deserialize both classifier and mapping
            var (untrainedClassifier, deserializedMapping) = BayesPointMachineClassifier.WithReader(UntrainedFileName, reader =>
            {
                var deserializedMapping1 = new BinaryNativeBayesPointMachineClassifierTestMapping(reader);
                var untrainedClassifier1 = BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(reader, deserializedMapping1);
                return (untrainedClassifier1, deserializedMapping1);
            });

            // Check predictions of deserialized classifiers
            CheckDeserializedNativePrediction(trainedClassifier, untrainedClassifier, deserializedMapping, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
        }

        /// <summary>
        /// Checks that all files with the specified names contain binary Bayes point machine classifiers  
        /// which are backward compatible with data in native format.
        /// </summary>
        /// <param name="fileNames">The file names of the serialized classifiers to test.</param>
        /// <param name="binaryNativeMapping">The mapping to native data format.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckBinaryNativeClassiferCustomSerializationBackwardCompatibility(
            IEnumerable<string> fileNames,
            NativeBayesPointMachineClassifierTestMapping<bool> binaryNativeMapping,
            NativeDataset trainingData,
            NativeDataset testData,
            IEnumerable<Bernoulli> expectedLabelDistributions,
            IEnumerable<Bernoulli> expectedIncrementalLabelDistributions,
            Action<IEnumerable<Bernoulli>, IEnumerable<Bernoulli>, double> checkPrediction)
        {
            foreach (string fileName in fileNames)
            {
                var deserializedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(fileName, binaryNativeMapping);
                CheckDeserializedNativePrediction(deserializedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
            }
        }

        const string extension = ".txt";

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier and data in native format.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestMulticlassNativeClassifierCustomSerializationRegression(
            NativeDataset trainingData,
            NativeDataset testData,
            IEnumerable<Discrete> expectedLabelDistributions,
            IEnumerable<Discrete> expectedIncrementalLabelDistributions,
            Action<IEnumerable<Discrete>, IEnumerable<Discrete>, double> checkPrediction)
        {
            var mapping = new MulticlassNativeBayesPointMachineClassifierTestMapping();
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(mapping);

            const string TrainedFileName = "trainedMulticlassNativeClassifier" + extension;
            const string UntrainedFileName = "untrainedMulticlassNativeClassifier" + extension;

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            // Save untrained classifier together with its mapping
            mapping.SaveForwardCompatible(UntrainedFileName);
            classifier.SaveForwardCompatible(UntrainedFileName, FileMode.Append);

            classifier.Train(trainingData);

            // Save trained classifier without its mapping
            classifier.SaveForwardCompatible(TrainedFileName);

            // Check wrong versions throw a serialization exception
            CheckCustomSerializationVersionException(reader => BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(reader, mapping));

            // Deserialize classifier alone
            var trainedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(TrainedFileName, mapping);

            // Deserialize both classifier and mapping
            var (untrainedClassifier, deserializedMapping) = BayesPointMachineClassifier.WithReader(UntrainedFileName, reader =>
            {
                var deserializedMapping1 = new MulticlassNativeBayesPointMachineClassifierTestMapping(reader);
                var untrainedClassifier1 = BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(reader, deserializedMapping1);
                return (untrainedClassifier1, deserializedMapping1);
            });

            // Check predictions of deserialized classifiers
            CheckDeserializedNativePrediction(trainedClassifier, untrainedClassifier, deserializedMapping, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
        }

        /// <summary>
        /// Checks that all files with the specified names contain multi-class Bayes point machine classifiers  
        /// which are backward compatible with data in native format.
        /// </summary>
        /// <param name="fileNames">The file names of the serialized classifiers to test.</param>
        /// <param name="multiclassNativeMapping">The mapping to native data format.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckMulticlassNativeClassiferCustomSerializationBackwardCompatibility(
            IEnumerable<string> fileNames,
            NativeBayesPointMachineClassifierTestMapping<int> multiclassNativeMapping,
            NativeDataset trainingData,
            NativeDataset testData,
            IEnumerable<Discrete> expectedLabelDistributions,
            IEnumerable<Discrete> expectedIncrementalLabelDistributions,
            Action<IEnumerable<Discrete>, IEnumerable<Discrete>, double> checkPrediction)
        {
            foreach (string fileName in fileNames)
            {
                var deserializedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(fileName, multiclassNativeMapping);
                CheckDeserializedNativePrediction(deserializedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
            }
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier and data in standard format.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestBinarySimpleClassifierCustomSerializationRegression<TLabel>(
            IReadOnlyList<Vector> trainingData,
            IReadOnlyList<TLabel> trainingLabels,
            IReadOnlyList<Vector> testData,
            IEnumerable<IDictionary<TLabel, double>> expectedLabelDistributions,
            IEnumerable<IDictionary<TLabel, double>> expectedIncrementalLabelDistributions,
            Action<IEnumerable<IDictionary<TLabel, double>>, IEnumerable<IDictionary<TLabel, double>>, double> checkPrediction)
        {
            var mapping = new BinarySimpleTestMapping<TLabel>();
            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(mapping);

            const string TrainedFileName = "trainedBinarySimpleClassifier" + extension;
            const string UntrainedFileName = "untrainedBinarySimpleClassifier" + extension;

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            // Save untrained classifier
            classifier.SaveForwardCompatible(UntrainedFileName);

            classifier.Train(trainingData, trainingLabels);

            // Save trained classifier without its mapping
            classifier.SaveForwardCompatible(TrainedFileName);

            // Check wrong versions throw a serialization exception
            CheckCustomSerializationVersionException(reader => BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(reader, mapping));

            // Deserialize classifier alone
            var trainedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(TrainedFileName, mapping);

            // Deserialize both classifier and mapping
            var untrainedClassifier = BayesPointMachineClassifier.WithReader(UntrainedFileName, reader =>
            {
                return BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(reader, mapping);
            });

            // Check predictions of deserialized classifiers
            CheckDeserializedStandardPrediction(trainedClassifier, untrainedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction, trainingLabels);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier and data in standard format.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestBinaryStandardClassifierCustomSerializationRegression(
            StandardDataset trainingData,
            StandardDataset testData,
            IEnumerable<IDictionary<string, double>> expectedLabelDistributions,
            IEnumerable<IDictionary<string, double>> expectedIncrementalLabelDistributions,
            Action<IEnumerable<IDictionary<string, double>>, IEnumerable<IDictionary<string, double>>, double> checkPrediction)
        {
            var mapping = new BinaryStandardBayesPointMachineClassifierTestMapping();
            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(mapping);

            const string TrainedFileName = "trainedBinaryStandardClassifier" + extension;
            const string UntrainedFileName = "untrainedBinaryStandardClassifier" + extension;

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            // Save untrained classifier together with its mapping
            mapping.SaveForwardCompatible(UntrainedFileName);
            classifier.SaveForwardCompatible(UntrainedFileName, FileMode.Append);

            classifier.Train(trainingData);

            // Save trained classifier without its mapping
            classifier.SaveForwardCompatible(TrainedFileName);

            // Check wrong versions throw a serialization exception
            CheckCustomSerializationVersionException(reader => BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(reader, mapping));

            // Deserialize classifier alone
            var trainedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(TrainedFileName, mapping);

            // Deserialize both classifier and mapping
            var untrainedClassifier = BayesPointMachineClassifier.WithReader(UntrainedFileName, reader =>
            {
                var deserializedMapping = new BinaryStandardBayesPointMachineClassifierTestMapping(reader);
                return BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(reader, deserializedMapping);
            });

            // Check predictions of deserialized classifiers
            CheckDeserializedStandardPrediction(trainedClassifier, untrainedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
        }

        /// <summary>
        /// Checks that all files with the specified names contain binary Bayes point machine classifiers  
        /// which are backward compatible with data in standard format.
        /// </summary>
        /// <param name="fileNames">The file names of the serialized classifiers to test.</param>
        /// <param name="binaryStandardMapping">The mapping to standard data format.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckBinaryStandardClassiferCustomSerializationBackwardCompatibility(
            IEnumerable<string> fileNames,
            BinaryStandardBayesPointMachineClassifierTestMapping binaryStandardMapping,
            StandardDataset trainingData,
            StandardDataset testData,
            IEnumerable<IDictionary<string, double>> expectedLabelDistributions,
            IEnumerable<IDictionary<string, double>> expectedIncrementalLabelDistributions,
            Action<IEnumerable<IDictionary<string, double>>, IEnumerable<IDictionary<string, double>>, double> checkPrediction)
        {
            foreach (string fileName in fileNames)
            {
                var deserializedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier(fileName, binaryStandardMapping);
                CheckDeserializedStandardPrediction(deserializedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
            }
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier and data in native format.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestMulticlassSimpleClassifierCustomSerializationRegression<TLabel>(
            IReadOnlyList<Vector> trainingData,
            IReadOnlyList<TLabel> trainingLabels,
            IReadOnlyList<Vector> testData,
            IEnumerable<IDictionary<TLabel, double>> expectedLabelDistributions,
            IEnumerable<IDictionary<TLabel, double>> expectedIncrementalLabelDistributions,
            Action<IEnumerable<IDictionary<TLabel, double>>, IEnumerable<IDictionary<TLabel, double>>, double> checkPrediction)
        {
            var mapping = new MulticlassSimpleBayesPointMachineClassifierTestMapping<TLabel>();
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(mapping);

            const string TrainedFileName = "trainedMulticlassStandardClassifier" + extension;
            const string UntrainedFileName = "untrainedMulticlassStandardClassifier" + extension;

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            // Save untrained classifier
            classifier.SaveForwardCompatible(UntrainedFileName);

            classifier.Train(trainingData, trainingLabels);

            // Save trained classifier without its mapping
            classifier.SaveForwardCompatible(TrainedFileName);

            // Check wrong versions throw a serialization exception
            CheckCustomSerializationVersionException(reader => BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(reader, mapping));

            // Deserialize classifier alone
            var trainedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(TrainedFileName, mapping);

            // Deserialize both classifier and mapping
            var untrainedClassifier = BayesPointMachineClassifier.WithReader(UntrainedFileName, reader =>
            {
                return BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(reader, mapping);
            });

            // Check predictions of deserialized classifiers
            CheckDeserializedStandardPrediction(trainedClassifier, untrainedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction, trainingLabels);
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier and data in native format.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestMulticlassStandardClassifierCustomSerializationRegression(
            StandardDataset trainingData,
            StandardDataset testData,
            IEnumerable<IDictionary<string, double>> expectedLabelDistributions,
            IEnumerable<IDictionary<string, double>> expectedIncrementalLabelDistributions,
            Action<IEnumerable<IDictionary<string, double>>, IEnumerable<IDictionary<string, double>>, double> checkPrediction)
        {
            var mapping = new MulticlassStandardBayesPointMachineClassifierTestMapping();
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(mapping);

            const string TrainedFileName = "trainedMulticlassStandardClassifier" + extension;
            const string UntrainedFileName = "untrainedMulticlassStandardClassifier" + extension;

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            // Save untrained classifier together with its mapping
            mapping.SaveForwardCompatible(UntrainedFileName);
            classifier.SaveForwardCompatible(UntrainedFileName, FileMode.Append);

            classifier.Train(trainingData);

            // Save trained classifier without its mapping
            classifier.SaveForwardCompatible(TrainedFileName);

            // Check wrong versions throw a serialization exception
            CheckCustomSerializationVersionException(reader => BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(reader, mapping));

            // Deserialize classifier alone
            var trainedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(TrainedFileName, mapping);

            // Deserialize both classifier and mapping
            var untrainedClassifier = BayesPointMachineClassifier.WithReader(UntrainedFileName, reader =>
            {
                var deserializedMapping = new MulticlassStandardBayesPointMachineClassifierTestMapping(reader);
                return BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(reader, deserializedMapping);
            });

            // Check predictions of deserialized classifiers
            CheckDeserializedStandardPrediction(trainedClassifier, untrainedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
        }

        /// <summary>
        /// Checks that all files with the specified names contain multi-class Bayes point machine classifiers  
        /// which are backward compatible with data in standard format.
        /// </summary>
        /// <param name="fileNames">The file names of the serialized classifiers to test.</param>
        /// <param name="multiclassStandardMapping">The mapping to standard data format.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckMulticlassStandardClassiferCustomSerializationBackwardCompatibility(
            IEnumerable<string> fileNames,
            MulticlassStandardBayesPointMachineClassifierTestMapping multiclassStandardMapping,
            StandardDataset trainingData,
            StandardDataset testData,
            IEnumerable<IDictionary<string, double>> expectedLabelDistributions,
            IEnumerable<IDictionary<string, double>> expectedIncrementalLabelDistributions,
            Action<IEnumerable<IDictionary<string, double>>, IEnumerable<IDictionary<string, double>>, double> checkPrediction)
        {
            foreach (string fileName in fileNames)
            {
                var deserializedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleMulticlassClassifier(fileName, multiclassStandardMapping);
                CheckDeserializedStandardPrediction(deserializedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
            }
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over weights and data in native format.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestGaussianPriorBinaryNativeClassifierCustomSerializationRegression(
            NativeDataset trainingData,
            NativeDataset testData,
            IEnumerable<Bernoulli> expectedLabelDistributions,
            IEnumerable<Bernoulli> expectedIncrementalLabelDistributions,
            Action<IEnumerable<Bernoulli>, IEnumerable<Bernoulli>, double> checkPrediction)
        {
            var mapping = new BinaryNativeBayesPointMachineClassifierTestMapping();
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(mapping);

            const string TrainedFileName = "trainedGaussianPriorBinaryNativeClassifier" + extension;
            const string UntrainedFileName = "untrainedGaussianPriorBinaryNativeClassifier" + extension;

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            // Save untrained classifier together with its mapping
            mapping.SaveForwardCompatible(UntrainedFileName);
            classifier.SaveForwardCompatible(UntrainedFileName, FileMode.Append);

            classifier.Train(trainingData);

            // Save trained classifier without its mapping
            classifier.SaveForwardCompatible(TrainedFileName);

            // Check wrong versions throw a serialization exception
            CheckCustomSerializationVersionException(reader => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(reader, mapping));

            // Deserialize classifier alone
            var trainedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(TrainedFileName, mapping);

            // Deserialize both classifier and mapping
            var (untrainedClassifier, deserializedMapping) = BayesPointMachineClassifier.WithReader(UntrainedFileName, reader =>
            {
                var deserializedMapping1 = new BinaryNativeBayesPointMachineClassifierTestMapping(reader);
                var untrainedClassifier1 = BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(reader, deserializedMapping1);
                return (untrainedClassifier1, deserializedMapping1);
            });

            // Check predictions of deserialized classifiers
            CheckDeserializedNativePrediction(trainedClassifier, untrainedClassifier, deserializedMapping, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
        }

        /// <summary>
        /// Checks that all files with the specified names contain binary Bayes point machine classifiers  
        /// with <see cref="Gaussian"/> prior distributions over weights which are backward compatible with data in native format.
        /// </summary>
        /// <param name="fileNames">The file names of the serialized classifiers to test.</param>
        /// <param name="binaryNativeMapping">The mapping to native data format.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckGaussianPriorBinaryNativeClassiferCustomSerializationBackwardCompatibility(
            IEnumerable<string> fileNames,
            NativeBayesPointMachineClassifierTestMapping<bool> binaryNativeMapping,
            NativeDataset trainingData,
            NativeDataset testData,
            IEnumerable<Bernoulli> expectedLabelDistributions,
            IEnumerable<Bernoulli> expectedIncrementalLabelDistributions,
            Action<IEnumerable<Bernoulli>, IEnumerable<Bernoulli>, double> checkPrediction)
        {
            foreach (string fileName in fileNames)
            {
                var deserializedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(fileName, binaryNativeMapping);
                CheckDeserializedNativePrediction(deserializedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
            }
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights and data in native format.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestGaussianPriorMulticlassNativeClassifierCustomSerializationRegression(
            NativeDataset trainingData,
            NativeDataset testData,
            IEnumerable<Discrete> expectedLabelDistributions,
            IEnumerable<Discrete> expectedIncrementalLabelDistributions,
            Action<IEnumerable<Discrete>, IEnumerable<Discrete>, double> checkPrediction)
        {
            var mapping = new MulticlassNativeBayesPointMachineClassifierTestMapping();
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(mapping);

            const string TrainedFileName = "trainedGaussianPriorMulticlassNativeClassifier" + extension;
            const string UntrainedFileName = "untrainedGaussianPriorMulticlassNativeClassifier" + extension;

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            // Save untrained classifier together with its mapping
            mapping.SaveForwardCompatible(UntrainedFileName);
            classifier.SaveForwardCompatible(UntrainedFileName, FileMode.Append);

            classifier.Train(trainingData);

            // Save trained classifier without its mapping
            classifier.SaveForwardCompatible(TrainedFileName);

            // Check wrong versions throw a serialization exception
            CheckCustomSerializationVersionException(reader => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(reader, mapping));

            // Deserialize classifier alone
            var trainedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(TrainedFileName, mapping);

            // Deserialize both classifier and mapping
            var (untrainedClassifier, deserializedMapping) = BayesPointMachineClassifier.WithReader(UntrainedFileName, reader =>
            {
                var deserializedMapping1 = new MulticlassNativeBayesPointMachineClassifierTestMapping(reader);
                var untrainedClassifier1 = BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(reader, deserializedMapping1);
                return (untrainedClassifier1, deserializedMapping1);
            });

            // Check predictions of deserialized classifiers
            CheckDeserializedNativePrediction(trainedClassifier, untrainedClassifier, deserializedMapping, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
        }

        /// <summary>
        /// Checks that all files with the specified names contain multi-class Bayes point machine classifiers  
        /// with <see cref="Gaussian"/> prior distributions over weights which are backward compatible with data in native format.
        /// </summary>
        /// <param name="fileNames">The file names of the serialized classifiers to test.</param>
        /// <param name="multiclassNativeMapping">The mapping to native data format.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckGaussianPriorMulticlassNativeClassiferCustomSerializationBackwardCompatibility(
            IEnumerable<string> fileNames,
            NativeBayesPointMachineClassifierTestMapping<int> multiclassNativeMapping,
            NativeDataset trainingData,
            NativeDataset testData,
            IEnumerable<Discrete> expectedLabelDistributions,
            IEnumerable<Discrete> expectedIncrementalLabelDistributions,
            Action<IEnumerable<Discrete>, IEnumerable<Discrete>, double> checkPrediction)
        {
            foreach (string fileName in fileNames)
            {
                var deserializedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(fileName, multiclassNativeMapping);
                CheckDeserializedNativePrediction(deserializedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
            }
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the binary Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights and data in standard format.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestGaussianPriorBinaryStandardClassifierCustomSerializationRegression(
            StandardDataset trainingData,
            StandardDataset testData,
            IEnumerable<IDictionary<string, double>> expectedLabelDistributions,
            IEnumerable<IDictionary<string, double>> expectedIncrementalLabelDistributions,
            Action<IEnumerable<IDictionary<string, double>>, IEnumerable<IDictionary<string, double>>, double> checkPrediction)
        {
            var mapping = new BinaryStandardBayesPointMachineClassifierTestMapping();
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(mapping);

            const string TrainedFileName = "trainedGaussianPriorBinaryStandardClassifier" + extension;
            const string UntrainedFileName = "untrainedGaussianPriorBinaryStandardClassifier" + extension;

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            // Save untrained classifier together with its mapping
            mapping.SaveForwardCompatible(UntrainedFileName);
            classifier.SaveForwardCompatible(UntrainedFileName, FileMode.Append);

            classifier.Train(trainingData);

            // Save trained classifier without its mapping
            classifier.SaveForwardCompatible(TrainedFileName);

            // Check wrong versions throw a serialization exception
            CheckCustomSerializationVersionException(reader => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(reader, mapping));

            // Deserialize classifier alone
            var trainedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(TrainedFileName, mapping);

            // Deserialize both classifier and mapping
            var untrainedClassifier = BayesPointMachineClassifier.WithReader(UntrainedFileName, reader =>
            {
                var deserializedMapping = new BinaryStandardBayesPointMachineClassifierTestMapping(reader);
                return BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(reader, deserializedMapping);
            });

            // Check predictions of deserialized classifiers
            CheckDeserializedStandardPrediction(trainedClassifier, untrainedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
        }

        /// <summary>
        /// Checks that all files with the specified names contain binary Bayes point machine classifiers  
        /// with <see cref="Gaussian"/> prior distributions over weights which are backward compatible with data in standard format.
        /// </summary>
        /// <param name="fileNames">The file names of the serialized classifiers to test.</param>
        /// <param name="binaryStandardMapping">The mapping to standard data format.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckGaussianPriorBinaryStandardClassiferCustomSerializationBackwardCompatibility(
            IEnumerable<string> fileNames,
            BinaryStandardBayesPointMachineClassifierTestMapping binaryStandardMapping,
            StandardDataset trainingData,
            StandardDataset testData,
            IEnumerable<IDictionary<string, double>> expectedLabelDistributions,
            IEnumerable<IDictionary<string, double>> expectedIncrementalLabelDistributions,
            Action<IEnumerable<IDictionary<string, double>>, IEnumerable<IDictionary<string, double>>, double> checkPrediction)
        {
            foreach (string fileName in fileNames)
            {
                var deserializedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorBinaryClassifier(fileName, binaryStandardMapping);
                CheckDeserializedStandardPrediction(deserializedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
            }
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of the multi-class Bayes point machine classifier 
        /// with <see cref="Gaussian"/> prior distributions over weights and data in native format.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void TestGaussianPriorMulticlassStandardClassifierCustomSerializationRegression(
            StandardDataset trainingData,
            StandardDataset testData,
            IEnumerable<IDictionary<string, double>> expectedLabelDistributions,
            IEnumerable<IDictionary<string, double>> expectedIncrementalLabelDistributions,
            Action<IEnumerable<IDictionary<string, double>>, IEnumerable<IDictionary<string, double>>, double> checkPrediction)
        {
            var mapping = new MulticlassStandardBayesPointMachineClassifierTestMapping();
            var classifier = BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(mapping);

            const string TrainedFileName = "trainedGaussianPriorMulticlassStandardClassifier" + extension;
            const string UntrainedFileName = "untrainedGaussianPriorMulticlassStandardClassifier" + extension;

            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;

            // Save untrained classifier together with its mapping
            mapping.SaveForwardCompatible(UntrainedFileName);
            classifier.SaveForwardCompatible(UntrainedFileName, FileMode.Append);

            classifier.Train(trainingData);

            // Save trained classifier without its mapping
            classifier.SaveForwardCompatible(TrainedFileName);

            // Check wrong versions throw a serialization exception
            CheckCustomSerializationVersionException(reader => BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(reader, mapping));

            // Deserialize classifier alone
            var trainedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(TrainedFileName, mapping);

            // Deserialize both classifier and mapping
            var untrainedClassifier = BayesPointMachineClassifier.WithReader(UntrainedFileName, reader =>
            {
                var deserializedMapping = new MulticlassStandardBayesPointMachineClassifierTestMapping(reader);
                return BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(reader, deserializedMapping);
            });

            // Check predictions of deserialized classifiers
            CheckDeserializedStandardPrediction(trainedClassifier, untrainedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
        }

        /// <summary>
        /// Checks that all files with the specified names contain multi-class Bayes point machine classifiers  
        /// with <see cref="Gaussian"/> prior distributions over weights which are backward compatible with data in standard format.
        /// </summary>
        /// <param name="fileNames">The file names of the serialized classifiers to test.</param>
        /// <param name="multiclassStandardMapping">The mapping to standard data format.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckGaussianPriorMulticlassStandardClassiferCustomSerializationBackwardCompatibility(
            IEnumerable<string> fileNames,
            MulticlassStandardBayesPointMachineClassifierTestMapping multiclassStandardMapping,
            StandardDataset trainingData,
            StandardDataset testData,
            IEnumerable<IDictionary<string, double>> expectedLabelDistributions,
            IEnumerable<IDictionary<string, double>> expectedIncrementalLabelDistributions,
            Action<IEnumerable<IDictionary<string, double>>, IEnumerable<IDictionary<string, double>>, double> checkPrediction)
        {
            foreach (string fileName in fileNames)
            {
                var deserializedClassifier = BayesPointMachineClassifier.LoadBackwardCompatibleGaussianPriorMulticlassClassifier(fileName, multiclassStandardMapping);
                CheckDeserializedStandardPrediction(deserializedClassifier, trainingData, testData, expectedLabelDistributions, expectedIncrementalLabelDistributions, checkPrediction);
            }
        }

        /// <summary>
        /// Checks the predictions on data in native format after custom binary deserialization of Bayes point machine classifiers.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="trainedClassifier">A trained Bayes point machine classifier.</param>
        /// <param name="untrainedClassifier">An untrained Bayes point machine classifier.</param>
        /// <param name="deserializedMapping">The deserialized mapping to data in native format.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckDeserializedNativePrediction<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> trainedClassifier,
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> untrainedClassifier,
            NativeBayesPointMachineClassifierTestMapping<TLabel> deserializedMapping,
            TInstanceSource trainingData,
            TInstanceSource testData,
            IEnumerable<TLabelDistribution> expectedLabelDistributions,
            IEnumerable<TLabelDistribution> expectedIncrementalLabelDistributions,
            Action<IEnumerable<TLabelDistribution>, IEnumerable<TLabelDistribution>, double> checkPrediction)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Check that trained classifier still protects settings
            Assert.Equal(IterationCount, trainedClassifier.Settings.Training.IterationCount);
            Assert.Equal(IterationCount, untrainedClassifier.Settings.Training.IterationCount);
            Assert.Throws<InvalidOperationException>(() => { trainedClassifier.Settings.Training.ComputeModelEvidence = false; }); // Guarded: Throw
            untrainedClassifier.Settings.Training.ComputeModelEvidence = false;

            // Batched training for untrained classifier
            untrainedClassifier.Settings.Training.BatchCount = 2;
            deserializedMapping.BatchCount = untrainedClassifier.Settings.Training.BatchCount;
            untrainedClassifier.Train(trainingData);
            untrainedClassifier.Settings.Training.BatchCount = 1;
            deserializedMapping.BatchCount = untrainedClassifier.Settings.Training.BatchCount;

            checkPrediction(expectedLabelDistributions, trainedClassifier.PredictDistribution(testData), Tolerance);
            checkPrediction(expectedLabelDistributions, untrainedClassifier.PredictDistribution(testData), Tolerance);

            // Incremental training
            trainedClassifier.TrainIncremental(trainingData);
            untrainedClassifier.TrainIncremental(trainingData);

            checkPrediction(expectedIncrementalLabelDistributions, trainedClassifier.PredictDistribution(testData), Tolerance);
            checkPrediction(expectedIncrementalLabelDistributions, untrainedClassifier.PredictDistribution(testData), Tolerance);
        }

        /// <summary>
        /// Checks the predictions on data in native format after custom binary deserialization of Bayes point machine classifiers.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="trainedClassifier">A trained Bayes point machine classifier.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckDeserializedNativePrediction<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> trainedClassifier,
            TInstanceSource trainingData,
            TInstanceSource testData,
            IEnumerable<TLabelDistribution> expectedLabelDistributions,
            IEnumerable<TLabelDistribution> expectedIncrementalLabelDistributions,
            Action<IEnumerable<TLabelDistribution>, IEnumerable<TLabelDistribution>, double> checkPrediction)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Check that trained classifier still protects settings
            Assert.Equal(IterationCount, trainedClassifier.Settings.Training.IterationCount);
            Assert.Throws<InvalidOperationException>(() => { trainedClassifier.Settings.Training.ComputeModelEvidence = false; }); // Guarded: Throw

            checkPrediction(expectedLabelDistributions, trainedClassifier.PredictDistribution(testData), Tolerance);

            // Incremental training
            trainedClassifier.TrainIncremental(trainingData);

            checkPrediction(expectedIncrementalLabelDistributions, trainedClassifier.PredictDistribution(testData), Tolerance);
        }

        /// <summary>
        /// Checks the predictions on data in standard format after custom binary deserialization of Bayes point machine classifiers.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="trainedClassifier">A trained Bayes point machine classifier.</param>
        /// <param name="untrainedClassifier">An untrained Bayes point machine classifier.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckDeserializedStandardPrediction<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> trainedClassifier,
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> untrainedClassifier,
            TInstanceSource trainingData,
            TInstanceSource testData,
            IEnumerable<TLabelDistribution> expectedLabelDistributions,
            IEnumerable<TLabelDistribution> expectedIncrementalLabelDistributions,
            Action<IEnumerable<TLabelDistribution>, IEnumerable<TLabelDistribution>, double> checkPrediction, 
            TLabelSource labelSource = default)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Check that trained classifier still protects settings
            Assert.Equal(IterationCount, trainedClassifier.Settings.Training.IterationCount);
            Assert.Equal(IterationCount, untrainedClassifier.Settings.Training.IterationCount);
            Assert.Throws<InvalidOperationException>(() => { trainedClassifier.Settings.Training.ComputeModelEvidence = false; }); // Guarded: Throw
            untrainedClassifier.Settings.Training.ComputeModelEvidence = false;

            // Batched training for untrained classifier
            untrainedClassifier.Settings.Training.BatchCount = 2;
            untrainedClassifier.Train(trainingData, labelSource);

            checkPrediction(expectedLabelDistributions, trainedClassifier.PredictDistribution(testData), Tolerance);
            checkPrediction(expectedLabelDistributions, untrainedClassifier.PredictDistribution(testData), Tolerance);

            // Incremental training
            trainedClassifier.TrainIncremental(trainingData, labelSource);
            untrainedClassifier.TrainIncremental(trainingData, labelSource);

            checkPrediction(expectedIncrementalLabelDistributions, trainedClassifier.PredictDistribution(testData), Tolerance);
            checkPrediction(expectedIncrementalLabelDistributions, untrainedClassifier.PredictDistribution(testData), Tolerance);
        }

        /// <summary>
        /// Checks the predictions on data in standard format after custom binary deserialization of Bayes point machine classifiers.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="trainedClassifier">A trained Bayes point machine classifier.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        private static void CheckDeserializedStandardPrediction<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> trainedClassifier,
            TInstanceSource trainingData,
            TInstanceSource testData,
            IEnumerable<TLabelDistribution> expectedLabelDistributions,
            IEnumerable<TLabelDistribution> expectedIncrementalLabelDistributions,
            Action<IEnumerable<TLabelDistribution>, IEnumerable<TLabelDistribution>, double> checkPrediction)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Check that trained classifier still protects settings
            Assert.Equal(IterationCount, trainedClassifier.Settings.Training.IterationCount);
            Assert.Throws<InvalidOperationException>(() => { trainedClassifier.Settings.Training.ComputeModelEvidence = false; }); // Guarded: Throw

            checkPrediction(expectedLabelDistributions, trainedClassifier.PredictDistribution(testData), Tolerance);

            // Incremental training
            trainedClassifier.TrainIncremental(trainingData);

            checkPrediction(expectedIncrementalLabelDistributions, trainedClassifier.PredictDistribution(testData), Tolerance);
        }

        /// <summary>
        /// Checks that the specified deserialization action throws a serialization exception for invalid versions.
        /// </summary>
        /// <param name="deserialize">The action which deserializes from a binary reader.</param>
        private static void CheckCustomSerializationVersionException(Action<IReader> deserialize)
        {
            using (var stream = new MemoryStream())
            {
                var writer = new WrappedBinaryWriter(new BinaryWriter(stream));
                writer.Write(new Guid("57CB3182-7C83-49F3-B56D-C3C7661439F5")); // Invalid serialization guid
                
                stream.Seek(0, SeekOrigin.Begin);

                using (var reader = new WrappedBinaryReader(new BinaryReader(stream)))
                {
                    Assert.Throws<SerializationException>(() => deserialize(reader));
                }
            }
        }

        #endregion

        /// <summary>
        /// Tests batching of the Bayes point machine classifier.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TNativeLabel">The type of a label in native data format.</typeparam>
        /// <typeparam name="TStandardLabel">The type of a label in standard data format.</typeparam>
        /// <typeparam name="TStandardLabelDistribution">The type of a distribution over labels in standard data format.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        /// <param name="trainingData">The training data.</param>
        /// <param name="testData">The prediction data.</param>
        /// <param name="expectedLabelDistributions">The expected label distributions.</param>
        /// <param name="expectedIncrementalLabelDistributions">The expected label distributions for incremental training.</param>
        /// <param name="checkPrediction">A method which asserts the equality of expected and predicted distributions.</param>
        /// <param name="mapping">An optional mapping into native data format.</param>
        private static void TestRegressionBatching<TInstanceSource, TInstance, TLabelSource, TNativeLabel, TStandardLabel, TStandardLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TStandardLabel, TStandardLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TStandardLabel>> classifier,
            TInstanceSource trainingData,
            TInstanceSource testData,
            IEnumerable<TStandardLabelDistribution> expectedLabelDistributions,
            IEnumerable<TStandardLabelDistribution> expectedIncrementalLabelDistributions,
            Action<IEnumerable<TStandardLabelDistribution>, IEnumerable<TStandardLabelDistribution>, double> checkPrediction,
            NativeBayesPointMachineClassifierTestMapping<TNativeLabel> mapping = null)
            where TInstanceSource : IDataset
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Train and serialize
            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.Settings.Training.BatchCount = 3;
            if (mapping != null)
            {
                mapping.BatchCount = classifier.Settings.Training.BatchCount;
            }

            classifier.Train(trainingData);

            // Check predictions (no batching)
            if (mapping != null)
            {
                mapping.BatchCount = 1;
            }

            checkPrediction(expectedLabelDistributions, classifier.PredictDistribution(testData), Tolerance);
            Assert.Equal(trainingData.InstanceCount, classifier.Predict(trainingData).Count());

            // Incremental training
            classifier.Settings.Training.BatchCount = 2;
            if (mapping != null)
            {
                mapping.BatchCount = classifier.Settings.Training.BatchCount;
            }

            classifier.TrainIncremental(trainingData);

            // Check predictions (no batching)
            if (mapping != null)
            {
                mapping.BatchCount = 1;
            }

            checkPrediction(expectedIncrementalLabelDistributions, classifier.PredictDistribution(testData), Tolerance);
            Assert.Equal(trainingData.InstanceCount, classifier.Predict(trainingData).Count());

            // Too few instances for requested number of batches
            classifier.Settings.Training.BatchCount = 4;
            if (mapping != null)
            {
                mapping.BatchCount = classifier.Settings.Training.BatchCount;
            }

            Assert.Throws<ArgumentException>(() => classifier.TrainIncremental(trainingData));

            if (mapping != null)
            {
                mapping.BatchCount = 1;
            }
        }

        /// <summary>
        /// Tests if classifier has the correct exception behavior in prediction.
        /// </summary>
        /// <typeparam name="TException">The type of exception expected to be thrown.</typeparam>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="mustThrow">Indicates whether the tested code must throw an exception of type <typeparamref name="TException"/>. Defaults to true.</param>
        private static void CheckPredictionException<TException, TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(
            IPredictor<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution> classifier,
            TInstanceSource instanceSource,
            bool mustThrow) where TException : Exception
        {
            if (mustThrow)
            {
                Assert.Throws<TException>(() => classifier.PredictDistribution(instanceSource));
                Assert.Throws<TException>(() => classifier.Predict(instanceSource));
            }
            else
            {
                classifier.PredictDistribution(instanceSource);
            }
        }

        /// <summary>
        /// Tests if classifier has the correct exception behavior in predicting a single instance.
        /// </summary>
        /// <typeparam name="TException">The type of the expected exception.</typeparam>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="instance">The instance.</param>
        /// <param name="mustThrow">Indicates whether the tested code must throw an exception of type <typeparamref name="TException"/>. Defaults to true.</param>
        private static void CheckSinglePredictionException<TException, TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(
            IPredictor<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution> classifier,
            TInstanceSource instanceSource, 
            TInstance instance,
            bool mustThrow) where TException : Exception
        {
            if (mustThrow)
            {
                Assert.Throws<TException>(() => classifier.PredictDistribution(instance, instanceSource));
                Assert.Throws<TException>(() => classifier.Predict(instance, instanceSource));
            }
            else
            {
                classifier.PredictDistribution(instance, instanceSource);
                classifier.Predict(instance, instanceSource);
            }
        }

        /// <summary>
        /// Tests prediction of a Bayes point machine classifier on empty features in dense native format.
        /// </summary>
        /// <typeparam name="TLabel">The type of the label in native data format.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of the distribution over labels in native data format.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private static void TestDenseNativePredictionEmptyFeatures<TLabel, TLabelDistribution>(
            IPredictor<NativeDataset, int, NativeDataset, TLabel, TLabelDistribution> classifier)
        {
            // Train on empty feature vectors
            var data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 0,
                FeatureValues = new[] { new double[] { }, new double[] { } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            classifier.Train(data);

            // Prediction on empty feature values: OK
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);
        }

        /// <summary>
        /// Tests prediction of a Bayes point machine classifier on empty features in sparse native format.
        /// </summary>
        /// <typeparam name="TLabel">The type of the label in native data format.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of the distribution over labels in native data format.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private static void TestSparseNativePredictionEmptyFeatures<TLabel, TLabelDistribution>(
            IPredictor<NativeDataset, int, NativeDataset, TLabel, TLabelDistribution> classifier)
        {
            // Train on empty feature vectors
            var data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 0,
                FeatureValues = new[] { new double[] { }, new double[] { } },
                FeatureIndexes = new[] { new int[] { }, new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            classifier.Train(data);

            // Prediction on empty feature values: OK
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);
        }

        /// <summary>
        /// Tests prediction of a Bayes point machine classifier on empty features in dense standard format.
        /// </summary>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private static void TestDenseStandardPredictionEmptyFeatures(
            IPredictor<StandardDataset, string, StandardDataset, string, IDictionary<string, double>> classifier)
        {
            // Train on empty feature vectors
            var data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray() }, { "Second", DenseVector.FromArray() } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "B" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            classifier.Train(data);

            // Prediction on empty feature values: OK
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);
        }

        /// <summary>
        /// Tests prediction of a Bayes point machine classifier on empty features in sparse standard format.
        /// </summary>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private static void TestSparseStandardPredictionEmptyFeatures(
            IPredictor<StandardDataset, string, StandardDataset, string, IDictionary<string, double>> classifier)
        {
            // Train on empty feature vectors
            var data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray() }, { "Second", SparseVector.FromArray() } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "B" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            classifier.Train(data);

            // Prediction on empty feature values: OK
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);
        }

        /// <summary>
        /// Tests the settings of a binary Bayes point machine classifier.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The classifier to test.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        private static void CheckBinaryClassifierSettings<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>> classifier,
            TInstanceSource instanceSource,
            TLabelSource labelSource = default(TLabelSource))
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Before training
            classifier.Settings.Training.ComputeModelEvidence = false; // OK
            classifier.Settings.Training.IterationCount = 5; // OK
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.IterationCount = -1; }); // Negative iterations: Throw
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.IterationCount = 0; }); // No iterations: Throw

            classifier.Settings.Training.BatchCount = 1; // OK
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.BatchCount = -1; }); // Negative batches: Throw
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.BatchCount = 0; }); // No iterations: Throw

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Absolute); // OK
            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.ZeroOneError); // OK
            Assert.Throws<ArgumentNullException>(
                 () => classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom)); // No custom loss function: Throw
            Assert.Throws<InvalidOperationException>(
                 () => classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Absolute, Metrics.ZeroOneError)); // Inconsistent settings: Throw

            IReadOnlyList<IReadOnlyList<Gaussian>> weightPosteriorDistributions;
            Assert.Throws<InvalidOperationException>(() => { weightPosteriorDistributions = classifier.WeightPosteriorDistributions; }); // Posterior weights distribution: Throw

            // Training
            classifier.Train(instanceSource, labelSource);

            // After training
            Assert.Throws<InvalidOperationException>(() => { classifier.Settings.Training.ComputeModelEvidence = false; }); // Guarded: Throw

            classifier.Settings.Training.IterationCount = 3; // OK
            classifier.Settings.Training.BatchCount = 3; // OK
            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.ZeroOne); // OK
            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.ZeroOneError); // OK
            weightPosteriorDistributions = classifier.WeightPosteriorDistributions; // OK
            Assert.True(weightPosteriorDistributions != null);
        }

        /// <summary>
        /// Tests the settings of a binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over weights.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="classifier">The classifier to test.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        private static void CheckGaussianPriorBinaryClassifierSettings<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>> classifier,
            TInstanceSource instanceSource,
            TLabelSource labelSource = default(TLabelSource))
        {
            // Before training
            classifier.Settings.Training.Advanced.WeightPriorVariance = 5; // OK
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.Advanced.WeightPriorVariance = -1; }); // Negative variance: Throw
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.Advanced.WeightPriorVariance = double.PositiveInfinity; }); // Variance is infinite: Throw
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.Advanced.WeightPriorVariance = double.NaN; }); // Variance is not a number: Throw

            // Check other settings
            CheckBinaryClassifierSettings(classifier, instanceSource, labelSource);

            // Aftertraining
            Assert.Throws<InvalidOperationException>(() => { classifier.Settings.Training.Advanced.WeightPriorVariance = 2.0; }); // Guarded: Throw
        }

        /// <summary>
        /// Tests the settings of a multi-class Bayes point machine classifier.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The classifier to test.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        private static void CheckMulticlassClassifierSettings<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>> classifier,
            TInstanceSource instanceSource,
            TLabelSource labelSource = default(TLabelSource))
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            classifier.Settings.Training.ComputeModelEvidence = false; // OK
            classifier.Settings.Training.IterationCount = 5; // OK
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.IterationCount = -1; }); // Negative iterations: Throw
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.IterationCount = 0; }); // No iterations: Throw

            classifier.Settings.Training.BatchCount = 1; // OK
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.BatchCount = -1; }); // Negative batches: Throw
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.BatchCount = 0; }); // No iterations: Throw

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Absolute); // OK
            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.ZeroOneError); // OK
            Assert.Throws<ArgumentNullException>(
                 () => classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom)); // No custom loss function: Throw
            Assert.Throws<InvalidOperationException>(
                 () => classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Absolute, Metrics.ZeroOneError)); // Inconsistent settings: Throw

            IReadOnlyList<IReadOnlyList<Gaussian>> weightPosteriorDistributions;
            Assert.Throws<InvalidOperationException>(() => { weightPosteriorDistributions = classifier.WeightPosteriorDistributions; }); // Posterior weights distribution: Throw

            // Training
            classifier.Train(instanceSource, labelSource);

            // After training
            Assert.Throws<InvalidOperationException>(() => { classifier.Settings.Training.ComputeModelEvidence = false; }); // Guarded: Throw

            classifier.Settings.Training.IterationCount = 3; // OK
            classifier.Settings.Training.BatchCount = 3; // OK
            classifier.Settings.Prediction.IterationCount = 5; // OK
            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.ZeroOne); // OK
            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.ZeroOneError); // OK
            weightPosteriorDistributions = classifier.WeightPosteriorDistributions; // OK
            Assert.True(weightPosteriorDistributions != null);
        }

        /// <summary>
        /// Tests the settings of a multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over weights.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <param name="classifier">The classifier to test.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        private static void CheckGaussianPriorMulticlassClassifierSettings<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>> classifier,
            TInstanceSource instanceSource,
            TLabelSource labelSource = default(TLabelSource))
        {
            // Before training
            classifier.Settings.Training.Advanced.WeightPriorVariance = 5; // OK
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.Advanced.WeightPriorVariance = -1; }); // Negative variance: Throw
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.Advanced.WeightPriorVariance = double.PositiveInfinity; }); // Variance is infinite: Throw
            Assert.Throws<ArgumentOutOfRangeException>(
                 () => { classifier.Settings.Training.Advanced.WeightPriorVariance = double.NaN; }); // Variance is not a number: Throw

            // Check other settings
            CheckMulticlassClassifierSettings(classifier, instanceSource, labelSource);

            // Aftertraining
            Assert.Throws<InvalidOperationException>(() => { classifier.Settings.Training.Advanced.WeightPriorVariance = 2.0; }); // Guarded: Throw
        }

        /// <summary>
        /// The check classifier capabilities.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The classifier to test.</param>
        private static void CheckClassifierCapabilities<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // ILearner
            Assert.True(classifier.Capabilities.IsPrecompiled);
            Assert.False(classifier.Capabilities.SupportsMissingData);
            Assert.True(classifier.Capabilities.SupportsSparseData);
            Assert.False(classifier.Capabilities.SupportsStreamedData);
            Assert.True(classifier.Capabilities.SupportsBatchedTraining);
            Assert.False(classifier.Capabilities.SupportsDistributedTraining);
            Assert.True(classifier.Capabilities.SupportsIncrementalTraining);
            Assert.True(classifier.Capabilities.SupportsModelEvidenceComputation);

            // IClassifier
            Assert.True(classifier.Capabilities.SupportsCustomPredictionLossFunction);

            Assert.True(((ILearner)classifier).Capabilities.IsPrecompiled);
        }

        /// <summary>
        /// Tests if the binary native data format classifier has the correct exception behavior in training.
        /// </summary>
        /// <typeparam name="TException">The type of exception expected to be thrown.</typeparam>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
        /// <typeparam name="TLabel">The type of a label in standard data format.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of a distribution over labels in standard data format.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="mustThrow">Indicates whether the tested code must throw an exception of type <typeparamref name="TException"/>.</param>
        private static void CheckTrainException<TException, TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> classifier,
            TInstanceSource instanceSource,
            bool mustThrow)
            where TException : Exception
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            if (mustThrow)
            {
                Assert.Throws<TException>(() => classifier.Train(instanceSource));
                Assert.Throws<TException>(() => classifier.TrainIncremental(instanceSource));
            }
            else
            {
                classifier.Train(instanceSource);
                classifier.TrainIncremental(instanceSource);
            }
        }

        /// <summary>
        /// Tests incremental training of the Bayes point machine classifier for data in native format and
        /// features in a dense representation.
        /// </summary>
        /// <typeparam name="TLabel">The type of the label in native data format.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of the distribution over labels in native data format.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private void TestDenseNativeIncrementalTraining<TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<NativeDataset, int, NativeDataset, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.TrainIncremental(this.denseNativeTrainingData);

            classifier.Settings.Training.IterationCount = 1;
            classifier.TrainIncremental(this.denseNativeTrainingData);

            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.TrainIncremental(this.denseNativeTrainingData, this.denseNativeTrainingData);

            // Incremental training using Train: Throw
            var trainedClassifier = classifier;
            Assert.Throws<InvalidOperationException>(() => trainedClassifier.Train(this.denseNativeTrainingData));

            // Check different number of instances: OK
            var nativeData = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 0.0, 1.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            classifier.TrainIncremental(nativeData);

            // Check different number of features: Throw
            nativeData = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(nativeData));
        }

        /// <summary>
        /// Tests incremental training of the Bayes point machine classifier for data in native format and
        /// features in a sparse representation.
        /// </summary>
        /// <typeparam name="TLabel">The type of the label in native data format.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of the distribution over labels in native data format.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private void TestSparseNativeIncrementalTraining<TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<NativeDataset, int, NativeDataset, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.TrainIncremental(this.sparseNativeTrainingData);

            classifier.Settings.Training.IterationCount = 1;
            classifier.TrainIncremental(this.sparseNativeTrainingData);

            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.TrainIncremental(this.sparseNativeTrainingData, this.sparseNativeTrainingData);

            // Incremental training using Train: Throw
            var trainedClassifier = classifier;
            Assert.Throws<InvalidOperationException>(() => trainedClassifier.Train(this.sparseNativeTrainingData));

            // Check different number of instances: OK
            var nativeData = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 1.0 } },
                FeatureIndexes = new[] { new[] { 1 } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            classifier.TrainIncremental(nativeData);

            // Check different labels: Throw
            nativeData = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 0.0, 1.0 } },
                FeatureIndexes = new[] { new[] { 1 } },
                ClassCount = 4,
                Labels = new[] { 4 }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(nativeData));

            // Check different number of features: Throw
            nativeData = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(nativeData));
        }

        /// <summary>
        /// Tests incremental training of the Bayes point machine classifier for data in standard format and
        /// features in a dense representation.
        /// </summary>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private void TestDenseStandardIncrementalTraining<TTrainingSettings>(
            IBayesPointMachineClassifier<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<string>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.TrainIncremental(this.denseStandardTrainingData);

            classifier.Settings.Training.IterationCount = 1;
            classifier.TrainIncremental(this.denseStandardTrainingData);

            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.TrainIncremental(this.denseStandardTrainingData, this.denseStandardTrainingData);

            // Incremental training using Train: Throw
            var trainedClassifier = classifier;
            Assert.Throws<InvalidOperationException>(() => trainedClassifier.Train(this.denseStandardTrainingData));

            // Check different number of instances: OK
            var standardData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 1) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            classifier.TrainIncremental(standardData);

            // Check different number of features: Throw
            standardData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(standardData));
        }

        /// <summary>
        /// Tests incremental training of the Bayes point machine classifier for data in standard format and
        /// features in a sparse representation.
        /// </summary>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private void TestSparseStandardIncrementalTraining<TTrainingSettings>(
            IBayesPointMachineClassifier<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<string>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.TrainIncremental(this.sparseStandardTrainingData);

            classifier.Settings.Training.IterationCount = 1;
            classifier.TrainIncremental(this.sparseStandardTrainingData);

            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.TrainIncremental(this.sparseStandardTrainingData, this.sparseStandardTrainingData);

            // Incremental training using Train: Throw
            var trainedClassifier = classifier;
            Assert.Throws<InvalidOperationException>(() => trainedClassifier.Train(this.sparseStandardTrainingData));

            // Check different number of instances: OK
            var standardData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 1) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            classifier.TrainIncremental(standardData);

            // Check different number of features: Throw
            standardData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            Assert.Throws<BayesPointMachineClassifierException>(() => classifier.TrainIncremental(standardData));
        }

        /// <summary>
        /// Tests prediction of the binary Bayes point machine classifier on data in native format.
        /// </summary>
        /// <param name="nativeTrainingData">The training set in native data format.</param>
        /// <param name="nativePredictionData">The prediction data set in native data format.</param>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestRegressionBinaryNativePrediction(NativeDataset nativeTrainingData, NativeDataset nativePredictionData, bool useCompoundWeightPriorDistribution = true)
        {
            var classifier = useCompoundWeightPriorDistribution
                                 ? BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping)
                                 : BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping);

            var expectedDistributions = useCompoundWeightPriorDistribution
                                                              ? this.expectedPredictiveBernoulliDistributions
                                                              : this.gaussianPriorExpectedPredictiveBernoulliDistributions;

            classifier.Settings.Training.IterationCount = IterationCount;

            classifier.Train(nativeTrainingData);

            // Check bulk predictions
            var predictedDistributions = classifier.PredictDistribution(nativePredictionData);
            CheckPredictedBernoulliDistributionNativeTestingDataset(expectedDistributions, predictedDistributions);

            // Check singleton predictions
            predictedDistributions = new[] { classifier.PredictDistribution(0, nativePredictionData), classifier.PredictDistribution(1, nativePredictionData) };
            CheckPredictedBernoulliDistributionNativeTestingDataset(expectedDistributions, predictedDistributions);

            // Check point estimates
            Assert.True(classifier.Predict(0, nativePredictionData));
            Assert.True(classifier.Predict(nativePredictionData).First());
            Assert.Equal(nativePredictionData.BinaryLabels[0], classifier.Predict(0, nativePredictionData));

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Squared);
            Assert.True(classifier.Predict(0, nativePredictionData));
            Assert.True(classifier.Predict(nativePredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Absolute);
            Assert.True(classifier.Predict(0, nativePredictionData));
            Assert.True(classifier.Predict(nativePredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.ZeroOneError);
            Assert.True(classifier.Predict(0, nativePredictionData));
            Assert.True(classifier.Predict(nativePredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.SquaredError);
            Assert.True(classifier.Predict(0, nativePredictionData));
            Assert.True(classifier.Predict(nativePredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.AbsoluteError);
            Assert.True(classifier.Predict(0, nativePredictionData));
            Assert.True(classifier.Predict(nativePredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Squared);
            foreach (var estimate in classifier.Predict(nativePredictionData))
            {
                Assert.True(estimate);
            }

            // ILearner settings
            Assert.NotNull(((ILearner)classifier).Settings);

            // Check dimensions of posterior distributions over weights
            Assert.Equal(1, classifier.WeightPosteriorDistributions.Count);
            Assert.Equal(nativePredictionData.FeatureCount, classifier.WeightPosteriorDistributions[0].Count);
        }

        /// <summary>
        /// Tests prediction of the multi-class Bayes point machine classifier on data in native format.
        /// </summary>
        /// <param name="nativeTrainingData">The training set in native data format.</param>
        /// <param name="nativePredictionData">The prediction data set in native data format.</param>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestRegressionMulticlassNativePrediction(NativeDataset nativeTrainingData, NativeDataset nativePredictionData, bool useCompoundWeightPriorDistribution = true)
        {
            var classifier = useCompoundWeightPriorDistribution
                                 ? BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping)
                                 : BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping);

            var expectedDistributions = useCompoundWeightPriorDistribution
                                                              ? this.expectedPredictiveDiscreteDistributions
                                                              : this.gaussianPriorExpectedPredictiveDiscreteDistributions;

            classifier.Settings.Training.IterationCount = IterationCount;

            classifier.Train(nativeTrainingData);

            // Check bulk predictions
            var predictedDistributions = classifier.PredictDistribution(nativePredictionData);
            CheckPredictedDiscreteDistributionNativeTestingDataset(expectedDistributions, predictedDistributions);

            // Check singleton predictions
            predictedDistributions = new[] { classifier.PredictDistribution(0, nativePredictionData), classifier.PredictDistribution(1, nativePredictionData) };
            CheckPredictedDiscreteDistributionNativeTestingDataset(expectedDistributions, predictedDistributions);

            // Check point estimates
            Assert.Equal(2, classifier.Predict(0, nativePredictionData));
            Assert.Equal(2, classifier.Predict(nativePredictionData).First());
            Assert.Equal(nativePredictionData.Labels[0], classifier.Predict(0, nativePredictionData));

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Squared);
            Assert.Equal(1, classifier.Predict(0, nativePredictionData));
            Assert.Equal(1, classifier.Predict(nativePredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Absolute);
            Assert.Equal(2, classifier.Predict(0, nativePredictionData));
            Assert.Equal(2, classifier.Predict(nativePredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.ZeroOneError);
            Assert.Equal(2, classifier.Predict(0, nativePredictionData));
            Assert.Equal(2, classifier.Predict(nativePredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.SquaredError);
            Assert.Equal(1, classifier.Predict(0, nativePredictionData));
            Assert.Equal(1, classifier.Predict(nativePredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.AbsoluteError);
            Assert.Equal(2, classifier.Predict(0, nativePredictionData));
            Assert.Equal(2, classifier.Predict(nativePredictionData).First());

            // ILearner settings
            Assert.NotNull(((ILearner)classifier).Settings);

            // Check dimensions of posterior distributions over weights
            Assert.Equal(nativePredictionData.ClassCount, classifier.WeightPosteriorDistributions.Count);
            Assert.Equal(nativePredictionData.FeatureCount, classifier.WeightPosteriorDistributions[0].Count);
        }

        /// <summary>
        /// Tests prediction of the binary Bayes point machine classifier on data in standard format.
        /// </summary>
        /// <param name="standardTrainingData">The training set in standard data format.</param>
        /// <param name="standardPredictionData">The prediction data set in standard data format.</param>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestRegressionBinaryStandardPrediction(StandardDataset standardTrainingData, StandardDataset standardPredictionData, bool useCompoundWeightPriorDistribution = true)
        {
            var classifier = useCompoundWeightPriorDistribution
                                 ? BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping)
                                 : BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping);

            var expectedDistributions = useCompoundWeightPriorDistribution
                                                              ? this.expectedPredictiveBernoulliStandardDistributions
                                                              : this.gaussianPriorExpectedPredictiveBernoulliStandardDistributions;

            classifier.Settings.Training.IterationCount = IterationCount;

            classifier.Train(standardTrainingData);

            // Check bulk predictions
            var predictedDistributions = classifier.PredictDistribution(standardPredictionData);
            CheckPredictedBernoulliDistributionStandardTestingDataset(expectedDistributions, predictedDistributions);

            // Check singleton predictions
            predictedDistributions = new[] { classifier.PredictDistribution("First", standardPredictionData), classifier.PredictDistribution("Second", standardPredictionData) };
            CheckPredictedBernoulliDistributionStandardTestingDataset(expectedDistributions, predictedDistributions);

            // Check point estimates
            Assert.Equal("True", classifier.Predict("First", standardPredictionData));
            Assert.Equal("True", classifier.Predict(standardPredictionData).First());
            Assert.Equal(standardPredictionData.BinaryLabels["First"], classifier.Predict("First", standardPredictionData));

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, (a, b) => a.Equals(b) ? 0.0 : 1.0);
            Assert.Equal("True", classifier.Predict("First", standardPredictionData));
            Assert.Equal("True", classifier.Predict(standardPredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.ZeroOneError);
            Assert.Equal("True", classifier.Predict("First", standardPredictionData));
            Assert.Equal("True", classifier.Predict(standardPredictionData).First());

            // ILearner settings
            Assert.NotNull(((ILearner)classifier).Settings);

            // Check dimensions of posterior distributions over weights
            Assert.Equal(1, classifier.WeightPosteriorDistributions.Count);
            Assert.Equal(standardPredictionData.FeatureVectors["First"].Count, classifier.WeightPosteriorDistributions[0].Count);
        }

        /// <summary>
        /// Tests prediction of the multi-class Bayes point machine classifier on data in standard format.
        /// </summary>
        /// <param name="standardTrainingData">The training set in standard data format.</param>
        /// <param name="standardPredictionData">The prediction data set in standard data format.</param>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestRegressionMulticlassStandardPrediction(StandardDataset standardTrainingData, StandardDataset standardPredictionData, bool useCompoundWeightPriorDistribution = true)
        {
            var classifier = useCompoundWeightPriorDistribution
                                 ? BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping)
                                 : BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping);

            var expectedDistributions = useCompoundWeightPriorDistribution
                                                              ? this.expectedPredictiveDiscreteStandardDistributions
                                                              : this.gaussianPriorExpectedPredictiveDiscreteStandardDistributions;

            classifier.Settings.Training.IterationCount = IterationCount;

            classifier.Train(standardTrainingData);

            // Check bulk predictions
            var predictedDistributions = classifier.PredictDistribution(standardPredictionData);
            CheckPredictedDiscreteDistributionStandardTestingDataset(expectedDistributions, predictedDistributions);

            // Check singleton predictions
            predictedDistributions = new[] { classifier.PredictDistribution("First", standardPredictionData), classifier.PredictDistribution("Second", standardPredictionData) };
            CheckPredictedDiscreteDistributionStandardTestingDataset(expectedDistributions, predictedDistributions);

            // Check point estimates
            Assert.Equal("C", classifier.Predict("First", standardPredictionData));
            Assert.Equal("C", classifier.Predict(standardPredictionData).First());
            Assert.Equal(standardPredictionData.Labels["First"], classifier.Predict("First", standardPredictionData));

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.ZeroOneError);
            Assert.Equal("C", classifier.Predict("First", standardPredictionData));
            Assert.Equal("C", classifier.Predict(standardPredictionData).First());

            classifier.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, (a, b) => a.Equals(b) ? 0.0 : 1.0);
            Assert.Equal("C", classifier.Predict("First", standardPredictionData));
            Assert.Equal("C", classifier.Predict(standardPredictionData).First());

            // ILearner settings
            Assert.NotNull(((ILearner)classifier).Settings);

            // Check dimensions of posterior distributions over weights
            Assert.Equal(standardPredictionData.ClassLabels.Length, classifier.WeightPosteriorDistributions.Count);
            Assert.Equal(standardPredictionData.FeatureVectors["First"].Count, classifier.WeightPosteriorDistributions[0].Count);
        }

        /// <summary>
        /// Tests whether the binary dense native data format classifier can correctly detect problems in training.
        /// </summary>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestDenseBinaryNativeTraining(bool useCompoundWeightPriorDistribution = true)
        {
            // Default training data: OK
            var data = this.denseNativeTrainingData;
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with more classes than training data: OK (ignored)
            data.ClassCount = 4;
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Feature values of dimensionality 0: OK
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 0,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with less classes than training data: OK (ignored)
            data = this.denseNativeTrainingData;
            data.ClassCount = 1;
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution); 

            // Feature values and labels have inconsistent lengths: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Label out of range: OK (impossible for bool)
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 3 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model for a single class: OK (ignored)
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 1,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, false, useCompoundWeightPriorDistribution);

            // All feature values null: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = null,
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = null
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Empty feature values: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Empty labels: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new int[] { }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Negative labels: OK (ignored)
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { -1 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, false, useCompoundWeightPriorDistribution); 

            // Inconsistent number of feature values: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 0.0 }, new[] { 0.0, 1.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values of a single instance are null: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 }, null },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution); 

            // The feature values contain infinite values: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.NegativeInfinity } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.PositiveInfinity } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution); 

            // The feature values must not contain NaNs: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.NaN } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Data is null: Throw (in user code/user data mapping)
            this.CheckBinaryNativeTrainException<ArgumentNullException>(null, true, useCompoundWeightPriorDistribution); 
        }

        /// <summary>
        /// Tests whether the binary sparse native data format classifier can correctly detect problems in training.
        /// </summary>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestSparseBinaryNativeTraining(bool useCompoundWeightPriorDistribution = true)
        {
            // Default training data: OK
            var data = this.sparseNativeTrainingData;
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with more classes than training data: OK (ignored)
            data.ClassCount = 4;
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Feature values of dimensionality 0: OK
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 0,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with less classes than training data: OK (ignored)
            data = this.sparseNativeTrainingData;
            data.ClassCount = 2;
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution); 

            // Feature values and labels have inconsistent lengths: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Label out of range: OK (impossible for bool)
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 3 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model for a single class: OK (ignored)
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 1,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, false, useCompoundWeightPriorDistribution);

            // All feature values null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = null,
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // All feature indexes null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = null
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Empty feature values and indexes: OK
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 0,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, false, useCompoundWeightPriorDistribution);

            // Empty labels: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new int[] { }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Negative labels: OK (impossible)
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { -1 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Inconsistent number of features: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 1.0, 1.0, 1.0 } },
                FeatureIndexes = new[] { new[] { 0, 1, 2 } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckBinaryNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values for a single instance are null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { }, null },
                FeatureIndexes = new[] { new int[] { }, new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // The feature indexes for a single instance are null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { }, new double[] { } },
                FeatureIndexes = new[] { new int[] { }, null },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.NegativeInfinity } },
                FeatureIndexes = new[] { new[] { 0 } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.PositiveInfinity } },
                FeatureIndexes = new[] { new[] { 0 } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values contain NaNs: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.NaN } },
                FeatureIndexes = new[] { new[] { 0 } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckBinaryNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Data is null: Throw (in user code/user data mapping)
            this.CheckBinaryNativeTrainException<ArgumentNullException>(null, true, useCompoundWeightPriorDistribution); 
        }

        /// <summary>
        /// Tests whether the multi-class dense native data format classifier can correctly detect problems in training.
        /// </summary>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestDenseMulticlassNativeTraining(bool useCompoundWeightPriorDistribution = true)
        {
            // Default training data: OK
            var data = this.denseNativeTrainingData;
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with more classes than training data: OK
            data.ClassCount = 4;
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution); 

            // Feature values of dimensionality 0: OK
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 0,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with less classes than training data: Throw
            data = this.denseNativeTrainingData;
            data.ClassCount = 2;
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution); 

            // Feature values and labels have inconsistent lengths: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Label out of range: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 3 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Model for a single class: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 1,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // All feature values are null: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = null,
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = null
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Empty feature values: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Empty labels: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new int[] { }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Negative labels: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { -1 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Inconsistent number of feature values: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 0.0 }, new[] { 0.0, 1.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values of a single instance are null: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { 0.0 }, null },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.NegativeInfinity } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.PositiveInfinity } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution); 

            // The feature values contain NaNs: Throw
            data = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.NaN } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Data is null: Throw (in user code/user data mapping)
            this.CheckMulticlassNativeTrainException<NullReferenceException>(null, true, useCompoundWeightPriorDistribution); 
        }

        /// <summary>
        /// Tests whether the multi-class sparse native data format classifier can correctly detect problems in training.
        /// </summary>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestSparseMulticlassNativeTraining(bool useCompoundWeightPriorDistribution = true)
        {
            // Default training data: OK
            var data = this.sparseNativeTrainingData;
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with more classes than training data: OK
            data.ClassCount = 4;
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Feature values of dimensionality 0: OK
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 0,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with less classes than training data: Throw
            data = this.sparseNativeTrainingData;
            data.ClassCount = 2;
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution); 

            // Feature values and labels have inconsistent lengths: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Label out of range: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 3 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Model for a single class: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 1,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // All feature values are null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = null,
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // All feature indexes are null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = null
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Empty feature values: OK
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 0,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Empty labels: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new int[] { }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Negative labels: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { -1 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Inconsistent number of features: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 1.0, 1.0, 1.0 } },
                FeatureIndexes = new[] { new[] { 0, 1, 2 } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckMulticlassNativeTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values of a single instance are null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { }, null },
                FeatureIndexes = new[] { new int[] { }, new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // The feature indexes of a single instance are null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { }, new double[] { } },
                FeatureIndexes = new[] { new int[] { }, null },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.NegativeInfinity } },
                FeatureIndexes = new[] { new[] { 0 } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.PositiveInfinity } },
                FeatureIndexes = new[] { new[] { 0 } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution); 

            // The feature values contain NaNs: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new[] { double.NaN } },
                FeatureIndexes = new[] { new[] { 0 } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            this.CheckMulticlassNativeTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Data is null: Throw (in user code/user data mapping)
            this.CheckMulticlassNativeTrainException<NullReferenceException>(null, true, useCompoundWeightPriorDistribution); 
        }

        /// <summary>
        /// Tests whether the binary dense standard data format classifier can correctly detect problems in training.
        /// </summary>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestDenseBinaryStandardTraining(bool useCompoundWeightPriorDistribution = true)
        {
            // Feature vector of dimensionality 0: OK
            var data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray() } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Default training data: OK
            data = this.denseStandardTrainingData;
            this.CheckBinaryStandardTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with more classes than training data: Throw
            data.BinaryClassLabels = new[] { "False", "True", "Neither" };
            this.CheckBinaryStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Model with less classes than training data: Throw
            data.BinaryClassLabels = new[] { "False" }; 
            this.CheckBinaryStandardTrainException<MappingException>(data, true);

            // Model with different classes than training data: Throw
            data.BinaryClassLabels = new[] { "A", "B" };
            this.CheckBinaryStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Model classes is empty: Throw
            data.BinaryClassLabels = new string[] { };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Duplicate class labels: Throw
            data.BinaryClassLabels = new[] { "False", "False" };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Duplicate class labels: Throw
            data.BinaryClassLabels = new[] { "False", "False", "True" };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Class label is null: Throw
            data.BinaryClassLabels = new[] { "False", null };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // All class labels are null: Throw
            data.BinaryClassLabels = null;
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Model for a single class: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                BinaryClassLabels = new[] { "False" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = null,
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", null } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = null,
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", null } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Instances null: Throw
            data = new StandardDataset
            {
                FeatureVectors = null,
                Labels = null,
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Empty feature vectors (and labels): Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // More labels than feature vectors: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // More feature vectors than labels: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) }, { "Second", DenseVector.FromArray(1.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Inconsistent number of features: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) }, { "Second", DenseVector.FromArray(0, 1) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(double.NegativeInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(double.PositiveInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain NaNs: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(double.NaN) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not be null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) }, { "Second", null } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Data is null: Throw
            this.CheckBinaryStandardTrainException<ArgumentNullException>(null, true, useCompoundWeightPriorDistribution);
        }

        /// <summary>
        /// Tests whether the binary sparse standard data format classifier can correctly detect problems in training.
        /// </summary>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestSparseBinaryStandardTraining(bool useCompoundWeightPriorDistribution = true)
        {
            // Feature vector of dimensionality 0: OK
            var data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray() } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Default training data: OK
            data = this.denseStandardTrainingData;
            this.CheckBinaryStandardTrainException<BayesPointMachineClassifierException>(data, false);

            // Model with more classes than training data: Throw
            data.BinaryClassLabels = new[] { "False", "True", "Neither" };
            this.CheckBinaryStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Model with less classes than training data: Throw
            data.BinaryClassLabels = new[] { "False" };
            this.CheckBinaryStandardTrainException<MappingException>(data, true);

            // Model with different classes than training data: Throw
            data.BinaryClassLabels = new[] { "A", "B" };
            this.CheckBinaryStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Model classes is empty: Throw
            data.BinaryClassLabels = new string[] { };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Duplicate class labels: Throw
            data.BinaryClassLabels = new[] { "False", "False" };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Duplicate class labels: Throw
            data.BinaryClassLabels = new[] { "False", "False", "True" };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Class label is null: Throw
            data.BinaryClassLabels = new[] { "False", null };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // All class labels are null: Throw
            data.BinaryClassLabels = null;
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Model for a single class: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                BinaryClassLabels = new[] { "False" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = null,
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", null } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = null,
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", null } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Instances null: Throw
            data = new StandardDataset
            {
                FeatureVectors = null,
                Labels = null,
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Empty feature vectors (and labels): Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // More labels than feature vectors: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // More feature vectors than labels: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) }, { "Second", SparseVector.FromArray(1.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Inconsistent number of features: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) }, { "Second", SparseVector.FromArray(0, 1, 2) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(double.NegativeInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(double.PositiveInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain NaNs: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(double.NaN) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not be null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) }, { "Second", null } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckBinaryStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Data is null: Throw
            this.CheckBinaryStandardTrainException<ArgumentNullException>(null, true, useCompoundWeightPriorDistribution);
        }

        /// <summary>
        /// Tests whether the multi-class dense standard data format classifier can correctly detect problems in training.
        /// </summary>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestDenseMulticlassStandardTraining(bool useCompoundWeightPriorDistribution = true)
        {
            // Feature vector of dimensionality 0: OK
            var data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray() } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Default training data: OK
            data = this.denseStandardTrainingData;
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with more classes than training data: OK
            data.ClassLabels = new[] { "A", "B", "C", "D" };
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with less classes than training data: Throw
            data.ClassLabels = new[] { "A", "B" };
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Model with different classes than training data: Throw
            data.ClassLabels = new[] { "D", "E", "F" };
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Model classes is empty: Throw
            data.ClassLabels = new string[] { };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Duplicate class labels: Throw
            data.ClassLabels = new[] { "E", "E" };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Class label is null: Throw
            data.ClassLabels = new[] { "A", null, "C" };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Model classes is null: Throw
            data.ClassLabels = null;
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Model for a single class: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = null,
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", null } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = null,
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", null } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Instances null: Throw
            data = new StandardDataset
            {
                FeatureVectors = null,
                Labels = null,
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Empty feature vectors (and labels): Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // More labels than feature vectors: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // More feature vectors than labels: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) }, { "Second", DenseVector.FromArray(1.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Inconsistent number of features: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) }, { "Second", DenseVector.FromArray(0, 1) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(double.NegativeInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(double.PositiveInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain NaNs: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(double.NaN) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not be null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) }, { "Second", null } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Data is null: Throw
            this.CheckMulticlassStandardTrainException<ArgumentNullException>(null, true, useCompoundWeightPriorDistribution); 
        }

        /// <summary>
        /// Tests whether the multi-class sparse standard data format classifier can correctly detect problems in training.
        /// </summary>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void TestSparseMulticlassStandardTraining(bool useCompoundWeightPriorDistribution = true)
        {
            // Feature vector of dimensionality 0: OK
            var data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray() } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Default training data: OK
            data = this.denseStandardTrainingData;
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with more classes than training data: OK
            data.ClassLabels = new[] { "A", "B", "C", "D" };
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, false, useCompoundWeightPriorDistribution);

            // Model with less classes than training data: Throw
            data.ClassLabels = new[] { "A", "B" };
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Model with different classes than training data: Throw
            data.ClassLabels = new[] { "D", "E", "F" };
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // Model classes is empty: Throw
            data.ClassLabels = new string[] { };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Duplicate class labels: Throw
            data.ClassLabels = new[] { "E", "E" };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Class label is null: Throw
            data.ClassLabels = new[] { "A", null, "C" };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Model classes is null: Throw
            data.ClassLabels = null;
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Model for a single class: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = null,
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", null } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = null,
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", null } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Instances null: Throw
            data = new StandardDataset
            {
                FeatureVectors = null,
                Labels = null,
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Feature vectors null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Labels null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Empty feature vectors (and labels): Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // More labels than feature vectors: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // More feature vectors than labels: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) }, { "Second", SparseVector.FromArray(1.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Inconsistent number of features: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) }, { "Second", SparseVector.FromArray(0, 1, 2) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<BayesPointMachineClassifierException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(double.NegativeInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(double.PositiveInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not contain NaNs: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(double.NaN) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // A feature vector must not be null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) }, { "Second", null } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            this.CheckMulticlassStandardTrainException<MappingException>(data, true, useCompoundWeightPriorDistribution);

            // Data is null: Throw
            this.CheckMulticlassStandardTrainException<ArgumentNullException>(null, true, useCompoundWeightPriorDistribution);
        }

        /// <summary>
        /// Tests if the binary native data format classifier has the correct exception behavior in training.
        /// </summary>
        /// <typeparam name="TException">The type of exception expected to be thrown.</typeparam>
        /// <param name="data">The data to test in native format.</param>
        /// <param name="mustThrow">Indicates whether the tested code must throw an exception of type <typeparamref name="TException"/>.</param>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void CheckBinaryNativeTrainException<TException>(NativeDataset data, bool mustThrow, bool useCompoundWeightPriorDistribution = true) 
            where TException : Exception
        {
            var classifier = useCompoundWeightPriorDistribution
                                 ? BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryNativeMapping)
                                 : BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryNativeMapping);

            CheckTrainException<TException, NativeDataset, int, NativeDataset, bool, Bernoulli, BayesPointMachineClassifierTrainingSettings>(classifier, data, mustThrow);
        }

        /// <summary>
        /// Tests if the multi-class native data format classifier has the correct exception behavior in training.
        /// </summary>
        /// <typeparam name="TException">The type of exception expected to be thrown.</typeparam>
        /// <param name="data">The data to test in native format.</param>
        /// <param name="mustThrow">Indicates whether the tested code must throw an exception of type <typeparamref name="TException"/>.</param>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void CheckMulticlassNativeTrainException<TException>(NativeDataset data, bool mustThrow, bool useCompoundWeightPriorDistribution = true) 
            where TException : Exception
        {
            var classifier = useCompoundWeightPriorDistribution
                                 ? BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassNativeMapping)
                                 : BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassNativeMapping);

            CheckTrainException<TException, NativeDataset, int, NativeDataset, int, Discrete, BayesPointMachineClassifierTrainingSettings>(classifier, data, mustThrow);
        }

        /// <summary>
        /// Tests if the multi-class standard data format classifier has the correct exception behavior in training.
        /// </summary>
        /// <typeparam name="TException">The type of exception expected to be thrown.</typeparam>
        /// <param name="data">The data to test in standard format.</param>
        /// <param name="mustThrow">Indicates whether the tested code must throw an exception of type <typeparamref name="TException"/>.</param>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void CheckBinaryStandardTrainException<TException>(StandardDataset data, bool mustThrow, bool useCompoundWeightPriorDistribution = true) 
            where TException : Exception
        {
            var classifier = useCompoundWeightPriorDistribution
                                 ? BayesPointMachineClassifier.CreateBinaryClassifier(this.binaryStandardMapping)
                                 : BayesPointMachineClassifier.CreateGaussianPriorBinaryClassifier(this.binaryStandardMapping);

            CheckTrainException<TException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(classifier, data, mustThrow);
        }

        /// <summary>
        /// Tests if the multi-class standard data format classifier has the correct exception behavior in training.
        /// </summary>
        /// <typeparam name="TException">The type of exception expected to be thrown.</typeparam>
        /// <param name="data">The data to test in standard format.</param>
        /// <param name="mustThrow">Indicates whether the tested code must throw an exception of type <typeparamref name="TException"/>.</param>
        /// <param name="useCompoundWeightPriorDistribution">If true, a classifier with compound prior distribution over weights is used. Defaults to true.</param>
        private void CheckMulticlassStandardTrainException<TException>(StandardDataset data, bool mustThrow, bool useCompoundWeightPriorDistribution = true) 
            where TException : Exception
        {
            var classifier = useCompoundWeightPriorDistribution
                                 ? BayesPointMachineClassifier.CreateMulticlassClassifier(this.multiclassStandardMapping)
                                 : BayesPointMachineClassifier.CreateGaussianPriorMulticlassClassifier(this.multiclassStandardMapping);

            CheckTrainException<TException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>, BayesPointMachineClassifierTrainingSettings>(classifier, data, mustThrow);
        }

        /// <summary>
        /// Tests prediction of a Bayes point machine classifier on dense data in native format.
        /// </summary>
        /// <typeparam name="TLabel">The type of the label in native data format.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of the distribution over labels in native data format.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private void TestDenseNativePrediction<TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<NativeDataset, int, NativeDataset, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Train classifier
            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.Train(this.denseNativeTrainingData);
            var data = this.denseNativePredictionData;

            // Default prediction data: OK
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // Wrong number of classes: OK
            data.ClassCount = 0;
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);
            data.ClassCount = 1;
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);
            data.ClassCount = 2;
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // Labels null: OK
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = new[] { new[] { 0.0, 2.0 } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = null
                       };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // New labels in test set: OK
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = new[] { new[] { 0.0, 2.0 } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 666 }
                       };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // Empty labels: OK
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = new[] { new[] { 0.0, 2.0 } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new int[] { }
                       };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // Feature values and labels have inconsistent lengths: OK
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = new[] { new[] { 0.0, 2.0 } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 0, 0 }
                       };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // The number of features is inconsistent with training: Throw
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 0,
                           FeatureValues = new[] { new double[] { } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 0 }
                       };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // The number of features is inconsistent with training: Throw
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 1,
                           FeatureValues = new[] { new[] { 0.0 } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 0 }
                       };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // All feature values null: Throw
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = null,
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 0 }
                       };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // Empty feature values: Throw
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = new[] { new double[] { } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 0 }
                       };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // Inconsistent feature value dimensions: Throw
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = new[] { new[] { 0.0, 2.0 }, new[] { 0.0 } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 0, 0 }
                       };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = new[] { new[] { 0.0, 2.0 }, new[] { 0.0, double.NegativeInfinity } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 0, 0 }
                       };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = new[] { new[] { 0.0, 2.0 }, new[] { 0.0, double.PositiveInfinity } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 0, 0 }
                       };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // The feature values contain NaNs: Throw
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = new[] { new[] { 0.0, 2.0 }, new[] { 0.0, double.NaN } },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 0, 0 }
                       };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // The feature values of a single instance are null: Throw
            data = new NativeDataset
                       {
                           IsSparse = false,
                           FeatureCount = 2,
                           FeatureValues = new[] { new[] { 0.0, 2.0 }, null },
                           FeatureIndexes = null,
                           ClassCount = 3,
                           Labels = new[] { 0, 0 }
                       };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // Data is null: Throw (in user code/user data mapping)
            CheckPredictionException<ArgumentNullException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, null, true);
            CheckSinglePredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, null, 0, true);
        }

        /// <summary>
        /// Tests prediction of a Bayes point machine classifier on dense data in native format.
        /// </summary>
        /// <typeparam name="TLabel">The type of the label in native data format.</typeparam>
        /// <typeparam name="TLabelDistribution">The type of the distribution over labels in native data format.</typeparam>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private void TestSparseNativePrediction<TLabel, TLabelDistribution, TTrainingSettings>(
            IBayesPointMachineClassifier<NativeDataset, int, NativeDataset, TLabel, TLabelDistribution, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<TLabel>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Train classifier
            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.Train(this.sparseNativeTrainingData);
            var data = this.sparseNativePredictionData;

            // Default prediction data: OK
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // Wrong number of classes: OK
            data.ClassCount = 0;
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false); 
            data.ClassCount = 1;
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false); 
            data.ClassCount = 2;
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // Labels null: OK
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 } },
                FeatureIndexes = new[] { new[] { 1 } },
                ClassCount = 3,
                Labels = null
            };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // New labels in test set: OK
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 } },
                FeatureIndexes = new[] { new[] { 1 } },
                ClassCount = 3,
                Labels = new[] { 666 }
            };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // Empty labels: OK
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 } },
                FeatureIndexes = new[] { new[] { 1 } },
                ClassCount = 3,
                Labels = new int[] { }
            };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false);

            // Feature values and labels have inconsistent lengths: OK
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 } },
                FeatureIndexes = new[] { new[] { 1 } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, false); 

            // The number of features is inconsistent with training: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 0,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new int[] { } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true); 

             // The number of features is inconsistent with training: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 1,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new[] { 0 } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // All feature values are null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = null,
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // Empty feature values: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new double[] { } },
                FeatureIndexes = new[] { new[] { 0 } },
                ClassCount = 3,
                Labels = new[] { 0 }
            };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // Inconsistent feature (index) dimensions: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } },
                FeatureIndexes = new[] { new[] { 1 }, new[] { 2 } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            CheckPredictionException<BayesPointMachineClassifierException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 }, new[] { double.NegativeInfinity } },
                FeatureIndexes = new[] { new[] { 1 }, new[] { 1 } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // The feature values contain infinite values: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 }, new[] { double.PositiveInfinity } },
                FeatureIndexes = new[] { new[] { 1 }, new[] { 1 } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // The feature values contain NaNs: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 }, new[] { double.NaN } },
                FeatureIndexes = new[] { new[] { 1 }, new[] { 1 } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // The feature values of a single instance are null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 }, null },
                FeatureIndexes = new[] { new[] { 1 }, new[] { 1 } },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // The feature indexes of a single instance are null: Throw
            data = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 }, new[] { 2.0 } },
                FeatureIndexes = new[] { new[] { 1 }, null },
                ClassCount = 3,
                Labels = new[] { 0, 0 }
            };
            CheckPredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, data, true);

            // Data is null: Throw (in user code/user data mapping)
            CheckPredictionException<ArgumentNullException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, null, true);
            CheckSinglePredictionException<MappingException, NativeDataset, int, NativeDataset, TLabel, TLabelDistribution>(classifier, null, 0, true);
        }

        /// <summary>
        /// Tests prediction of a Bayes point machine classifier on dense data in standard format.
        /// </summary>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private void TestDenseStandardPrediction<TTrainingSettings>(
            IBayesPointMachineClassifier<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<string>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Train classifier
            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.Train(this.denseStandardTrainingData);

            // Default prediction data: OK
            var data = this.denseStandardPredictionData;
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // Wrong number of classes: OK
            data.ClassLabels = new string[] { };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false); 
            data.ClassLabels = new[] { "A" };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false); 
            data.ClassLabels = new[] { "A", "B" };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // A single label is null: OK
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 2) } },
                Labels = new Dictionary<string, string> { { "First", null } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // All labels are null: OK
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 2) } },
                Labels = null,
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // The labels are empty: OK
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 2) } },
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // New label in test set: OK
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 2) } },
                Labels = new Dictionary<string, string> { { "First", "D" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // More labels than feature vectors: OK
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0, 0.0) }, { "Second", DenseVector.FromArray(0.0, 1.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // The number of features is inconsistent with training: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray() } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // The number of features is inconsistent with training: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // More instances / labels than feature vectors: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0.0, 2.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // A single feature vector is null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", null } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // All feature vectors are null: Throw
            data = new StandardDataset
            {
                FeatureVectors = null,
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // All feature vectors are null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // Instances / labels are null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // Unknown instance: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "FortyTwo", DenseVector.FromArray(0, 2) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // Inconsistent feature vector dimensions: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 2) }, { "Second", DenseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // A feature vector contains infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 2) }, { "Second", DenseVector.FromArray(0, double.NegativeInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // A feature vector contains infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 2) }, { "Second", DenseVector.FromArray(0, double.PositiveInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // A feature vector contains NaNs: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 2) }, { "Second", DenseVector.FromArray(0, double.NaN) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // A single feature vector is null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", DenseVector.FromArray(0, 2) }, { "Second", null } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true); 

            // Instance or instance source are null: Throw
            Assert.Throws<ArgumentNullException>(() => classifier.PredictDistribution(null));
            Assert.Throws<ArgumentNullException>(() => classifier.PredictDistribution(null, null));
            Assert.Throws<ArgumentNullException>(() => classifier.Predict(null));
            Assert.Throws<ArgumentNullException>(() => classifier.Predict(null, null));

            Assert.Throws<MappingException>(() => classifier.PredictDistribution("First"));
            Assert.Throws<MappingException>(() => classifier.Predict("First"));
        }

        /// <summary>
        /// Tests prediction of a Bayes point machine classifier on sparse data in standard format.
        /// </summary>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        private void TestSparseStandardPrediction<TTrainingSettings>(
            IBayesPointMachineClassifier<StandardDataset, string, StandardDataset, string, IDictionary<string, double>, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<string>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Train classifier
            classifier.Settings.Training.IterationCount = IterationCount;
            classifier.Train(this.sparseStandardTrainingData);

            // Default prediction data: OK
            var data = this.sparseStandardPredictionData;
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // Wrong number of classes: OK
            data.ClassLabels = new string[] { };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);
            data.ClassLabels = new[] { "A" };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);
            data.ClassLabels = new[] { "A", "B" };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // A single label is null: OK
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 2) } },
                Labels = new Dictionary<string, string> { { "First", null } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // All labels are null: OK
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 2) } },
                Labels = null,
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // The labels are empty: OK
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 2) } },
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // New label in test set: OK
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 2) } },
                Labels = new Dictionary<string, string> { { "First", "D" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // More labels than feature vectors: OK
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0, 0.0) }, { "Second", SparseVector.FromArray(0.0, 1.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, false);

            // The number of features is inconsistent with training: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray() } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // The number of features is inconsistent with training: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // More instances / labels than feature vectors: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0.0, 2.0) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // A single feature vector is null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", null } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // All feature vectors are null: Throw
            data = new StandardDataset
            {
                FeatureVectors = null,
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // All feature vectors are null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // Instances / labels are null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>(),
                Labels = new Dictionary<string, string>(),
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // Unknown instance: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "FortyTwo", SparseVector.FromArray(0, 2) } },
                Labels = new Dictionary<string, string> { { "First", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // Inconsistent feature vector dimensions: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 2) }, { "Second", SparseVector.FromArray(0, 1, 2) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<BayesPointMachineClassifierException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // A feature vector contains infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 2) }, { "Second", SparseVector.FromArray(0, double.NegativeInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // A feature vector contains infinite values: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 2) }, { "Second", SparseVector.FromArray(0, double.PositiveInfinity) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // A feature vector contains NaNs: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 2) }, { "Second", SparseVector.FromArray(0, double.NaN) } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // A single feature vector is null: Throw
            data = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector> { { "First", SparseVector.FromArray(0, 2) }, { "Second", null } },
                Labels = new Dictionary<string, string> { { "First", "A" }, { "Second", "A" } },
                ClassLabels = new[] { "A", "B", "C" }
            };
            CheckPredictionException<MappingException, StandardDataset, string, StandardDataset, string, IDictionary<string, double>>(classifier, data, true);

            // Instance or instance source are null: Throw
            Assert.Throws<ArgumentNullException>(() => classifier.PredictDistribution(null));
            Assert.Throws<ArgumentNullException>(() => classifier.PredictDistribution(null, null));
            Assert.Throws<ArgumentNullException>(() => classifier.Predict(null));
            Assert.Throws<ArgumentNullException>(() => classifier.Predict(null, null));

            Assert.Throws<MappingException>(() => classifier.PredictDistribution("First"));
            Assert.Throws<MappingException>(() => classifier.Predict("First"));
        }

        #endregion

        #region Data initialization

        /// <summary>
        /// Creates the expected predictive distributions.
        /// </summary>
        private void InitializePredictiveDistributions()
        {
            // Expected Bernoulli predictive distributions for data in native format
            this.expectedPredictiveBernoulliDistributions = 
                CreateBernoulliDistributionsFromProbabilities(this.expectedBernoulliModeProbabilities);

            this.expectedIncrementalPredictiveBernoulliDistributions = 
                CreateBernoulliDistributionsFromProbabilities(this.expectedIncrementalBernoulliModeProbabilities);

            this.gaussianPriorExpectedPredictiveBernoulliDistributions = 
                CreateBernoulliDistributionsFromProbabilities(this.gaussianPriorExpectedBernoulliModeProbabilities);

            this.gaussianPriorExpectedIncrementalPredictiveBernoulliDistributions = 
                CreateBernoulliDistributionsFromProbabilities(this.gaussianPriorExpectedIncrementalBernoulliModeProbabilities);

            // Expected Discrete predictive distributions for data in native format
            this.expectedPredictiveDiscreteDistributions = 
                CreateDiscreteDistributionsFromProbabilities(this.expectedDiscreteModeProbabilities);

            this.expectedIncrementalPredictiveDiscreteDistributions = 
                CreateDiscreteDistributionsFromProbabilities(this.expectedIncrementalDiscreteModeProbabilities);

            this.gaussianPriorExpectedPredictiveDiscreteDistributions = 
                CreateDiscreteDistributionsFromProbabilities(this.gaussianPriorExpectedDiscreteModeProbabilities);

            this.gaussianPriorExpectedIncrementalPredictiveDiscreteDistributions = 
                CreateDiscreteDistributionsFromProbabilities(this.gaussianPriorExpectedIncrementalDiscreteModeProbabilities);

            // Expected Bernoulli predictive distributions for data in simple format
            this.expectedPredictiveBernoulliSimpleDistributions =
                CreateSimpleDistributionsFromBernoulliDistributions(this.expectedPredictiveBernoulliDistributions);
            this.expectedPredictiveBernoulliSimpleIntDistributions =
                CreateIntDistributionsFromBernoulliDistributions(this.expectedPredictiveBernoulliDistributions);
            this.expectedPredictiveDiscreteIntDistributions =
                CreateIntDistributionsFromDiscreteDistributions(this.expectedPredictiveDiscreteDistributions);

            this.expectedIncrementalPredictiveBernoulliSimpleDistributions =
                CreateSimpleDistributionsFromBernoulliDistributions(this.expectedIncrementalPredictiveBernoulliDistributions);
            this.expectedIncrementalPredictiveBernoulliSimpleIntDistributions =
                CreateIntDistributionsFromBernoulliDistributions(this.expectedIncrementalPredictiveBernoulliDistributions);
            this.expectedIncrementalPredictiveDiscreteIntDistributions =
                CreateIntDistributionsFromDiscreteDistributions(this.expectedIncrementalPredictiveDiscreteDistributions);

            // Expected Bernoulli predictive distributions for data in standard format
            this.expectedPredictiveBernoulliStandardDistributions =
                CreateStandardDistributionsFromBernoulliDistributions(this.expectedPredictiveBernoulliDistributions);

            this.expectedIncrementalPredictiveBernoulliStandardDistributions =
                CreateStandardDistributionsFromBernoulliDistributions(this.expectedIncrementalPredictiveBernoulliDistributions);

            this.gaussianPriorExpectedPredictiveBernoulliStandardDistributions =
                CreateStandardDistributionsFromBernoulliDistributions(this.gaussianPriorExpectedPredictiveBernoulliDistributions);

            this.gaussianPriorExpectedIncrementalPredictiveBernoulliStandardDistributions =
                CreateStandardDistributionsFromBernoulliDistributions(this.gaussianPriorExpectedIncrementalPredictiveBernoulliDistributions);

            // Expected Discrete predictive distributions for data in standard format
            this.expectedPredictiveDiscreteStandardDistributions =
                CreateStandardDistributionsFromDiscreteDistributions(this.expectedPredictiveDiscreteDistributions);

            this.expectedIncrementalPredictiveDiscreteStandardDistributions =
                CreateStandardDistributionsFromDiscreteDistributions(this.expectedIncrementalPredictiveDiscreteDistributions);

            this.gaussianPriorExpectedPredictiveDiscreteStandardDistributions =
                CreateStandardDistributionsFromDiscreteDistributions(this.gaussianPriorExpectedPredictiveDiscreteDistributions);

            this.gaussianPriorExpectedIncrementalPredictiveDiscreteStandardDistributions =
                CreateStandardDistributionsFromDiscreteDistributions(this.gaussianPriorExpectedIncrementalPredictiveDiscreteDistributions);
        }

        /// <summary>
        /// Creates the datasets in native format.
        /// </summary>
        private void InitializeNativeData()
        {
            this.denseNativeTrainingData = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 0, 1, 2 }
            };

            this.sparseNativeTrainingData = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new double[] { }, new[] { 1.0 }, new[] { 1.0 } },
                FeatureIndexes = new[] { new int[] { }, new[] { 0 }, new[] { 1 } },
                ClassCount = 3,
                Labels = new[] { 0, 1, 2 }
            };

            this.denseNativePredictionData = new NativeDataset
            {
                IsSparse = false,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 0.0, 2.0 }, new[] { 4.0, 0.0 } },
                FeatureIndexes = null,
                ClassCount = 3,
                Labels = new[] { 2, 1 }
            };

            this.sparseNativePredictionData = new NativeDataset
            {
                IsSparse = true,
                FeatureCount = 2,
                FeatureValues = new[] { new[] { 2.0 }, new[] { 4.0 } },
                FeatureIndexes = new[] { new[] { 1 }, new[] { 0 } },
                ClassCount = 3,
                Labels = new[] { 2, 1 }
            };

            this.binaryNativeMapping = new BinaryNativeBayesPointMachineClassifierTestMapping();
            this.multiclassNativeMapping = new MulticlassNativeBayesPointMachineClassifierTestMapping();
        }

        /// <summary>
        /// Creates the datasets in standard format.
        /// </summary>
        private void InitializeStandardData()
        {
            var classLabels = new[] { "A", "B", "C" };
            var trainingLabels = new Dictionary<string, string> { { "First", "A" }, { "Second", "B" }, { "Third", "C" } };
            var predictionLabels = new Dictionary<string, string> { { "First", "C" }, { "Second", "B" } };

            this.denseSimpleTrainingData = new List<Vector>()
            {
                DenseVector.FromArray(0, 0),
                DenseVector.FromArray(1, 0),
                DenseVector.FromArray(0, 1)
            };
            this.denseSimpleTrainingLabels = new List<bool>()
            {
                false,
                true,
                true
            };
            this.denseSimpleIntTrainingLabels = new List<int>()
            {
                0,
                1,
                1
            };
            this.denseSimpleMulticlassTrainingLabels = new List<int>()
            {
                0,
                1,
                2
            };

            this.denseStandardTrainingData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>
                {
                    { "First", DenseVector.FromArray(0, 0) },
                    { "Second", DenseVector.FromArray(1, 0) },
                    { "Third", DenseVector.FromArray(0, 1) }
                },

                Labels = trainingLabels,

                ClassLabels = classLabels
            };

            this.sparseStandardTrainingData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>
                {
                    { "First", SparseVector.FromArray(0, 0) },
                    { "Second", SparseVector.FromArray(1, 0) },
                    { "Third", SparseVector.FromArray(0, 1) }
                },

                Labels = trainingLabels,

                ClassLabels = classLabels
            };

            this.denseSimplePredictionData = new List<Vector>()
            {
                DenseVector.FromArray(0, 2),
                DenseVector.FromArray(4, 0)
            };

            this.denseStandardPredictionData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>
                {
                    { "First", DenseVector.FromArray(0, 2) },
                    { "Second", DenseVector.FromArray(4, 0) }
                },

                Labels = predictionLabels,

                ClassLabels = classLabels
            };

            this.sparseStandardPredictionData = new StandardDataset
            {
                FeatureVectors = new Dictionary<string, Vector>
                {
                    { "First", SparseVector.FromArray(0, 2) },
                    { "Second", SparseVector.FromArray(4, 0) }
                },

                Labels = predictionLabels,

                ClassLabels = classLabels
            };

            this.binaryStandardMapping = new BinaryStandardBayesPointMachineClassifierTestMapping();
            this.multiclassStandardMapping = new MulticlassStandardBayesPointMachineClassifierTestMapping();
        }

        #endregion

        #region Datasets

        /// <summary>
        /// Represents a dataset in the native format of the Bayes point machine classifier.
        /// </summary>
        private class NativeDataset : IDataset
        {
            /// <summary>
            /// The label of the negative class.
            /// </summary>
            private const int NegativeClassLabel = 0;

            /// <summary>
            /// The multi-class labels.
            /// </summary>
            private int[] multiclassLabels;

            /// <summary>
            /// Gets or sets a value indicating whether the features are in a sparse or a dense representation.
            /// </summary>
            public bool IsSparse { get; set; }

            /// <summary>
            /// Gets the number of instances.
            /// </summary>
            public int InstanceCount 
            {
                get
                {
                    Assert.Equal(this.FeatureValues.Length, this.Labels.Length);
                    return this.FeatureValues.Length;
                }
            }

            /// <summary>
            /// Gets or sets the total number of features.
            /// </summary>
            public int FeatureCount { get; set; }

            /// <summary>
            /// Gets or sets the number of classes of the classifier.
            /// </summary>
            public int ClassCount { get; set; }

            /// <summary>
            /// Gets or sets the feature values.
            /// </summary>
            public double[][] FeatureValues { get; set; }

            /// <summary>
            /// Gets or sets the feature indexes.
            /// </summary>
            public int[][] FeatureIndexes { get; set; }

            /// <summary>
            /// Gets or sets the multi-class labels.
            /// </summary>
            public int[] Labels 
            {
                get
                {
                    return this.multiclassLabels;
                }

                set
                {
                    this.SetLabels(value);
                }
            }

            /// <summary>
            /// Gets the binary labels.
            /// </summary>
            public bool[] BinaryLabels { get; private set; }

            /// <summary>
            /// Sets both multi-class and binary labels.
            /// </summary>
            /// <param name="labels">The multi-class labels.</param>
            /// <remarks>
            /// The binary labels are created from the multi-class labels 
            /// where the value 0 is interpreted as false and everything else as true.
            /// </remarks>
            private void SetLabels(int[] labels)
            {
                this.multiclassLabels = labels;

                if (labels == null)
                {
                    this.BinaryLabels = null;
                }
                else
                {
                    this.BinaryLabels = new bool[labels.Length];
                    for (int i = 0; i < labels.Length; i++)
                    {
                        if (labels[i] != NegativeClassLabel)
                        {
                            this.BinaryLabels[i] = true;
                        }
                        else
                        {
                            this.BinaryLabels[i] = false;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Represents a dataset in the standard format of the Bayes point machine classifier.
        /// </summary>
        private class StandardDataset : IDataset
        {
            /// <summary>
            /// The label of the negative class.
            /// </summary>
            private const string NegativeClassLabel = "A";

            /// <summary>
            /// The multi-class labels.
            /// </summary>
            private Dictionary<string, string> multiclassLabels;

            /// <summary>
            /// Initializes a new instance of the <see cref="StandardDataset"/> class.
            /// </summary>
            public StandardDataset()
            {
                this.BinaryClassLabels = new[] { "False", "True" };
            }

            /// <summary>
            /// Gets the number of instances.
            /// </summary>
            public int InstanceCount
            {
                get
                {
                    Assert.Equal(this.FeatureVectors.Count, this.Labels.Count);
                    return this.FeatureVectors.Count;
                }
            }

            /// <summary>
            /// Gets or sets the feature vectors for instances.
            /// </summary>
            public Dictionary<string, Vector> FeatureVectors { get; set; }

            /// <summary>
            /// Gets or sets the multi-class labels.
            /// </summary>
            public Dictionary<string, string> Labels
            {
                get
                {
                    return this.multiclassLabels;
                }

                set
                {
                    this.SetLabels(value);
                }
            }

            /// <summary>
            /// Gets the binary labels.
            /// </summary>
            public Dictionary<string, string> BinaryLabels { get; private set; }

            /// <summary>
            /// Gets or sets the class labels of the multi-class classifier.
            /// </summary>
            public string[] ClassLabels { get; set; }

            /// <summary>
            /// Gets or sets the class labels of the binary classifier.
            /// </summary>
            public string[] BinaryClassLabels { get; set; }

            /// <summary>
            /// Sets both multi-class and binary labels.
            /// </summary>
            /// <param name="labels">The multi-class labels.</param>
            /// <remarks>
            /// The binary labels are created from the multi-class labels where
            /// the string "A" is interpreted as false and everything else as true.
            /// </remarks>
            private void SetLabels(Dictionary<string, string> labels)
            {
                this.multiclassLabels = labels;

                if (labels == null)
                {
                    this.BinaryLabels = null;
                }
                else
                {
                    this.BinaryLabels = new Dictionary<string, string>();
                    foreach (KeyValuePair<string, string> label in labels)
                    {
                        string binaryLabel = null;
                        if (label.Value != null)
                        {
                            binaryLabel = label.Value.Equals(NegativeClassLabel) ? this.BinaryClassLabels[0] : this.BinaryClassLabels[1];
                        }

                        this.BinaryLabels.Add(label.Key, binaryLabel);
                    }
                }
            }
        }

        #endregion

        #region Mappings

        /// <summary>
        /// An abstract base class implementation of <see cref="IBayesPointMachineClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel}"/> 
        /// for <see cref="NativeDataset"/>.
        /// </summary>
        /// <typeparam name="TLabel">The type of a label in native data format.</typeparam>
        [Serializable]
        private abstract class NativeBayesPointMachineClassifierTestMapping<TLabel> 
            : IBayesPointMachineClassifierMapping<NativeDataset, int, NativeDataset, TLabel>, ICustomSerializable
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="NativeBayesPointMachineClassifierTestMapping{TLabel}"/> class.
            /// </summary>
            protected NativeBayesPointMachineClassifierTestMapping()
            {
                this.BatchCount = 1;
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeBayesPointMachineClassifierTestMapping{TLabel}"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            protected NativeBayesPointMachineClassifierTestMapping(IReader reader)
            {
                this.BatchCount = reader.ReadInt32();
            }

            /// <summary>
            /// Gets or sets the number of batches.
            /// </summary>
            public int BatchCount { get; set; }

            /// <summary>
            /// Indicates whether the feature representation provided by the instance source is sparse or dense.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>True, if the feature representation is sparse and false if it is dense.</returns>
            public bool IsSparse(NativeDataset instanceSource)
            {
                return instanceSource.IsSparse;
            }

            /// <summary>
            /// Provides the total number of features for the specified instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>The total number of features.</returns>
            public int GetFeatureCount(NativeDataset instanceSource)
            {
                return instanceSource.FeatureCount;
            }

            /// <summary>
            /// Provides the number of classes that the Bayes point machine classifier is used for.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">The label source.</param>
            /// <returns>The number of classes that the Bayes point machine classifier is used for.</returns>
            public int GetClassCount(NativeDataset instanceSource, NativeDataset labelSource)
            {
                return instanceSource.ClassCount;
            }

            /// <summary>
            /// Provides the feature values for a specified instance.
            /// </summary>
            /// <param name="instance">The instance.</param>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>The feature values for the specified instance.</returns>
            public double[] GetFeatureValues(int instance, NativeDataset instanceSource)
            {
                if (instanceSource == null || instanceSource.FeatureValues == null)
                {
                    return null;
                }

                return instanceSource.FeatureValues[instance];
            }

            /// <summary>
            /// Provides the feature indexes for a specified instance.
            /// </summary>
            /// <param name="instance">The instance.</param>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>The feature indexes for the specified instance. Null if feature values are in a dense representation.</returns>
            public int[] GetFeatureIndexes(int instance, NativeDataset instanceSource)
            {
                if (instanceSource == null || instanceSource.FeatureIndexes == null)
                {
                    return null;
                }

                return instanceSource.FeatureIndexes[instance];
            }

            /// <summary>
            /// Provides the feature values of all instances from the specified batch of the instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.</param>
            /// <returns>The feature values provided by the specified batch of the instance source.</returns>
            public double[][] GetFeatureValues(NativeDataset instanceSource, int batchNumber = 0)
            {
                return instanceSource == null || instanceSource.FeatureValues == null ? null : Utilities.GetBatch(batchNumber, instanceSource.FeatureValues, this.BatchCount).ToArray();
            }

            /// <summary>
            /// Provides the feature indexes of all instances from the specified batch of the instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance source is divided into batches.</param>
            /// <returns>
            /// The feature indexes provided by the specified batch of the instance source. Null if feature values are in a dense representation.
            /// </returns>
            public int[][] GetFeatureIndexes(NativeDataset instanceSource, int batchNumber = 0)
            {
                return instanceSource == null || instanceSource.FeatureIndexes == null ? null : Utilities.GetBatch(batchNumber, instanceSource.FeatureIndexes, this.BatchCount).ToArray();
            }

            /// <summary>
            /// Provides the labels of all instances from the specified batch of the instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">A label source.</param>
            /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance and label sources are divided into batches.</param>
            /// <returns>The labels provided by the specified batch of the sources.</returns>
            public abstract TLabel[] GetLabels(NativeDataset instanceSource, NativeDataset labelSource, int batchNumber = 0);

            /// <summary>
            /// Saves the state of the native data mapping using a writer to a binary stream.
            /// </summary>
            /// <param name="writer">The writer to save the state of the native data mapping to.</param>
            public void SaveForwardCompatible(IWriter writer)
            {
                writer.Write(this.BatchCount);
            }
        }

        /// <summary>
        /// An implementation of <see cref="IBayesPointMachineClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel}"/> 
        /// for <see cref="NativeDataset"/> and <see cref="bool"/> labels.
        /// </summary>
        [Serializable]
        private class BinaryNativeBayesPointMachineClassifierTestMapping :
            NativeBayesPointMachineClassifierTestMapping<bool>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="BinaryNativeBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            public BinaryNativeBayesPointMachineClassifierTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="BinaryNativeBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public BinaryNativeBayesPointMachineClassifierTestMapping(IReader reader) : base(reader)
            {
            }

            /// <summary>
            /// Provides the labels of all instances from the specified batch of the instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">A label source.</param>
            /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance and label sources are divided into batches.</param>
            /// <returns>The labels provided by the specified batch of the sources.</returns>
            public override bool[] GetLabels(NativeDataset instanceSource, NativeDataset labelSource, int batchNumber = 0)
            {
                return instanceSource == null || instanceSource.BinaryLabels == null ? null : Utilities.GetBatch(batchNumber, instanceSource.BinaryLabels, this.BatchCount).ToArray();
            }
        }

        /// <summary>
        /// An implementation of <see cref="IBayesPointMachineClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel}"/> 
        /// for <see cref="NativeDataset"/> and <see cref="int"/> labels.
        /// </summary>
        [Serializable]
        private class MulticlassNativeBayesPointMachineClassifierTestMapping :
            NativeBayesPointMachineClassifierTestMapping<int>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="MulticlassNativeBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            public MulticlassNativeBayesPointMachineClassifierTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="MulticlassNativeBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public MulticlassNativeBayesPointMachineClassifierTestMapping(IReader reader) : base(reader)
            {
            }

            /// <summary>
            /// Provides the labels of all instances from the specified batch of the instance source.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">A label source.</param>
            /// <param name="batchNumber">An optional batch number. Defaults to 0 and is used only if the instance and label sources are divided into batches.</param>
            /// <returns>The labels provided by the specified batch of the sources.</returns>
            public override int[] GetLabels(NativeDataset instanceSource, NativeDataset labelSource, int batchNumber = 0)
            {
                return instanceSource == null || instanceSource.Labels == null ? null : Utilities.GetBatch(batchNumber, instanceSource.Labels, this.BatchCount).ToArray();
            }
        }

        private class BinarySimpleTestMapping<TLabel> : ClassifierMapping<IReadOnlyList<Vector>, int, IReadOnlyList<TLabel>, TLabel, Vector>
        {
            public override IEnumerable<int> GetInstances(IReadOnlyList<Vector> featureVectors)
            {
                for (int instance = 0; instance < featureVectors.Count; instance++)
                {
                    yield return instance;
                }
            }

            public override Vector GetFeatures(int instance, IReadOnlyList<Vector> featureVectors)
            {
                return featureVectors[instance];
            }

            public override TLabel GetLabel(int instance, IReadOnlyList<Vector> featureVectors, IReadOnlyList<TLabel> labels)
            {
                return labels[instance];
            }

            public override IEnumerable<TLabel> GetClassLabels(
                IReadOnlyList<Vector> featureVectors = null, IReadOnlyList<TLabel> labels = null)
            {
                if (true is TLabel t && false is TLabel f)
                {
                    // This array is intentionally out of order
                    return new[] { t, f };
                }
                else if (0 is TLabel zero && 1 is TLabel one)
                {
                    // This array is intentionally out of order
                    return new[] { one, zero };
                }
                else throw new NotImplementedException();
            }
        }

        /// <summary>
        /// An implementation of <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues}"/> 
        /// for <see cref="StandardDataset"/> and multiple class labels.
        /// </summary>
        [Serializable]
        private class MulticlassSimpleBayesPointMachineClassifierTestMapping<TLabel> : BinarySimpleTestMapping<TLabel>
        {
            public override IEnumerable<TLabel> GetClassLabels(
                IReadOnlyList<Vector> featureVectors = null, IReadOnlyList<TLabel> labels = null)
            {
                if (0 is TLabel zero && 1 is TLabel one && 2 is TLabel two)
                {
                    // This array is intentionally out of order
                    return new[] { one, two, zero };
                }
                else throw new NotImplementedException();
            }
        }

        /// <summary>
        /// An abstract base class implementation of <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues}"/> 
        /// for <see cref="StandardDataset"/>.
        /// </summary>
        [Serializable]
        private abstract class StandardBayesPointMachineClassifierTestMapping :
            ClassifierMapping<StandardDataset, string, StandardDataset, string, Vector>, ICustomSerializable
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="StandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            protected StandardBayesPointMachineClassifierTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="StandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            protected StandardBayesPointMachineClassifierTestMapping(IReader reader)
            {
            }

            /// <summary>
            /// Retrieves a list of instances from a given source.
            /// </summary>
            /// <param name="instanceSource">The source of instances.</param>
            /// <returns>The retrieved list of instances.</returns>
            public override IEnumerable<string> GetInstances(StandardDataset instanceSource)
            {
                if (instanceSource == null)
                {
                    return null;
                }

                if (instanceSource.FeatureVectors == null)
                {
                    return instanceSource.Labels == null ? null : instanceSource.Labels.Keys;
                }

                if (instanceSource.Labels == null)
                {
                    return instanceSource.FeatureVectors.Keys;
                }

                return instanceSource.FeatureVectors.Keys.Count > instanceSource.Labels.Keys.Count
                           ? (IEnumerable<string>)instanceSource.FeatureVectors.Keys
                           : instanceSource.Labels.Keys;
            }

            /// <summary>
            /// Provides the features for a given instance.
            /// </summary>
            /// <param name="instance">The instance to provide features for.</param>
            /// <param name="instanceSource">The source of instances.</param>
            /// <returns>The features for the given instance.</returns>
            public override Vector GetFeatures(string instance, StandardDataset instanceSource)
            {
                if (instance == null || instanceSource == null || instanceSource.FeatureVectors == null)
                {
                    return null;
                }

                Vector featureVector;

                if (!instanceSource.FeatureVectors.TryGetValue(instance, out featureVector))
                {
                    // Return null for test purposes
                    return null;
                }

                return featureVector;
            }

            /// <summary>
            /// Saves the state of the standard data mapping using a writer to a binary stream.
            /// </summary>
            /// <param name="writer">The writer to save the state of the standard data mapping to.</param>
            public void SaveForwardCompatible(IWriter writer)
            {
                // Nothing to serialize.
            }
        }

        /// <summary>
        /// An implementation of <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues}"/> 
        /// for <see cref="StandardDataset"/> and binary class labels.
        /// </summary>
        [Serializable]
        private class BinaryStandardBayesPointMachineClassifierTestMapping : StandardBayesPointMachineClassifierTestMapping
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="BinaryStandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            public BinaryStandardBayesPointMachineClassifierTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="BinaryStandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public BinaryStandardBayesPointMachineClassifierTestMapping(IReader reader) : base(reader)
            {
            }

            /// <summary>
            /// Gets all class labels.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">The label source.</param>
            /// <returns>All possible values of a label.</returns>
            public override IEnumerable<string> GetClassLabels(StandardDataset instanceSource, StandardDataset labelSource)
            {
                if (instanceSource == null)
                {
                    return null;
                }

                return instanceSource.BinaryClassLabels;
            }

            /// <summary>
            /// Provides the label for a given instance.
            /// </summary>
            /// <param name="instance">The instance to provide the label for.</param>
            /// <param name="instanceSource">The source of instances.</param>
            /// <param name="labelSource">The source of labels.</param>
            /// <returns>The label of the given instance.</returns>
            public override string GetLabel(string instance, StandardDataset instanceSource, StandardDataset labelSource)
            {
                if (instance == null || instanceSource == null || instanceSource.Labels == null)
                {
                    return null;
                }

                string label;
                if (!instanceSource.BinaryLabels.TryGetValue(instance, out label))
                {
                    // Return null for test purposes
                    return null;
                }

                return label;
            }
        }

        /// <summary>
        /// An implementation of <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatureValues}"/> 
        /// for <see cref="StandardDataset"/> and multiple class labels.
        /// </summary>
        [Serializable]
        private class MulticlassStandardBayesPointMachineClassifierTestMapping : StandardBayesPointMachineClassifierTestMapping
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="MulticlassStandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            public MulticlassStandardBayesPointMachineClassifierTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="MulticlassStandardBayesPointMachineClassifierTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public MulticlassStandardBayesPointMachineClassifierTestMapping(IReader reader) : base(reader)
            {
            }

            /// <summary>
            /// Gets all class labels.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <param name="labelSource">The label source.</param>
            /// <returns>All possible values of a label.</returns>
            public override IEnumerable<string> GetClassLabels(StandardDataset instanceSource, StandardDataset labelSource)
            {
                if (instanceSource == null)
                {
                    return null;
                }

                return instanceSource.ClassLabels;
            }

            /// <summary>
            /// Provides the label for a given instance.
            /// </summary>
            /// <param name="instance">The instance to provide the label for.</param>
            /// <param name="instanceSource">The source of instances.</param>
            /// <param name="labelSource">The source of labels.</param>
            /// <returns>The label of the given instance.</returns>
            public override string GetLabel(string instance, StandardDataset instanceSource, StandardDataset labelSource)
            {
                if (instance == null || instanceSource == null || instanceSource.Labels == null)
                {
                    return null;
                }

                string label;
                if (!instanceSource.Labels.TryGetValue(instance, out label))
                {
                    // Return null for test purposes
                    return null;
                }

                return label;
            }
        }

        #endregion
    }
}
