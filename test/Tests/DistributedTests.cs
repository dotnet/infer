// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Xunit;
using System.IO;
using System.Linq;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    using Assert = Xunit.Assert;
    using BernoulliArray = DistributionStructArray<Bernoulli, bool>;
    using BernoulliArrayArray = DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>;
    using BetaArray = DistributionStructArray<Beta, double>;
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using GaussianArrayArray = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using DirichletArray = DistributionRefArray<Dirichlet, Vector>;
    using System.Threading.Tasks;
    using Microsoft.ML.Probabilistic.Compiler;
    using Microsoft.ML.Probabilistic.Algorithms;
    using Microsoft.ML.Probabilistic.Models.Attributes;

    public class DistributedTests
    {
        // trust_rep_F, trust_rep_F_fileBlock__marginal, isMalware_rep_F, isMalware_rep_F_clientBlock__marginal, isMalware_itemfileOfDownload_clientBlock__fileBlock__download__F should be re-created each loop
        // original: 10990 file reads 4806 file writes
        // 7456 file reads 5170 file writes
        internal void DistributedTrustTest()
        {
            Variable<int> numClientBlocks = Variable.New<int>().Named("numClientBlocks");
            Range clientBlock = new Range(numClientBlocks).Named("clientBlock");
            VariableArray<int> numClientsInBlock = Variable.Array<int>(clientBlock).Named("numClientsInBlock");
            Range client = new Range(numClientsInBlock[clientBlock]).Named("client");
            var trust = Variable.Array(Variable.Array<double>(client), clientBlock).Named("trust");
            trust[clientBlock][client] = Variable.Beta(1, 1).ForEach(clientBlock, client);

            Variable<int> numFileBlocks = Variable.New<int>().Named("numFileBlocks");
            Range fileBlock = new Range(numFileBlocks).Named("fileBlock");
            VariableArray<int> numFilesInBlock = Variable.Array<int>(fileBlock).Named("numFilesInBlock");
            Range file = new Range(numFilesInBlock[fileBlock]).Named("file");
            var isMalwarePrior = Variable.IArray(Variable.Array<Bernoulli>(file), fileBlock).Named("isMalwarePrior");
            var isMalware = Variable.Array(Variable.Array<bool>(file), fileBlock).Named("isMalware");
            isMalware[fileBlock][file] = Variable<bool>.Random(isMalwarePrior[fileBlock][file]);

            var numDownloadsInBlock = Variable.Array<int>(clientBlock, fileBlock).Named("numDownloadsInBlock");
            Range download = new Range(numDownloadsInBlock[clientBlock, fileBlock]).Named("download");
            var clientOfDownload = Variable.IArray(Variable.Array<int>(download), clientBlock, fileBlock).Named("clientOfDownload");
            var fileOfDownload = Variable.IArray(Variable.Array<int>(download), clientBlock, fileBlock).Named("fileOfDownload");
            using (Variable.ForEach(clientBlock))
            using (Variable.ForEach(fileBlock))
            using (Variable.ForEach(download))
            {
                Variable<int> c = clientOfDownload[clientBlock, fileBlock][download];
                Variable<bool> thisIsMalware = isMalware[fileBlock][fileOfDownload[clientBlock, fileBlock][download]];
                using (Variable.If(thisIsMalware))
                {
                    Variable.ConstrainFalse(Variable.Bernoulli(trust[clientBlock][c]));
                }
                using (Variable.IfNot(thisIsMalware))
                {
                    Variable.ConstrainTrue(Variable.Bernoulli(trust[clientBlock][c]));
                }
            }

            var clientData = new int[] {0, 0, 1, 1, 2, 2, 3, 3};
            var fileData = new int[] {0, 1, 2, 3, 1, 3, 0, 2};
            var malwarePrior = new Bernoulli[]
                {
                    new Bernoulli(0.99), // file 0 is malware
                    new Bernoulli(0.5),
                    new Bernoulli(0.5),
                    new Bernoulli(0.01) // file 3 is not malware
                };
            // partition prior and data into blocks
            int clientBlockSize = 2;
            int fileBlockSize = 2;
            numClientBlocks.ObservedValue = 2;
            numFileBlocks.ObservedValue = 2;
            numClientsInBlock.ObservedValue = new int[] {2, 2};
            numFilesInBlock.ObservedValue = new int[] {2, 2};
            isMalwarePrior.ObservedValue = Util.IArrayFromFunc(numFileBlocks.ObservedValue,
                                                               fb => Util.ArrayInit(numFilesInBlock.ObservedValue[fb], f => malwarePrior[fb*2 + f]));
            int[,][] downloadsInBlock = Util.ArrayInit(numClientBlocks.ObservedValue, numFileBlocks.ObservedValue, (cb, fb) =>
                                                                                                                   System.Linq.Enumerable.Range(0, clientData.Length)
                                                                                                                         .Where(i =>
                                                                                                                                (cb*clientBlockSize <= clientData[i]) &&
                                                                                                                                (clientData[i] < (cb + 1)*clientBlockSize) &&
                                                                                                                                (fb*fileBlockSize <= fileData[i]) &&
                                                                                                                                (fileData[i] < (fb + 1)*fileBlockSize))
                                                                                                                         .ToArray());
            numDownloadsInBlock.ObservedValue = Util.ArrayInit(numClientBlocks.ObservedValue, numFileBlocks.ObservedValue, (cb, fb) => downloadsInBlock[cb, fb].Length);
            // convert global client indices into local indices within a block
            clientOfDownload.ObservedValue = Util.IArrayFromFunc(numClientBlocks.ObservedValue, numFileBlocks.ObservedValue, (cb, fb) =>
                                                                                                                             downloadsInBlock[cb, fb].Select(
                                                                                                                                 i => clientData[i]%clientBlockSize).ToArray());
            fileOfDownload.ObservedValue = Util.IArrayFromFunc(numClientBlocks.ObservedValue, numFileBlocks.ObservedValue, (cb, fb) =>
                                                                                                                           downloadsInBlock[cb, fb].Select(
                                                                                                                               i => fileData[i]%fileBlockSize).ToArray());

            InferenceEngine engine = new InferenceEngine();
            engine.ShowTimings = true;
            download.AddAttribute(new Sequential());
            IList<BernoulliArray> isMalwareExpected = engine.Infer<IList<BernoulliArray>>(isMalware);
            IList<BetaArray> trustExpected = engine.Infer<IList<BetaArray>>(trust);

            clientBlock.AddAttribute(new Partitioned());
            fileBlock.AddAttribute(new Partitioned());
            IList<BernoulliArray> isMalwarePost = engine.Infer<IList<BernoulliArray>>(isMalware);
            IList<BetaArray> trustPost = engine.Infer<IList<BetaArray>>(trust);
            if (false)
            {
                Console.WriteLine("isMalware = ");
                Console.WriteLine(isMalwarePost);
                Console.WriteLine("trust = ");
                Console.WriteLine(trustPost);
            }
            else
            {
                Console.WriteLine(StringUtil.JoinColumns("isMalware = ", StringUtil.VerboseToString(isMalwarePost), " should be ", isMalwareExpected));
                Console.WriteLine(StringUtil.JoinColumns("trust = ", StringUtil.VerboseToString(trustPost), " should be ", trustExpected));
                for (int fb = 0; fb < isMalwarePost.Count; fb++)
                {
                    Assert.True(isMalwareExpected[fb].MaxDiff(isMalwarePost[fb]) < 1e-10);
                }
                for (int cb = 0; cb < trustPost.Count; cb++)
                {
                    Assert.True(trustExpected[cb].MaxDiff(trustPost[cb]) < 1e-10);
                }
            }
        }

        // example created for Christian Seifert <chriseif@microsoft.com>
        internal void TrustTest()
        {
            Variable<int> nClients = Variable.New<int>().Named("nClients");
            Range client = new Range(nClients).Named("client");
            VariableArray<double> trust = Variable.Array<double>(client).Named("trust");
            trust[client] = Variable.Beta(1, 1).ForEach(client);

            Variable<int> nFiles = Variable.New<int>().Named("nFiles");
            Range file = new Range(nFiles).Named("file");
            VariableArray<Bernoulli> isMalwarePrior = Variable.Array<Bernoulli>(file).Named("isMalwarePrior");
            VariableArray<bool> isMalware = Variable.Array<bool>(file).Named("isMalware");
            isMalware[file] = Variable<bool>.Random(isMalwarePrior[file]);

            Variable<int> nDownloads = Variable.New<int>().Named("nDownloads");
            Range download = new Range(nDownloads).Named("download");
            VariableArray<int> clientOfDownload = Variable.Array<int>(download).Named("clientOfDownload");
            VariableArray<int> fileOfDownload = Variable.Array<int>(download).Named("fileOfDownload");
            using (Variable.ForEach(download))
            {
                Variable<int> c = clientOfDownload[download];
                Variable<bool> thisIsMalware = isMalware[fileOfDownload[download]];
                using (Variable.If(thisIsMalware))
                {
                    Variable.ConstrainFalse(Variable.Bernoulli(trust[c]));
                }
                using (Variable.IfNot(thisIsMalware))
                {
                    Variable.ConstrainTrue(Variable.Bernoulli(trust[c]));
                }
            }

            nClients.ObservedValue = 2;
            nFiles.ObservedValue = 4;
            nDownloads.ObservedValue = 4;
            clientOfDownload.ObservedValue = new int[] {0, 0, 1, 1};
            fileOfDownload.ObservedValue = new int[] {0, 1, 2, 3};
            isMalwarePrior.ObservedValue = new Bernoulli[]
                {
                    new Bernoulli(0.99), // file 0 is malware
                    new Bernoulli(0.5),
                    new Bernoulli(0.5),
                    new Bernoulli(0.01) // file 3 is not malware
                };
            if (false)
            {
                var isMalwarePrior2 = new DistributionStructArray<Bernoulli, bool>(new Bernoulli[]
                    {
                        new Bernoulli(0.99), // file 0 is malware
                        new Bernoulli(0.5),
                        new Bernoulli(0.5),
                        new Bernoulli(0.01) // file 3 is not malware
                    });
                Console.WriteLine(isMalwarePrior2);
            }

            InferenceEngine engine = new InferenceEngine();
            download.AddAttribute(new Sequential());
            IList<Bernoulli> isMalwarePost = engine.Infer<IList<Bernoulli>>(isMalware);
            Console.WriteLine("isMalware = ");
            Console.WriteLine(isMalwarePost);
            Console.WriteLine("trust = ");
            Console.WriteLine(engine.Infer(trust));
        }

        [Fact]
        [Trait("Category", "DistributedTest")]
        public void DistributedDirichletArrayTest()
        {
            // 2 file reads 6 file writes
            DistributedDirichletArray(new VariationalMessagePassing());
            // 2 file reads 10 file writes
            DistributedDirichletArray(new ExpectationPropagation());
        }

        private void DistributedDirichletArray(IAlgorithm algorithm)
        {
            int[][][] data = new int[][][]
                {
                    new int[][]
                        {
                            new int[] {1, 5, 5, 8, 3},
                            new int[] {1, 2, 1, 5, 8}
                        },
                    new int[][]
                        {
                            new int[] {5, 3, 3, 3, 3},
                            new int[] {6, 7, 6, 5, 8}
                        }
                };
            int maxV = 9;
            Range block = new Range(data.Length).Named("block");
            Range outer = new Range(data[0].Length).Named("outer");
            Range inner = new Range(data[0][0].Length).Named("inner");

            DirichletArray prior = new DirichletArray(new Dirichlet[]
                {
                    Dirichlet.Uniform(maxV),
                    Dirichlet.Uniform(maxV),
                    Dirichlet.Uniform(maxV),
                    Dirichlet.Uniform(maxV),
                    Dirichlet.Uniform(maxV)
                });

            VariableArray<Vector> phi = Variable.Array<Vector>(inner).Named("phi");
            //phi[inner] = Variable.DirichletUniform(maxV).ForEach(inner);
            phi.SetTo(Variable<Vector[]>.Random(prior));
            var x = Variable.IArray(Variable.Array(Variable.Array<int>(inner), outer), block).Named("x");
            using (Variable.ForEach(block))
            {
                using (Variable.ForEach(outer))
                {
                    using (Variable.ForEach(inner))
                    {
                        x[block][outer][inner] = Variable.Discrete(phi[inner]);
                    }
                }
            }
            if (true)
            {
                x.ObservedValue = Util.IArrayFromFunc(data.Length, b => data[b]);
            }
            else
            {
                var LoadBlock = Variable.New<Func<int, int[][]>>().Named("LoadBlock");
                ConstrainFromIndex(x, b => VariableEvaluate(LoadBlock, b));
                LoadBlock.ObservedValue = b => data[b];
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            var phiExpected = engine.Infer<IDistribution<Vector[]>>(phi);

            InferenceEngine engine2 = new InferenceEngine();
            engine2.ShowTimings = true;
            engine2.Algorithm = algorithm;
            SetAttribute(block, new Partitioned());
            var phiActual = engine2.Infer<IDistribution<Vector[]>>(phi);
            Console.WriteLine(StringUtil.JoinColumns("phi = ", phiActual, " should be ", phiExpected));
            Assert.True(phiExpected.MaxDiff(phiActual) < 1e-8);
        }

        [Fact]
        [Trait("Category", "DistributedTest")]
        public void DistributedDirichletTest()
        {
            // original: 7 file reads 12 file writes
            DistributedDirichlet(new VariationalMessagePassing());
            // Divide: 35 file reads 60 file writes
            // phi_rep0_B doesn't merge with phi_rep0_rep0_B_toDef if they have different types
            // evidence_selector_cases_0_rep1_uses_B has an extra write
            // evidence_selector_cases_0_rep1_uses_B__ind_local_user_local should be declared inside of _ind loop
            // evidence_selector_cases_0_rep1_uses_B should be indexed by user first (from Channel2)
            DistributedDirichlet(new ExpectationPropagation());
        }

        private void DistributedDirichlet(IAlgorithm algorithm)
        {
            int[][] dataSets = new int[][]
                {
                    new int[] {1, 5, 8, 3},
                    new int[] {7},
                    new int[] {1, 2, 1, 5, 8}
                };
            int maxV = 9;

            // The model
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Dirichlet priorMean = Dirichlet.Uniform(maxV);
            Variable<Vector> phi = Variable<Vector>.Random(priorMean).Named("phi");
            Range user = new Range(dataSets.Length).Named("user");
            var dataCount = Variable.IArray<int>(user).Named("dataCount");
            Range item = new Range(dataCount[user]).Named("item");
            var data = Variable.IArray(Variable.Array<int>(item), user).Named("data");
            data[user][item] = Variable.Discrete(phi).ForEach(user, item);
            if (true)
            {
                data.ObservedValue = Util.IArrayFromFunc(dataSets.Length, u => dataSets[u]);
                dataCount.ObservedValue = Util.IArrayFromFunc(dataSets.Length, u => dataSets[u].Length);
            }
            else
            {
                Variable<Func<int, int[]>> LoadUserData = Variable.New<Func<int, int[]>>().Named("LoadUserData");
                ConstrainFromIndex(data, b => VariableEvaluate(LoadUserData, b));
                Variable<Func<int, int>> LoadUserDataCount = Variable.New<Func<int, int>>().Named("LoadUserDataCount");
                SetFromIndex(dataCount, b => VariableEvaluate(LoadUserDataCount, b));
                LoadUserData.ObservedValue = u => dataSets[u];
                LoadUserDataCount.ObservedValue = u => dataSets[u].Length;
            }
            block.CloseBlock();

            // Set the inference algorithm
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = algorithm;
            Dirichlet phiExpected = engine.Infer<Dirichlet>(phi);
            double evExpected = engine.Infer<Bernoulli>(evidence).LogOdds;

            InferenceEngine engine2 = new InferenceEngine();
            engine2.ShowTimings = true;
            engine2.Algorithm = algorithm;
            //phi.AddAttribute(new GivePriorityTo(typeof(ReplicateOp_NoDivide)));
            SetAttribute(user, new Partitioned());
            Dirichlet phiActual = engine2.Infer<Dirichlet>(phi);
            double evActual = engine2.Infer<Bernoulli>(evidence).LogOdds;

            Console.WriteLine("phi = {0} should be {1}", phiActual, phiExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Assert.True(phiExpected.MaxDiff(phiActual) < 1e-4);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-6) < 1e-4);
        }

        [Fact]
        [Trait("Category", "DistributedTest")]
        public void DistributedRegressionTest()
        {
            double[][] inputs = new double[][] {new double[] {1, 2}, new double[] {3, 4}};
            double[] outputs = new double[] {7.9, 1.3};

            int ndims = inputs[0].Length;
            Range dim = new Range(ndims).Named("dim");

            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(dim);
            int dataCount = outputs.Length;
            Range item = new Range(dataCount).Named("item");
            var x = Variable.IArray(Variable.Array<double>(dim), item).Named("x");
            var product = Variable.Array(Variable.Array<double>(dim), item).Named("product");
            product[item][dim] = w[dim]*x[item][dim];
            VariableArray<double> y = Variable.Array<double>(item).Named("y");
            y[item] = Variable.Sum(product[item]);
            var yNoisy = Variable.IArray<double>(item).Named("yNoisy");
            yNoisy[item] = Variable.GaussianFromMeanAndVariance(y[item], 0.1).Named("yNoisy");

            if (true)
            {
                x.ObservedValue = Util.IArrayFromFunc(inputs.Length, i => inputs[i]);
                yNoisy.ObservedValue = Util.IArrayFromFunc(outputs.Length, i => outputs[i]);
            }
            else
            {
                Variable<Func<int, double[]>> LoadInput = Variable.New<Func<int, double[]>>();
                SetFromIndex(x, b => VariableEvaluate(LoadInput, b));
                Variable<Func<int, double>> LoadOutput = Variable.New<Func<int, double>>();
                ConstrainFromIndex(yNoisy, b => VariableEvaluate(LoadOutput, b));
                LoadInput.ObservedValue = i => inputs[i];
                LoadOutput.ObservedValue = i => outputs[i];
            }
            w.AddAttribute(new InitialiseBackward());

            // does not work with VMP because Sum(array) cannot take a derived argument
            InferenceEngine engine = new InferenceEngine(); //new VariationalMessagePassing());
            engine.Compiler.GivePriorityTo(typeof (ReplicateOp_Divide));
            engine.NumberOfIterations = 100;
            var expectedW = engine.Infer<IDistribution<double[]>>(w);
            Console.WriteLine(expectedW);
            Console.WriteLine("should be:\n[0] Gaussian(-8.106, 0.0748)\n[1] Gaussian(6.69, 0.03759)");

            InferenceEngine engine2 = new InferenceEngine();
            engine2.ShowTimings = true;
            engine2.Compiler.GivePriorityTo(typeof (ReplicateOp_Divide));
            engine2.OptimiseForVariables = new List<IVariable>() {w};
            engine2.NumberOfIterations = 100;
            SetAttribute(item, new Partitioned());
            var actualW = engine2.Infer<IDistribution<double[]>>(w);
            Console.WriteLine(StringUtil.JoinColumns("w = ", actualW, " should be ", expectedW));
            Assert.True(expectedW.MaxDiff(actualW) < 1e-4);

            FileStats.Clear();
            var gen = new DistributedRegression_EP();
            gen.x = x.ObservedValue;
            gen.yNoisy = yNoisy.ObservedValue;
            gen.Execute(100);
            var actualW2 = gen.WMarginal();
            Console.WriteLine(StringUtil.JoinColumns("w = ", actualW2, " should be ", expectedW));
            Assert.True(expectedW.MaxDiff(actualW2) < 1e-4);
            Console.WriteLine("{0} file reads {1} file writes", FileStats.ReadCount, FileStats.WriteCount);
        }

        internal void SparseFactorizedBayesPointMachineExample2()
        {
            int numObsBlocks = 1;
            int numFeatureBlocks = 1;
            Range obsBlock = new Range(numObsBlocks);
            var sizeOfObsBlock = Variable.IArray<int>(obsBlock);
            Range obs = new Range(sizeOfObsBlock[obsBlock]);
            Range featureBlock = new Range(numFeatureBlocks);
            var sizeOfFeatureBlock = Variable.IArray<int>(featureBlock);
            Range feature = new Range(sizeOfFeatureBlock[featureBlock]);
            var w = Variable.Array(Variable.Array<double>(feature), featureBlock);
            w[featureBlock][feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(featureBlock, feature);
            var xValueCount = Variable.IArray(Variable.Array(Variable.Array<int>(featureBlock), obs), obsBlock);
            Range featureOfObs = new Range(xValueCount[obsBlock][obs][featureBlock]).Named("userFeature");
            var y = Variable.IArray(Variable.Array<bool>(obs), obsBlock).Named("y");
            var numObsInFeatureBlock = Variable.Array(Variable.Array<int>(featureBlock), obsBlock);
            Range obsInFeatureBlock = new Range(numObsInFeatureBlock[obsBlock][featureBlock]);
            var xValues = Variable.IArray(Variable.Array(Variable.Array(Variable.Array<double>(featureOfObs), obsInFeatureBlock), featureBlock), obsBlock);
            var xIndices = Variable.IArray(Variable.Array(Variable.Array(Variable.Array<int>(featureOfObs), obsInFeatureBlock), featureBlock), obsBlock);
            var numFeatureBlocksOfObs = Variable.IArray(Variable.Array<int>(obs), obsBlock);
            Range featureBlockOfObs = new Range(numFeatureBlocksOfObs[obsBlock][obs]); // not partitioned
            //var featureBlockOf = Variable.Array(Variable.Array(Variable.Array<int>(featureBlockOfObs), obs), obsBlock);
            //var obsIndexInFeatureBlock = Variable.Array(Variable.Array(Variable.Array<int>(featureBlockOfObs), obs), obsBlock);
            var obsIndex = Variable.IArray(Variable.Array(Variable.Array<int>(obsInFeatureBlock), featureBlock), obsBlock);
            using (Variable.ForEach(obsBlock))
            {
                var sumOfBlock = Variable.Array(Variable.Array<double>(obsInFeatureBlock), featureBlock);
                using (Variable.ForEach(featureBlock))
                using (Variable.ForEach(obsInFeatureBlock))
                {
                    VariableArray<double> wSparse = Variable.Subarray(w[featureBlock], xIndices[obsBlock][featureBlock][obsInFeatureBlock]);
                    VariableArray<double> product = Variable.Array<double>(featureOfObs);
                    product[featureOfObs] = xValues[obsBlock][featureBlock][obsInFeatureBlock][featureOfObs]*wSparse[featureOfObs];
                    sumOfBlock[featureBlock][obsInFeatureBlock] = Variable.Sum(product);
                }
                var sumOfBlockTransposed = Transpose<double>(sumOfBlock, obsIndex[obsBlock], obs, featureBlockOfObs);
                using (Variable.ForEach(obs))
                {
                    Variable<double> score = Variable.Sum(sumOfBlockTransposed[obs]);
                    y[obsBlock][obs] = (score > 0);
                }
            }
        }

        private VariableArray<VariableArray<T>, T[][]> Transpose<T>(
            VariableArray<VariableArray<T>, T[][]> array,
            VariableArray<VariableArray<int>, int[][]> indices,
            Range r1,
            Range r2)
        {
            return array;
        }

        internal void DistributedSFBPMTest()
        {
            int numObs = 4;
            int numFeatures = 4;
            double[][] xValuesData = Util.ArrayInit(numObs, i => Util.ArrayInit(numFeatures, f => i + f + 1.0));
            int[][] xIndicesData = Util.ArrayInit(numObs, i => Util.ArrayInit(numFeatures, f => f));
            bool[] yData = Util.ArrayInit(numObs, i => true);
            int numObsBlocks = 2;
            int numFeatureBlocks = 2;
            Range obsBlock = new Range(numObsBlocks);
            var sizeOfObsBlock = Variable.IArray<int>(obsBlock);
            Range obs = new Range(sizeOfObsBlock[obsBlock]);
            Range featureBlock = new Range(numFeatureBlocks);
            var sizeOfFeatureBlock = Variable.IArray<int>(featureBlock);
            Range feature = new Range(sizeOfFeatureBlock[featureBlock]);
            var w = Variable.Array(Variable.Array<double>(feature), featureBlock);
            w[featureBlock][feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(featureBlock, feature);
            var xValueCount = Variable.IArray(Variable.Array(Variable.Array<int>(featureBlock), obs), obsBlock);
            Range featureOfObs = new Range(xValueCount[obsBlock][obs][featureBlock]).Named("userFeature");
            var xValues = Variable.IArray(Variable.Array(Variable.Array(Variable.Array<double>(featureOfObs), featureBlock), obs), obsBlock);
            var xIndices = Variable.IArray(Variable.Array(Variable.Array(Variable.Array<int>(featureOfObs), featureBlock), obs), obsBlock);
            var y = Variable.IArray(Variable.Array<bool>(obs), obsBlock).Named("y");
            using (Variable.ForEach(obsBlock))
            using (Variable.ForEach(obs))
            {
                VariableArray<double> sumOfBlock = Variable.Array<double>(featureBlock);
                using (Variable.ForEach(featureBlock))
                {
                    VariableArray<double> wSparse = Variable.Subarray(w[featureBlock], xIndices[obsBlock][obs][featureBlock]);
                    VariableArray<double> product = Variable.Array<double>(featureOfObs);
                    product[featureOfObs] = xValues[obsBlock][obs][featureBlock][featureOfObs]*wSparse[featureOfObs];
                    sumOfBlock[featureBlock] = Variable.Sum(product);
                }
                Variable<double> score = Variable.Sum(sumOfBlock);
                y[obsBlock][obs] = (score > 0);
            }
            InferenceEngine engine = new InferenceEngine();
            obsBlock.AddAttribute(new Partitioned());
            featureBlock.AddAttribute(new Partitioned());
            int obsBlockSize = numObs/numObsBlocks;
            int featureBlockSize = numFeatures/numFeatureBlocks;
            sizeOfObsBlock.ObservedValue = Util.IArrayFromFunc(numObsBlocks, ob => obsBlockSize);
            sizeOfFeatureBlock.ObservedValue = Util.IArrayFromFunc(numFeatureBlocks, fb => featureBlockSize);
            // divide the data into blocks
            y.ObservedValue = Util.IArrayFromFunc(numObsBlocks, ob => Util.ArrayInit(obsBlockSize, i => yData[ob*obsBlockSize + i]));
            int[,][] indicesInBlock = Util.ArrayInit(numObs, numFeatureBlocks, (i, fb) =>
                                                                               System.Linq.Enumerable.Range(0, xIndicesData[i].Length).Where(j =>
                                                                                                                                             (fb*featureBlockSize <=
                                                                                                                                              xIndicesData[i][j]) &&
                                                                                                                                             (xIndicesData[i][j] <
                                                                                                                                              (fb + 1)*featureBlockSize))
                                                                                     .ToArray());
            xValueCount.ObservedValue = Util.IArrayFromFunc(numObsBlocks, ob => Util.ArrayInit(obsBlockSize, i => Util.ArrayInit(numFeatureBlocks, fb =>
                                                                                                                                                   indicesInBlock[
                                                                                                                                                       ob*obsBlockSize + i, fb]
                                                                                                                                                       .Length)));
            xValues.ObservedValue = Util.IArrayFromFunc(numObsBlocks, ob => Util.ArrayInit(obsBlockSize, i => Util.ArrayInit(numFeatureBlocks, fb =>
                                                                                                                                               indicesInBlock[
                                                                                                                                                   ob*obsBlockSize + i, fb]
                                                                                                                                                   .Select(
                                                                                                                                                       j =>
                                                                                                                                                       xValuesData[
                                                                                                                                                           ob*obsBlockSize + i]
                                                                                                                                                           [j]).ToArray())));
            xIndices.ObservedValue = Util.IArrayFromFunc(numObsBlocks, ob => Util.ArrayInit(obsBlockSize, i => Util.ArrayInit(numFeatureBlocks, fb =>
                                                                                                                                                indicesInBlock[
                                                                                                                                                    ob*obsBlockSize + i, fb]
                                                                                                                                                    .Select(
                                                                                                                                                        j =>
                                                                                                                                                        xIndicesData[
                                                                                                                                                            ob*obsBlockSize + i
                                                                                                                                                            ][j]%
                                                                                                                                                        featureBlockSize)
                                                                                                                                                    .ToArray())));
            IList<GaussianArray> wActualB = engine.Infer<IList<GaussianArray>>(w);
            GaussianArray wActual = new GaussianArray(numFeatures, f => wActualB[f/featureBlockSize][f%featureBlockSize]);
            GaussianArray wExpected;
            SparseFactorizedBayesPointMachineExample(yData, numFeatures, xValuesData, xIndicesData, out wExpected);
            Console.WriteLine(StringUtil.JoinColumns("w = ", wActual, " should be ", wExpected));
            Assert.True(wExpected.MaxDiff(wActual) < 1e-8);
        }

        private void SparseFactorizedBayesPointMachineExample(bool[] yData, int nFeatures, double[][] xValuesData, int[][] xIndicesData, out GaussianArray wPost)
        {
            Variable<int> numObservations = Variable.New<int>();
            Range obs = new Range(numObservations);
            //Variable<int> nFeatures = Variable.New<int>().Named("nFeatures");
            Range feature = new Range(nFeatures).Named("feature");
            VariableArray<double> w = Variable.Array<double>(feature).Named("w");
            w[feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(feature);
            VariableArray<int> xValueCount = Variable.Array<int>(obs).Named("xCount");
            Range featureOfObs = new Range(xValueCount[obs]).Named("userFeature");
            var xValues = Variable.Array(Variable.Array<double>(featureOfObs), obs).Named("xValues");
            var xIndices = Variable.Array(Variable.Array<int>(featureOfObs), obs).Named("xIndices");
            VariableArray<bool> y = Variable.Array<bool>(obs).Named("y");
            using (Variable.ForEach(obs))
            {
                VariableArray<double> product = Variable.Array<double>(featureOfObs).Named("product");
                VariableArray<double> wSparse = Variable.Subarray(w, xIndices[obs]);
                product[featureOfObs] = xValues[obs][featureOfObs]*wSparse[featureOfObs];
                Variable<double> score = Variable.Sum(product).Named("score");
                y[obs] = (score > 0);
            }
            y.ObservedValue = yData;
            xValues.ObservedValue = xValuesData;
            xIndices.ObservedValue = xIndicesData;
            numObservations.ObservedValue = yData.Length;
            xValueCount.ObservedValue = Util.ArrayInit(xValuesData.Length, i => xValuesData[i].Length);
            InferenceEngine engine = new InferenceEngine();
            wPost = engine.Infer<GaussianArray>(w);
        }

        internal void DistributedTrueSkillTest2()
        {
            int nPlayers = 4;
            int[] winnerData = new int[] {2, 2, 3, 3};
            int[] loserData = new int[] {0, 1, 0, 2};
            int numBlocks = 2;
            Range block = new Range(numBlocks).Named("block");
            var sizeOfBlock = Variable.IArray<int>(block).Named("sizeOfBlock");
            Range player = new Range(sizeOfBlock[block]).Named("player");
            Range block2 = block.Clone().Named("block2");
            Range player2 = new Range(sizeOfBlock[block2]).Named("player2");
            var winnerSkill = Variable.Array(Variable.Array<double>(player), block).Named("winnerSkill");
            winnerSkill[block][player] = Variable.GaussianFromMeanAndVariance(0, 100*2).ForEach(block, player);
            var loserSkill = Variable.Array(Variable.Array<double>(player2), block2).Named("loserSkill");
            loserSkill[block2][player2] = Variable.GaussianFromMeanAndVariance(0, 100*2).ForEach(block2, player2);
            var numGames = Variable.IArray<int>(block, block2).Named("numGames");
            Range game = new Range(numGames[block, block2]).Named("game");
            var winner = Variable.IArray(Variable.Array<int>(game), block, block2).Named("winner");
            var loser = Variable.IArray(Variable.Array<int>(game), block, block2).Named("loser");
            using (Variable.ForEach(block))
            {
                using (Variable.ForEach(block2))
                {
                    using (Variable.ForEach(game))
                    {
                        Variable<double> winner_performance = Variable.GaussianFromMeanAndVariance(winnerSkill[block][winner[block, block2][game]], 1);
                        Variable<double> loser_performance = Variable.GaussianFromMeanAndVariance(loserSkill[block2][loser[block, block2][game]], 1);
                        Variable.ConstrainTrue(winner_performance > loser_performance);
                    }
                }
                using (var fb = Variable.ForEach(player))
                {
                    Variable.ConstrainEqual(winnerSkill[block][player], loserSkill[block][fb.Index]);
                }
            }
            InferenceEngine engine = new InferenceEngine();
            SetAttribute(block, new Partitioned());
            SetAttribute(block2, new Partitioned());
            int blockSize = nPlayers/numBlocks;
            sizeOfBlock.ObservedValue = Util.IArrayFromFunc(numBlocks, b => blockSize);
            // partition data into blocks
            int[,][] gamesInBlock = Util.ArrayInit(numBlocks, numBlocks, (wb, lb) =>
                                                                         System.Linq.Enumerable.Range(0, winnerData.Length).Where(i =>
                                                                                                                                  (wb*blockSize <= winnerData[i]) &&
                                                                                                                                  (winnerData[i] < (wb + 1)*blockSize) &&
                                                                                                                                  (lb*blockSize <= loserData[i]) &&
                                                                                                                                  (loserData[i] < (lb + 1)*blockSize))
                                                                               .ToArray());
            numGames.ObservedValue = Util.IArrayFromFunc(numBlocks, numBlocks, (wb, lb) => gamesInBlock[wb, lb].Length);
            winner.ObservedValue = Util.IArrayFromFunc(numBlocks, numBlocks, (wb, lb) => gamesInBlock[wb, lb].Select(i => winnerData[i]%blockSize).ToArray());
            loser.ObservedValue = Util.IArrayFromFunc(numBlocks, numBlocks, (wb, lb) => gamesInBlock[wb, lb].Select(i => loserData[i]%blockSize).ToArray());
            IList<GaussianArray> skillActualB = engine.Infer<IList<GaussianArray>>(winnerSkill);
            GaussianArray skillActual = new GaussianArray(nPlayers, p => skillActualB[p/blockSize][p%blockSize]);
            IDistribution<double[]> skillExpected;
            TrueSkillExample(nPlayers, winnerData, loserData, out skillExpected);
            Console.WriteLine(StringUtil.JoinColumns("skill = ", skillActual, " should be ", skillExpected));
            Assert.True(skillExpected.MaxDiff(skillActual) < 1e-8);
        }

        // 8760 file reads 4498 file writes
        // should prune: loserPerf_F, winnerPerf_use_B
        // these don't get pruned because their initialization needs to be duplicated
        internal void DistributedTrueSkillTest()
        {
            int nPlayers = 4;
            int[] winnerData = new int[] {2, 2, 3, 3};
            int[] loserData = new int[] {0, 1, 0, 2};
            int numBlocks = 2;
            Range block = new Range(numBlocks).Named("block");
            var sizeOfBlock = Variable.IArray<int>(block).Named("sizeOfBlock");
            Range player = new Range(sizeOfBlock[block]).Named("player");
            Range block2 = block.Clone().Named("block2");
            Range player2 = new Range(sizeOfBlock[block2]).Named("player2");
            var skill = Variable.Array(Variable.Array<double>(player), block).Named("skill");
            skill[block][player] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(block, player);
            var numGames = Variable.IArray<int>(block, block2).Named("numGames");
            Range game = new Range(numGames[block, block2]).Named("game");
            var winner = Variable.IArray(Variable.Array<int>(game), block, block2).Named("winner");
            var loser = Variable.IArray(Variable.Array<int>(game), block, block2).Named("loser");
            using (Variable.ForEach(block))
            using (Variable.ForEach(block2))
            using (Variable.ForEach(game))
            {
                Variable<double> winner_performance = Variable.GaussianFromMeanAndVariance(skill[block][winner[block, block2][game]], 1).Named("winnerPerf");
                Variable<double> loser_performance = Variable.GaussianFromMeanAndVariance(skill[block2][loser[block, block2][game]], 1).Named("loserPerf");
                Variable.ConstrainTrue(winner_performance > loser_performance);
            }
            InferenceEngine engine = new InferenceEngine();
            engine.ShowTimings = true;
            SetAttribute(block, new Partitioned());
            SetAttribute(block2, new Partitioned());
            int blockSize = nPlayers/numBlocks;
            sizeOfBlock.ObservedValue = Util.IArrayFromFunc(numBlocks, b => blockSize);
            // partition data into blocks
            int[,][] gamesInBlock = Util.ArrayInit(numBlocks, numBlocks, (wb, lb) =>
                                                                         System.Linq.Enumerable.Range(0, winnerData.Length).Where(i =>
                                                                                                                                  (wb*blockSize <= winnerData[i]) &&
                                                                                                                                  (winnerData[i] < (wb + 1)*blockSize) &&
                                                                                                                                  (lb*blockSize <= loserData[i]) &&
                                                                                                                                  (loserData[i] < (lb + 1)*blockSize))
                                                                               .ToArray());
            numGames.ObservedValue = Util.IArrayFromFunc(numBlocks, numBlocks, (wb, lb) => gamesInBlock[wb, lb].Length);
            winner.ObservedValue = Util.IArrayFromFunc(numBlocks, numBlocks, (wb, lb) => gamesInBlock[wb, lb].Select(i => winnerData[i]%blockSize).ToArray());
            loser.ObservedValue = Util.IArrayFromFunc(numBlocks, numBlocks, (wb, lb) => gamesInBlock[wb, lb].Select(i => loserData[i]%blockSize).ToArray());
            IList<GaussianArray> skillActualB = engine.Infer<IList<GaussianArray>>(skill);
            GaussianArray skillActual = new GaussianArray(nPlayers, p => skillActualB[p/blockSize][p%blockSize]);
            IDistribution<double[]> skillExpected;
            TrueSkillExample(nPlayers, winnerData, loserData, out skillExpected);
            Console.WriteLine(StringUtil.JoinColumns("skill = ", skillActual, " should be ", skillExpected));
            Assert.True(skillExpected.MaxDiff(skillActual) < 1e-8);
        }

        private void TrueSkillExample(int nPlayers, int[] winnerData, int[] loserData, out IDistribution<double[]> skillPost)
        {
            int nGames = winnerData.Length;
            Range player = new Range(nPlayers).Named("player");
            VariableArray<double> skill = Variable.Array<double>(player).Named("skill");
            skill[player] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(player);
            Range game = new Range(nGames).Named("game");
            VariableArray<int> winner = Variable.Observed(winnerData, game).Named("winner");
            VariableArray<int> loser = Variable.Observed(loserData, game).Named("loser");
            Variable<double> winner_performance = Variable.GaussianFromMeanAndVariance(skill[winner[game]], 1);
            Variable<double> loser_performance = Variable.GaussianFromMeanAndVariance(skill[loser[game]], 1);
            Variable.ConstrainTrue(winner_performance > loser_performance);
            InferenceEngine engine = new InferenceEngine();
            skillPost = engine.Infer<IDistribution<double[]>>(skill);
        }

        internal void DistributedMatchboxExample()
        {
            bool[] data = new bool[1];
            int numTraits = 1;
            double affinityNoiseVar = 1;
            int numUserBlocks = 1, numItemBlocks = 1;
            Range userBlock = new Range(numUserBlocks);
            VariableArray<int> sizeOfUserBlock = Variable.Array<int>(userBlock);
            Range user = new Range(sizeOfUserBlock[userBlock]).Named("user");
            Range itemBlock = new Range(numItemBlocks);
            VariableArray<int> sizeOfItemBlock = Variable.Array<int>(itemBlock);
            Range item = new Range(sizeOfItemBlock[itemBlock]).Named("item");
            Range trait = new Range(numTraits).Named("trait");
            VariableArray2D<int> numObservationsInBlock = Variable.Array<int>(userBlock, itemBlock);
            Range obs = new Range(numObservationsInBlock[userBlock, itemBlock]).Named("obs");
            var userOfObsB = Variable.Array(Variable.Array<int>(obs), userBlock, itemBlock).Named("userOfObs");
            var userOfObs = userOfObsB[userBlock, itemBlock];
            var itemOfObsB = Variable.Array(Variable.Array<int>(obs), userBlock, itemBlock).Named("itemOfObs");
            var itemOfObs = itemOfObsB[userBlock, itemBlock];
            var ratingOfObsB = Variable.Array(Variable.Array<bool>(obs), userBlock, itemBlock).Named("ratingOfObs");
            var ratingOfObs = ratingOfObsB[userBlock, itemBlock];
            var userTraitsB = Variable.Array(Variable.Array(Variable.Array<double>(trait), user), userBlock).Named("userTraitsB");
            var userTraits = userTraitsB[userBlock];
            userTraits[user][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(userBlock, user, trait);
            var itemTraitsB = Variable.Array(Variable.Array(Variable.Array<double>(trait), item), itemBlock).Named("itemTraits");
            var itemTraits = itemTraitsB[itemBlock];
            itemTraits[item][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(itemBlock, item, trait);
            using (Variable.ForEach(userBlock))
            using (Variable.ForEach(itemBlock))
            using (Variable.ForEach(obs))
            {
                VariableArray<double> products = Variable.Array<double>(trait).Named("products");
                products[trait] = userTraits[userOfObs[obs]][trait]*itemTraits[itemOfObs[obs]][trait];
                Variable<double> affinity = Variable.Sum(products).Named("sum");
                Variable<double> affinityNoisy = Variable.GaussianFromMeanAndVariance(affinity, affinityNoiseVar).Named("affinityNoisy");
                ratingOfObs[obs] = (affinityNoisy > 0);
            }
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            //SetAttribute(userBlock, new Partitioned());
            //SetAttribute(itemBlock, new Partitioned());
            sizeOfUserBlock.ObservedValue = Util.ArrayInit(numUserBlocks, ub => 1);
            sizeOfItemBlock.ObservedValue = Util.ArrayInit(numItemBlocks, ib => 1);
            numObservationsInBlock.ObservedValue = Util.ArrayInit(numUserBlocks, numItemBlocks, (ub, ib) => 1);
            userOfObsB.ObservedValue = Util.ArrayInit(numUserBlocks, numItemBlocks, (ub, ib) => new int[1]);
            itemOfObsB.ObservedValue = Util.ArrayInit(numUserBlocks, numItemBlocks, (ub, ib) => new int[1]);
            ratingOfObsB.ObservedValue = Util.ArrayInit(numUserBlocks, numItemBlocks, (ub, ib) => new bool[1]);
            //engine.SetPartitionedRange(userBlock);
            //engine.SetPartitionedRange(itemBlock);
            //engine.SetObservedValueByItem(sizeOfUserBlock, ub => GetSizeOfUserBlock(ub));
            //engine.SetObservedValueByItem(sizeOfItemBlock, ib => GetSizeOfItemBlock(ib));
            //engine.SetObservedValueByItem(numObservationsInBlock, (ub, ib) => GetNumObs(ub, ib));
            //engine.SetObservedValueByItem(userOfObs, (ub,ib) => LoadUserOfObs(ub,ib));
            //engine.SetObservedValueByItem(itemOfObs, (ub,ib) => LoadItemOfObs(ub,ib));
            //engine.SetObservedValueByItem(ratingOfObs, (ub, ib) => LoadRatingOfObs(ub, ib));
            Console.WriteLine(engine.Infer(userTraitsB));
        }

        // 2236 file reads 5008 file writes
        internal void DistributedMatchboxTest()
        {
            int numUsers = 4, numItems = 4, numTraits = 2;
            int[] userData = new int[] {0, 1, 2, 3};
            int[] itemData = new int[] {0, 1, 2, 3};
            bool[] ratingData = new bool[] {true, false, true, false};
            int numUserBlocks = 2, numItemBlocks = 2;
            double affinityNoiseVar = 1;
            Range userBlock = new Range(numUserBlocks).Named("userBlock");
            var sizeOfUserBlock = Variable.IArray<int>(userBlock);
            Range user = new Range(sizeOfUserBlock[userBlock]).Named("user");
            Range itemBlock = new Range(numItemBlocks).Named("itemBlock");
            var sizeOfItemBlock = Variable.IArray<int>(itemBlock).Named("sizeOfItemBlock");
            Range item = new Range(sizeOfItemBlock[itemBlock]).Named("item");
            Range trait = new Range(numTraits).Named("trait");
            var numObservationsInBlock = Variable.IArray<int>(userBlock, itemBlock).Named("numObs");
            Range obs = new Range(numObservationsInBlock[userBlock, itemBlock]).Named("obs");
            var userOfObs = Variable.IArray(Variable.Array<int>(obs), userBlock, itemBlock).Named("userOfObs");
            var itemOfObs = Variable.IArray(Variable.Array<int>(obs), userBlock, itemBlock).Named("itemOfObs");
            var ratingOfObs = Variable.IArray(Variable.Array<bool>(obs), userBlock, itemBlock).Named("ratingOfObs");
            var userTraits = Variable.Array(Variable.Array(Variable.Array<double>(trait), user), userBlock).Named("userTraits");
            userTraits[userBlock][user][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(userBlock, user, trait);
            var itemTraits = Variable.Array(Variable.Array(Variable.Array<double>(trait), item), itemBlock).Named("itemTraits");
            itemTraits[itemBlock][item][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(itemBlock, item, trait);
            using (Variable.ForEach(userBlock))
            using (Variable.ForEach(itemBlock))
            using (Variable.ForEach(obs))
            {
                VariableArray<double> products = Variable.Array<double>(trait).Named("products");
                products[trait] = userTraits[userBlock][userOfObs[userBlock, itemBlock][obs]][trait]*itemTraits[itemBlock][itemOfObs[userBlock, itemBlock][obs]][trait];
                Variable<double> affinity = Variable.Sum(products).Named("sum");
                Variable<double> affinityNoisy = Variable.GaussianFromMeanAndVariance(affinity, affinityNoiseVar).Named("affinityNoisy");
                ratingOfObs[userBlock, itemBlock][obs] = (affinityNoisy > 0);
            }
            InferenceEngine engine = new InferenceEngine();
            engine.ShowTimings = true;
            engine.Compiler.GivePriorityTo(typeof (GaussianProductOp_SHG09));
            //engine.Algorithm = new VariationalMessagePassing();
            SetAttribute(userBlock, new Partitioned());
            SetAttribute(itemBlock, new Partitioned());
            userTraits.AddAttribute(new InitialiseBackward());
            itemTraits.AddAttribute(new InitialiseBackward());
            int userBlockSize = numUsers/numUserBlocks;
            int itemBlockSize = numItems/numItemBlocks;
            sizeOfUserBlock.ObservedValue = Util.IArrayFromFunc(numUserBlocks, ub => userBlockSize);
            sizeOfItemBlock.ObservedValue = Util.IArrayFromFunc(numItemBlocks, ib => itemBlockSize);
            // partition data into blocks
            int[,][] ratingsInBlock = Util.ArrayInit(numUserBlocks, numItemBlocks, (ub, ib) =>
                                                                                   System.Linq.Enumerable.Range(0, ratingData.Length).Where(i =>
                                                                                                                                            (ub*userBlockSize <= userData[i]) &&
                                                                                                                                            (userData[i] <
                                                                                                                                             (ub + 1)*userBlockSize) &&
                                                                                                                                            (ib*itemBlockSize <= itemData[i]) &&
                                                                                                                                            (itemData[i] <
                                                                                                                                             (ib + 1)*itemBlockSize)).ToArray());
            numObservationsInBlock.ObservedValue = Util.IArrayFromFunc(numUserBlocks, numItemBlocks, (ub, ib) => ratingsInBlock[ub, ib].Length);
            userOfObs.ObservedValue = Util.IArrayFromFunc(numUserBlocks, numItemBlocks, (ub, ib) => ratingsInBlock[ub, ib].Select(i => userData[i]%userBlockSize).ToArray());
            itemOfObs.ObservedValue = Util.IArrayFromFunc(numUserBlocks, numItemBlocks, (ub, ib) => ratingsInBlock[ub, ib].Select(i => itemData[i]%itemBlockSize).ToArray());
            ratingOfObs.ObservedValue = Util.IArrayFromFunc(numUserBlocks, numItemBlocks, (ub, ib) => ratingsInBlock[ub, ib].Select(i => ratingData[i]).ToArray());
            IList<GaussianArrayArray> userTraitsActualB = engine.Infer<IList<GaussianArrayArray>>(userTraits);
            GaussianArrayArray userTraitsActual = new GaussianArrayArray(numUsers, u => userTraitsActualB[u/userBlockSize][u%userBlockSize]);
            GaussianArrayArray userTraitsExpected;
            MatchboxExample(numUsers, numItems, numTraits, userData, itemData, ratingData, out userTraitsExpected);
            Console.WriteLine(StringUtil.JoinColumns("userTraits = ", userTraitsActual, " should be ", userTraitsExpected));
            Assert.True(userTraitsExpected.MaxDiff(userTraitsActual) < 1e-8);
        }

        private void MatchboxExample(int numUsers, int numItems, int numTraits, int[] userData, int[] itemData, bool[] ratingData, out GaussianArrayArray userTraitsPost)
        {
            int numObservations = ratingData.Length;
            double affinityNoiseVar = 1;
            Range user = new Range(numUsers).Named("user");
            Range item = new Range(numItems).Named("item");
            Range trait = new Range(numTraits).Named("trait");
            Range obs = new Range(numObservations).Named("obs");
            VariableArray<int> userOfObs = Variable.Array<int>(obs).Named("userOfObs");
            VariableArray<int> itemOfObs = Variable.Array<int>(obs).Named("itemOfObs");
            VariableArray<bool> ratingOfObs = Variable.Array<bool>(obs).Named("ratingOfObs");
            var userTraits = Variable.Array(Variable.Array<double>(trait), user).Named("userTraits");
            userTraits[user][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(user, trait);
            var itemTraits = Variable.Array(Variable.Array<double>(trait), item).Named("itemTraits");
            itemTraits[item][trait] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(item, trait);
            using (Variable.ForEach(obs))
            {
                VariableArray<double> products = Variable.Array<double>(trait).Named("products");
                products[trait] = userTraits[userOfObs[obs]][trait]*itemTraits[itemOfObs[obs]][trait];
                Variable<double> affinity = Variable.Sum(products).Named("sum");
                Variable<double> affinityNoisy = Variable.GaussianFromMeanAndVariance(affinity, affinityNoiseVar).Named("affinityNoisy");
                ratingOfObs[obs] = (affinityNoisy > 0);
            }
            ratingOfObs.ObservedValue = ratingData;
            userOfObs.ObservedValue = userData;
            itemOfObs.ObservedValue = itemData;
            InferenceEngine engine = new InferenceEngine();
            userTraits.AddAttribute(new InitialiseBackward());
            itemTraits.AddAttribute(new InitialiseBackward());
            //engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.GivePriorityTo(typeof (GaussianProductOp_SHG09));
            userTraitsPost = engine.Infer<GaussianArrayArray>(userTraits);
        }

        [Fact]
        [Trait("Category", "DistributedTest")]
        public void DistributedHierarchicalGaussianTest3()
        {
            int numRows = 4;
            double[][] data = Util.ArrayInit(numRows, r => Util.ArrayInit(1 + r, i => r + i + 1.0));
            int numRowBlocks = 1;
            Range rowBlock = new Range(numRowBlocks).Named("rowBlock");
            VariableArray<int> sizeOfRowBlock = Variable.Array<int>(rowBlock).Named("sizeOfRowBlock");
            Range row = new Range(sizeOfRowBlock[rowBlock]).Named("row");
            var mean = Variable.Array(Variable.Array<double>(row), rowBlock).Named("mean");
            mean[rowBlock][row] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(rowBlock, row);
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            VariableArray<int> numItemBlocks = Variable.Array<int>(rowBlock).Named("numItemBlocks");
            Range itemBlock = new Range(numItemBlocks[rowBlock]).Named("itemBlock");
            var sizeOfRow = Variable.IArray(Variable.IArray(Variable.Array<int>(row), itemBlock), rowBlock).Named("sizeOfRow");
            Range item = new Range(sizeOfRow[rowBlock][itemBlock][row]).Named("n");
            var x = Variable.IArray(Variable.IArray(Variable.Array(Variable.Array<double>(item), row), itemBlock), rowBlock).Named("x");
            using (Variable.ForEach(rowBlock))
            using (Variable.ForEach(itemBlock))
            {
                x[rowBlock][itemBlock][row][item] = Variable.GaussianFromMeanAndPrecision(mean[rowBlock][row], precision).ForEach(item);
            }

            sizeOfRowBlock.ObservedValue = Util.ArrayInit(numRowBlocks, rb => numRows);
            numItemBlocks.ObservedValue = Util.ArrayInit(numRowBlocks, rb => 1);
            x.ObservedValue = Util.IArrayFromFunc(numRowBlocks,
                                                  rb =>
                                                  Util.IArrayFromFunc(numItemBlocks.ObservedValue[rb],
                                                                      b => Util.ArrayInit(numRows, r => Util.ArrayInit(data[r].Length, i => data[r][i]))));
            sizeOfRow.ObservedValue = Util.IArrayFromFunc(numRowBlocks,
                                                          rb => Util.IArrayFromFunc(numItemBlocks.ObservedValue[rb], b => Util.ArrayInit(numRows, r => data[r].Length)));
            //precision.ObservedValue = 1.0;

            SetAttribute(rowBlock, new Partitioned());
            SetAttribute(itemBlock, new Partitioned());
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            var meanActualB = engine.Infer<IList<GaussianArray>>(mean);
            GaussianArray meanActual = new GaussianArray(numRows, r => meanActualB[0][r]);
            Gamma precActual = engine.Infer<Gamma>(precision);
            IDistribution<double[]> meanExpected;
            Gamma precExpected;
            HierarchicalGaussianExample(data, out meanExpected, out precExpected);
            Console.WriteLine(StringUtil.JoinColumns("mean = ", meanActual, " should be ", meanExpected));
            Console.WriteLine("prec = {0} should be {1}", precActual, precExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-8);
            Assert.True(precExpected.MaxDiff(precActual) < 1e-8);
        }

        [Fact]
        [Trait("Category", "DistributedTest")]
        public void DistributedHierarchicalGaussianTest2()
        {
            int numRows = 4;
            double[][] data = Util.ArrayInit(numRows, r => Util.ArrayInit(1 + r, i => r + i + 1.0));
            int numItemBlocks = 1;
            Range row = new Range(numRows).Named("row");
            VariableArray<double> mean = Variable.Array<double>(row).Named("mean");
            mean[row] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(row);
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            Range itemBlock = new Range(numItemBlocks).Named("itemBlock"); // fixed
            var sizeOfRow = Variable.IArray(Variable.Array<int>(row), itemBlock).Named("sizeOfRow");
            Range item = new Range(sizeOfRow[itemBlock][row]).Named("n");
            var x = Variable.IArray(Variable.Array(Variable.Array<double>(item), row), itemBlock).Named("x");
            using (Variable.ForEach(itemBlock))
            {
                x[itemBlock][row][item] = Variable.GaussianFromMeanAndPrecision(mean[row], precision).ForEach(item);
            }

            x.ObservedValue = Util.IArrayFromFunc(numItemBlocks, b => Util.ArrayInit(numRows, r => Util.ArrayInit(data[r].Length, i => data[r][i])));
            sizeOfRow.ObservedValue = Util.IArrayFromFunc(numItemBlocks, b => Util.ArrayInit(numRows, r => data[r].Length));

            SetAttribute(itemBlock, new Partitioned());
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            IDistribution<double[]> meanActual = engine.Infer<IDistribution<double[]>>(mean);
            Gamma precActual = engine.Infer<Gamma>(precision);
            IDistribution<double[]> meanExpected;
            Gamma precExpected;
            HierarchicalGaussianExample(data, out meanExpected, out precExpected);
            Console.WriteLine(StringUtil.JoinColumns("mean = ", meanActual, " should be ", meanExpected));
            Console.WriteLine("prec = {0} should be {1}", precActual, precExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-8);
            Assert.True(precExpected.MaxDiff(precActual) < 1e-8);
        }

        internal void DistributedHierarchicalGaussianTest()
        {
            int numRows = 4;
            double[][] data = Util.ArrayInit(numRows, r => Util.ArrayInit(1 + r, i => r + i + 1.0));
            Range row = new Range(numRows).Named("row");
            VariableArray<double> mean = Variable.Array<double>(row).Named("mean");
            mean[row] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(row);
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            VariableArray<int> numItemBlocks = Variable.Array<int>(row).Named("numItemBlocks");
            Range itemBlock = new Range(numItemBlocks[row]).Named("itemBlock"); // problem!
            var sizeOfRow = Variable.IArray(Variable.IArray<int>(itemBlock), row).Named("sizeOfRow");
            Range item = new Range(sizeOfRow[row][itemBlock]).Named("n");
            var x = Variable.IArray(Variable.IArray(Variable.Array<double>(item), itemBlock), row).Named("x");
            x[row][itemBlock][item] = Variable.GaussianFromMeanAndPrecision(mean[row], precision).ForEach(itemBlock, item);

            x.ObservedValue = Util.IArrayFromFunc(numRows, r => Util.IArrayFromFunc(data[r].Length, b => Util.ArrayInit(1, i => data[r][b])));
            sizeOfRow.ObservedValue = Util.IArrayFromFunc(numRows, r => Util.IArrayFromFunc(data[r].Length, b => 1));
            numItemBlocks.ObservedValue = Util.ArrayInit(numRows, r => data[r].Length);

            // this test should fail if row is not partitioned
            SetAttribute(row, new Partitioned());
            SetAttribute(itemBlock, new Partitioned());
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            IDistribution<double[]> meanActual = engine.Infer<IDistribution<double[]>>(mean);
            Gamma precActual = engine.Infer<Gamma>(precision);
            IDistribution<double[]> meanExpected;
            Gamma precExpected;
            HierarchicalGaussianExample(data, out meanExpected, out precExpected);
            Console.WriteLine(StringUtil.JoinColumns("mean = ", meanActual, " should be ", meanExpected));
            Console.WriteLine("prec = {0} should be {1}", precActual, precExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-8);
            Assert.True(precExpected.MaxDiff(precActual) < 1e-8);
        }

        private void HierarchicalGaussianExample(double[][] data, out IDistribution<double[]> meanPost, out Gamma precPost)
        {
            int numRows = data.Length;
            Range row = new Range(numRows);
            VariableArray<double> mean = Variable.Array<double>(row);
            mean[row] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(row);
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            VariableArray<int> sizeOfRow = Variable.Array<int>(row);
            Range item = new Range(sizeOfRow[row]).Named("n");
            var x = Variable.Array(Variable.Array<double>(item), row).Named("x");
            x[row][item] = Variable.GaussianFromMeanAndPrecision(mean[row], precision).ForEach(item);
            x.ObservedValue = data;
            sizeOfRow.ObservedValue = Util.ArrayInit(numRows, r => data[r].Length);

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            meanPost = engine.Infer<IDistribution<double[]>>(mean);
            precPost = engine.Infer<Gamma>(precision);
        }

        // TODO: This model has a performance bug in DependencyAnalysis
        [Fact]
        [Trait("Category", "DistributedTest")]
        public void LearningAGaussianCaseTest()
        {
            LearningAGaussianCase(false);
        }

        internal void DistributedLearningAGaussianCaseTest()
        {
            LearningAGaussianCase(true);
        }

        private void LearningAGaussianCase(bool distributed)
        {
            double[] data = Util.ArrayInit(10, i => i + 1.0);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            Range block = new Range(data.Length).Named("block");
            using (ForEachBlock fb = Variable.ForEach(block))
            {
                for (int i = 0; i < data.Length; i++)
                {
                    using (Variable.Case(fb.Index, i))
                    {
                        Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x" + i);
                        x.ObservedValue = data[i];
                    }
                }
            }
            FileStats.Clear();
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.OptimiseForVariables = new List<IVariable>() {mean, precision};
            if (distributed) SetAttribute(block, new Partitioned());
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Gamma precActual = engine.Infer<Gamma>(precision);

            Gaussian meanExpected;
            Gamma precExpected;
            LearningGaussianExample(false, data, out meanExpected, out precExpected);
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            Console.WriteLine("prec = {0} should be {1}", precActual, precExpected);
            Console.WriteLine("{0} file reads {1} file writes", FileStats.ReadCount, FileStats.WriteCount);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-8);
            Assert.True(precExpected.MaxDiff(precActual) < 1e-8);
        }

        // mean_rep0_rep0_B could be re-created each time
        // original: 1821 file reads 1252 file writes
        // 1405 file reads 840 file writes
        [Fact]
        [Trait("Category", "DistributedTest")]
        public void DistributedLearningAGaussianNoisy()
        {
            double[] data = Util.ArrayInit(12, i => i + 1.0);
            int numBlocks = 4;
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            Range block = new Range(numBlocks).Named("block");
            var sizeOfBlock = Variable.IArray<int>(block).Named("sizeOfBlock");
            Range dataRange = new Range(sizeOfBlock[block]).Named("n");
            var x = Variable.Array(Variable.Array<double>(dataRange), block).Named("x");
            x[block][dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(block, dataRange);
            var y = Variable.IArray(Variable.Array<double>(dataRange), block).Named("y");
            y[block][dataRange] = Variable.GaussianFromMeanAndPrecision(x[block][dataRange], 1.0);

            FileStats.Clear();
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowTimings = true;
            block.AddAttribute(new Partitioned());
            int blockSize = data.Length/numBlocks;
            sizeOfBlock.ObservedValue = Util.IArrayFromFunc(numBlocks, b => blockSize);
            y.ObservedValue = Util.IArrayFromFunc(numBlocks, b => Util.ArrayInit(blockSize, i => data[b*blockSize + i]));
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Gamma precActual = engine.Infer<Gamma>(precision);
            Gaussian meanExpected;
            Gamma precExpected;
            LearningGaussianExample(true, data, out meanExpected, out precExpected);
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            Console.WriteLine("prec = {0} should be {1}", precActual, precExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-8);
            Assert.True(precExpected.MaxDiff(precActual) < 1e-8);
        }

        internal void DistributedLearningAGaussianNoisyConstraint()
        {
            double[] data = Util.ArrayInit(12, i => i + 1.0);
            int numBlocks = 4;
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            Range block = new Range(numBlocks).Named("block");
            var sizeOfBlock = Variable.Array<int>(block).Named("sizeOfBlock");
            Range dataRange = new Range(sizeOfBlock[block]).Named("n");
            var x = Variable.Array(Variable.Array<double>(dataRange), block).Named("x");
            x[block][dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(block, dataRange);
            var y = Variable.Array(Variable.Array<double>(dataRange), block).Named("y");
            y[block][dataRange] = Variable.GaussianFromMeanAndPrecision(x[block][dataRange], 1.0);

            Variable<Func<int, double[]>> LoadDataBlock = Variable.New<Func<int, double[]>>();
            ConstrainFromIndex(y, b => VariableEvaluate(LoadDataBlock, b));
            Variable<Func<int, int>> GetSizeOfBlock = Variable.New<Func<int, int>>();
            SetFromIndex(sizeOfBlock, b => VariableEvaluate(GetSizeOfBlock, b));

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            SetAttribute(block, new Partitioned());
            int blockSize = data.Length/numBlocks;
            GetSizeOfBlock.ObservedValue = b => blockSize;
            LoadDataBlock.ObservedValue = b => Util.ArrayInit(blockSize, i => data[b*blockSize + i]);
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Gamma precActual = engine.Infer<Gamma>(precision);
            Gaussian meanExpected;
            Gamma precExpected;
            LearningGaussianExample(true, data, out meanExpected, out precExpected);
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            Console.WriteLine("prec = {0} should be {1}", precActual, precExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-8);
            Assert.True(precExpected.MaxDiff(precActual) < 1e-8);
        }

        private void LearningGaussianExample(bool addNoise, double[] data, out Gaussian meanPost, out Gamma precPost)
        {
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            Range dataRange = new Range(data.Length).Named("n");
            VariableArray<double> x = Variable.Array<double>(dataRange).Named("x");
            x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(dataRange);
            if (addNoise)
            {
                VariableArray<double> y = Variable.Array<double>(dataRange).Named("y");
                y[dataRange] = Variable.GaussianFromMeanAndPrecision(x[dataRange], 1.0);
                y.ObservedValue = data;
            }
            else
            {
                x.ObservedValue = data;
            }

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            meanPost = engine.Infer<Gaussian>(mean);
            precPost = engine.Infer<Gamma>(precision);
            //Console.WriteLine("mean=" + engine.Infer(mean));
            //Console.WriteLine("prec=" + engine.Infer(precision));
        }

        [Fact]
        [Trait("Category", "DistributedTest")]
        public void FileArrayTest()
        {
            string folder = FileArray<double>.GetTempFolder("doubles");
            FileArray<double> doubles = new FileArray<double>(folder, 4, i => 1.0 + i);
            for (int i = 0; i < doubles.Count; i++)
            {
                double x = doubles[i];
                Assert.Equal(x, 1.0 + i);
            }
        }

        private static void SetAttribute(Range range, ICompilerAttribute attr)
        {
            range.AddAttribute(attr);
        }

        private static void SetToFunctionOfIndex<T, TItem>(VariableArray<TItem, T[]> array, Variable<Func<int, T>> func)
            where TItem : Variable<T>, SettableTo<TItem>
        {
            using (var fb = Variable.ForEach(array.Range))
            {
                array[fb.Index].SetTo(Variable<T>.Factor(func.ObservedValue, fb.Index));
            }
        }

        private static void ConstrainEqualToFunctionOfIndex<T, TItem>(VariableArray<TItem, T[]> array, Variable<Func<int, T>> func)
            where TItem : Variable<T>, SettableTo<TItem>
        {
            using (var fb = Variable.ForEach(array.Range))
            {
                Variable.ConstrainEqual(array[fb.Index], Variable<T>.Factor(func.ObservedValue, fb.Index));
            }
        }

        private static void SetFromIndex<T, TItem, TArray>(VariableArray<TItem, TArray> array, Func<Variable<int>, Variable<T>> func)
            where TItem : Variable<T>, SettableTo<TItem>
        {
            using (var fb = Variable.ForEach(array.Range))
            {
                array[fb.Index].SetTo(func(fb.Index));
            }
        }

        private static void ConstrainFromIndex<T, TItem, TArray>(VariableArray<TItem, TArray> array, Func<Variable<int>, Variable<T>> func)
            where TItem : Variable<T>, SettableTo<TItem>
        {
            using (var fb = Variable.ForEach(array.Range))
            {
                Variable.ConstrainEqual(array[fb.Index], func(fb.Index));
            }
        }

        private static Variable<TResult> VariableEvaluate<T, TResult>(Variable<Func<T, TResult>> func, Variable<T> arg)
        {
            return Variable<TResult>.Factor(Evaluate, func, arg);
        }

        public static TResult Evaluate<T, TResult>(Func<T, TResult> func, T arg)
        {
            return func(arg);
        }

        public class DistributedRegression_EP : IGeneratedAlgorithm
        {
            #region Fields

            /// <summary>Field backing the NumberOfIterationsDone property</summary>
            private int numberOfIterationsDone;

            /// <summary>Field backing the x property</summary>
            private IArray<double[]> X;

            /// <summary>Field backing the yNoisy property</summary>
            private IArray<double> YNoisy;

            /// <summary>The number of iterations last computed by Changed_numberOfIterationsDecreased_x_yNoisy. Set this to zero to force re-execution of Changed_numberOfIterationsDecreased_x_yNoisy</summary>
            public int Changed_numberOfIterationsDecreased_x_yNoisy_iterationsDone;

            /// <summary>The number of iterations last computed by Constant. Set this to zero to force re-execution of Constant</summary>
            public int Constant_iterationsDone;

            /// <summary>The number of iterations last computed by Init_numberOfIterationsDecreased_x_yNoisy. Set this to zero to force re-execution of Init_numberOfIterationsDecreased_x_yNoisy</summary>
            public int Init_numberOfIterationsDecreased_x_yNoisy_iterationsDone;

            /// <summary>True if Init_numberOfIterationsDecreased_x_yNoisy has performed initialisation. Set this to false to force re-execution of Init_numberOfIterationsDecreased_x_yNoisy</summary>
            public bool Init_numberOfIterationsDecreased_x_yNoisy_isInitialised;

            public DistributionFileArray<GaussianArray, double[]> w_rep0_B;

            /// <summary>Message to marginal of 'w'</summary>
            public DistributionStructArray<Gaussian, double> w_marginal_F;

            #endregion

            #region Properties

            /// <summary>The number of iterations done from the initial state</summary>
            public int NumberOfIterationsDone
            {
                get { return this.numberOfIterationsDone; }
            }

            /// <summary>The externally-specified value of 'x'</summary>
            public IArray<double[]> x
            {
                get { return this.X; }
                set
                {
                    if ((value != null) && (value.Count != 2))
                    {
                        throw new ArgumentException(((("Provided array of length " + value.Count) + " when length ") + 2) + " was expected for variable \'x\'");
                    }
                    this.X = value;
                    this.numberOfIterationsDone = 0;
                    this.Init_numberOfIterationsDecreased_x_yNoisy_isInitialised = false;
                    this.Changed_numberOfIterationsDecreased_x_yNoisy_iterationsDone = 0;
                }
            }

            /// <summary>The externally-specified value of 'yNoisy'</summary>
            public IArray<double> yNoisy
            {
                get { return this.YNoisy; }
                set
                {
                    if ((value != null) && (value.Count != 2))
                    {
                        throw new ArgumentException(((("Provided array of length " + value.Count) + " when length ") + 2) + " was expected for variable \'yNoisy\'");
                    }
                    this.YNoisy = value;
                    this.numberOfIterationsDone = 0;
                    this.Init_numberOfIterationsDecreased_x_yNoisy_isInitialised = false;
                    this.Changed_numberOfIterationsDecreased_x_yNoisy_iterationsDone = 0;
                }
            }

            #endregion

            #region Methods

            /// <summary>Get the observed value of the specified variable.</summary>
            /// <param name="variableName">Variable name</param>
            public object GetObservedValue(string variableName)
            {
                if (variableName == "x")
                {
                    return this.x;
                }
                if (variableName == "yNoisy")
                {
                    return this.yNoisy;
                }
                throw new ArgumentException("Not an observed variable name: " + variableName);
            }

            /// <summary>Set the observed value of the specified variable.</summary>
            /// <param name="variableName">Variable name</param>
            /// <param name="value">Observed value</param>
            public void SetObservedValue(string variableName, object value)
            {
                if (variableName == "x")
                {
                    this.x = (IArray<double[]>) value;
                    return;
                }
                if (variableName == "yNoisy")
                {
                    this.yNoisy = (IArray<double>) value;
                    return;
                }
                throw new ArgumentException("Not an observed variable name: " + variableName);
            }

            /// <summary>The marginal distribution of the specified variable.</summary>
            /// <param name="variableName">Variable name</param>
            public object Marginal(string variableName)
            {
                if (variableName == "w")
                {
                    return this.WMarginal();
                }
                throw new ArgumentException("This class was not built to infer " + variableName);
            }

            public T Marginal<T>(string variableName)
            {
                return Distribution.ChangeType<T>(this.Marginal(variableName));
            }

            /// <summary>The query-specific marginal distribution of the specified variable.</summary>
            /// <param name="variableName">Variable name</param>
            /// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
            public object Marginal(string variableName, string query)
            {
                if (query == "Marginal")
                {
                    return this.Marginal(variableName);
                }
                throw new ArgumentException(((("This class was not built to infer \'" + variableName) + "\' with query \'") + query) + "\'");
            }

            public T Marginal<T>(string variableName, string query)
            {
                return Distribution.ChangeType<T>(this.Marginal(variableName, query));
            }

            /// <summary>The output message of the specified variable.</summary>
            /// <param name="variableName">Variable name</param>
            public object GetOutputMessage(string variableName)
            {
                throw new ArgumentException("This class was not built to compute an output message for " + variableName);
            }

            /// <summary>Update all marginals, by iterating message passing the given number of times</summary>
            /// <param name="numberOfIterations">The number of times to iterate each loop</param>
            /// <param name="initialise">If true, messages that initialise loops are reset when observed values change</param>
            private void Execute(int numberOfIterations, bool initialise)
            {
                if (numberOfIterations < this.Changed_numberOfIterationsDecreased_x_yNoisy_iterationsDone)
                {
                    this.Init_numberOfIterationsDecreased_x_yNoisy_isInitialised = false;
                    this.Changed_numberOfIterationsDecreased_x_yNoisy_iterationsDone = 0;
                }
                this.Constant();
                this.Init_numberOfIterationsDecreased_x_yNoisy(initialise);
                this.Changed_numberOfIterationsDecreased_x_yNoisy(numberOfIterations);
                this.numberOfIterationsDone = numberOfIterations;
            }

            public void Execute(int numberOfIterations)
            {
                this.Execute(numberOfIterations, true);
            }

            public void Update(int additionalIterations)
            {
                this.Execute(this.numberOfIterationsDone + additionalIterations, false);
            }

            private void OnProgressChanged(ProgressChangedEventArgs e)
            {
                // Make a temporary copy of the event to avoid a race condition
                // if the last subscriber unsubscribes immediately after the null check and before the event is raised.
                this.ProgressChanged?.Invoke(this, e);
            }

            /// <summary>Reset all messages to their initial values.  Sets NumberOfIterationsDone to 0.</summary>
            public void Reset()
            {
                this.Execute(0);
            }

            /// <summary>Computations that do not depend on observed values</summary>
            public void Constant()
            {
                if (this.Constant_iterationsDone == 1)
                {
                    return;
                }
                this.w_rep0_B = new DistributionFileArray<GaussianArray, double[]>(FileArray<bool>.GetTempFolder("w_rep0_B"), 2);
                for (int item = 0; item < 2; item++)
                {
                    this.w_rep0_B[item] = new GaussianArray(2);
                }
                this.Constant_iterationsDone = 1;
                this.Init_numberOfIterationsDecreased_x_yNoisy_iterationsDone = 0;
                this.Changed_numberOfIterationsDecreased_x_yNoisy_iterationsDone = 0;
            }

            /// <summary>Computations that must reset on changes to numberOfIterationsDecreased and x and yNoisy</summary>
            /// <param name="initialise">If true, reset messages that initialise loops</param>
            public void Init_numberOfIterationsDecreased_x_yNoisy(bool initialise)
            {
                if ((this.Init_numberOfIterationsDecreased_x_yNoisy_iterationsDone == 1) && ((!initialise) || this.Init_numberOfIterationsDecreased_x_yNoisy_isInitialised))
                {
                    return;
                }
                this.Init_numberOfIterationsDecreased_x_yNoisy_iterationsDone = 1;
                this.Init_numberOfIterationsDecreased_x_yNoisy_isInitialised = true;
                this.Changed_numberOfIterationsDecreased_x_yNoisy_iterationsDone = 0;
            }

            /// <summary>Computations that depend on the observed value of numberOfIterationsDecreased and x and yNoisy</summary>
            /// <param name="numberOfIterations">The number of times to iterate each loop</param>
            public void Changed_numberOfIterationsDecreased_x_yNoisy(int numberOfIterations)
            {
                if (this.Changed_numberOfIterationsDecreased_x_yNoisy_iterationsDone == numberOfIterations)
                {
                    return;
                }
                this.w_marginal_F = new DistributionStructArray<Gaussian, double>(2);
                for (int dim = 0; dim < 2; dim++)
                {
                    this.w_marginal_F[dim] = ArrayHelper.MakeUniform<Gaussian>(Gaussian.Uniform());
                }
                DistributionStructArray<Gaussian, double> w_use_B = default(DistributionStructArray<Gaussian, double>);
                w_use_B = new DistributionStructArray<Gaussian, double>(2);
                for (int dim = 0; dim < 2; dim++)
                {
                    w_use_B[dim] = ArrayHelper.MakeUniform<Gaussian>(Gaussian.Uniform());
                }
                GaussianArray _hoist = new GaussianArray(2, i => GaussianFromMeanAndVarianceOp.SampleAverageConditional(0, 1));
                GaussianArray w_rep0_B_marginal = new GaussianArray(2);
                for (int dim = 0; dim < 2; dim++)
                {
                    w_rep0_B_marginal[dim] = _hoist[dim];
                }
                GaussianArray w_rep0_B_toDef = new GaussianArray(2);
                for (int iteration = this.Changed_numberOfIterationsDecreased_x_yNoisy_iterationsDone; iteration < numberOfIterations; iteration++)
                {
                    w_rep0_B_toDef = ReplicateOp_Divide.ToDef(this.w_rep0_B, w_rep0_B_toDef);
                    w_rep0_B_marginal = ReplicateOp_Divide.Marginal(w_rep0_B_toDef, _hoist, w_rep0_B_marginal);
                    for (int item = 0; item < 2; item++)
                    {
                        double[] thisDotX_item_local;
                        thisDotX_item_local = this.X[item];
                        GaussianArray w_rep0_B_local = this.w_rep0_B[item];
                        GaussianArray w_rep0_F_dim_local_item_local2;
                        w_rep0_F_dim_local_item_local2 = new GaussianArray(2);
                        w_rep0_F_dim_local_item_local2 = ReplicateOp_Divide.UsesAverageConditional(w_rep0_B_local, w_rep0_B_marginal, item, w_rep0_F_dim_local_item_local2);
                        DistributionStructArray<Gaussian, double> product_F_item_local2;
                        product_F_item_local2 = new DistributionStructArray<Gaussian, double>(2);
                        for (int dim = 0; dim < 2; dim++)
                        {
                            product_F_item_local2[dim] = GaussianProductOp.ProductAverageConditional(w_rep0_F_dim_local_item_local2[dim], thisDotX_item_local[dim]);
                        }
                        DistributionStructArray<Gaussian, double> product_F_item_local3;
                        product_F_item_local3 = product_F_item_local2;
                        Gaussian y_F_item_local2;
                        y_F_item_local2 = FastSumOp.SumAverageConditional(product_F_item_local3);
                        Gaussian y_B_item_local2;
                        double thisDotYNoisy_item_local;
                        thisDotYNoisy_item_local = this.YNoisy[item];
                        y_B_item_local2 = GaussianFromMeanAndVarianceOp.MeanAverageConditional(thisDotYNoisy_item_local, 0.1);
                        DistributionStructArray<Gaussian, double> product_B_item_local2;
                        product_B_item_local2 = new DistributionStructArray<Gaussian, double>(2);
                        product_B_item_local2 = FastSumOp.ArrayAverageConditional<DistributionStructArray<Gaussian, double>>(y_B_item_local2, y_F_item_local2,
                                                                                                                             product_F_item_local3, product_B_item_local2);
                        Gaussian thisDotw_rep0_B_dim_local_item_local2;
                        for (int dim = 0; dim < 2; dim++)
                        {
                            thisDotw_rep0_B_dim_local_item_local2 = GaussianProductOp.AAverageConditional(product_B_item_local2[dim], thisDotX_item_local[dim]);
                            w_rep0_B_local[dim] = thisDotw_rep0_B_dim_local_item_local2;
                        }
                        this.w_rep0_B[item] = w_rep0_B_local;
                    }
                    this.OnProgressChanged(new ProgressChangedEventArgs(iteration));
                }
                //w_use_B = ReplicateOp_NoDivide.DefAverageConditional(this.w_rep0_B, w_use_B);
                w_use_B = ReplicateOp_Divide.DefAverageConditional(w_rep0_B_toDef, w_use_B);
                this.w_marginal_F = VariableOp.MarginalAverageConditional(w_use_B, _hoist, this.w_marginal_F);
                this.Changed_numberOfIterationsDecreased_x_yNoisy_iterationsDone = numberOfIterations;
            }

            /// <summary>
            /// Returns the marginal distribution for 'w' given by the current state of the
            /// message passing algorithm.
            /// </summary>
            /// <returns>The marginal distribution</returns>
            public DistributionStructArray<Gaussian, double> WMarginal()
            {
                return this.w_marginal_F;
            }

            #endregion

            #region Events

            /// <summary>Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.</summary>
            public event EventHandler<ProgressChangedEventArgs> ProgressChanged;

            #endregion
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}