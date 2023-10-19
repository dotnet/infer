// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.Serialization;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Serialization;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    using GaussianArray = DistributionStructArray<Gaussian, double>;

    /// <summary>
    /// Measure the speed of different implementations of the Bayes Point Machine.
    /// </summary>
    public class BpmSpeedTests
    {
        public class Instance
        {
            public bool label;
            public int[] featureIndices;
            public double[] featureValues;
        }

        public sealed class VwReader : IDisposable, IEnumerable<Instance>
        {
            private StreamReader reader;

            public VwReader(string fileName)
            {
                FileStream fileStream = File.OpenRead(fileName);
                var stream = new GZipStream(fileStream, CompressionMode.Decompress);
                reader = new StreamReader(stream);
            }

            public Instance Read()
            {
                string line = reader.ReadLine();
                if (line == null) return null;
                string[] fields = line.Split(' ');
                Instance instance = new Instance();
                instance.label = (int.Parse(fields[0]) > 0);
                int numFeatures = fields.Length - 2;
                instance.featureIndices = new int[numFeatures];
                instance.featureValues = new double[numFeatures];
                for (int i = 0; i < numFeatures; i++)
                {
                    string[] parts = fields[i + 2].Split(':');
                    instance.featureIndices[i] = int.Parse(parts[0]) - 1;
                    instance.featureValues[i] = double.Parse(parts[1]);
                }
                return instance;
            }

            public IEnumerator<Instance> GetEnumerator()
            {
                while (true)
                {
                    Instance instance = Read();
                    if (instance == null) break;
                    yield return instance;
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }

            public void Dispose()
            {
                reader.Dispose();
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        static readonly string dataFolder = @"c:\Users\minka\Downloads\rcv1";

        public static void Rcv1Test(double wVariance, double biasVariance)
        {
            int count = 0;
            if (false)
            {
                int maxFeatureIndex = 0;
                foreach (Instance instance in new VwReader(Path.Combine(dataFolder, "rcv1.train.vw.gz")))
                {
                    count++;
                    if (count%10000 == 0) Console.WriteLine(count);
                    foreach (int index in instance.featureIndices)
                    {
                        if (index > maxFeatureIndex) maxFeatureIndex = index;
                    }
                }
                Console.WriteLine("{0} features", maxFeatureIndex + 1);
            }
            int nf = 47152;
            var train = new BpmTrain2();
            var predict = new BpmPredict2();
            train.SetPriors(nf, wVariance, biasVariance);

            StreamWriter writer = new StreamWriter(Path.Combine(dataFolder, "log.txt"));
            int errors = 0;
            //int errors2 = 0;
            //StreamReader reader = new StreamReader(Path.Combine(dataFolder, "preds.txt");
            // takes 92s to train
            // takes 74s just to read the data
            // takes 15s just to do 'wc' on the data
            // there are 781265 data points in train, 23149 in test
            foreach (Instance instance in new VwReader(Path.Combine(dataFolder, "rcv1.train.vw.gz")))
            {
                predict.SetPriors(train.wPost, train.biasPost);
                bool yPred = predict.Predict(instance);
                if (yPred != instance.label) errors++;
                //double pred2 = double.Parse(reader.ReadLine());
                //if ((pred2 > 0.5) != instance.label) errors2++;
                train.Train(instance);
                count++;
                if (count%1000 == 0)
                {
                    Console.WriteLine("{0} {1} {2}", count, (double) errors/count, train.biasPost);
                    //Console.WriteLine("{0} {1} {2} {3}", count, (double)errors/count, (double)errors2/count, train.biasPost);
                    writer.WriteLine("{0} {1}", count, (double) errors/count);
                    writer.Flush();
                    //if (count == 10000) break;
                }
            }
            writer.Dispose();

            using (Stream stream = File.Create(Path.Combine(dataFolder, "weights.bin")))
            {
                var gaussianArraySerializer = new DataContractSerializer(typeof(GaussianArray), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });
                gaussianArraySerializer.WriteObject(stream, train.wPost);

                var gaussianSerializer = new DataContractSerializer(typeof(Gaussian), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });
                gaussianSerializer.WriteObject(stream, train.biasPost);
            }
        }

        // (0.5,0.5):
        // weight distribution = Gaussian(-0.02787, 0.2454)
        // error rate = 0.0527452589744697 = 1221/23149
        // (1,1):
        // weight distribution = Gaussian(-0.03117, 0.3967)
        // error rate = 0.0522268780508877 = 1209/23149
        // (2,2):
        // weight distribution = Gaussian(-0.03522, 0.6794)
        // error rate = 0.0530476478465593 = 1228/23149
        // (10,10):
        // weight distribution = Gaussian(-0.05455, 2.96)
        // error rate = 0.0580586634411854 = 1344/23149

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        public static void Rcv1Test2()
        {
            GaussianArray wPost;
            Gaussian biasPost;
            using (Stream stream = File.OpenRead(Path.Combine(dataFolder, "weights.bin")))
            {
                var gaussianArraySerializer = new DataContractSerializer(typeof(GaussianArray), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });
                wPost = (GaussianArray)gaussianArraySerializer.ReadObject(stream);
                var gaussianSerializer = new DataContractSerializer(typeof(Gaussian), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });
                biasPost = (Gaussian)gaussianSerializer.ReadObject(stream);
            }
            if (true)
            {
                GaussianEstimator est = new GaussianEstimator();
                foreach (Gaussian item in wPost) est.Add(item.GetMean());
                Console.WriteLine("weight distribution = {0}", est.GetDistribution(new Gaussian()));
            }
            var predict = new BpmPredict2();
            predict.SetPriors(wPost, biasPost);
            int count = 0;
            int errors = 0;
            foreach (Instance instance in new VwReader(Path.Combine(dataFolder, "rcv1.test.vw.gz")))
            {
                bool yPred = predict.Predict(instance);
                if (yPred != instance.label) errors++;
                count++;
            }
            Console.WriteLine("error rate = {0} = {1}/{2}", (double)errors/count, errors, count);
        }

        public static void Rcv1Test3()
        {
            int nf = 47152;
            var train = new BpmTrain2();
            double wVariance = 10;
            double biasVariance = 10;
            train.SetPriors(nf, wVariance, biasVariance);
            int count = 0;
            foreach (Instance instance in new VwReader(Path.Combine(dataFolder, "rcv1.train.vw.gz")))
            {
                train.Train(instance);
                count++;
                if (count%1000 == 0)
                {
                    Console.WriteLine("{0} {1}", count, train.biasPost);
                    if (count == 10000) break;
                }
            }
        }

        public class BpmTrain
        {
            private Variable<int> nFeatures;
            private VariableArray<double> w, wSparse;
            private Variable<GaussianArray> wPrior;
            private Variable<Gaussian> biasPrior;
            private Variable<double> bias;
            private Variable<int> xValueCount;
            private VariableArray<double> xValues;
            private VariableArray<int> xIndices;
            private Variable<bool> y;
            public GaussianArray wPost;
            public Gaussian biasPost;
            private InferenceEngine engine;

            public BpmTrain()
            {
                nFeatures = Variable.New<int>().Named("nFeatures");
                Range feature = new Range(nFeatures).Named("feature");
                w = Variable.Array<double>(feature).Named("w");
                //VariableArray<Gaussian> wPrior = Variable.Array<Gaussian>(feature).Named("wPrior");
                //w[feature] = Variable<double>.Random(wPrior[feature]);
                wPrior = Variable.New<GaussianArray>().Named("wPrior");
                w.SetTo(Variable<double[]>.Random(wPrior));
                biasPrior = Variable.New<Gaussian>().Named("biasPrior");
                bias = Variable<double>.Random(biasPrior).Named("bias");
                xValueCount = Variable.New<int>().Named("xValueCount");
                Range userFeature = new Range(xValueCount).Named("userFeature");
                xValues = Variable.Array<double>(userFeature).Named("xValues");
                xIndices = Variable.Array<int>(userFeature).Named("xIndices");
                VariableArray<double> product = Variable.Array<double>(userFeature).Named("product");
                wSparse = Variable.Subarray(w, xIndices).Named("wSparse");
                wSparse.AddAttribute(new MarginalPrototype(new Gaussian()));
                product[userFeature] = xValues[userFeature]*wSparse[userFeature];
                Variable<double> score = Variable.Sum(product).Named("score");
                y = (Variable.GaussianFromMeanAndVariance(score + bias, 1.0) > 0);
                y.Name = "y";
                engine = new InferenceEngine();
                engine.Compiler.FreeMemory = false;
                engine.Compiler.ReturnCopies = false;
                engine.OptimiseForVariables = new IVariable[] {wSparse, bias};
                engine.ModelName = "BpmTrain";
            }

            public void SetPriors(int nf, double wVariance, double biasVariance)
            {
                nFeatures.ObservedValue = nf;
                wPost = new GaussianArray(nf, f => new Gaussian(0, wVariance));
                biasPost = new Gaussian(0, biasVariance);
            }

            public void Train(Instance instance)
            {
                wPrior.ObservedValue = wPost;
                biasPrior.ObservedValue = biasPost;
                xValueCount.ObservedValue = instance.featureIndices.Length;
                xIndices.ObservedValue = instance.featureIndices;
                xValues.ObservedValue = instance.featureValues;
                y.ObservedValue = instance.label;
                GaussianArray wSparsePost = engine.Infer<GaussianArray>(wSparse);
                for (int i = 0; i < wSparsePost.Count; i++)
                {
                    int index = instance.featureIndices[i];
                    wPost[index] = wSparsePost[i];
                }
                biasPost = engine.Infer<Gaussian>(bias);
            }
        }

        public class BpmPredict
        {
            private Variable<int> nFeatures;
            private VariableArray<double> w;
            private Variable<GaussianArray> wPrior;
            private Variable<Gaussian> biasPrior;
            private Variable<double> bias;
            private Variable<int> xValueCount;
            private VariableArray<double> xValues;
            private VariableArray<int> xIndices;
            private Variable<bool> y;
            private InferenceEngine engine;

            public BpmPredict()
            {
                nFeatures = Variable.New<int>().Named("nFeatures");
                Range feature = new Range(nFeatures).Named("feature");
                w = Variable.Array<double>(feature).Named("w");
                //VariableArray<Gaussian> wPrior = Variable.Array<Gaussian>(feature).Named("wPrior");
                //w[feature] = Variable<double>.Random(wPrior[feature]);
                wPrior = Variable.New<GaussianArray>().Named("wPrior");
                w.SetTo(Variable<double[]>.Random(wPrior));
                biasPrior = Variable.New<Gaussian>().Named("biasPrior");
                bias = Variable<double>.Random(biasPrior).Named("bias");
                xValueCount = Variable.New<int>().Named("xValueCount");
                Range userFeature = new Range(xValueCount).Named("userFeature");
                xValues = Variable.Array<double>(userFeature).Named("xValues");
                xIndices = Variable.Array<int>(userFeature).Named("xIndices");
                y = Variable.New<bool>().Named("y");
                VariableArray<double> product = Variable.Array<double>(userFeature).Named("product");
                VariableArray<double> wSparse = Variable.Subarray(w, xIndices);
                product[userFeature] = xValues[userFeature]*wSparse[userFeature];
                Variable<double> score = Variable.Sum(product).Named("score");
                y = (Variable.GaussianFromMeanAndVariance(score + bias, 1.0) > 0);
                engine = new InferenceEngine();
                engine.Compiler.FreeMemory = false;
                engine.Compiler.ReturnCopies = false;
                engine.OptimiseForVariables = new IVariable[] {y};
                engine.ModelName = "BpmPredict";
            }

            public void SetPriors(GaussianArray wDist, Gaussian biasDist)
            {
                nFeatures.ObservedValue = wDist.Count;
                wPrior.ObservedValue = wDist;
                biasPrior.ObservedValue = biasDist;
            }

            public bool Predict(Instance instance)
            {
                xValueCount.ObservedValue = instance.featureIndices.Length;
                xIndices.ObservedValue = instance.featureIndices;
                xValues.ObservedValue = instance.featureValues;
                Bernoulli yProb = engine.Infer<Bernoulli>(y);
                return (yProb.LogOdds > 0);
            }
        }

        public class BpmPredict2
        {
            private GaussianArray wDist;
            private Gaussian biasDist;

            public void SetPriors(GaussianArray wDist, Gaussian biasDist)
            {
                this.wDist = wDist;
                this.biasDist = biasDist;
            }

            public bool Predict(Instance instance)
            {
                double score = biasDist.GetMean();
                for (int i = 0; i < instance.featureIndices.Length; i++)
                {
                    score += wDist[instance.featureIndices[i]].GetMean()*instance.featureValues[i];
                }
                return (score > 0.0);
            }
        }

        public class BpmTrain2
        {
            private BpmTrain_EP gen;
            public GaussianArray wPost;
            public Gaussian biasPost;

            public BpmTrain2()
            {
                gen = new BpmTrain_EP();
            }

            public void SetPriors(int nf, double wVariance, double biasVariance)
            {
                gen.nFeatures = nf;
                wPost = new GaussianArray(nf, f => new Gaussian(0, wVariance));
                biasPost = new Gaussian(0, biasVariance);
            }

            public void Train(Instance instance)
            {
                gen.wPrior = wPost;
                gen.biasPrior = biasPost;
                gen.xValueCount = instance.featureIndices.Length;
                gen.xIndices = instance.featureIndices;
                gen.xValues = instance.featureValues;
                gen.y = instance.label;
                gen.Execute(1);
                GaussianArray wSparsePost = gen.WSparseMarginal();
                for (int i = 0; i < wSparsePost.Count; i++)
                {
                    int index = instance.featureIndices[i];
                    wPost[index] = wSparsePost[i];
                }
                biasPost = gen.BiasMarginal();
            }
        }

        public class BpmTrain_EP : IGeneratedAlgorithm
        {
            #region Fields

            /// <summary>Field backing the NumberOfIterationsDone property</summary>
            private int numberOfIterationsDone;

            /// <summary>Field backing the nFeatures property</summary>
            private int NFeatures;

            /// <summary>Field backing the wPrior property</summary>
            private DistributionStructArray<Gaussian, double> WPrior;

            /// <summary>Field backing the biasPrior property</summary>
            private Gaussian BiasPrior;

            /// <summary>Field backing the xValueCount property</summary>
            private int XValueCount;

            /// <summary>Field backing the xValues property</summary>
            private double[] XValues;

            /// <summary>Field backing the xIndices property</summary>
            private int[] XIndices;

            /// <summary>Field backing the y property</summary>
            private bool Y;

            /// <summary>The number of iterations last computed by Changed_biasPrior. Set this to zero to force re-execution of Changed_biasPrior</summary>
            public int Changed_biasPrior_iterationsDone;

            /// <summary>The number of iterations last computed by Constant. Set this to zero to force re-execution of Constant</summary>
            public int Constant_iterationsDone;

            /// <summary>The number of iterations last computed by Changed_xValueCount. Set this to zero to force re-execution of Changed_xValueCount</summary>
            public int Changed_xValueCount_iterationsDone;

            /// <summary>The number of iterations last computed by Changed_wPrior_xIndices_xValueCount. Set this to zero to force re-execution of Changed_wPrior_xIndices_xValueCount</summary>
            public int Changed_wPrior_xIndices_xValueCount_iterationsDone;

            /// <summary>The number of iterations last computed by Changed_xValueCount_xValues_wPrior_xIndices. Set this to zero to force re-execution of Changed_xValueCount_xValues_wPrior_xIndices</summary>
            public int Changed_xValueCount_xValues_wPrior_xIndices_iterationsDone;

            /// <summary>The number of iterations last computed by Changed_biasPrior_xValueCount_xValues_wPrior_xIndices. Set this to zero to force re-execution of Changed_biasPrior_xValueCount_xValues_wPrior_xIndices</summary>
            public int Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone;

            /// <summary>The number of iterations last computed by Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices. Set this to zero to force re-execution of Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices</summary>
            public int Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone;

            public DistributionStructArray<Gaussian, double> wSparse_F;
            public DistributionStructArray<Gaussian, double> product_F;
            public Gaussian score_F;
            public Gaussian vdouble13_F;

            /// <summary>Message from use of 'vdouble13'</summary>
            public Gaussian vdouble13_use_B;

            public DistributionStructArray<Gaussian, double> product_B;

            /// <summary>Message to marginal of 'wSparse'</summary>
            public DistributionStructArray<Gaussian, double> wSparse_marginal_F;

            /// <summary>Message from use of 'wSparse'</summary>
            public DistributionStructArray<Gaussian, double> wSparse_use_B;

            public Gaussian score_B;

            /// <summary>Message to marginal of 'bias'</summary>
            public Gaussian bias_marginal_F;

            /// <summary>Message from use of 'bias'</summary>
            public Gaussian bias_use_B;

            public Gaussian vdouble11_F;
            public Gaussian vdouble11_B;

            #endregion

            #region Properties

            /// <summary>The number of iterations done from the initial state</summary>
            public int NumberOfIterationsDone
            {
                get { return this.numberOfIterationsDone; }
            }

            /// <summary>The externally-specified value of 'nFeatures'</summary>
            public int nFeatures
            {
                get { return this.NFeatures; }
                set
                {
                    if (this.NFeatures != value)
                    {
                        this.NFeatures = value;
                        this.numberOfIterationsDone = 0;
                    }
                }
            }

            /// <summary>The externally-specified value of 'wPrior'</summary>
            public DistributionStructArray<Gaussian, double> wPrior
            {
                get { return this.WPrior; }
                set
                {
                    this.WPrior = value;
                    this.numberOfIterationsDone = 0;
                    this.Changed_wPrior_xIndices_xValueCount_iterationsDone = 0;
                    this.Changed_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                    this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                    this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                }
            }

            /// <summary>The externally-specified value of 'biasPrior'</summary>
            public Gaussian biasPrior
            {
                get { return this.BiasPrior; }
                set
                {
                    if (this.BiasPrior != value)
                    {
                        this.BiasPrior = value;
                        this.numberOfIterationsDone = 0;
                        this.Changed_biasPrior_iterationsDone = 0;
                        this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                        this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                    }
                }
            }

            /// <summary>The externally-specified value of 'xValueCount'</summary>
            public int xValueCount
            {
                get { return this.XValueCount; }
                set
                {
                    if (this.XValueCount != value)
                    {
                        this.XValueCount = value;
                        this.numberOfIterationsDone = 0;
                        this.Changed_xValueCount_iterationsDone = 0;
                        this.Changed_wPrior_xIndices_xValueCount_iterationsDone = 0;
                        this.Changed_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                        this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                        this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                    }
                }
            }

            /// <summary>The externally-specified value of 'xValues'</summary>
            public double[] xValues
            {
                get { return this.XValues; }
                set
                {
                    if ((value != null) && (value.Length != this.XValueCount))
                    {
                        throw new ArgumentException(((("Provided array of length " + value.Length) + " when length ") + this.XValueCount) +
                                                    " was expected for variable \'xValues\'");
                    }
                    this.XValues = value;
                    this.numberOfIterationsDone = 0;
                    this.Changed_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                    this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                    this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                }
            }

            /// <summary>The externally-specified value of 'xIndices'</summary>
            public int[] xIndices
            {
                get { return this.XIndices; }
                set
                {
                    if ((value != null) && (value.Length != this.XValueCount))
                    {
                        throw new ArgumentException(((("Provided array of length " + value.Length) + " when length ") + this.XValueCount) +
                                                    " was expected for variable \'xIndices\'");
                    }
                    this.XIndices = value;
                    this.numberOfIterationsDone = 0;
                    this.Changed_wPrior_xIndices_xValueCount_iterationsDone = 0;
                    this.Changed_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                    this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                    this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                }
            }

            /// <summary>The externally-specified value of 'y'</summary>
            public bool y
            {
                get { return this.Y; }
                set
                {
                    if (this.Y != value)
                    {
                        this.Y = value;
                        this.numberOfIterationsDone = 0;
                        this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                    }
                }
            }

            #endregion

            #region Methods

            /// <summary>Get the observed value of the specified variable.</summary>
            /// <param name="variableName">Variable name</param>
            public object GetObservedValue(string variableName)
            {
                if (variableName == "nFeatures")
                {
                    return this.nFeatures;
                }
                if (variableName == "wPrior")
                {
                    return this.wPrior;
                }
                if (variableName == "biasPrior")
                {
                    return this.biasPrior;
                }
                if (variableName == "xValueCount")
                {
                    return this.xValueCount;
                }
                if (variableName == "xValues")
                {
                    return this.xValues;
                }
                if (variableName == "xIndices")
                {
                    return this.xIndices;
                }
                if (variableName == "y")
                {
                    return this.y;
                }
                throw new ArgumentException("Not an observed variable name: " + variableName);
            }

            /// <summary>Set the observed value of the specified variable.</summary>
            /// <param name="variableName">Variable name</param>
            /// <param name="value">Observed value</param>
            public void SetObservedValue(string variableName, object value)
            {
                if (variableName == "nFeatures")
                {
                    this.nFeatures = (int) value;
                    return;
                }
                if (variableName == "wPrior")
                {
                    this.wPrior = (DistributionStructArray<Gaussian, double>) value;
                    return;
                }
                if (variableName == "biasPrior")
                {
                    this.biasPrior = (Gaussian) value;
                    return;
                }
                if (variableName == "xValueCount")
                {
                    this.xValueCount = (int) value;
                    return;
                }
                if (variableName == "xValues")
                {
                    this.xValues = (double[]) value;
                    return;
                }
                if (variableName == "xIndices")
                {
                    this.xIndices = (int[]) value;
                    return;
                }
                if (variableName == "y")
                {
                    this.y = (bool) value;
                    return;
                }
                throw new ArgumentException("Not an observed variable name: " + variableName);
            }

            /// <summary>The marginal distribution of the specified variable.</summary>
            /// <param name="variableName">Variable name</param>
            public object Marginal(string variableName)
            {
                if (variableName == "bias")
                {
                    return this.BiasMarginal();
                }
                if (variableName == "wSparse")
                {
                    return this.WSparseMarginal();
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
                this.Constant();
                this.Changed_xValueCount();
                this.Changed_wPrior_xIndices_xValueCount();
                this.Changed_xValueCount_xValues_wPrior_xIndices();
                this.Changed_biasPrior();
                this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices();
                this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices();
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
                this.score_F = ArrayHelper.MakeUniform<Gaussian>(new Gaussian());
                this.vdouble13_F = ArrayHelper.MakeUniform<Gaussian>(Gaussian.Uniform());
                this.vdouble13_use_B = ArrayHelper.MakeUniform<Gaussian>(Gaussian.Uniform());
                this.score_B = ArrayHelper.MakeUniform<Gaussian>(new Gaussian());
                this.Constant_iterationsDone = 1;
                this.Changed_xValueCount_iterationsDone = 0;
                this.Changed_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
            }

            /// <summary>Computations that depend on the observed value of xValueCount</summary>
            public void Changed_xValueCount()
            {
                if (this.Changed_xValueCount_iterationsDone == 1)
                {
                    return;
                }
                this.wSparse_F = new DistributionStructArray<Gaussian, double>(this.XValueCount);
                this.product_F = new DistributionStructArray<Gaussian, double>(this.XValueCount);
                this.product_B = new DistributionStructArray<Gaussian, double>(this.XValueCount);
                this.wSparse_marginal_F = new DistributionStructArray<Gaussian, double>(this.XValueCount);
                this.wSparse_use_B = new DistributionStructArray<Gaussian, double>(this.XValueCount);
                this.Changed_xValueCount_iterationsDone = 1;
                this.Changed_wPrior_xIndices_xValueCount_iterationsDone = 0;
                this.Changed_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
            }

            /// <summary>Computations that depend on the observed value of wPrior and xIndices and xValueCount</summary>
            public void Changed_wPrior_xIndices_xValueCount()
            {
                if (this.Changed_wPrior_xIndices_xValueCount_iterationsDone == 1)
                {
                    return;
                }
                this.wSparse_F = SubarrayOp<double>.ItemsAverageConditional<Gaussian, DistributionStructArray<Gaussian, double>>(this.WPrior, this.XIndices, this.wSparse_F);
                this.Changed_wPrior_xIndices_xValueCount_iterationsDone = 1;
                this.Changed_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
            }

            /// <summary>Computations that depend on the observed value of xValueCount and xValues and wPrior and xIndices</summary>
            public void Changed_xValueCount_xValues_wPrior_xIndices()
            {
                if (this.Changed_xValueCount_xValues_wPrior_xIndices_iterationsDone == 1)
                {
                    return;
                }
                for (int userFeature = 0; userFeature < this.XValueCount; userFeature++)
                {
                    this.product_F[userFeature] = GaussianProductOp.ProductAverageConditional(this.XValues[userFeature], this.wSparse_F[userFeature]);
                }
                this.score_F = FastSumOp.SumAverageConditional(this.product_F);
                this.Changed_xValueCount_xValues_wPrior_xIndices_iterationsDone = 1;
                this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
            }

            /// <summary>Computations that depend on the observed value of biasPrior</summary>
            public void Changed_biasPrior()
            {
                if (this.Changed_biasPrior_iterationsDone == 1)
                {
                    return;
                }
                this.Changed_biasPrior_iterationsDone = 1;
                this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
                this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
            }

            /// <summary>Computations that depend on the observed value of biasPrior and xValueCount and xValues and wPrior and xIndices</summary>
            public void Changed_biasPrior_xValueCount_xValues_wPrior_xIndices()
            {
                if (this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone == 1)
                {
                    return;
                }
                this.vdouble11_F = DoublePlusOp.SumAverageConditional(this.score_F, this.BiasPrior);
                this.vdouble13_F = GaussianFromMeanAndVarianceOp.SampleAverageConditional(this.vdouble11_F, 1);
                this.Changed_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 1;
                this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 0;
            }

            /// <summary>Computations that depend on the observed value of y and biasPrior and xValueCount and xValues and wPrior and xIndices</summary>
            public void Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices()
            {
                if (this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone == 1)
                {
                    return;
                }
                this.vdouble13_use_B = IsPositiveOp.XAverageConditional(this.Y, this.vdouble13_F);
                this.vdouble11_B = GaussianFromMeanAndVarianceOp.MeanAverageConditional(this.vdouble13_use_B, 1);
                this.bias_use_B = DoublePlusOp.BAverageConditional(this.vdouble11_B, this.score_F);
                this.bias_marginal_F = VariableOp.MarginalAverageConditional<Gaussian>(this.bias_use_B, this.BiasPrior, this.bias_marginal_F);
                this.score_B = DoublePlusOp.AAverageConditional(this.vdouble11_B, this.BiasPrior);
                this.product_B = FastSumOp.ArrayAverageConditional<DistributionStructArray<Gaussian, double>>(this.score_B, this.score_F, this.product_F, this.product_B);
                for (int userFeature = 0; userFeature < this.XValueCount; userFeature++)
                {
                    this.wSparse_use_B[userFeature] = GaussianProductOp.BAverageConditional(this.product_B[userFeature], this.XValues[userFeature]);
                }
                this.wSparse_marginal_F = DerivedVariableOp.MarginalAverageConditional<DistributionStructArray<Gaussian, double>>(this.wSparse_use_B, this.wSparse_F,
                                                                                                                                  this.wSparse_marginal_F);
                this.Changed_y_biasPrior_xValueCount_xValues_wPrior_xIndices_iterationsDone = 1;
            }

            /// <summary>
            /// Returns the marginal distribution for 'bias' given by the current state of the
            /// message passing algorithm.
            /// </summary>
            /// <returns>The marginal distribution</returns>
            public Gaussian BiasMarginal()
            {
                return this.bias_marginal_F;
            }

            /// <summary>
            /// Returns the marginal distribution for 'wSparse' given by the current state of the
            /// message passing algorithm.
            /// </summary>
            /// <returns>The marginal distribution</returns>
            public DistributionStructArray<Gaussian, double> WSparseMarginal()
            {
                return this.wSparse_marginal_F;
            }

            #endregion

            #region Events

            /// <summary>Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.</summary>
            public event EventHandler<ProgressChangedEventArgs> ProgressChanged;

            #endregion
        }
    }
}