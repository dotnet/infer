// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using System.Xml;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using Xunit;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Serialization;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Xunit.Assert;
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using GaussianArrayArray = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;

    /// <summary>
    /// Tests of the various examples in the documentation.  If these lines need to be changed in the future, 
    /// then the documentation must be updated as well.
    /// </summary>
    public class DocumentationTests
    {
        [Fact]
        public void SimpleGaussianExample()
        {
            // WARNING: If you change anything here, you must update the documentation            
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
            VariableArray<double> data = Variable.Observed(new double[] { 11, 5, 8, 9 });
            Range i = data.Range;
            data[i] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(i);

            // Create an inference engine for VMP
            InferenceEngine engine = new InferenceEngine(new Algorithms.VariationalMessagePassing());
            // Retrieve the posterior distributions
            Gaussian marginalMean = engine.Infer<Gaussian>(mean);
            Gamma marginalPrecision = engine.Infer<Gamma>(precision);
            Console.WriteLine("mean=" + marginalMean);
            Console.WriteLine("prec=" + marginalPrecision);
        }

        [Fact]
        public void ControllingInference()
        {
            // WARNING: If you change anything here, you must update the documentation            

            // The model
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
            Variable<int> dataCount = Variable.Observed(0);
            Range item = new Range(dataCount);
            VariableArray<double> data = Variable.Observed<double>(null, item).Named("data");
            data[item] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(item);

            // The data
            double[][] dataSets = new double[][]
                {
                    new double[] {11, 5, 8, 9},
                    new double[] {-1, -3, 2, 3, -5}
                };

            // Set the inference algorithm
            InferenceEngine engine = new InferenceEngine(new Algorithms.VariationalMessagePassing());

            // Get the compiled inference algorithm
            var ca = engine.GetCompiledInferenceAlgorithm(mean, precision);

            Gaussian[] meanExpected = new Gaussian[]
                {
                    new Gaussian(8.165, 1.026),
                    new Gaussian(-0.7877, 1.532)
                };
            Gamma[] precExpected = new Gamma[]
                {
                    new Gamma(3, 0.08038),
                    new Gamma(3.5, 0.03672)
                };

            // Run the inference on each data set
            for (int j = 0; j < dataSets.Length; j++)
            {
                // Set the data and the size of the range
                ca.SetObservedValue(dataCount.NameInGeneratedCode, dataSets[j].Length);
                ca.SetObservedValue(data.NameInGeneratedCode, dataSets[j]);

                // Execute the inference, running 10 iterations
                ca.Execute(10);
                // Retrieve the posterior distributions
                Gaussian marginalMean = ca.Marginal<Gaussian>(mean.NameInGeneratedCode);
                Gamma marginalPrecision = ca.Marginal<Gamma>(precision.NameInGeneratedCode);
                Console.WriteLine("mean = {0} should be {1}", marginalMean, meanExpected[j]);
                Console.WriteLine("prec = {0} should be {1}", marginalPrecision, precExpected[j]);
                Assert.True(meanExpected[j].MaxDiff(marginalMean) < 1e-2);
                Assert.True(precExpected[j].MaxDiff(marginalPrecision) < 1e-2);
            }
        }

        [Fact]
        public void PrecompiledAlgorithm()
        {
            // Data
            double[] dataSet = new double[100];
            for (int i = 0; i < dataSet.Length; i++)
                dataSet[i] = Rand.Normal(0, 1);

            // Observed variables for data and data count
            Variable<int> dataCount = Variable.Observed(dataSet.Length).Named("dataCount");
            Range N = new Range(dataCount);
            VariableArray<double> data = Variable.Observed<double>(dataSet, N).Named("data");

            // Observations are assumed to be sampled from a Gaussian with unknown parameters
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            data[N] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(N);

            // Create an inference engine for VMP
            InferenceEngine engine = new InferenceEngine(new Algorithms.VariationalMessagePassing());

            // Retrieve the posterior distributions
            Console.WriteLine("mean=" + engine.Infer(mean));
            Console.WriteLine("prec=" + engine.Infer(precision));
        }

        /// <summary>
        /// Test that the compiler throws an exception on missing operator methods.
        /// </summary>
        [Fact]
        public void MissingOperatorExample()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                // WARNING: If you change anything here, you must update the documentation            
                Variable<double> meanA = Variable.GaussianFromMeanAndVariance(0, 100);
                Variable<double> precisionA = Variable.GammaFromShapeAndScale(1, 1);
                Variable<double> meanB = Variable.GaussianFromMeanAndVariance(0, 100);
                Variable<double> precisionB = Variable.GammaFromShapeAndScale(1, 1);
                VariableArray<double> dataAB = Variable.Constant(new double[] { 20, 12, 10, 12 });
                Range i = dataAB.Range;
                VariableArray<double> dataA = Variable.Array<double>(i);
                VariableArray<double> dataB = Variable.Array<double>(i);
                dataA[i] = Variable.GaussianFromMeanAndPrecision(meanA, precisionA).ForEach(i);
                dataB[i] = Variable.GaussianFromMeanAndPrecision(meanB, precisionB).ForEach(i);
                dataAB[i] = Variable<double>.Factor(Factor.Product, dataA[i], dataB[i]);

                // Create an inference engine for VMP
                InferenceEngine engine = new InferenceEngine(new Algorithms.VariationalMessagePassing());
                //InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
                // Retrieve the posterior distributions
                Gaussian marginalMeanA = engine.Infer<Gaussian>(meanA);
                Gamma marginalPrecisionA = engine.Infer<Gamma>(precisionA);
                Gaussian marginalMeanB = engine.Infer<Gaussian>(meanB);
                Gamma marginalPrecisionB = engine.Infer<Gamma>(precisionB);
                Console.WriteLine("mean A = " + marginalMeanA);
                Console.WriteLine("prec A = " + marginalPrecisionA);
                Console.WriteLine("mean B = " + marginalMeanB);
                Console.WriteLine("prec B = " + marginalPrecisionB);
            });
        }

        /// <summary>
        /// Test that the compiler throws an exception on missing operator methods.
        /// </summary>
        [Fact]
        public void MissingOperatorTest2()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                InferenceEngine engine = new InferenceEngine();
                Variable<double> V = Variable.GaussianFromMeanAndVariance(0, 1);
                Variable<double> X = Variable.GaussianFromMeanAndVariance(0, V);
                Gaussian marginalX = engine.Infer<Gaussian>(X); // fails
                Console.WriteLine("Posterior X = " + marginalX);
                Assert.True(false, "Missing operator exception not thrown");
            });
        }

        [Fact]
        public void SharedVariablesExample()
        {
            // WARNING: If you change anything here, you must update the documentation            
            // The data
            double[][] dataSets = new double[][]
                {
                    new double[] {11, 5, 8, 9},
                    new double[] {-1, -3, 2, 3, -5}
                };
            int numChunks = dataSets.Length;
            // The model
            Gaussian priorMean = Gaussian.FromMeanAndVariance(0, 100);
            Gamma priorPrec = Gamma.FromShapeAndScale(1, 1);
            SharedVariable<double> mean = SharedVariable<double>.Random(priorMean);
            SharedVariable<double> precision = SharedVariable<double>.Random(priorPrec);
            Model model = new Model(numChunks);
            Variable<int> dataCount = Variable.New<int>();
            Range item = new Range(dataCount);
            VariableArray<double> data = Variable.Array<double>(item);
            data[item] = Variable.GaussianFromMeanAndPrecision(mean.GetCopyFor(model), precision.GetCopyFor(model)).ForEach(item);
            // Set the inference algorithm
            InferenceEngine engine = new InferenceEngine(new Algorithms.VariationalMessagePassing());
            for (int pass = 0; pass < 5; pass++)
            {
                // Run the inference on each data set
                for (int c = 0; c < numChunks; c++)
                {
                    dataCount.ObservedValue = dataSets[c].Length;
                    data.ObservedValue = dataSets[c];
                    model.InferShared(engine, c);
                }
            }
            // Retrieve the posterior distributions
            Gaussian marginalMean = mean.Marginal<Gaussian>();
            Gamma marginalPrec = precision.Marginal<Gamma>();
            Console.WriteLine("mean=" + marginalMean);
            Console.WriteLine("prec=" + marginalPrec);
        }

        [Fact]
        public void SimpleGaussianExampleEP()
        {
            // WARNING: If you change anything here, you must update the documentation            
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
            VariableArray<double> data = Variable.Constant(new double[] { 11, 5, 8, 9 });
            Range i = data.Range;
            data[i] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(i);

            try
            {
                // Create an inference engine for EP
                InferenceEngine engine = new InferenceEngine(new Algorithms.ExpectationPropagation());
                // Retrieve the posterior distributions
                Gaussian marginalMean = engine.Infer<Gaussian>(mean);
                Gamma marginalPrecision = engine.Infer<Gamma>(precision);
                Console.WriteLine("mean=" + marginalMean);
                Console.WriteLine("prec=" + marginalPrecision);
            }
            catch (ImproperMessageException)
            {
            }
        }


        [Fact]
        public void CreatingVariables()
        {
            // WARNING: If you change anything here, you must update the documentation!
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<bool> b = Variable.Bernoulli(0.5).Named("b");
            Variable<double> y = Variable.Random<double>(new Gaussian(0, 1));
            Variable<double> z = Variable.GaussianFromMeanAndPrecision(x, 1);
            Variable<double> one = Variable.Constant(1.0);
            Variable<double> anotherOne = Variable.New<double>();
            anotherOne.ObservedValue = 1;
            anotherOne.IsReadOnly = true;
            Variable<double> z2 = Variable.GaussianFromMeanAndPrecision(x, Variable.Constant(1.0));
            Variable<int> size = Variable.New<int>().Named("size");
            size.ObservedValue = 10;
            Variable<int> size2 = Variable.Observed(10).Named("size");
            Variable<Gaussian> priorOnX = Variable.New<Gaussian>().Named("priorOnX");
            Variable<double> x2 = Variable<double>.Random(priorOnX);
            priorOnX.ObservedValue = Gaussian.FromMeanAndPrecision(1.0, 2.0);
        }

        [Fact]
        public void FactorsEtc()
        {
            // WARNING: If you change anything here, you must update the documentation!
            Variable<double> var = Variable<double>.Factor(Factor.Product, 1.0, 5.0);
            Variable<double> a = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<double> b = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
            Variable<double> x = Variable<double>.Factor(Factor.Gaussian, mean, precision);
            Variable<double> x2 = Variable.GaussianFromMeanAndPrecision(mean, precision);
            Variable<double> y = Variable<double>.Factor(Factor.Plus, a, b);
            Variable<double> y2 = a + b;
            Variable<double> z = (a + b) * Variable.GaussianFromMeanAndPrecision(mean + x, precision);
            Variable<bool> c = a > b;
            Variable.Constrain(Constrain.Equal, x, y);
            Variable.ConstrainPositive(x);
            Variable.ConstrainTrue(a + b < x);
            Variable.ConstrainEqual(y - z, 5.0);
        }

        [Fact]
        public void ArraysAndRanges()
        {
            // WARNING: If you change anything here, you must update the documentation!
            Range pixel = new Range(10);
            Range image = new Range(4).Named("image");
            Variable<int> nImages = Variable.New<int>().Named("nImages");
            nImages.ObservedValue = 10;
            Range image2 = new Range(nImages);
            VariableArray<bool> bools = Variable.Array<bool>(pixel);
            VariableArray2D<double> doubles2D = Variable.Array<double>(pixel, image);
            VariableArray2D<double> someArray2D = Variable.Array<double>(pixel, image2);
            VariableArray2D<double> anotherArray2D = Variable.Array<double>(image2, image);
            doubles2D.SetTo(Factor.MatrixMultiply, someArray2D, anotherArray2D);
            VariableArray2D<double> doubles2D2 = Variable.MatrixMultiply(someArray2D, anotherArray2D);
            bools[pixel] = Variable.Bernoulli(0.7).ForEach(pixel);
            // TODO: update this to reflect the new documentation
            //y[C, D] = Variable.GaussianFromMeanAndPrecision(x[C], 1);
            //x[C] = Variable.GaussianFromMeanAndPrecision(y[C,D], 1); // runtime error

            // Constant array example
            VariableArray<double> data = Variable.Constant(new double[] { 1, 2, 3, 4 });
            VariableArray2D<double> data2D = Variable.Constant(new double[,] { { 5, 6 }, { 7, 9 } });

            // Observed array example 1
            VariableArray<double> obs = Variable.Observed(new double[] { 1, 2, 3, 4 });
            VariableArray2D<double> obs2D = Variable.Observed(new double[,] { { 5, 6 }, { 7, 9 } });
            Range r = obs.Range;
            Range r0 = obs2D.Range0;
            Range r1 = obs2D.Range1;

            // Observed array example 2
            Variable<int> sizeX = Variable.New<int>().Named("NX");
            Variable<int> sizeY = Variable.New<int>().Named("NY");
            Range x = new Range(sizeX).Named("x");
            Range y = new Range(sizeX).Named("y");
            VariableArray2D<double> arrayXY = Variable.Array<double>(x, y).Named("ArrayXY");

            double[,] obs2DData = new double[,] { { 5, 6 }, { 7, 9 }, { 6, 7 } };
            sizeX.ObservedValue = obs2DData.GetLength(0);
            sizeY.ObservedValue = obs2DData.GetLength(1);
            arrayXY.ObservedValue = obs2DData;
        }

        [Fact]
        public void ForEachBlockExample()
        {
            Range pixel = new Range(10);
            Range image = new Range(4);
            VariableArray<bool> bools2 = Variable.Array<bool>(pixel);
            VariableArray2D<double> doubles2D2 = Variable.Array<double>(pixel, image);
            bools2[pixel] = Variable.Bernoulli(0.7).ForEach(pixel) | Variable.Bernoulli(0.4).ForEach(pixel);
            doubles2D2[pixel, image] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(pixel, image);
            VariableArray<bool> bools = Variable.Array<bool>(pixel);
            VariableArray2D<double> doubles2D = Variable.Array<double>(pixel, image);
            using (Variable.ForEach(pixel))
            {
                bools[pixel] = Variable.Bernoulli(0.7) | Variable.Bernoulli(0.4);
                using (Variable.ForEach(image))
                {
                    doubles2D[pixel, image] = Variable.GaussianFromMeanAndVariance(0, 1);
                }
            }
            VariableArray2D<double> doubles2D3 = Variable.Array<double>(pixel, image);
            ForEachBlock pixelBlock = Variable.ForEach(pixel);
            ForEachBlock imageBlock = Variable.ForEach(image);
            doubles2D3[pixel, image] = Variable.GaussianFromMeanAndVariance(0, 1);
            imageBlock.CloseBlock();
            pixelBlock.CloseBlock();
            VariableArray2D<double> doubles2D4 = Variable.Array<double>(pixel, image);
            using (ForEachBlock pixelBlock2 = Variable.ForEach(pixel))
            {
                using (ForEachBlock imageBlock2 = Variable.ForEach(image))
                {
                    using (Variable.If(imageBlock2.Index < 2))
                    {
                        doubles2D4[pixel, image] = Variable.GaussianFromMeanAndVariance(0, 1);
                    }
                    using (Variable.IfNot(imageBlock2.Index < 2))
                    {
                        doubles2D4[pixel, image] = Variable.GaussianFromMeanAndVariance(1, 2);
                    }
                }
            }
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(doubles2D4));
        }

        [Fact]
        public void CloneRangeExample()
        {
            Range i = new Range(3);
            Range j = i.Clone();
            VariableArray<double> y = Variable.Array<double>(i);
            y.ObservedValue = new double[] { 1, 2, 3 };
            VariableArray2D<double> outerProduct = Variable.Array<double>(i, j);
            outerProduct[i, j] = y[i] * y[j];
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(outerProduct));
        }

        [Fact]
        public void JaggedArrayDocTests()
        {
            int[] sizes = new int[] { 2, 3 };
            Range item = new Range(sizes.Length).Named("item");
            VariableArray<int> sizesVar = Variable.Constant(sizes, item).Named("sizes");
            Range feature = new Range(sizesVar[item]).Named("feature");
            VariableArray<VariableArray<double>, double[][]> x =
                Variable.Array(Variable.Array<double>(feature), item).Named("x");

            Gaussian xPrior = new Gaussian(1.2, 3.4);
            x[item][feature] = Variable<double>.Random(xPrior).ForEach(item, feature);
            Variable.ConstrainPositive(x[item][feature]);

            // Implicit jagged array
            VariableArray<double> x1;
            using (Variable.ForEach(item))
            {
                Range feature1 = new Range(sizesVar[item]).Named("feature1");
                x1 = Variable.Array<double>(feature1).Named("x1");
                x1[feature1] = Variable<double>.Random(xPrior).ForEach(feature1);
                Variable.ConstrainPositive(x1[feature1]);
            }

            // .NET  constant array of array of doubles
            double[][] a = new double[][] { new double[] { 1.1, 3.3 }, new double[] { 1.1, 2.2, 4.4 } };
            int[] innerSizes = new int[a.Length];
            for (int i = 0; i < a.Length; i++)
                innerSizes[i] = a[i].Length;
            Range outer = new Range(a.Length).Named("outer");
            VariableArray<int> innerSizesVar =
                Variable.Constant(innerSizes, outer).Named("innerSizes");
            Range inner = new Range(innerSizesVar[outer]).Named("outer");
            VariableArray<VariableArray<double>, double[][]> aConst = Variable.Constant<double>(a, outer, inner);

            // 2D jagged array
            int[,] sizes2D = new int[,] { { 2, 3 }, { 4, 2 }, { 3, 1 } };
            Range rx = new Range(sizes2D.GetLength(0)).Named("rx");
            Range ry = new Range(sizes2D.GetLength(1)).Named("ry");
            VariableArray2D<int> sizes2DVar = Variable.Constant(sizes2D, rx, ry);
            Range rz = new Range(sizes2DVar[rx, ry]).Named("rz");
            VariableArray2D<VariableArray<double>, double[,][]> zVar =
                Variable.Array(Variable.Array<double>(rz), rx, ry).Named("zVar");

            // .NET array of array of array of doubles
            //double[][][] a = new double[][][] 
            //{
            //    new double[][] {new double[] {1.1, 3.3}, new double[] {1.1, 2.2, 4.4}},
            //    new double[][] {new double[] {2.2}, new double[] {2.2, 5.5, 4.4, 1.1}, new double[] {3.2, 2.2}}
            //};
            //int[] middleSizes = new int[a.Length];
            //int[][] innerSizes = new int[a.Length][];
            //for (int i=0; i < a.Length; i++)
            //{
            //    int len = a[i].Length;
            //    middleSizes[i] = len;
            //    innerSizes[i] = new int[len];
            //    for (int j = 0; j < len; j++)
            //        innerSizes[i][j] = a[i][j].Length;
            //}
            //Range outer = new Range(a.Length).Named("outer");
            //VariableArray<int> middleSizesVar = Variable.Constant(middleSizes, outer).Named("sizes");
            //Range middle = new Range(middleSizesVar[outer]).Named("middle");
            //var innerSizesVar = Variable.Constant(innerSizes, outer, middle);
            //Range inner = new Range(innerSizesVar[outer][middle]).Named("inner");
            //var aConst = Variable.Constant<double>(a, outer, middle, inner);
        }

        [Fact]
        public void DeepJaggedArrayExample()
        {
            var a = Variable.Array<Vector>(new Range(1));
            //var b = Variable.Array<VariableArray<Vector>, Vector[][]>(a, new Range(2));
            //var b = Variable<Vector[]>.Array(a, new Range(2));
            var b = Variable.Array(a, new Range(2));
            //var c = Variable.Array<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]>(b, new Range(3));
            var c = Variable.Array(b, new Range(3));
            //var d = Variable.Array<VariableArray<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]>, Vector[][][][]>(c, new Range(4));
            var d = Variable.Array(c, new Range(4));
        }

        [Fact]
        public void GetItemExample()
        {
            Range item = new Range(4);
            VariableArray<bool> bools = Variable.Array<bool>(item);
            bools[item] = Variable.Bernoulli(0.7).ForEach(item);
            Variable<int> index = Variable.New<int>();
            Variable.ConstrainTrue(bools[index]);

            InferenceEngine engine = new InferenceEngine();
            index.ObservedValue = 2;
            Console.WriteLine(engine.Infer(bools));
            // Result is:
            // [0] Bernoulli(0.7)
            // [1] Bernoulli(0.7)
            // [2] Bernoulli(1)
            // [3] Bernoulli(0.7)
            index.ObservedValue = 3;
            Console.WriteLine(engine.Infer(bools));
            // Result is:
            // [0] Bernoulli(0.7)
            // [1] Bernoulli(0.7)
            // [2] Bernoulli(0.7)
            // [3] Bernoulli(1)
        }

        [Fact]
        public void GetItemsExample()
        {
            Range item = new Range(4);
            VariableArray<bool> bools = Variable.Array<bool>(item);
            bools[item] = Variable.Bernoulli(0.7).ForEach(item);
            Variable<int> numIndices = Variable.New<int>();
            Range indexed_item = new Range(numIndices);
            VariableArray<int> indices = Variable.Array<int>(indexed_item);
            Variable.ConstrainTrue(bools[indices[indexed_item]]);

            InferenceEngine engine = new InferenceEngine();
            numIndices.ObservedValue = 2;
            indices.ObservedValue = new int[] { 1, 2 };
            Console.WriteLine(engine.Infer(bools));
            // Result is:
            // [0] Bernoulli(0.7)
            // [1] Bernoulli(1)
            // [2] Bernoulli(1)
            // [3] Bernoulli(0.7)
            numIndices.ObservedValue = 3;
            indices.ObservedValue = new int[] { 1, 2, 3 };
            Console.WriteLine(engine.Infer(bools));
            // Result is:
            // [0] Bernoulli(0.7)
            // [1] Bernoulli(1)
            // [2] Bernoulli(1)
            // [3] Bernoulli(1)
        }

        [Fact]
        public void SubarrayExample()
        {
            Range item = new Range(4);
            VariableArray<bool> bools = Variable.Array<bool>(item);
            bools[item] = Variable.Bernoulli(0.7).ForEach(item);
            Variable<int> numIndices = Variable.New<int>();
            Range indexed_item = new Range(numIndices);
            VariableArray<int> indices = Variable.Array<int>(indexed_item);
            VariableArray<bool> indexedBools = Variable.Subarray(bools, indices);
            // indexedBools automatically has range 'indexed_item'
            Variable.ConstrainTrue(indexedBools[indexed_item]);

            InferenceEngine engine = new InferenceEngine();
            numIndices.ObservedValue = 2;
            indices.ObservedValue = new int[] { 1, 2 };
            Console.WriteLine(engine.Infer(bools));
            // Result is:
            // [0] Bernoulli(0.7)
            // [1] Bernoulli(1)
            // [2] Bernoulli(1)
            // [3] Bernoulli(0.7)
            numIndices.ObservedValue = 3;
            indices.ObservedValue = new int[] { 1, 2, 3 };
            Console.WriteLine(engine.Infer(bools));
            // Result is:
            // [0] Bernoulli(0.7)
            // [1] Bernoulli(1)
            // [2] Bernoulli(1)
            // [3] Bernoulli(1)
        }


        //[Fact]
        internal void JaggedSubarrayExample()
        {
            Range item = new Range(4);
            VariableArray<bool> bools = Variable.Array<bool>(item);
            bools[item] = Variable.Bernoulli(0.7).ForEach(item);
            Range outer = new Range(2);
            Range inner = new Range(1);
            var indices = Variable.Array(Variable.Array<int>(inner), outer);
            using (Variable.ForEach(outer))
            {
                VariableArray<bool> indexedBools = Variable.Subarray(bools, indices[outer]);
                // indexedBools automatically has range 'inner'
                Variable.ConstrainTrue(indexedBools[inner]);
            }
            InferenceEngine engine = new InferenceEngine();
            indices.ObservedValue = new int[][] { new int[] { 0 }, new int[] { 2 } };
            Console.WriteLine(engine.Infer(bools));
            // Result is:
            // [0] Bernoulli(1)
            // [1] Bernoulli(0.7)
            // [2] Bernoulli(1)
            // [3] Bernoulli(0.7)
        }

        [Fact]
        public void JaggedGetItemExample()
        {
            Range item = new Range(2);
            VariableArray<int> sizes = Variable.Constant(new int[] { 3, 4 }, item);
            Range inner = new Range(sizes[item]);
            var bools = Variable.Array(Variable.Array<bool>(inner), item);
            bools[item][inner] = Variable.Bernoulli(0.7).ForEach(item, inner);
            Variable<int> index = Variable.New<int>();
            var boolsIndexed = bools[index];
            Range innerIndexed = boolsIndexed.Range;
            Variable.ConstrainTrue(boolsIndexed[innerIndexed]);

            InferenceEngine engine = new InferenceEngine();
            index.ObservedValue = 0;
            Console.WriteLine(engine.Infer(bools));
            // Result is:
            // [0] Bernoulli(0.7)
            // [1] Bernoulli(0.7)
            // [2] Bernoulli(1)
            // [3] Bernoulli(0.7)
            index.ObservedValue = 1;
            Console.WriteLine(engine.Infer(bools));
            // Result is:
            // [0] Bernoulli(0.7)
            // [1] Bernoulli(0.7)
            // [2] Bernoulli(0.7)
            // [3] Bernoulli(1)
        }

        [Fact]
        public void GameGraphUnrolled()
        {
            int nPlayers = 4;
            int nGames = 4;
            int[] winner = new int[] { 2, 2, 3, 3 };
            int[] loser = new int[] { 0, 1, 0, 2 };
            Variable<double>[] skill = new Variable<double>[nPlayers];
            for (int player = 0; player < nPlayers; player++)
            {
                skill[player] = Variable.GaussianFromMeanAndVariance(0, 100);
            }
            for (int game = 0; game < nGames; game++)
            {
                Variable<double> winner_performance = Variable.GaussianFromMeanAndVariance(skill[winner[game]], 1);
                Variable<double> loser_performance = Variable.GaussianFromMeanAndVariance(skill[loser[game]], 1);
                Variable.ConstrainTrue(winner_performance > loser_performance);
            }
            InferenceEngine engine = new InferenceEngine();
            for (int player = 0; player < nPlayers; player++)
            {
                Console.WriteLine("player {0} skill = {1}", player, engine.Infer(skill[player]));
            }
        }

        [Fact]
        public void GameGraphCompact()
        {
            int nPlayers = 4;
            int nGames = 4;
            int[] winner = new int[] { 2, 2, 3, 3 };
            int[] loser = new int[] { 0, 1, 0, 2 };
            Range player = new Range(nPlayers).Named("player");
            VariableArray<double> skill = Variable.Array<double>(player).Named("skill");
            skill[player] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(player);
            Range game = new Range(nGames).Named("game");
            VariableArray<int> winnerVar = Variable.Observed(winner, game).Named("winner");
            VariableArray<int> loserVar = Variable.Observed(loser, game).Named("loser");
            Variable<double> winner_performance = Variable.GaussianFromMeanAndVariance(skill[winnerVar[game]], 1);
            Variable<double> loser_performance = Variable.GaussianFromMeanAndVariance(skill[loserVar[game]], 1);
            Variable.ConstrainTrue(winner_performance > loser_performance);
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(skill));
        }

        [Fact]
        public void IfCaseAndSwitch()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<double> y = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<bool> b = Variable.Bernoulli(0.5);
            using (Variable.If(b))
            {
                Variable.ConstrainPositive(x);
            } // the block is now closed
            IfBlock ifb = Variable.If(b);
            Variable.ConstrainPositive(x);
            ifb.CloseBlock();
            using (Variable.IfNot(b))
            {
                Variable.ConstrainPositive(y);
            }
            Variable<double> z = Variable.New<double>();
            using (Variable.If(b))
            {
                z.SetTo(Variable.GaussianFromMeanAndVariance(0, 1));
            }
        }

        [Fact]
        public void AddingAttributes()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1);
            x.AddAttribute(new MyAttribute());
            Variable<bool> b = Variable.Bernoulli(0.5).Attrib(new MyAttribute());
            x.AddAttribute(new MarginalPrototype(new Gaussian()));
            x.AddAttribute(QueryTypes.MarginalDividedByPrior);
        }

        public class MyAttribute : Microsoft.ML.Probabilistic.Compiler.ICompilerAttribute
        {
        }


        [Fact]
        public void CaseExample()
        {
            Variable<int> c = Variable.Discrete(new double[] { 0.5, 0.5 });
            Variable<double> x = Variable.New<double>();
            using (Variable.Case(c, 0))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(1, 1));
            }
            using (Variable.Case(c, 1))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(2, 1));
            }
            InferenceEngine engine = new InferenceEngine();
            Gaussian expected = new Gaussian(1.5, 1.25);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }


        [Fact]
        public void CaseArrayExample()
        {
            Range i = new Range(4).Named("i");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");
            VariableArray<int> c = Variable.Array<int>(i).Named("c");
            using (Variable.ForEach(i))
            {
                c[i] = Variable.Discrete(new double[] { 0.5, 0.5 });
                using (Variable.Case(c[i], 0))
                {
                    x[i] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.Case(c[i], 1))
                {
                    x[i] = Variable.GaussianFromMeanAndVariance(2, 1);
                }
            }

            // check inference
            VariableArray<double> data = Variable.Constant(new double[] { 0.9, 1.1, 1.9, 2.1 }, i).Named("data");
            Variable.ConstrainEqual(x[i], data[i]);
            InferenceEngine engine = new InferenceEngine(new Algorithms.VariationalMessagePassing());
            Console.WriteLine(engine.Infer(c));
        }

        [Fact]
        public void SwitchExample()
        {
            int mixtureSize = 2;
            Range k = new Range(mixtureSize).Named("k");
            Variable<int> c = Variable.Discrete(k, new double[] { 0.5, 0.5 }).Named("c");
            VariableArray<double> means = Variable.Observed(new double[] { 1, 2 }, k).Named("means");
            Variable<double> x = Variable.New<double>().Named("x");
            using (Variable.Switch(c))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(means[c], 1));
            }

            means.ObservedValue = new double[] { 1, 2 };
            InferenceEngine engine = new InferenceEngine();
            Gaussian expected = new Gaussian(1.5, 1.25);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void AttributesExample()
        {
            Variable<int> mixtureSize = Variable.New<int>();
            Range k = new Range(mixtureSize);
            Variable<Vector> weights = Variable.New<Vector>();
            VariableArray<double> means = Variable.Array<double>(k);
            Variable<int> c = Variable.Discrete(k, weights);
            Variable<double> x = Variable.New<double>();
            x.AddAttribute(new TraceMessages());
            using (Variable.Switch(c))
            {
                x.SetTo(Variable.GaussianFromMeanAndVariance(means[c], 1));
            }

            means.ObservedValue = new double[] { 1, 2 };
            weights.ObservedValue = Vector.FromArray(new double[] { 0.5, 0.5 });
            mixtureSize.ObservedValue = 2;
            InferenceEngine engine = new InferenceEngine();
            Gaussian expected = new Gaussian(1.5, 1.25);
            Gaussian actual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} (should be {1})", actual, expected);
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        [Fact]
        public void SetToExample()
        {
            Variable<int> c = Variable.Discrete(new double[] { 0.5, 0.5 });
            Variable<double> x = Variable.New<double>();
            using (Variable.Case(c, 0))
            {
                // incorrect!!
                x = Variable.GaussianFromMeanAndVariance(1, 1);
            }
            using (Variable.Case(c, 1))
            {
                // clobbers the assignment to x above
                x = Variable.GaussianFromMeanAndVariance(2, 1);
            }
            InferenceEngine engine = new InferenceEngine();
            Gaussian g = engine.Infer<Gaussian>(x);
        }

        [Fact]
        public void EvidenceExample()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            // start of model
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable.ConstrainTrue(x > 0.5);
            // end of model
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            double logEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("The probability that a Gaussian(0,1) > 0.5 is {0}", System.Math.Exp(logEvidence));
            Assert.True(MMath.AbsDiff(System.Math.Exp(logEvidence), 0.308537538725987) < 1e-10);
        }

        [Fact]
        public void Initialisation()
        {
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1);
            x.InitialiseTo(Gaussian.FromMeanAndVariance(0, 10));

            Range r = new Range(10);
            VariableArray<double> y = Variable.Array<double>(r);
            y[r].InitialiseTo(Gaussian.FromMeanAndVariance(0, 1));

            Gaussian[] inity = new Gaussian[r.SizeAsInt];
            for (int i = 0; i < inity.Length; i++)
                inity[i] = Gaussian.FromMeanAndVariance(Rand.Normal(), 1);
            y.InitialiseTo(Distribution<double>.Array(inity));

            VariableArray<Gaussian> initVar = Variable.Observed(inity, r);
            y[r].InitialiseTo(initVar[r]);
        }

        internal void GivenExample1()
        {
            Vector data = Vector.Zero(1);
            Variable<Vector> dataGiven = Variable.New<Vector>();
            dataGiven.ObservedValue = data;
        }

        internal void GivenExamplePriorOnX()
        {
            Variable<Gaussian> priorOnX = Variable.New<Gaussian>();
            Variable<double> x = Variable.Random<double, Gaussian>(priorOnX);
            priorOnX.ObservedValue = Gaussian.FromMeanAndPrecision(1.0, 2.0);
        }

        internal void GivenExample2()
        {
            int nItems = 10;
            Vector[] data = new Vector[nItems];

            Range item = new Range(nItems);
            VariableArray<Vector> dataGiven = Variable.Array<Vector>(item);
            dataGiven.ObservedValue = data;
        }

        internal void GivenExample3()
        {
            int nItems = 10;
            int nGroups = 5;
            Vector[][] data = new Vector[nGroups][];
            for (int g = 0; g < nGroups; g++)
            {
                data[g] = new Vector[nItems];
            }
            Range item = new Range(nItems);
            VariableArray<Vector>[] dataGiven = new VariableArray<Vector>[nGroups];
            for (int g = 0; g < nGroups; g++)
            {
                dataGiven[g] = Variable.Array<Vector>(item);
                dataGiven[g].ObservedValue = data[g];
            }
        }


        internal void GivenExample4()
        {
            int nItems = 10;
            int nFeatures = 5;
            Vector[,] data = new Vector[nItems, nFeatures];

            Range item = new Range(nItems);
            Range feature = new Range(nFeatures);
            VariableArray2D<Vector> dataGiven = Variable.Array<Vector>(item, feature);
            dataGiven.ObservedValue = data;
        }

        internal void GivenExample5()
        {
            int nItems = 10;
            int nFeatures = 5;
            int nGroups = 3;
            Vector[][,] data = new Vector[nGroups][,];
            for (int g = 0; g < nGroups; g++)
            {
                data[g] = new Vector[nItems, nFeatures];
            }
            Range item = new Range(nItems);
            Range feature = new Range(nFeatures);

            VariableArray2D<Vector>[] dataGiven = new VariableArray2D<Vector>[nGroups];
            for (int g = 0; g < nGroups; g++)
            {
                dataGiven[g] = Variable.Array<Vector>(item, feature);
            }
            for (int g = 0; g < nGroups; g++)
            {
                dataGiven[g].ObservedValue = data[g];
            }
        }

        internal void GivenExample6()
        {
            int nItems = 10;
            int nPlaces = 5;
            int nGroups = 3;
            Vector[,][] data = new Vector[nPlaces, nGroups][];
            for (int p = 0; p < nPlaces; p++)
            {
                for (int g = 0; g < nGroups; g++)
                {
                    data[p, g] = new Vector[nItems];
                }
            }
            Range item = new Range(nItems);

            VariableArray<Vector>[,] dataGiven = new VariableArray<Vector>[nPlaces, nGroups];
            for (int p = 0; p < nPlaces; p++)
            {
                for (int g = 0; g < nGroups; g++)
                {
                    dataGiven[p, g] = Variable.Array<Vector>(item);
                }
            }
            for (int p = 0; p < nPlaces; p++)
            {
                for (int g = 0; g < nGroups; g++)
                {
                    dataGiven[p, g].ObservedValue = data[p, g];
                }
            }
        }

        [Fact]
        public void TrueSkillSmall()
        {
            Variable<double> skill1 = Variable.GaussianFromMeanAndVariance(3, 2);
            Variable<double> skill2 = Variable.GaussianFromMeanAndVariance(4, 1);
            Variable.ConstrainTrue(skill1 > skill2);
            Console.WriteLine(new InferenceEngine().Infer(skill1));

            //InferenceEngine engine = new InferenceEngine();
            //engine.ShowFactorGraph = true;
            //Console.WriteLine(engine.Infer(skill1));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ChannelTransformExample()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(ChannelTransformInput, 2, new int[] { 0, 0 });
            ca.Execute(engine.NumberOfIterations);
            Console.WriteLine(ca.Marginal("array2D"));
        }

        internal void ChannelTransformInput(int scalarGiven, int[] arrayGiven)
        {
            // constants.
            double precision = 1.0;
            int arrayLength = 2;
            double scalar = Gaussian.Sample(0.0, precision);
            double[] array = new double[arrayLength];
            for (int i = 0; i < arrayLength; i++)
            {
                // defining an array via its elements.
                // using a scalar in a plate.
                // using a scalar constant in a plate.
                array[i] = Gaussian.Sample(scalar, precision);
            }
            // using given for array size.
            double[] items = new double[scalarGiven];
            // defining whole array at once.
            // passing whole array to method.
            // passing given array to method.
            items = Collection.GetItems(array, arrayGiven);
            double[,] array2D = new double[arrayLength, scalarGiven];
            for (int j = 0; j < arrayLength; j++)
            {
                // using given for loop bound.
                for (int k = 0; k < scalarGiven; k++)
                {
                    // using a scalar in two plates.
                    // using an array element in another plate.
                    array2D[j, k] = Factor.Plus(scalar, items[k]);
                }
            }
            InferNet.Infer(array2D, nameof(array2D));
        }

#if false
        public void ChannelTransformOutput(int scalarGiven, int[] arrayGiven)
        {
            // defining a constant.
            double precision = 1.0;
            int arrayLength = 2;

            double scalar = Gaussian.Sample(0.0, precision);
            double scalar_marginal = 0;
            double[] scalar_uses = new double[2];
            scalar_uses = Factor.UsesEqualDef<double>(scalar, scalar_marginal);

            double[] array = new double[arrayLength];
            double[] array_marginal = new double[arrayLength];
            double[][] array_uses = new double[1][];
            for (int _ind = 0; _ind < 1; _ind++)
            {
                array_uses[_ind] = new double[arrayLength];
            }
            array_uses = Factor.UsesEqualDef<double[]>(array, array_marginal);

            double[] scalar__uses0 = new double[array.Length];
            scalar__uses0 = Factor.Replicate<double>(scalar_uses[0], array.Length);
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = Gaussian.Sample(scalar__uses0[i], precision);
            }

            // using given for array size.
            double[] items = new double[scalarGiven];
            double[] items_marginal = new double[scalarGiven];
            double[][] items_uses = new double[1][];
            for (int _ind = 0; _ind < 1; _ind++)
            {
                items_uses[_ind] = new double[scalarGiven];
            }
            items_uses = Factor.UsesEqualDef<double[]>(items, items_marginal);

            // defining whole array at once.
            // passing whole array to method.
            // passing given array to method.
            items = Factor.GetItems(array_uses[0], arrayGiven);
            double[,] array2D = new double[arrayLength, scalarGiven];
            double[,] array2D_marginal = new double[arrayLength, scalarGiven];
            double[][,] array2D_uses = new double[0][,];
            for (int _ind = 0; _ind < 0; _ind++)
            {
                array2D_uses[_ind] = new double[arrayLength, scalarGiven];
            }
            array2D_uses = Factor.UsesEqualDef<double[,]>(array2D, array2D_marginal);
            double[,] scalar__uses1 = new double[array2D.GetLength(0), scalarGiven];
            scalar__uses1 = (double[,])Factor.ReplicateNd<double>(scalar_uses[1]);
            double[][] items__uses0 = new double[scalarGiven][];
            for (int i = 0; i < scalarGiven; i++)
            {
                items__uses0[i] = new double[array2D.GetLength(0)];
                items__uses0[i] = Factor.Replicate<double>(items_uses[0][i], array2D.GetLength(0));
            }
            for (int j = 0; j < array2D.GetLength(0); j++)
            {
                // using given for loop bound.
                for (int k = 0; k < scalarGiven; k++)
                {
                    // using a scalar in two plates.
                    // using an array element in another plate.
                    array2D[j, k] = Factor.Plus(scalar__uses1[j, k], items__uses0[k][j]);
                }
            }
            InferNet.Infer(array2D, nameof(array2D));
        }
        public void ChannelTransformOutputNew(int scalarGiven, int[] arrayGiven)
        {
            // defining a constant.
            double precision = 1.0;
            int arrayLength = 2;

            double scalar = Gaussian.Sample(0.0, precision);
            double scalar_marginal = new double();
            double[] scalar_uses = new double[2];
            scalar_uses = Factor.UsesEqualDef<double>(scalar, scalar_marginal);

            double[] array = new double[arrayLength];
            double[] array_marginal = new double[arrayLength];
            double[][] array_uses = new double[1][];
            for (int _ind = 0; _ind < 1; _ind++)
            {
                array_uses[_ind] = new double[arrayLength];
            }
            array_uses = Factor.UsesEqualDef<double[]>(array, array_marginal);

            double[] scalar_rep0 = new double[array.Length];
            scalar_rep0 = Factor.Replicate<double>(scalar_uses[0], array.Length);
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = Gaussian.Sample(scalar_rep0[i], precision);
            }

            // using given for array size.
            double[] items = new double[scalarGiven];
            double[] items_marginal = new double[scalarGiven];
            double[][] items_uses = new double[1][];
            for (int _ind = 0; _ind < 1; _ind++)
            {
                items_uses[_ind] = new double[scalarGiven];
            }
            items_uses = Factor.UsesEqualDef<double[]>(items, items_marginal);
            // defining whole array at once.
            // passing whole array to method.
            // passing given array to method.
            items = Factor.GetItems(array_uses[0], arrayGiven);

            double[,] array2D = new double[arrayLength, scalarGiven];
            double[,] array2D_marginal = new double[arrayLength, scalarGiven];
            double[][,] array2D_uses = new double[0][,];
            for (int _ind = 0; _ind < 0; _ind++)
            {
                array2D_uses[_ind] = new double[arrayLength, scalarGiven];
            }
            array2D_uses = Factor.UsesEqualDef<double[,]>(array2D, array2D_marginal);

            double[,] scalar_rep1 = new double[array2D.GetLength(0), scalarGiven];
            scalar_rep1 = (double[,])Factor.ReplicateNd<double>(scalar_uses[1]);
            double[][] items_rep0 = new double[array2D.GetLength(0)][];
            items_rep0 = Factor.Replicate<double[]>(items_uses[0], array2D.GetLength(0));
            for (int j = 0; j < array2D.GetLength(0); j++)
            {
                // using given for loop bound.
                for (int k = 0; k < scalarGiven; k++)
                {
                    // using a scalar in two plates.
                    // using an array element in another plate.
                    array2D[j, k] = Factor.Plus(scalar_rep1[j, k], items_rep0[j][k]);
                }
            }
            InferNet.Infer(array2D, nameof(array2D));
        }
        /// <summary>
        /// Indexing each element of an array a different number of times (via literal indexing).
        /// </summary>
        /// <param name="data"></param>
        public void ChannelTransformInput2(double[] data0, double[] data1)
        {
            double[] means = new double[2];
            for (int i = 0; i < means.Length; i++)
            {
                means[i] = Factor.Random(new Gaussian(0, 100));
            }
            for (int j = 0; j < data0.Length; j++)
            {
                data0[j] = Factor.Gaussian(means[0], 1.0);
            }
            for (int j = 0; j < data1.Length; j++)
            {
                data1[j] = Factor.Gaussian(means[1], 1.0);
            }
        }
        public void ChannelTransformOutput2(double[] data0, double[] data1)
        {
            double[] means = new double[2];
            double[] means_marginal = new double[means.Length];
            double[][] means_uses = new double[1][];
            for (int _ind = 0; _ind < 1; _ind++)
            {
                means_uses[_ind] = new double[means.Length];
            }
            means_uses = Factor.UsesEqualDef<double[]>(means, means_marginal);
            for (int i = 0; i < means.Length; i++)
            {
                means[i] = Factor.Random(new Gaussian(0, 100));
            }
            double[][] means_item_uses = new double[means.Length][];
            means_item_uses[0] = new double[data0.Length];  // number of times means[0] is used.
            means_item_uses[0] = Factor.Replicate<double>(means_uses[0][0], data0.Length);
            means_item_uses[1] = new double[data1.Length];  // number of times means[1] is used.
            means_item_uses[1] = Factor.Replicate<double>(means_uses[0][1], data1.Length);
            for (int j = 0; j < data0.Length; j++)
            {
                data0[j] = Factor.Gaussian(means_item_uses[0][j], 1.0);
            }
            for (int j = 0; j < data1.Length; j++)
            {
                data1[j] = Factor.Gaussian(means_item_uses[1][j], 1.0);
            }
        }
        /// <summary>
        /// Indexing each element of an array a different number of times (via loop indexing).
        /// </summary>
        /// <param name="data"></param>
        public void ChannelTransformInput3(double[][] data)
        {
            double[] means = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                means[i] = Factor.Random(new Gaussian(0, 100));
            }
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = Factor.Gaussian(means[i], 1.0);
                }
            }
        }
        public void ChannelTransformOutput3(double[][] data)
        {
            double[] means = new double[data.Length];
            double[] means_marginal = new double[data.Length];
            double[][] means_uses = new double[1][];
            for (int _ind = 0; _ind < 1; _ind++)
            {
                means_uses[_ind] = new double[data.Length];
            }
            means_uses = Factor.UsesEqualDef<double[]>(means, means_marginal);
            double[][] means_item_uses = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                means_item_uses[i] = new double[data[i].Length];  // number of times means[i] is used.
                means_item_uses[i] = Factor.Replicate<double>(means_uses[0][i], data[i].Length);
            }
            for (int i = 0; i < data.Length; i++)
            {
                means[i] = Factor.Random(new Gaussian(0, 100));
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = Factor.Gaussian(means_item_uses[i][j], 1.0);
                }
            }
        }
        /// <summary>
        /// Replicating to a jagged array.
        /// </summary>
        /// <param name="data"></param>
        public void ChannelTransformInput4(double[][] data)
        {
            double mean = Factor.Gaussian(0, 100);
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = Factor.Gaussian(mean, 1.0);
                }
            }
        }
        public void ChannelTransformOutput4(double[][] data)
        {
            double mean = Factor.Gaussian(0, 100);
            double mean_marginal = new double();
            double[] mean_uses = new double[1];
            mean_uses = Factor.UsesEqualDef<double>(mean, mean_marginal);
            double[] mean_uses0 = new double[data.Length];
            mean_uses0 = Factor.Replicate<double>(mean_uses[0], data.Length);
            for (int i = 0; i < data.Length; i++)
            {
                double[] mean_uses0_i = new double[data[i].Length];
                mean_uses0_i = Factor.Replicate<double>(mean_uses0[i], data[i].Length);
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = Factor.Gaussian(mean_uses0_i[j], 1.0);
                }
            }
        }
#endif

        [Fact]
        public void HybridModelExample()
        {
            double[] dataSet = new double[] { 5, 5.1, 5.2, 4.9, -5.1, -5.2, -5.3, -4.9 };
            Gaussian actualMean, expectedMean;
            Gamma actualPrec, expectedPrec;
            EstimateGaussian(new InferenceEngine(new Algorithms.VariationalMessagePassing()), dataSet,
                             out expectedMean, out expectedPrec);
            InferenceEngine engine1 = new InferenceEngine() { NumberOfIterations = 10 };
            InferenceEngine engine2 = new InferenceEngine(new Algorithms.VariationalMessagePassing()) { NumberOfIterations = 10 };
            EstimateGaussianSharedDefinition(dataSet, engine1, engine2, out actualMean, out actualPrec);
            Assert.True(expectedMean.MaxDiff(actualMean) < 1e-4);
            Assert.True(expectedPrec.MaxDiff(actualPrec) < 1e-2);
            EstimateGaussianSharedDefinitionParallel(dataSet, engine1, engine2, out actualMean, out actualPrec);
            Assert.True(expectedMean.MaxDiff(actualMean) < 1e-4);
            Assert.True(expectedPrec.MaxDiff(actualPrec) < 1e-2);
        }

        private void EstimateGaussianSharedDefinition(
            double[] data, InferenceEngine engine1, InferenceEngine engine2,
            out Gaussian meanPosterior, out Gamma precPosterior)
        {
            Range n = new Range(data.Length);
            Model meanModel = new Model(1);
            Model precModel = new Model(1);
            Model dataModel = new Model(1);
            var sharedMean = SharedVariable<double>.Random(Gaussian.Uniform());
            var sharedPrec = SharedVariable<double>.Random(Gamma.Uniform());
            var mean = Variable.GaussianFromMeanAndPrecision(0, 1);
            var prec = Variable.GammaFromShapeAndRate(10, 10);

            sharedMean.SetDefinitionTo(meanModel, mean);
            sharedPrec.SetDefinitionTo(precModel, prec);
            var x = Variable.Array<double>(n);
            x[n] = Variable.GaussianFromMeanAndPrecision(
                sharedMean.GetCopyFor(dataModel), sharedPrec.GetCopyFor(dataModel)).ForEach(n);
            x.ObservedValue = data;

            for (int pass = 0; pass < 10; pass++)
            {
                meanModel.InferShared(engine1, 0);
                precModel.InferShared(engine1, 0);
                dataModel.InferShared(engine2, 0);
            }
            meanPosterior = sharedMean.Marginal<Gaussian>();
            precPosterior = sharedPrec.Marginal<Gamma>();
        }

        private void EstimateGaussianSharedDefinitionParallel(
            double[] data, InferenceEngine engine1, InferenceEngine engine2,
            out Gaussian meanPosterior, out Gamma precPosterior)
        {
            Range n = new Range(data.Length);
            Model meanModel = new Model(1);
            Model precModel = new Model(1);
            Model dataModel = new Model(1);
            var sharedMean = SharedVariable<double>.Random(Gaussian.Uniform());
            var sharedPrec = SharedVariable<double>.Random(Gamma.Uniform());
            var mean = Variable.GaussianFromMeanAndPrecision(0, 1);
            var prec = Variable.GammaFromShapeAndRate(10, 10);

            sharedMean.SetDefinitionTo(meanModel, mean);
            sharedPrec.SetDefinitionTo(precModel, prec);
            var x = Variable.Array<double>(n);
            x[n] = Variable.GaussianFromMeanAndPrecision(
                sharedMean.GetCopyFor(dataModel), sharedPrec.GetCopyFor(dataModel)).ForEach(n);
            x.ObservedValue = data;

            InferenceEngine engine1b = new InferenceEngine()
            {
                NumberOfIterations = 10
            };
            for (int pass = 0; pass < 10; pass++)
            {
                System.Threading.Tasks.Parallel.Invoke(
                    () => meanModel.InferShared(engine1, 0),
                    () => precModel.InferShared(engine1b, 0));
                dataModel.InferShared(engine2, 0);
            }
            meanPosterior = sharedMean.Marginal<Gaussian>();
            precPosterior = sharedPrec.Marginal<Gamma>();
        }

        private void EstimateGaussian(
            InferenceEngine engine, double[] data, out Gaussian expectedMean, out Gamma expectedPrec)
        {
            Range n = new Range(data.Length).Named("n");
            var mean = Variable.GaussianFromMeanAndPrecision(0, 1).Named("mean");
            var prec = Variable.GammaFromShapeAndRate(10, 10).Named("prec");

            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(mean, prec).ForEach(n);
            x.ObservedValue = data;
            expectedMean = engine.Infer<Gaussian>(mean);
            expectedPrec = engine.Infer<Gamma>(prec);
        }

        [Fact]
        public void JaggedSharedVariableArray()
        {
            double[][][] data = new double[][][]
                {
                    new double[][] {new double[] {1, 2}, new double[] {3, 4}},
                    new double[][] {new double[] {5, 6}, new double[] {7, 8}}
                };
            int numChunks = data.Length;
            Range outer = new Range(data[0].Length);
            Range inner = new Range(data[0][0].Length);
            GaussianArrayArray priorW = new GaussianArrayArray(new GaussianArray(new Gaussian(0, 1), inner.SizeAsInt), outer.SizeAsInt);

            var w = SharedVariable<double>.Random(Variable.Array<double>(inner), outer, priorW).Named("w");
            Model model = new Model(numChunks);
            var x = Variable.Array<double>(Variable.Array<double>(inner), outer).Named("x");
            x[outer][inner] = Variable.GaussianFromMeanAndPrecision(w.GetCopyFor(model)[outer][inner], 1.0);
            InferenceEngine engine = new InferenceEngine();

            for (int pass = 0; pass < 5; pass++)
            {
                for (int c = 0; c < numChunks; c++)
                {
                    x.ObservedValue = data[c];
                    model.InferShared(engine, c);
                }
            }
            var posteriorW = w.Marginal<Gaussian[][]>();
        }

        //[Fact]
        internal void AllZeroExceptionExample()
        {
            try
            {
                Variable<bool> b = Variable.Bernoulli(0.1).Named("b");
                Variable<bool> x = Variable.New<bool>().Named("x");
                using (Variable.If(b))
                {
                    x.SetTo(Variable.Bernoulli(1));
                }
                using (Variable.IfNot(b))
                {
                    x.SetTo(Variable.Bernoulli(0));
                }
                Variable.ConstrainTrue(x);
                InferenceEngine engine = new InferenceEngine(new Algorithms.VariationalMessagePassing());
                Console.WriteLine(engine.Infer(b));
            }
            catch (AllZeroException ex)
            {
                Console.WriteLine("Correctly threw exception: " + ex);
            }
        }

        //[Fact]
        internal void ImproperMessageExceptionExample()
        {
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            // this prior gives ImproperMessageException
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");
            // this prior works
            //Variable<double> precision = Variable.GammaFromShapeAndScale(1, .1).Named("precision");
            Variable<double> x1 = Variable.GaussianFromMeanAndPrecision(mean, precision);
            Variable<double> y1 = Variable.GaussianFromMeanAndPrecision(x1, 1);
            Variable<double> x2 = Variable.GaussianFromMeanAndPrecision(mean, precision);
            Variable<double> y2 = Variable.GaussianFromMeanAndPrecision(x2, 1);
            y1.ObservedValue = -5;
            y2.ObservedValue = 5;
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(mean));
        }

        internal void XmlSerializationExample()
        {
            Dirichlet d = new Dirichlet(3.0, 1.0, 2.0);
            string fileName = "temp.xml";
            DataContractSerializer serializer = new DataContractSerializer(typeof(Dirichlet), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });
            // write to disk
            using (XmlDictionaryWriter writer = XmlDictionaryWriter.CreateTextWriter(new FileStream(fileName, FileMode.Create)))
            {
                serializer.WriteObject(writer, d);
            }
            // read from disk
            using (XmlDictionaryReader reader = XmlDictionaryReader.CreateTextReader(new FileStream(fileName, FileMode.Open), new XmlDictionaryReaderQuotas()))
            {
                Dirichlet d2 = (Dirichlet)serializer.ReadObject(reader);
                Console.WriteLine(d2);
            }
        }

        [Fact]
        public void RandomWalkExample()
        {
            Variable<int> numTimes = Variable.Observed(10);
            Range time = new Range(numTimes);
            VariableArray<double> x = Variable.Array<double>(time);
            using (var block = Variable.ForEach(time))
            {
                var t = block.Index;
                using (Variable.If(t == 0))
                {
                    x[t] = Variable.GaussianFromMeanAndVariance(0, 1);
                }
                using (Variable.If(t > 0))
                {
                    x[t] = Variable.GaussianFromMeanAndVariance(x[t - 1], 1);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(x));
        }

        [Fact]
        public void RandomWalkExample2()
        {
            Variable<int> numTimes = Variable.Observed(10);
            Range time = new Range(numTimes);
            VariableArray<double> x = Variable.Array<double>(time);
            using (var block = Variable.ForEach(time))
            {
                var t = block.Index;
                using (Variable.If(t == 0))
                {
                    x[t] = Variable.GaussianFromMeanAndVariance(0, 1);
                }
                using (Variable.If(t == 1))
                {
                    x[t] = Variable.GaussianFromMeanAndVariance(x[t - 1], 1);
                }
                using (Variable.If(t > 1))
                {
                    x[t] = Variable.GaussianFromMeanAndVariance(x[t - 1] + x[t - 2], 1);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(x));
        }

        [Fact]
        public void RandomWalkExampleUndirected()
        {
            Variable<int> numTimes = Variable.Observed(10);
            Range time = new Range(numTimes);
            VariableArray<double> x = Variable.Array<double>(time);
            x[time] = Variable.GaussianFromMeanAndVariance(0, 1000).ForEach(time);
            using (var block = Variable.ForEach(time))
            {
                var t = block.Index;
                using (Variable.If(t > 0))
                {
                    Variable.ConstrainEqualRandom(x[t] - x[t - 1], new Gaussian(0, 1));
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(x));
        }

        [Fact]
        public void MarginalDividedByPriorExample()
        {
            Variable<double> weight = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable<Gaussian> weightInbox = Variable.Observed(new Gaussian(3, 4));
            Variable.ConstrainEqualRandom(weight, weightInbox);
            Variable<double> weightCopy = Variable.Copy(weight);
            weightCopy.AddAttribute(QueryTypes.MarginalDividedByPrior);
            InferenceEngine engine = new InferenceEngine();
            var message = engine.Infer<Gaussian>(weightCopy, QueryTypes.MarginalDividedByPrior);
            Assert.True(message.IsUniform());
        }

        [Fact]
        public void DerivativeExample()
        {
            // Compute the derivative of the function log(x) at the point x = 0.5
            Variable<double> xPoint = Variable.Observed(0.5);
            Variable<double> x = Variable.GammaFromMeanAndVariance(0.5, 0.0);
            x.Name = nameof(x);
            x.AddAttribute(QueryTypes.MarginalDividedByPrior);
            Variable<double> f = Variable.Log(x);
            f.Name = nameof(f);
            f.AddAttribute(new MarginalPrototype(new Gaussian()));
            Variable.ConstrainEqualRandom(f, Gaussian.FromNatural(1, 0));
            InferenceEngine engine = new InferenceEngine();
            var xPost = engine.Infer<Gamma>(x, QueryTypes.MarginalDividedByPrior);
            xPost.GetDerivatives(xPoint.ObservedValue, out double derivative, out _);
            Assert.Equal(1.0 / xPoint.ObservedValue, derivative);
        }

        [Fact]
        public void DerivativeExample2()
        {
            // Compute the derivative of the function exp(x) at the point x = 0.5
            Variable<double> xPoint = Variable.Observed(0.5);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(xPoint, 0.0);
            x.Name = nameof(x);
            x.AddAttribute(QueryTypes.MarginalDividedByPrior);
            Variable<double> f = Variable.Exp(x);
            Variable.ConstrainEqualRandom(f, Gamma.FromNatural(0, -1));
            InferenceEngine engine = new InferenceEngine();
            var xPost = engine.Infer<Gaussian>(x, QueryTypes.MarginalDividedByPrior);
            xPost.GetDerivatives(xPoint.ObservedValue, out double derivative, out _);
            Assert.Equal(System.Math.Exp(xPoint.ObservedValue), derivative);
        }

        /// <summary>
        /// A simplified exmaple of the popular TrueSkill model.
        /// The data describes a competition of series of two-player games.
        /// Based on the outcome of the games, we want to infer the skills of the players.
        /// </summary>
        [Fact]
        public void TrueSkillMlNetGettingStartedTest()
        {
            // The winner and loser in each of 6 samples games
            var winnerData = new[] { 0, 0, 0, 1, 3, 4 };
            var loserData = new[] { 1, 3, 4, 2, 1, 2 };

            // Define the statistical model as a probabilistic program
            var game = new Range(winnerData.Length);
            var player = new Range(winnerData.Concat(loserData).Max() + 1);
            var playerSkills = Variable.Array<double>(player);
            playerSkills[player] = Variable.GaussianFromMeanAndVariance(6, 9).ForEach(player);
            var winners = Variable.Array<int>(game);
            var losers = Variable.Array<int>(game);
            using (Variable.ForEach(game))
            {
                // The player performance is a noisy version of their skill
                var winnerPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[winners[game]], 1.0);
                var loserPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[losers[game]], 1.0);

                // The winner performed better in this game
                Variable.ConstrainTrue(winnerPerformance > loserPerformance);
            }

            // Attach the data to the model
            winners.ObservedValue = winnerData;
            losers.ObservedValue = loserData;

            // Run inference
            var inferenceEngine = new InferenceEngine();
            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(playerSkills);

            // The inferred skills are uncertain, which is captured in their variance
            var orderedPlayerSkills = inferredSkills
                .Select((s, i) => new { Player = i, Skill = s })
                .OrderByDescending(ps => ps.Skill.GetMean());
            foreach (var playerSkill in orderedPlayerSkills)
            {
                Console.WriteLine($"Player {playerSkill.Player} skill: {playerSkill.Skill}");
            }

            Assert.True(inferredSkills[3].GetMean() > inferredSkills[4].GetMean());
        }
    }
}