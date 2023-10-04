// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Runtime.Serialization.Formatters.Binary;

    using Xunit;

    using Newtonsoft.Json;
    using Newtonsoft.Json.Serialization;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Distributions.Kernels;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Serialization;

    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    public class SerializableTests
    {
        // How to serialize derived classes: http://www.codeproject.com/Articles/8644/XmlSerializer-and-not-expected-Inherited-Types
        [Fact]
        public void DataContractAndDataContractJsonSerializersVectorTest()
        {
            for (var vectorType = 0; vectorType < 4; vectorType++)
            {
                Vector v;
                if (vectorType == 0) v = Vector.FromArray(3.5, 4.3);
                else
                {
                    var sparsity = Sparsity.Sparse;
                    if (vectorType == 1) sparsity = Sparsity.Sparse;
                    else if (vectorType == 2) sparsity = Sparsity.ApproximateWithTolerance(0.1);
                    else if (vectorType == 3) sparsity = Sparsity.Piecewise;
                    v = Vector.Constant(3, 1.4, sparsity);
                    v[1] = 2.5;
                }

                var v2 = CloneDataContract(v);
                Assert.Equal(0, v.MaxDiff(v2), 1e-10);
            }
        }

        [Fact]
        public void DataContractSerializerMatrixTest()
        {
            for (int matrixType = 0; matrixType < 2; matrixType++)
            {
                MemoryStream stream = new MemoryStream();
                //FileStream stream = new FileStream("temp.txt", FileMode.Create);
                Matrix m;
                if (matrixType == 0) m = new Matrix(new double[,] { { 2, 3, 4 }, { 5, 6, 7 } });
                else
                {
                    m = new PositiveDefiniteMatrix(new double[,] { { 2, 1 }, { 1, 2 } });
                }

                var m2 = CloneDataContract(m);
                Assert.True(m.MaxDiff(m2) < 1e-10);
                stream.Close();
            }
        }

        [Fact]
        public void DataContractSerializerTest()
        {
            var mc = new MyClass();
            mc.Initialize();

            var mc2 = CloneDataContract(mc);
            mc.AssertEqualTo(mc2);
        }

        [Fact]
        public void JsonNetSerializerTest()
        {
            var mc = new MyClass();
            mc.Initialize();

            var mc2 = CloneJsonNet(mc);
            mc.AssertEqualTo(mc2);
        }

        [Fact]
        public void VectorSerializeTests()
        {
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            double[] fromArray = new double[] {1.2, 2.3, 3.4, 1.2, 1.2, 2.3};
            Vector vdense = Vector.FromArray(fromArray);
            Vector vsparse = Vector.FromArray(fromArray, Sparsity.Sparse);
            Vector vapprox = Vector.FromArray(fromArray, approxSparsity);
            MemoryStream vdenseStream = new MemoryStream();
            MemoryStream vsparseStream = new MemoryStream();
            MemoryStream vapproxStream = new MemoryStream();
            var serializer = new DataContractSerializer(typeof(Vector), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });
            serializer.WriteObject(vdenseStream, vdense);
            serializer.WriteObject(vsparseStream, vsparse);
            serializer.WriteObject(vapproxStream, vapprox);

            vdenseStream.Position = 0;
            vsparseStream.Position = 0;
            vapproxStream.Position = 0;
            Vector vdense2 = (Vector)serializer.ReadObject(vdenseStream);
            SparseVector vsparse2 = (SparseVector)serializer.ReadObject(vsparseStream);
            ApproximateSparseVector vapprox2 = (ApproximateSparseVector)serializer.ReadObject(vapproxStream);

            Assert.Equal(6, vdense2.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(fromArray[i], vdense2[i]);
            Assert.Equal(vdense2.Sparsity, Sparsity.Dense);

            Assert.Equal(6, vsparse2.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(vsparse2[i], fromArray[i]);
            Assert.Equal(vsparse2.Sparsity, Sparsity.Sparse);
            Assert.Equal(1.2, vsparse2.CommonValue);
            Assert.Equal(3, vsparse2.SparseValues.Count);
            Assert.True(vsparse2.HasCommonElements);

            Assert.Equal(6, vapprox2.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(vapprox2[i], fromArray[i]);
            Assert.Equal(vapprox2.Sparsity, approxSparsity);
            Assert.Equal(1.2, vapprox2.CommonValue);
            Assert.Equal(3, vapprox2.SparseValues.Count);
            Assert.True(vapprox2.HasCommonElements);
        }

        [DataContract]
        [Serializable]
        public class MyClass
        {
            [DataMember] private Bernoulli bernoulli;
            [DataMember] private Beta beta;
            [DataMember] private Binomial binomial;
            [DataMember] private ConjugateDirichlet conjugateDirichlet;
            [DataMember] private Dirichlet dirichlet;
            [DataMember] private Discrete discrete;
            [DataMember] private Gamma gamma;
            [DataMember] private GammaPower gammaPower;
            [DataMember] private Gaussian gaussian;
            [DataMember] private NonconjugateGaussian nonconjugateGaussian;
            [DataMember] private PointMass<double> pointMass;
            [DataMember] private SparseBernoulliList sparseBernoulliList;
            [DataMember] private SparseBetaList sparseBetaList;
            [DataMember] private SparseGaussianList sparseGaussianList;
            [DataMember] private SparseGammaList sparseGammaList;
            [DataMember] private TruncatedGamma truncatedGamma;
            [DataMember] private TruncatedGaussian truncatedGaussian;
            [DataMember] private UnnormalizedDiscrete unnormalizedDiscrete;
            [DataMember] private VectorGaussian vectorGaussian;
            [DataMember] private Wishart wishart;
            [DataMember] private WrappedGaussian wrappedGaussian;
            [DataMember] private Pareto pareto;
            [DataMember] private Poisson poisson;
            [DataMember] private IDistribution<double[]> ga;
            [DataMember] private IDistribution<Vector[]> vga;
            [DataMember] private IDistribution<double[,]> ga2D;
            [DataMember] private IDistribution<Vector[,]> vga2D;
            [DataMember] private IDistribution<double[][]> gaJ;
            [DataMember] private IDistribution<Vector[][]> vgaJ;
            [DataMember] private SparseGP sparseGp;
            [DataMember] private QuantileEstimator quantileEstimator;
            [DataMember] private OuterQuantiles outerQuantiles;
            [DataMember] private InnerQuantiles innerQuantiles;
            [DataMember] private StringDistribution stringDistribution1;
            [DataMember] private StringDistribution stringDistribution2;

            public void Initialize(bool skipStringDistributions = false)
            {
                // DO NOT make this a constructor, because it makes the test not notice complete lack of serialization as an empty object is set up exactly as the thing
                // you are trying to deserialize.
                this.pareto = new Pareto(1.2, 3.5);
                this.poisson = new Poisson(2.3);
                this.wishart = new Wishart(20, new PositiveDefiniteMatrix(new double[,] { { 22, 21 }, { 21, 23 } }));
                this.vectorGaussian = new VectorGaussian(Vector.FromArray(13, 14), new PositiveDefiniteMatrix(new double[,] { { 16, 15 }, { 15, 17 } }));
                this.unnormalizedDiscrete = UnnormalizedDiscrete.FromLogProbs(DenseVector.FromArray(5.1, 5.2, 5.3));
                this.pointMass = PointMass<double>.Create(1.1);
                this.gaussian = new Gaussian(11.0, 12.0);
                this.nonconjugateGaussian = new NonconjugateGaussian(1.2, 2.3, 3.4, 4.5);
                this.gamma = new Gamma(9.0, 10.0);
                this.gammaPower = new GammaPower(5.6, 2.8, 3.4);
                this.discrete = new Discrete(6.0, 7.0, 8.0);
                this.conjugateDirichlet = new ConjugateDirichlet(1.2, 2.3, 3.4, 4.5);
                this.dirichlet = new Dirichlet(3.0, 4.0, 5.0);
                this.beta = new Beta(2.0, 1.0);
                this.binomial = new Binomial(5, 0.8);
                this.bernoulli = new Bernoulli(0.6);

                this.sparseBernoulliList = SparseBernoulliList.Constant(4, new Bernoulli(0.1));
                this.sparseBernoulliList[1] = new Bernoulli(0.9);
                this.sparseBernoulliList[3] = new Bernoulli(0.7);

                this.sparseBetaList = SparseBetaList.Constant(5, new Beta(2.0, 2.0));
                this.sparseBetaList[0] = new Beta(3.0, 4.0);
                this.sparseBetaList[1] = new Beta(5.0, 6.0);

                this.sparseGaussianList = SparseGaussianList.Constant(6, Gaussian.FromMeanAndPrecision(0.1, 0.2));
                this.sparseGaussianList[4] = Gaussian.FromMeanAndPrecision(0.3, 0.4);
                this.sparseGaussianList[5] = Gaussian.FromMeanAndPrecision(0.5, 0.6);

                this.sparseGammaList = SparseGammaList.Constant(1, Gamma.FromShapeAndRate(1.0, 2.0));

                this.truncatedGamma = new TruncatedGamma(1.2, 2.3, 3.4, 4.5);
                this.truncatedGaussian = new TruncatedGaussian(1.2, 3.4, 5.6, 7.8);
                this.wrappedGaussian = new WrappedGaussian(1.2, 2.3, 3.4);

                ga = Distribution<double>.Array(new[] { this.gaussian, this.gaussian });
                vga = Distribution<Vector>.Array(new[] { this.vectorGaussian, this.vectorGaussian });
                ga2D = Distribution<double>.Array(new[,] { { this.gaussian, this.gaussian }, { this.gaussian, this.gaussian } });
                vga2D = Distribution<Vector>.Array(new[,] { { this.vectorGaussian, this.vectorGaussian }, { this.vectorGaussian, this.vectorGaussian } });
                gaJ = Distribution<double>.Array(new[] { new[] { this.gaussian, this.gaussian }, new[] { this.gaussian, this.gaussian } });
                vgaJ = Distribution<Vector>.Array(new[] { new[] { this.vectorGaussian, this.vectorGaussian }, new[] { this.vectorGaussian, this.vectorGaussian } });

                var gp = new GaussianProcess(new ConstantFunction(0), new SquaredExponential(0));
                var basis = Util.ArrayInit(2, i => Vector.FromArray(1.0 * i));
                this.sparseGp = new SparseGP(new SparseGPFixed(gp, basis));

                this.quantileEstimator = new QuantileEstimator(0.01);
                this.quantileEstimator.Add(5);
                this.outerQuantiles = OuterQuantiles.FromDistribution(3, this.quantileEstimator);
                this.innerQuantiles = InnerQuantiles.FromDistribution(3, this.outerQuantiles);

                if (!skipStringDistributions)
                {
                    // String distributions can not be serialized by some formatters (namely BinaryFormatter)
                    // That is fine because this combination is never used in practice
                    this.stringDistribution1 = StringDistribution.String("aa")
                        .Append(StringDistribution.OneOf("b", "ccc")).Append("dddd");
                    this.stringDistribution2 = new StringDistribution();
                    this.stringDistribution2.SetToProduct(StringDistribution.OneOf("a", "b"),
                        StringDistribution.OneOf("b", "c"));
                }
            }

            public void AssertEqualTo(MyClass that)
            {
                Assert.Equal(0, this.bernoulli.MaxDiff(that.bernoulli));
                Assert.Equal(0, this.beta.MaxDiff(that.beta));
                Assert.Equal(0, this.binomial.MaxDiff(that.binomial));
                Assert.Equal(0, this.conjugateDirichlet.MaxDiff(that.conjugateDirichlet));
                Assert.Equal(0, this.dirichlet.MaxDiff(that.dirichlet));
                Assert.Equal(0, this.discrete.MaxDiff(that.discrete));
                Assert.Equal(0, this.gamma.MaxDiff(that.gamma));
                Assert.Equal(0, this.gammaPower.MaxDiff(that.gammaPower));
                Assert.Equal(0, this.gaussian.MaxDiff(that.gaussian));
                Assert.Equal(0, this.nonconjugateGaussian.MaxDiff(that.nonconjugateGaussian));
                Assert.Equal(0, this.pointMass.MaxDiff(that.pointMass));
                Assert.Equal(0, this.sparseBernoulliList.MaxDiff(that.sparseBernoulliList));
                Assert.Equal(0, this.sparseBetaList.MaxDiff(that.sparseBetaList));
                Assert.Equal(0, this.sparseGammaList.MaxDiff(that.sparseGammaList));
                Assert.Equal(0, this.truncatedGamma.MaxDiff(that.truncatedGamma));
                Assert.Equal(0, this.truncatedGaussian.MaxDiff(that.truncatedGaussian));
                Assert.Equal(0, this.wrappedGaussian.MaxDiff(that.wrappedGaussian));
                Assert.Equal(0, this.sparseGaussianList.MaxDiff(that.sparseGaussianList));
                Assert.Equal(0, this.unnormalizedDiscrete.MaxDiff(that.unnormalizedDiscrete));
                Assert.Equal(0, this.vectorGaussian.MaxDiff(that.vectorGaussian));
                Assert.Equal(0, this.wishart.MaxDiff(that.wishart));
                Assert.Equal(0, this.pareto.MaxDiff(that.pareto));
                Assert.Equal(0, this.poisson.MaxDiff(that.poisson));
                Assert.Equal(0, ga.MaxDiff(that.ga));
                Assert.Equal(0, vga.MaxDiff(that.vga));
                Assert.Equal(0, ga2D.MaxDiff(that.ga2D));
                Assert.Equal(0, vga2D.MaxDiff(that.vga2D));
                Assert.Equal(0, gaJ.MaxDiff(that.gaJ));
                Assert.Equal(0, vgaJ.MaxDiff(that.vgaJ));
                Assert.Equal(0, this.sparseGp.MaxDiff(that.sparseGp));
                Assert.True(this.quantileEstimator.ValueEquals(that.quantileEstimator));
                Assert.True(this.innerQuantiles.Equals(that.innerQuantiles));
                Assert.True(this.outerQuantiles.Equals(that.outerQuantiles));

                if (this.stringDistribution1 != null)
                {
                    Assert.Equal(0, this.stringDistribution1.MaxDiff(that.stringDistribution1));
                    Assert.Equal(0, this.stringDistribution2.MaxDiff(that.stringDistribution2));
                }
            }
        }

        private static T CloneDataContract<T>(T obj)
        {
            var ser = new DataContractSerializer(typeof(T), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });            
            using (var ms = new MemoryStream())
            {
                ser.WriteObject(ms, obj);
                ms.Position = 0;
                return (T)ser.ReadObject(ms);
            }
        }

        private static T CloneJsonNet<T>(T obj)
        {
            var serializerSettings = new JsonSerializerSettings
                                         {
                                             TypeNameHandling = TypeNameHandling.Auto,
                                             ContractResolver = new CollectionAsObjectResolver(),
                                             PreserveReferencesHandling = PreserveReferencesHandling.Objects
                                         };
            var serializer = JsonSerializer.Create(serializerSettings);

            using (var memoryStream = new MemoryStream())
            {
                var streamWriter = new StreamWriter(memoryStream);
                var jsonWriter = new JsonTextWriter(streamWriter);
                serializer.Serialize(jsonWriter, obj);
                jsonWriter.Flush();

                memoryStream.Position = 0;

                var streamReader = new StreamReader(memoryStream);
                var jsonReader = new JsonTextReader(streamReader);
                return serializer.Deserialize<T>(jsonReader);
            }
        }
    }

    /// <summary>
    /// Treats as objects distribution member types which implement <see cref="IList{T}"/>.
    /// </summary>
    public class CollectionAsObjectResolver : DefaultContractResolver
    {
        private static readonly HashSet<Type> SerializeAsObjectTypes = new HashSet<Type>
            {
                typeof(Vector),
                typeof(Matrix),
                typeof(IArray<>),
                typeof(ISparseList<>)
            };

        private static readonly ConcurrentDictionary<Type, JsonContract> ResolvedContracts = new ConcurrentDictionary<Type, JsonContract>();

        public override JsonContract ResolveContract(Type type) => ResolvedContracts.GetOrAdd(type, this.ResolveContractInternal);

        private JsonContract ResolveContractInternal(Type type) => IsExcludedType(type)
            ? this.CreateObjectContract(type)
            : this.CreateContract(type);

        private static bool IsExcludedType(Type type)
        {
            if (type == null) return false;
            if (SerializeAsObjectTypes.Contains(type)) return true;
            if (type.IsGenericType && SerializeAsObjectTypes.Contains(type.GetGenericTypeDefinition())) return true;
            return IsExcludedType(type.BaseType) || type.GetInterfaces().Any(IsExcludedType);
        }
    }
}