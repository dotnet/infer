// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests.Core
{
    using Microsoft.ML.Probabilistic.Algorithms;
    // put this here so that it overrides the System namespace instead of clashing
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Models.Attributes;
    using Microsoft.ML.Probabilistic.Utilities;
    using Assert = AssertHelper;

    // There is a fairly exhaustive set of tests for Vectors, SparseVectors
    // and ApproximateSparseVectors in MatrixTests

    public class SparseTests
    {
        [Fact]
        public void SparseVectorBasic()
        {
            var sv = SparseVector.Constant(10, 1);
            Console.WriteLine("sv = {0}, sum = {1}", sv, sv.Sum());
            Assert.Equal(10.0, sv.Sum());
            sv.SetToFunction(sv, System.Math.Log);
            Console.WriteLine("sv = {0}", sv);
            Assert.True(sv.EqualsAll(0.0));
        }

        [Fact]
        public void SparseVectorInPlace()
        {
            var sv1 = SparseVector.Constant(10, 1);
            var sv2 = SparseVector.Constant(10, 2);
            sv1.SetToSum(sv1, sv2);
            Console.WriteLine("sv1 = {0}, sv2 = {1}", sv1, sv2);
            sv2[5] = 0;
            sv1.SetToSum(sv1, sv2);
            Console.WriteLine("sv1 = {0}, sv2 = {1}", sv1, sv2);
        }

        [Fact]
        public void SparseVectorSum()
        {
            var sv1 = SparseVector.Zero(10);
            sv1[1] = 1;
            sv1[2] = 1;
            var sv2 = SparseVector.Constant(10, 2);
            sv2[1] = 1;
            sv2[3] = 1;
            var sv3 = sv1 + sv2;
            Console.WriteLine("sv1 = " + sv1);
            Console.WriteLine("sv2 = " + sv2);
            Console.WriteLine("sv3 = " + sv3);
        }

        [Fact]
        public void SparseVectorDifference()
        {
            var sv1 = SparseVector.Zero(10);
            sv1[1] = 1;
            sv1[2] = 1;
            Console.WriteLine("sv1: " + sv1);
            var sv2 = SparseVector.Constant(10, 2);
            sv2[1] = 1;
            sv2[3] = 1;
            Console.WriteLine("sv2: " + sv2);
            var sv3 = sv1 - sv2;
            Console.WriteLine("sv3: " + sv3);
            var sv4 = sv1 + (-sv2);
            Console.WriteLine("sv4: " + sv4);
            Assert.Equal(sv3, sv4);
        }

        [Fact]
        public void SparseBernoulliListTest()
        {
            // Sparse Bernoulli list
            var sbl = SparseBernoulliList.FromProbTrue(10, 0.1);
            sbl[3] = new Bernoulli(0.4);
            sbl[7] = new Bernoulli(0.8);

            // Non-sparse Bernoulli list
            Bernoulli[] b = new Bernoulli[sbl.Count];
            for (int i = 0; i < b.Length; i++) b[i] = new Bernoulli(0.1);
            b[3].SetProbTrue(0.4);
            b[7].SetProbTrue(0.8);

            // Observations
            int[] trueInds = {3, 5};
            SparseList<bool> obsSparse = SparseList<bool>.Constant(b.Length, false);
            foreach (int i in trueInds)
                obsSparse[i] = true;
            bool[] obs = obsSparse.ToArray();

            // Non-sparse log prob
            double logprob = 0;
            for (int i = 0; i < b.Length; i++) logprob += b[i].GetLogProb(obs[i]);

            Console.WriteLine("Non-sparse logprob=" + logprob);
            double logprob1 = sbl.GetLogProb(obsSparse);
            Console.WriteLine("Sparse logprob (bool[])=" + logprob1);
            //double logprob2 = ((CanGetLogProb<int[]>)sbl).GetLogProb(trueInds);
            //Console.WriteLine("Sparse logprob (int[])=" + logprob2);
            Assert.True(System.Math.Abs(logprob - logprob1) < 1E-10);
            //Assert.True(Math.Abs(logprob - logprob2) < 1E-10);
        }

        [Fact]
        public void SparseBetaListTest()
        {
            // Sparse Beta list
            var sbl = SparseBetaList.FromCounts(10, 2, 1);
            sbl[3] = new Beta(1, 1);
            sbl[7] = new Beta(2, 3);

            // Non-sparse Beta list
            Beta[] b = new Beta[sbl.Count];
            for (int i = 0; i < b.Length; i++) b[i] = new Beta(2, 1);
            b[3].TrueCount = 1;
            b[7].FalseCount = 3;

            // Compute mean
            var mean = Vector.Zero(b.Length);
            for (int i = 0; i < b.Length; i++) mean[i] = b[i].GetMean();
            Console.WriteLine("Non-sparse mean: " + mean);
            var mean2 = sbl.GetMean();
            Console.WriteLine("Sparse mean: " + mean2);

            Assert.True(mean2.ToVector().MaxDiff(mean) < 1E-10);

            // Compute log prob
            SparseVector obs = SparseVector.Constant(sbl.Count, 0.5);
            obs[2] = 0.3;
            obs[7] = 0.8;
            double logprob = 0;
            for (int i = 0; i < b.Length; i++) logprob += b[i].GetLogProb(obs[i]);
            Console.WriteLine("Non-sparse log prob: " + logprob);
            double logprob2 = sbl.GetLogProb(obs);
            Console.WriteLine("Sparse log prob: " + logprob2);

            Assert.True(MMath.AbsDiff(logprob, logprob2) < 1E-10);
        }

        [Fact]
        public void ArrayOfSparseListTest()
        {
            int count = 3;
            var sbl = SparseBetaList.FromSparseValues(count, Beta.Uniform(),
                new List<ValueAtIndex<Beta>>() { new ValueAtIndex<Beta>(0, new Beta(1, 2)), new ValueAtIndex<Beta>(2, new Beta(2, 1)) });
            Range outer = new Range(3);
            Range inner = new Range(2);
            var array = Variable.Array(Variable.ISparseList<double>(inner), outer).Named("array");
            array[outer].SetTo(Variable<ISparseList<double>>.Random(sbl).ForEach(outer));

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(array));
        }

        [Fact]
        public void BernoulliIntegerSubsetTest()
        {
            int count = 3;
            var sbl = SparseBetaList.FromSparseValues(count, Beta.Uniform(),
                new List<ValueAtIndex<Beta>>() { new ValueAtIndex<Beta>(0, new Beta(1, 2)), 
                                                 new ValueAtIndex<Beta>(2, new Beta(2, 1)) });
            Range item = new Range(count);
            var probs = new VariableArray<Variable<double>, ISparseList<double>>(new Variable<double>(), item).Named("probs");
            probs.SetTo(Variable<ISparseList<double>>.Random(sbl));
            //var subset = Variable.BernoulliIntegerSubset(probs).Named("subset");
            var subset = Variable.IList<int>(item).Named("subset");
            subset.SetTo(Variable.BernoulliIntegerSubset(probs));

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(subset));
            Console.WriteLine(engine.Infer(probs));
        }

        [Fact]
        public void SparseBernoulliBetaTest()
        {
            SparseBernoulliBeta(new InferenceEngine(new ExpectationPropagation())
            {
                ShowProgress = false
            });

            SparseBernoulliBeta(new InferenceEngine(new VariationalMessagePassing())
            {
                ShowProgress = false
            });
        }

        private void SparseBernoulliBeta(InferenceEngine engine)
        {
            ISparseList<bool>[] data = new ISparseList<bool>[] {
                SparseList<bool>.Constant(5, false),
                SparseList<bool>.Constant(5, false)
            };

            data[0][0] = true;
            data[0][1] = true;
            data[1][1] = true;
            data[1][2] = true;

            // Sparse model
            var pPrior = Variable.Observed(SparseBetaList.FromSize(data[0].Count));
            var p = Variable<ISparseList<double>>.Random(pPrior).Named("p");
            var x = Variable.Observed(data);
            var N = x.Range;
            x[N] = Variable.BernoulliList(p).ForEach(N);

            var pdist = engine.Infer<SparseBetaList>(p);
            var mean1 = pdist.GetMean().ToVector();
            Console.WriteLine("Dist over p: " + Environment.NewLine + pdist);
            Console.WriteLine("Mean of p: " + mean1);

            // Non-sparse model
            Range K = new Range(data[0].Count);
            var p2 = Variable.Array<double>(K);
            p2[K] = Variable.Beta(1, 1).ForEach(K);
            var x2 = Variable.Observed(new bool[][] {data[0].ToArray(), data[1].ToArray()}, N, K);
            x2[N][K] = Variable.Bernoulli(p2[K]).ForEach(N);
            var pdist2 = engine.Infer<Beta[]>(p2);
            var mean2 = Vector.Zero(pdist2.Length);
            for (int i = 0; i < mean2.Count; i++) mean2[i] = pdist2[i].GetMean();
            Console.WriteLine("Dist over p2: " + Environment.NewLine + StringUtil.CollectionToString(pdist2, ","));
            Console.WriteLine("Mean of p2: " + mean2);

            // Compare means
            Assert.True(mean1.MaxDiff(mean2) < 1E-10);

            var y = Variable.BernoulliList(p).Named("y");
            var ydist = engine.Infer<SparseBernoulliList>(y);
            Console.WriteLine("Dist over y: " + ydist);
        }

        [Fact]
        public void SparseBernoulliBetaMixtureTest()
        {
            //sparseBernoulliBetaMixtureTest(new InferenceEngine(new ExpectationPropagation())
            //{
            //    ShowProgress = false
            //});

            SparseBernoulliBetaMixture(new InferenceEngine(new VariationalMessagePassing())
            {
                ShowProgress = false
            });
        }

        private void SparseBernoulliBetaMixture(InferenceEngine engine)
        {
            ISparseList<bool>[] obs = Util.ArrayInit(9, i => SparseList<bool>.Constant(10, false));
            obs[0][0] = true;
            obs[0][1] = true;
            obs[1][0] = true;
            obs[1][1] = true;
            obs[2][0] = true;
            obs[2][1] = true;
            obs[3][1] = true;
            obs[3][2] = true;
            obs[3][3] = true;
            obs[4][1] = true;
            obs[4][2] = true;
            obs[4][3] = true;
            obs[5][1] = true;
            obs[5][2] = true;
            obs[5][3] = true;
            obs[6][0] = true;
            obs[6][1] = true;
            obs[7][0] = true;
            obs[7][1] = true;

            SparseBetaList groupPrior = SparseBetaList.FromSize(obs[0].Count);
            var init = new SparseBetaList[] { SparseBetaList.FromSize(obs[0].Count), SparseBetaList.FromSize(obs[0].Count) };
            init[0][0] = new Beta(2, init[0][0].FalseCount);
            SparseBernoulliBetaMixture(obs, Dirichlet.Uniform(2), new SparseBetaList[] {groupPrior, groupPrior}, init, engine);
        }

        private void SparseBernoulliBetaMixture(
            ISparseList<bool>[] data, 
            Dirichlet groupPrior, 
            SparseBetaList[] contactInGroupPrior, 
            SparseBetaList[] init,
            InferenceEngine engine)
        {
            Range C = new Range(data[0].Count).Named("C");
            Range G = new Range(groupPrior.Dimension).Named("G");
            Range N = new Range(data.Length).Named("N");

            var ProbGroupPrior = Variable.Observed(groupPrior).Named("ProbGroupPrior");
            var ProbGroup = Variable.Random<Vector, Dirichlet>(ProbGroupPrior).Named("ProbGroup");
            var SelectedGroup = Variable.Array<int>(N).Named("SelectedGroup");
            var ProbContactInGroupPrior = Variable.Observed(contactInGroupPrior, G).Named("pContactInGrpPrior");

            var ProbContactInGroup = Variable.Array<ISparseList<double>>(G).Named("pContactInGrp");
            ProbContactInGroup.InitialiseTo(Distribution<ISparseList<double>>.Array(init));
            ProbContactInGroup[G] = Variable.Random<ISparseList<double>, SparseBetaList>(ProbContactInGroupPrior[G]);
            var IntendedRecipients = Variable.Array<ISparseList<bool>>(N).Named("intendedRecipients");
            using (Variable.ForEach(N))
            {
                SelectedGroup[N] = Variable.Discrete(G, ProbGroup);
                using (Variable.Switch(SelectedGroup[N]))
                {
                    IntendedRecipients[N] = Variable.BernoulliList(ProbContactInGroup[SelectedGroup[N]]);
                }
            }
            IntendedRecipients.ObservedValue = data;
            IntendedRecipients.AddAttribute(new MarginalPrototype(SparseBernoulliList.FromSize(C.SizeAsInt)));

            engine.NumberOfIterations = 1;
            var probGroupExpected = engine.Infer(ProbGroup);
            Console.WriteLine(probGroupExpected);

            engine.NumberOfIterations = 10;
            var pdist = engine.Infer<SparseBetaList[]>(ProbContactInGroup);
            for (int i = 0; i < pdist.Length; i++)
            {
                Console.WriteLine("Dist over p: " + Environment.NewLine + pdist[i]);
                Console.WriteLine("Mean of p: " + pdist[i].GetMean());
            }

            // test resetting inference
            engine.NumberOfIterations = 1;
            var probGroupActual = engine.Infer<Diffable>(ProbGroup);
            Assert.True(probGroupActual.MaxDiff(probGroupExpected) < 1e-10);
        }

        private void SparseListEquality(bool useISparseList)
        {
            var dense = new bool[] {false, false, false, true, true, false, false, false, false, false};
            var sl = SparseList<bool>.Constant(10, false);
            sl[3] = true;
            sl[4] = true;
            Console.WriteLine(sl);
            Assert.True(sl.Equals(dense));
            var dense2 = new bool[] {false, false, false, false, true, true, false, false, false, false};
            var sl2 = SparseList<bool>.Constant(10, false);
            sl2[4] = true;
            sl2[5] = true;
            Console.WriteLine(sl2);
            Assert.True(sl2.Equals(dense2));
            var sl3 = SparseList<bool>.Constant(10, false);
            if (useISparseList)
                sl3.SetToFunction((ISparseList<bool>) sl, (ISparseList<bool>) sl2, (a, b) => a | b);
            else
                sl3.SetToFunction(sl, sl2, (a, b) => a | b);
            var answer = new bool[] {false, false, false, true, true, true, false, false, false, false};
            Assert.True(sl3.Equals(answer));
        }

        [Fact]
        public void SparseListEqualityTest()
        {
            SparseListEquality(false);
        }

        [Fact]
        public void ISparseListEqualityTest()
        {
            SparseListEquality(true);
        }

        [Fact]
        public void SparseListEqualityTest2()
        {
            var dense1 = new bool[] {false, false, false, true, true, false, false, false, false, false};
            var sl1 = SparseList<bool>.Constant(10, false);
            sl1[3] = true;
            sl1[4] = true;
            Console.WriteLine(sl1);
            Assert.True(sl1.Equals(dense1));

            var dense2 = new bool[] {false, false, false, false, true, false, true, false, false, false};
            var sl2 = SparseList<bool>.Constant(10, false);
            sl2[4] = true;
            sl2[6] = true;
            Console.WriteLine(sl2);
            Assert.True(sl2.Equals(dense2));

            var dense3 = new bool[] {false, true, false, false, false, false, true, false, false, false};
            var sl3 = SparseList<bool>.Constant(10, false);
            sl3[1] = true;
            sl3[6] = true;
            Console.WriteLine(sl3);
            Assert.True(sl3.Equals(dense3));


            var sparse = SparseList<bool>.Constant(10, false);
            sparse.SetToFunction(sl1, sl2, sl3, (a, b, c) => a | b | c);
            var answer = new bool[] {false, true, false, true, true, false, true, false, false, false};
            Assert.True(sparse.Equals(answer));
        }

        [Fact]
        public void SparseListContainsTest()
        {
            var sl = SparseList<int>.Constant(10, 0);
            sl[3] = 4;
            sl[4] = 5;
            Console.WriteLine(sl);
            Assert.Contains(0, sl);
            Assert.Contains(4, sl);
            Assert.True(sl.IndexOf(0) == 0);
            Assert.True(sl.IndexOf(5) == 4);
            Assert.True(sl.All(i => i < 7));
            Assert.True(!sl.All(i => i > 4));
            Assert.True(sl.Any(i => i > 4));
        }

        [Fact]
        public void SparseMultinomial()
        {
            var sl = SparseList<int>.FromSize(10);
            sl[4] = 10;
            sl[5] = 4;
            var probs = Variable.DirichletUniform(10);
            int sum = sl.ListSum();
            Assert.Equal(sum, sl.Sum());
            var counts = Variable<IList<int>>.Factor(Factor.MultinomialList, sum, probs);
            counts.ObservedValue = sl;

            var ie = new InferenceEngine {};
            var result = ie.Infer<Dirichlet>(probs);
            var dl = sl.ListSelect(x => x + 1.0).ToVector();
            var diff = result.PseudoCount - dl;
            Console.WriteLine("result=" + result);
            Console.WriteLine("dl=" + dl);
            Console.WriteLine("diff=" + diff);
            Assert.True(diff.EqualsAll(0.0));
        }

        private void SparseListReduce(SparseList<double> thisList, ISparseList<double> b, ISparseList<int> c)
        {
            double delta = 1e-9;
            double expected = 0.0;
            double actual = 0.0;
            double initial;
            // Single argument sum
            initial = 1.0;
            expected = initial;
            for (int i = 0; i < thisList.Count; i++)
                expected += thisList[i];

            actual = thisList.Reduce(initial, (x, y) => x + y, (x, y, count) => x + count*y);
            Assert.Equal(expected, actual, delta);

            // Single argument max
            initial = double.MinValue;
            expected = initial;
            for (int i = 0; i < thisList.Count; i++)
                expected = System.Math.Max(expected, thisList[i]);

            actual = thisList.Reduce(initial, (x, y) => System.Math.Max(x, y), (x, y, count) => System.Math.Max(x, y));
            Assert.Equal(expected, actual, delta);

            // Double argument sum of product
            initial = 1.0;
            expected = initial;
            for (int i = 0; i < thisList.Count; i++)
                expected += thisList[i]*b[i];
            actual = thisList.Reduce(initial, b, (x, y, z) => x + y*z, (x, y, z, count) => x + count*y*z);
            Assert.Equal(expected, actual, delta);

            // Double argument max of product
            initial = double.MinValue;
            expected = initial;
            for (int i = 0; i < thisList.Count; i++)
                expected = System.Math.Max(expected, thisList[i] * b[i]);
            actual = thisList.Reduce(initial, b, (x, y, z) => System.Math.Max(x, y* z), (x, y, z, count) => System.Math.Max(x, y* z));
            Assert.Equal(expected, actual, delta);

            // Triple argument - sum of product to a power
            initial = 1.0;
            expected = initial;
            for (int i = 0; i < thisList.Count; i++)
                expected += System.Math.Pow(thisList[i] * b[i], c[i]);
            actual = thisList.Reduce(initial, b, c, (w, x, y, z) => w + System.Math.Pow(x * y, z), (w, x, y, z, count) => w + count* System.Math.Pow(x * y, z));
            Assert.Equal(expected, actual, delta);

            // Triple argument - max of product to a power
            initial = 1.0;
            expected = initial;
            for (int i = 0; i < thisList.Count; i++)
                expected = System.Math.Max(expected, System.Math.Pow(thisList[i] * b[i], c[i]));
            actual = thisList.Reduce(initial, b, c, (w, x, y, z) => System.Math.Max(w, System.Math.Pow(x * y, z)), (w, x, y, z, count) => System.Math.Max(w, System.Math.Pow(x * y, z)));
            Assert.Equal(expected, actual, delta);
        }

        [Fact]
        public void SparseListReduceTest()
        {
            Rand.Restart(12347);
            int len = 200;
            for (int i = 0; i < 100; i++)
            {
                // Create random sparse vectors for the reduce operations
                // Random numbers of non-common values are in random positions. 
                double aCom = 2.0*Rand.Double() - 1.0;
                double bCom = 2.0*Rand.Double() - 1.0;
                int cCom = Rand.Int(0, 3);
                var a = SparseList<double>.Constant(len, aCom);
                var b = SparseList<double>.Constant(len, bCom);
                var c = SparseList<int>.Constant(len, cCom);

                int numA = Rand.Poisson(20);
                int numB = Rand.Poisson(20);
                int numC = Rand.Poisson(20);

                int[] permA = Rand.Perm(len);
                int[] permB = Rand.Perm(len);
                int[] permC = Rand.Perm(len);

                for (int j = 0; j < numA; j++)
                {
                    double aval = 2.0*Rand.Double() - 1.0;
                    a[permA[j]] = aval;
                }
                for (int j = 0; j < numB; j++)
                {
                    double bval = 2.0*Rand.Double() - 1.0;
                    b[permB[j]] = bval;
                }
                for (int j = 0; j < numC; j++)
                {
                    int cval = Rand.Int(0, 3);
                    c[permC[j]] = cval;
                }
                SparseListReduce(a, b, c);
            }
        }

        [Fact]
        public void ThreeWayMapTest()
        {
            var a = SparseList<double>.FromSize(10);
            var b = SparseList<double>.FromSize(10);
            var c = SparseList<double>.FromSize(10);
            a[1] = 3.4;
            a[4] = 1.2;
            b[4] = 2.1;
            b[5] = 4.2;
            c[0] = .1;
            c[5] = 1.1;
            c[8] = 2;
            Func<double, double, double, double> f = (x, y, z) => x + y + z;
            var ie = f.Map(a, b, c) as ISparseEnumerable<double>;
            var ise = ie.GetSparseEnumerator();
            int nonZero = 0;
            while (ise.MoveNext())
            {
                nonZero++;
                Assert.True(f(a[ise.CurrentIndex], b[ise.CurrentIndex], c[ise.CurrentIndex]) == ise.Current);
                Console.WriteLine(ise.CurrentIndex + " : " + ise.Current);
            }
            Assert.True(ise.CommonValueCount == (10 - nonZero));
        }

        private void SparseSetToFuncEnumerable(ISparseList<double> a, ISparseList<double> b, ISparseList<int> c, ISparseList<int> d)
        {
            //Func<double,double,double> f2 = Math.Min;
            //f2.Map(a, b);
            //Func<double, double, int, double> f = (x, y, z) => x + y + z;
            //f.Map(a, b, c);

            SparseList<double> result = SparseList<double>.FromSize(a.Count);
            double[] expected = new double[a.Count];
            SparseList<double> aCopy;

            // Function of one list
            Func<double, double> f = x => System.Math.Pow(x, 2.0);
            for (int i = 0; i < a.Count; i++)
                expected[i] = f(a[i]);

            result.SetTo(f.Map(a));
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], result[i]);

            aCopy = SparseList<double>.Copy(a);
            aCopy.SetTo(f.Map(aCopy));
            Assert.True(expected.Length == aCopy.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], aCopy[i]);

            // Function of two lists.
            Func<double, double, double> g = (x, y) => x*y;
            for (int i = 0; i < a.Count; i++)
                expected[i] = g(a[i], b[i]);

            result.SetTo(g.Map(a, b));
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], result[i]);

            aCopy = SparseList<double>.Copy(a);
            aCopy.SetTo(g.Map(aCopy, b));
            Assert.True(expected.Length == aCopy.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], aCopy[i]);

            // Function of three lists.
            Func<double, double, int, double> h = (x, y, z) => z > 0 ? x + y : x - y;
            for (int i = 0; i < a.Count; i++)
                expected[i] = h(a[i], b[i], c[i]);

            result.SetTo(h.Map(a, b, c));
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], result[i]);

            aCopy = SparseList<double>.Copy(a);
            aCopy.SetTo(h.Map(aCopy, b, c));
            Assert.True(expected.Length == aCopy.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], aCopy[i]);

            // Function of four lists.
            Func<double, double, int, int, double> l = (w, x, y, z) => y > z ? w - x : w + x;
            for (int i = 0; i < a.Count; i++)
                expected[i] = l(a[i], b[i], c[i], d[i]);

            result.SetTo(l.Map(a, b, c, d));
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], result[i]);

            aCopy = SparseList<double>.Copy(a);
            aCopy.SetTo(l.Map(aCopy, b, c, d));
            Assert.True(expected.Length == aCopy.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], aCopy[i]);

            // Function of five lists.
            Func<double, double, double, int, int, double> m = (w, x, x2, y, z) => y > z ? w*x2 - x : w + x*x2;
            for (int i = 0; i < a.Count; i++)
                expected[i] = m(a[i], b[i], b[i], c[i], d[i]);

            result.SetTo(m.Map(a, b, b, c, d));
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], result[i]);

            aCopy = SparseList<double>.Copy(a);
            aCopy.SetTo(m.Map(aCopy, b, b, c, d));
            Assert.True(expected.Length == aCopy.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], aCopy[i]);
        }

        private void SparseSetToFunction(ISparseList<double> a, ISparseList<double> b, ISparseList<int> c, ISparseList<int> d)
        {
            //Func<double,double,double> f2 = Math.Min;
            //f2.Map(a, b);
            //Func<double, double, int, double> f = (x, y, z) => x + y + z;
            //f.Map(a, b, c);

            SparseList<double> result = SparseList<double>.FromSize(a.Count);
            double[] expected = new double[a.Count];

            // Function of one list
            for (int i = 0; i < a.Count; i++)
                expected[i] = System.Math.Pow(a[i], 2.0);

            result.SetToFunction(a, x => System.Math.Pow(x, 2.0));
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], result[i]);

            // Function of two lists.
            for (int i = 0; i < a.Count; i++)
                expected[i] = a[i]*b[i];

            result.SetToFunction(a, b, (x, y) => x*y);
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], result[i]);

            // Function of three lists.
            for (int i = 0; i < a.Count; i++)
                expected[i] = c[i] > 0 ? a[i] + b[i] : a[i] - b[i];

            result.SetToFunction(a, b, c, (x, y, z) => z > 0 ? x + y : x - y);
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], result[i]);

            // Function of four lists.
            for (int i = 0; i < a.Count; i++)
                expected[i] = c[i] > d[i] ? a[i] + b[i] : a[i] - b[i];

            result.SetToFunction(a, b, c, d, (x, y, z, u) => z > u ? x + y : x - y);
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], result[i]);

            // ----------
            // In place
            // ----------

            // Function of one list
            for (int i = 0; i < a.Count; i++)
                expected[i] = System.Math.Pow(a[i], 2.0);

            var aCopy = SparseList<double>.Copy(a);
            aCopy.SetToFunction(aCopy, x => System.Math.Pow(x, 2.0));
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], aCopy[i]);

            // Function of two lists.
            for (int i = 0; i < a.Count; i++)
                expected[i] = a[i]*b[i];

            aCopy = SparseList<double>.Copy(a);
            aCopy.SetToFunction(aCopy, b, (x, y) => x*y);
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], aCopy[i]);

            // Function of three lists.
            for (int i = 0; i < a.Count; i++)
                expected[i] = c[i] > 0 ? a[i] + b[i] : a[i] - b[i];

            aCopy = SparseList<double>.Copy(a);
            aCopy.SetToFunction(aCopy, b, c, (x, y, z) => z > 0 ? x + y : x - y);
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], aCopy[i]);

            // Function of four lists.
            for (int i = 0; i < a.Count; i++)
                expected[i] = c[i] > d[i] ? a[i] + b[i] : a[i] - b[i];

            aCopy = SparseList<double>.Copy(a);
            aCopy.SetToFunction(aCopy, b, c, d, (x, y, z, u) => z > u ? x + y : x - y);
            Assert.True(expected.Length == result.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], aCopy[i]);
        }

        [Fact]
        public void SparseSetToFuncEnumerableTests()
        {
            SparseFunctionTestHelper(SparseSetToFuncEnumerable);
        }

        [Fact]
        public void SparseSetToFunctionTest()
        {
            SparseFunctionTestHelper(SparseSetToFunction);
        }

        private delegate void testFunction(ISparseList<double> a, ISparseList<double> b, ISparseList<int> c, ISparseList<int> d);

        private void SparseFunctionTestHelper(testFunction f)
        {
            Rand.Restart(12347);
            int len = 200;
            for (int i = 0; i < 100; i++)
            {
                // Create random sparse vectors for the reduce operations
                // Random numbers of non-common values are in random positions. 
                double aCom = 2.0*Rand.Double() - 1.0;
                double bCom = 2.0*Rand.Double() - 1.0;
                int cCom = Rand.Int(0, 3);
                int dCom = Rand.Int(-3, 3);
                var a = SparseList<double>.Constant(len, aCom);
                var b = SparseList<double>.Constant(len, bCom);
                var c = SparseList<int>.Constant(len, cCom);
                var d = SparseList<int>.Constant(len, cCom);

                int numA = Rand.Poisson(20);
                int numB = Rand.Poisson(20);
                int numC = Rand.Poisson(20);
                int numD = Rand.Poisson(20);

                int[] permA = Rand.Perm(len);
                int[] permB = Rand.Perm(len);
                int[] permC = Rand.Perm(len);
                int[] permD = Rand.Perm(len);

                for (int j = 0; j < numA; j++)
                {
                    double aval = 2.0*Rand.Double() - 1.0;
                    a[permA[j]] = aval;
                }
                for (int j = 0; j < numB; j++)
                {
                    double bval = 2.0*Rand.Double() - 1.0;
                    b[permB[j]] = bval;
                }
                for (int j = 0; j < numC; j++)
                {
                    int cval = Rand.Int(0, 3);
                    c[permC[j]] = cval;
                }
                for (int j = 0; j < numD; j++)
                {
                    int dval = Rand.Int(-3, 3);
                    d[permD[j]] = dval;
                }
                f(a, b, c, d);
            }
        }

        [Fact]
        public void ListSetToEnumerableGaussianTests()
        {
            Gaussian[] thisArray = new Gaussian[] {Gaussian.FromMeanAndVariance(1.2, 1), Gaussian.FromMeanAndVariance(2.3, 1)};
            Gaussian[] thatArray = new Gaussian[] {Gaussian.FromMeanAndVariance(5.6, 1), Gaussian.FromMeanAndVariance(1.2, 1)};
            Gaussian[] thisExtendedArray = new Gaussian[] {Gaussian.FromMeanAndVariance(5.6, 1), Gaussian.FromMeanAndVariance(1.2, 1), Gaussian.FromMeanAndVariance(2.3, 1)};
            Gaussian[] thatExtendedArray = new Gaussian[] {Gaussian.FromMeanAndVariance(5.6, 1), Gaussian.FromMeanAndVariance(5.6, 1), Gaussian.FromMeanAndVariance(1.2, 1)};
            var thisList = new List<IList<Gaussian>>();
            var that = new List<IEnumerable<Gaussian>>();
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            thisList.Add(thisArray.ToList());
            thisList.Add(SparseList<Gaussian>.Copy(thisArray.ToList()));
            thisList.Add(thisExtendedArray.ToList());
            thisList.Add(SparseList<Gaussian>.Copy(thisExtendedArray.ToList()));
            var sparseValues1 = thisArray.Select((x, i) => new ValueAtIndex<Gaussian>(i, x)).ToList();
            thisList.Add(ApproximateSparseList<Gaussian>.FromSparseValues(thisArray.Length, Gaussian.Uniform(), sparseValues1));

            that.Add(thatArray);
            that.Add(thatArray.ToList());
            var sparseValues = thatArray.Select((x, i) => new ValueAtIndex<Gaussian>(i, x)).ToList();
            that.Add(SparseList<Gaussian>.FromSparseValues(thatArray.Length, Gaussian.Uniform(), sparseValues));

            for (int i = 0; i < thisList.Count; i++)
                for (int j = 0; j < that.Count; j++)
                {
                    ListSetToEnumerableGaussian(thisList[i], that[j]);
                }
        }

        private void ListSetToEnumerableGaussian(IList<Gaussian> thisList, IEnumerable<Gaussian> that)
        {
            thisList.SetTo(that);
            var thatList = that.ToList();
            Assert.True(thisList.Count == thatList.Count);
            for (int i = 0; i < thisList.Count; i++)
            {
                Assert.True(thisList[i] == thatList[i]);
            }
        }


        [Fact]
        public void ListSetToEnumerableTests()
        {
            double[] thisArray = new double[] {1.2, 2.3, 3.4, 1.2, 1.2, 2.3};
            double[] thatArray = new double[] {5.6, 1.2, 1.2, 6.7, 7.8, 1.2};
            double[] thisExtendedArray = new double[] {5.6, 1.2, 2.3, 3.4, 1.2, 1.2, 2.3, 6.7};
            double[] thatExtendedArray = new double[] {5.6, 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 1.2};
            var thisList = new IList<double>[4];
            var that = new IEnumerable<double>[7];
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            thisList[0] = thisArray.ToList();
            thisList[1] = SparseList<double>.Copy(thisArray.ToList());
            thisList[2] = thisExtendedArray.ToList();
            thisList[3] = SparseList<double>.Copy(thisExtendedArray.ToList());

            that[0] = Vector.FromArray(thatArray);
            that[1] = Vector.FromArray(thatArray, Sparsity.Sparse);
            that[2] = Vector.FromArray(thatArray, approxSparsity);
            that[3] = DenseVector.FromArrayReference(6, thatExtendedArray, 1);
            that[4] = thatArray;
            that[5] = thatArray.ToList();
            var sparseValues = thatArray.Select((x, i) => new ValueAtIndex<double>(i, x)).ToList();
            that[6] = SparseList<double>.FromSparseValues(thatArray.Length, 0.0, sparseValues);

            for (int i = 0; i < thisList.Length; i++)
                for (int j = 0; j < that.Length; j++)
                {
                    ListSetToEnumerable(thisList[i], that[j]);
                }
        }

        private void ListSetToEnumerable(IList<double> thisList, IEnumerable<double> that)
        {
            thisList.SetTo(that);
            var thatList = that.ToList();
            Assert.True(thisList.Count == thatList.Count);
            for (int i = 0; i < thisList.Count; i++)
            {
                Assert.True(thisList[i] == thatList[i]);
            }
        }

        /// <summary>
        /// Tests sparsity attribute for Random dirichlet
        /// </summary>
        [Fact]
        public void SparseVariableFromRandomDirichlet()
        {
            int L = 2;
            Range K = new Range(2).Named("K");
            var d = new Dirichlet[][]
                {
                    new Dirichlet[2]
                        {
                            new Dirichlet(Vector.FromArray(1.2, 2.3)), new Dirichlet(Vector.FromArray(3.2, 7.6, 5.4, 3.2))
                        },
                    new Dirichlet[2]
                        {
                            new Dirichlet(Vector.FromArray(1.2, 2.3, 1.2)), new Dirichlet(Vector.FromArray(9.8, 7.6))
                        }
                };
            var dv = new VariableArray<Vector>[L];
            for (int l = 0; l < L; l++)
            {
                var dvprior = Variable.Observed<Dirichlet>(d[l], K).Named("dvprior" + l);
                // Mark with a sparse attribute
                dv[l] = Variable.Array<Vector>(K).Named("dv" + l).Attrib(new SparsityAttribute(Sparsity.Sparse));
                dv[l][K] = Variable.Random<Vector, Dirichlet>(dvprior[K]);
            }
            var engine = new InferenceEngine();
            var dvPost = new Dirichlet[2][];
            for (int l = 0; l < L; l++)
            {
                dvPost[l] = engine.Infer<Dirichlet[]>(dv[l]);
                for (int k = 0; k < K.SizeAsInt; k++)
                    Assert.True(dvPost[l][k].PseudoCount.IsSparse);
            }
        }

        /// <summary>
        /// Tests sparsity attribute for Random dirichlet uniform
        /// </summary>
        [Fact]
        public void SparseVariableFromRandomDirichletUniform()
        {
            int L = 2;
            Range K = new Range(2).Named("K");
            var dv = new VariableArray<Vector>[L];
            for (int l = 0; l < L; l++)
            {
                // Mark with a sparse attribute
                dv[l] = Variable.Array<Vector>(K).Named("dv" + l).Attrib(new SparsityAttribute(Sparsity.Sparse));
                dv[l][K] = Variable.Random<Vector, Dirichlet>(Dirichlet.Uniform(4)).ForEach(K);
            }
            var engine = new InferenceEngine();
            var dvPost = new Dirichlet[2][];
            for (int l = 0; l < L; l++)
            {
                dvPost[l] = engine.Infer<Dirichlet[]>(dv[l]);
                for (int k = 0; k < K.SizeAsInt; k++)
                    Assert.True(dvPost[l][k].PseudoCount.IsSparse);
            }
        }

        /// <summary>
        /// Tests sparse marginal prototype for Random dirichlet from mean log
        /// </summary>
        [Fact]
        public void SparseVariableFromRandomDirichletFromMeanLog()
        {
            int L = 2;
            Range K = new Range(2).Named("K");
            var dv = new VariableArray<Vector>[L];
            for (int l = 0; l < L; l++)
            {
                // Mark with a sparse attribute
                dv[l] = Variable.Array<Vector>(K).Named("dv" + l).Attrib(new MarginalPrototype(Dirichlet.Uniform(3, Sparsity.Sparse)));
                dv[l][K] = Variable<Vector>.Random(Dirichlet.FromMeanLog(Vector.FromArray(.1, .2, .1))).ForEach(K);
            }
            var engine = new InferenceEngine();
            var dvPost = new Dirichlet[2][];
            for (int l = 0; l < L; l++)
            {
                dvPost[l] = engine.Infer<Dirichlet[]>(dv[l]);
                for (int k = 0; k < K.SizeAsInt; k++)
                    Assert.True(dvPost[l][k].PseudoCount.IsSparse);
            }
        }

        /// <summary>
        /// Tests sparsity attribute for Random discrete
        /// </summary>
        [Fact]
        public void SparseVariableFromRandomDiscrete()
        {
            int L = 2;
            Range K = new Range(2).Named("K");
            var d = new Discrete[][]
                {
                    new Discrete[2]
                        {
                            new Discrete(Vector.FromArray(0.3, 0.7)), new Discrete(Vector.FromArray(0.4, 0.3, 0.2, 0.1))
                        },
                    new Discrete[2]
                        {
                            new Discrete(Vector.FromArray(0.2, 0.3, 0.5)), new Discrete(Vector.FromArray(0.7, 0.2, 0.1))
                        }
                };
            var dv = new VariableArray<int>[L];

            for (int l = 0; l < L; l++)
            {
                var dvprior = Variable.Observed<Discrete>(d[l], K).Named("dvprior" + l);

                dv[l] = Variable.Array<int>(K).Named("dv" + l).Attrib(new SparsityAttribute(Sparsity.Sparse));
                dv[l][K] = Variable.Random<int, Discrete>(dvprior[K]);
            }
            var engine = new InferenceEngine();
            var dvPost = new Discrete[2][];
            for (int l = 0; l < L; l++)
            {
                dvPost[l] = engine.Infer<Discrete[]>(dv[l]);
                for (int k = 0; k < K.SizeAsInt; k++)
                    Assert.True(dvPost[l][k].GetProbs().IsSparse);
            }
        }

        /// <summary>
        /// Tests sparsity attribute for Random discrete from uniform
        /// </summary>
        [Fact]
        public void SparseVariableFromRandomDiscreteUniform()
        {
            int L = 2;
            Range K = new Range(2).Named("K");
            var dv = new VariableArray<int>[L];

            for (int l = 0; l < L; l++)
            {
                dv[l] = Variable.Array<int>(K).Named("dv" + l).Attrib(new SparsityAttribute(Sparsity.Sparse));
                dv[l][K] = Variable.Random<int, Discrete>(Discrete.Uniform(3)).ForEach(K);
            }
            var engine = new InferenceEngine();
            var dvPost = new Discrete[2][];
            for (int l = 0; l < L; l++)
            {
                dvPost[l] = engine.Infer<Discrete[]>(dv[l]);
                for (int k = 0; k < K.SizeAsInt; k++)
                    Assert.True(dvPost[l][k].GetProbs().IsSparse);
            }
        }

        private void SparseVariableFromDirichlet(bool attribOnVec)
        {
            int L = 2;
            Range K = new Range(2).Named("K");
            var pc = new Vector[][]
                {
                    new Vector[] {Vector.FromArray(1.2, 2.3), Vector.FromArray(9.8, 7.6, 5.4, 3.2)},
                    new Vector[] {Vector.FromArray(1.2, 2.3, 3.4), Vector.FromArray(9.8, 7.6)}
                };
            var dv = new VariableArray<Vector>[L];
            for (int l = 0; l < L; l++)
            {
                var pcv = Variable.Observed<Vector>(pc[l], K).Named("pc" + l);
                if (attribOnVec) pcv.SetSparsity(Sparsity.Sparse);
                dv[l] = Variable.Array<Vector>(K).Named("dv" + l);
                if (!attribOnVec) dv[l].SetSparsity(Sparsity.Sparse);
                dv[l][K] = Variable.Dirichlet(pcv[K]);
            }
            var engine = new InferenceEngine();
            var dvPost = new Dirichlet[2][];
            for (int l = 0; l < L; l++)
            {
                dvPost[l] = engine.Infer<Dirichlet[]>(dv[l]);
                for (int k = 0; k < K.SizeAsInt; k++)
                    Assert.True(dvPost[l][k].PseudoCount.IsSparse);
            }
        }

        /// <summary>
        /// Tests sparsity attribute for Variable.Dirichlet
        /// </summary>
        internal void SparseVariableFromDirichletTest()
        {
            SparseVariableFromDirichlet(true);
            SparseVariableFromDirichlet(false);
        }

        private void SparseVariableFromDiscrete(bool attribOnVec)
        {
            int L = 2;
            Range K = new Range(2).Named("K");
            var probs = new Vector[][]
                {
                    new Vector[] {Vector.FromArray(0.3, 0.7), Vector.FromArray(0.4, 0.3, 0.2, 0.1)},
                    new Vector[] {Vector.FromArray(0.2, 0.3, 0.5), Vector.FromArray(0.7, 0.2, 0.1)}
                };
            var dv = new VariableArray<int>[L];
            for (int l = 0; l < L; l++)
            {
                var probsv = Variable.Observed<Vector>(probs[l], K).Named("probs" + l);
                if (attribOnVec) probsv.SetSparsity(Sparsity.Sparse);
                dv[l] = Variable.Array<int>(K).Named("dv" + l);
                if (!attribOnVec) dv[l].SetSparsity(Sparsity.Sparse);
                dv[l][K] = Variable.Discrete(probsv[K]);
            }
            var engine = new InferenceEngine();
            var dvPost = new Discrete[2][];
            for (int l = 0; l < L; l++)
            {
                dvPost[l] = engine.Infer<Discrete[]>(dv[l]);
                for (int k = 0; k < K.SizeAsInt; k++)
                    Assert.True(dvPost[l][k].GetProbs().IsSparse);
            }
        }

        /// <summary>
        /// Tests sparsity attribute for Variable.Discrete
        /// </summary>
        [Fact]
        public void SparseVariableFromDiscreteTest()
        {
            SparseVariableFromDiscrete(true);
            SparseVariableFromDiscrete(false);
        }

        private void SparseConstantVariable(bool sparse, bool over_ride)
        {
            Range K1 = new Range(2);
            Range K2 = new Range(2);
            Range K3 = new Range(2);
            Sparsity s = sparse ? Sparsity.Sparse : Sparsity.Dense;

            Vector v = Vector.FromArray(new double[] {0.1, 0.1, 0.8}, s);
            Vector[] v1 = new Vector[] {v, v};
            Vector[][] v11 = new Vector[][] {v1, v1};
            Vector[,] v2 = new Vector[,] {{v, v}, {v, v}};
            Vector[][,] v12 = new Vector[][,] {v2, v2};
            Vector[][][] v111 = new Vector[][][] {v11, v11};

            var vv = Variable.Constant(v);
            var vv1 = Variable.Constant(v1, K1);
            var vv11 = Variable.Constant(v11, K1, K2);
            var vv2 = Variable.Constant(v2, K1, K2);
            var vv12 = Variable.Constant(v12, K1, K2, K3);
            var vv111 = Variable.Constant(v111, K1, K2, K3);

            var d = Variable.Discrete(vv).Named("d");
            var d1 = Variable.Array<int>(K1).Named("d1");
            var d11 = Variable.Array(Variable.Array<int>(K2), K1).Named("d11");
            var d2 = Variable.Array<int>(K1, K2).Named("d2");
            var d12 = Variable.Array(Variable.Array<int>(K2, K3), K1).Named("d12");
            var d111 = Variable.Array(Variable.Array(Variable.Array<int>(K3), K2), K1).Named("d111");

            d1[K1] = Variable.Discrete(vv1[K1]);
            d11[K1][K2] = Variable.Discrete(vv11[K1][K2]);
            d2[K1, K2] = Variable.Discrete(vv2[K1, K2]);
            d12[K1][K2, K3] = Variable.Discrete(vv12[K1][K2, K3]);
            d111[K1][K2][K3] = Variable.Discrete(vv111[K1][K2][K3]);

            if (over_ride)
            {
                sparse = !sparse;
                s = sparse ? Sparsity.Sparse : Sparsity.Dense;
                d.SetSparsity(s);
                d1.SetSparsity(s);
                d11.SetSparsity(s);
                d2.SetSparsity(s);
                d12.SetSparsity(s);
                d111.SetSparsity(s);
            }

            var engine = new InferenceEngine();
            var dpost = engine.Infer<Discrete>(d);
            var d1post = engine.Infer<Discrete[]>(d1);
            var d11post = engine.Infer<Discrete[][]>(d11);
            var d2post = engine.Infer<Discrete[,]>(d2);
            var d12post = engine.Infer<Discrete[][,]>(d12);
            var d111post = engine.Infer<Discrete[][][]>(d111);

            Assert.Equal(dpost.GetProbs().IsSparse, sparse);
            foreach (Discrete de in d1post)
                Assert.Equal(de.GetProbs().IsSparse, sparse);
            foreach (Discrete[] dea in d11post)
                foreach (Discrete de in dea)
                    Assert.Equal(de.GetProbs().IsSparse, sparse);
            foreach (Discrete de in d2post)
                Assert.Equal(de.GetProbs().IsSparse, sparse);
            foreach (Discrete[,] dea in d12post)
                foreach (Discrete de in dea)
                    Assert.Equal(de.GetProbs().IsSparse, sparse);
            foreach (Discrete[][] deaa in d111post)
                foreach (Discrete[] dea in deaa)
                    foreach (Discrete de in dea)
                        Assert.Equal(de.GetProbs().IsSparse, sparse);
        }

        /// <summary>
        /// Tests sparsity is handled correctly for factors with Constant vector arguments
        /// </summary>
        internal void SparseConstantVariableTest()
        {
            SparseConstantVariable(false, false);
            SparseConstantVariable(false, true);
            SparseConstantVariable(true, false);
            SparseConstantVariable(true, true);
        }

        private void SparseGaussianListProduct(double tolerance)
        {
            var g1 = Gaussian.FromMeanAndPrecision(.1, .2);
            var g2 = Gaussian.FromMeanAndPrecision(.2, .3);
            var g3 = Gaussian.FromMeanAndPrecision(.3, .4);
            var g4 = Gaussian.FromMeanAndPrecision(.4, .5);
            var g5 = Gaussian.FromMeanAndPrecision(.5, .6);
            var g6 = Gaussian.FromMeanAndPrecision(.6, .7);

            SparseGaussianList sgl1 = SparseGaussianList.Constant(10, g1, 0);
            sgl1[3] = g2;
            sgl1[7] = g3;

            SparseGaussianList sgl2 = SparseGaussianList.Constant(10, g4, 0);
            sgl2[3] = g5;
            sgl2[8] = g6;

            SparseGaussianList sgl3 = SparseGaussianList.FromSize(10, tolerance);
            sgl3.SetToProduct(sgl1, sgl2);

            var gOf3 = g2*g5;
            var gOf7 = g3*g4;
            var gOf8 = g1*g6;
            var common = g1*g4;

            int diffcount = 0;
            if (gOf3.MaxDiff(common) <= tolerance) gOf3.SetTo(common);
            else diffcount++;
            if (gOf7.MaxDiff(common) <= tolerance) gOf7.SetTo(common);
            else diffcount++;
            if (gOf8.MaxDiff(common) <= tolerance) gOf8.SetTo(common);
            else diffcount++;
            Console.WriteLine(diffcount);

            Assert.Equal(diffcount, sgl3.SparseValues.Count);
            Assert.Equal(gOf3, sgl3[3]);
            Assert.Equal(gOf7, sgl3[7]);
            Assert.Equal(gOf8, sgl3[8]);
            Assert.Equal(common, sgl3.CommonValue);
        }

        private void SparseGaussianListRatio(double tolerance)
        {
            var g1 = Gaussian.FromMeanAndPrecision(.1, .2);
            var g2 = Gaussian.FromMeanAndPrecision(.2, .3);
            var g3 = Gaussian.FromMeanAndPrecision(.3, .4);
            var g4 = Gaussian.FromMeanAndPrecision(.4, .5);
            var g5 = Gaussian.FromMeanAndPrecision(.5, .6);
            var g6 = Gaussian.FromMeanAndPrecision(.6, .7);

            SparseGaussianList sgl1 = SparseGaussianList.Constant(10, g1, 0);
            sgl1[3] = g2;
            sgl1[7] = g3;

            SparseGaussianList sgl2 = SparseGaussianList.Constant(10, g4, 0);
            sgl2[3] = g5;
            sgl2[8] = g6;

            SparseGaussianList sgl3 = SparseGaussianList.FromSize(10, tolerance);
            sgl3.SetToRatio(sgl1, sgl2);

            var gOf3 = g2/g5;
            var gOf7 = g3/g4;
            var gOf8 = g1/g6;
            var common = g1/g4;

            int diffcount = 0;
            if (gOf3.MaxDiff(common) <= tolerance) gOf3.SetTo(common);
            else diffcount++;
            if (gOf7.MaxDiff(common) <= tolerance) gOf7.SetTo(common);
            else diffcount++;
            if (gOf8.MaxDiff(common) <= tolerance) gOf8.SetTo(common);
            else diffcount++;
            Console.WriteLine(diffcount);

            Assert.Equal(diffcount, sgl3.SparseValues.Count);
            Assert.Equal(gOf3, sgl3[3]);
            Assert.Equal(gOf7, sgl3[7]);
            Assert.Equal(gOf8, sgl3[8]);
            Assert.Equal(common, sgl3.CommonValue);
        }

        private void SparseGaussianListPower(double tolerance)
        {
            var g1 = Gaussian.FromMeanAndPrecision(.1, .2);
            var g2 = Gaussian.FromMeanAndPrecision(.2, .3);
            var g3 = Gaussian.FromMeanAndPrecision(.3, .4);
            var g4 = Gaussian.FromMeanAndPrecision(.4, .5);
            var g5 = Gaussian.FromMeanAndPrecision(.5, .6);
            var g6 = Gaussian.FromMeanAndPrecision(.6, .7);

            SparseGaussianList sgl1 = SparseGaussianList.Constant(10, g1, 0);
            sgl1[3] = g2;
            sgl1[7] = g3;

            SparseGaussianList sgl2 = SparseGaussianList.Constant(10, g4, 0);
            sgl2[3] = g5;
            sgl2[8] = g6;

            SparseGaussianList sgl3 = SparseGaussianList.FromSize(10, tolerance);
            double exp = 1.2;
            sgl3.SetToPower(sgl1, exp);

            var gOf3 = g2 ^ exp;
            var gOf7 = g3 ^ exp;
            var common = g1 ^ exp;

            int diffcount = 0;
            if (gOf3.MaxDiff(common) <= tolerance) gOf3.SetTo(common);
            else diffcount++;
            if (gOf7.MaxDiff(common) <= tolerance) gOf7.SetTo(common);
            else diffcount++;
            Console.WriteLine(diffcount);

            Assert.Equal(diffcount, sgl3.SparseValues.Count);
            Assert.Equal(gOf3, sgl3[3]);
            Assert.Equal(gOf7, sgl3[7]);
            Assert.Equal(common, sgl3.CommonValue);
        }

        private void SparseGaussianListSum(double tolerance)
        {
            var g1 = Gaussian.FromMeanAndPrecision(.1, .2);
            var g2 = Gaussian.FromMeanAndPrecision(.2, .3);
            var g3 = Gaussian.FromMeanAndPrecision(.3, .4);
            var g4 = Gaussian.FromMeanAndPrecision(.4, .5);
            var g5 = Gaussian.FromMeanAndPrecision(.5, .6);
            var g6 = Gaussian.FromMeanAndPrecision(.6, .7);

            SparseGaussianList sgl1 = SparseGaussianList.Constant(10, g1, 0);
            sgl1[3] = g2;
            sgl1[7] = g3;

            SparseGaussianList sgl2 = SparseGaussianList.Constant(10, g4, 0);
            sgl2[3] = g5;
            sgl2[8] = g6;

            SparseGaussianList sgl3 = SparseGaussianList.FromSize(10, tolerance);
            double w1 = 0.1;
            double w2 = 0.2;
            sgl3.SetToSum(w1, sgl1, w2, sgl2);

            var gOf3 = new Gaussian();
            var gOf7 = new Gaussian();
            var gOf8 = new Gaussian();
            var common = new Gaussian();
            gOf3.SetToSum(w1, g2, w2, g5);
            gOf7.SetToSum(w1, g3, w2, g4);
            gOf8.SetToSum(w1, g1, w2, g6);
            common.SetToSum(w1, g1, w2, g4);

            int diffcount = 0;
            if (gOf3.MaxDiff(common) <= tolerance) gOf3.SetTo(common);
            else diffcount++;
            if (gOf7.MaxDiff(common) <= tolerance) gOf7.SetTo(common);
            else diffcount++;
            if (gOf8.MaxDiff(common) <= tolerance) gOf8.SetTo(common);
            else diffcount++;
            Console.WriteLine("diffcount = {0} should be {1}", diffcount, sgl3.SparseValues.Count);

            Assert.Equal(diffcount, sgl3.SparseValues.Count);
            Assert.Equal(gOf3, sgl3[3]);
            Assert.Equal(gOf7, sgl3[7]);
            Assert.Equal(gOf8, sgl3[8]);
            Assert.Equal(common, sgl3.CommonValue);
        }

        private void SparseGaussianListIndex(double tolerance)
        {
            var g1 = Gaussian.FromMeanAndPrecision(.1, .2);
            var g2 = Gaussian.FromMeanAndPrecision(.2, .3);
            SparseGaussianList sgl1 = SparseGaussianList.Constant(10, g1, tolerance);
            double maxdiff = g1.MaxDiff(g2);
            sgl1[3] = g2;
            Console.WriteLine("tol = {0}, max diff = {1}", tolerance, maxdiff);

            if (maxdiff <= tolerance)
                Assert.Equal(sgl1[3], g1);
            else
                Assert.Equal(sgl1[3], g2);
        }

        [Fact]
        public void SparseGaussianListProductTest()
        {
            double tol = 0.05;
            double inc = 0.01;
            for (int i = 0; i < 20; i++, tol += inc)
                SparseGaussianListProduct(tol);
        }

        [Fact]
        public void SparseGaussianListRatioTest()
        {
            double tol = 0.05;
            double inc = 0.01;
            for (int i = 0; i < 20; i++, tol += inc)
                SparseGaussianListRatio(tol);
        }

        [Fact]
        public void SparseGaussianListPowerTest()
        {
            double tol = 0.05;
            double inc = 0.01;
            for (int i = 0; i < 20; i++, tol += inc)
                SparseGaussianListPower(tol);
        }

        [Fact]
        public void SparseGaussianListSumTest()
        {
            double tol = 0.05;
            double inc = 0.01;
            for (int i = 0; i < 20; i++, tol += inc)
                SparseGaussianListSum(tol);
        }

        [Fact]
        public void SparseGaussianListIndexTest()
        {
            double tol = 0.05;
            double inc = 0.01;
            for (int i = 0; i < 20; i++, tol += inc)
                SparseGaussianListIndex(tol);
        }

        [Fact]
        public void IsSparseTest()
        {
            IList<double> a = Vector.Zero(2);
            IList<double> b = SparseVector.Zero(2);
            IList<double> c = ApproximateSparseVector.Zero(2);
            IList<double> d = new List<double>(new double[] {0.0, 0.0});
            IList<double> e = SparseList<double>.Constant(2, 0.0);
            IList<Gaussian> f = SparseGaussianList.Constant(2, Gaussian.Uniform());

            Assert.False(a.IsSparse());
            Assert.True(b.IsSparse());
            Assert.True(c.IsSparse());
            Assert.False(d.IsSparse());
            Assert.True(e.IsSparse());
            Assert.True(f.IsSparse());
        }

        [Fact]
        public void GaussianOfSparseGaussianList()
        {
            int n = 5;
            var mmean = Variable.Constant(SparseVector.Constant(n, 0.0) as ISparseList<double>).Named("mmean");
            var mprec = Variable.Constant(SparseVector.Constant(n, 0.1) as ISparseList<double>).Named("mprec");
            var m = Variable.GaussianListFromMeanAndPrecision(mmean, mprec).Named("mean");
            m.Attrib(new MarginalPrototype(SparseGaussianList.FromSize(n)));
            //m.SetSparsity(Sparsity.ApproximateWithTolerance(0.0001));
            var p = SparseVector.Constant(n, 1.0) as ISparseList<double>;

            Range d = new Range(3).Named("d");
            var y = Variable.Array<ISparseList<double>>(d).Named("y");
            //y.SetSparsity(Sparsity.ApproximateWithTolerance(0.01));

            y[d] = Variable.GaussianListFromMeanAndPrecision(m, p).ForEach(d);
            y.ObservedValue = new ISparseList<double>[]
                {
                    SparseVector.Constant(n, 1.0), SparseVector.Constant(n, 1.0), SparseVector.Constant(n, 1.0)
                };
            y.ObservedValue[0][2] = 2.0;
            y.ObservedValue[1][2] = 3.0;
            y.ObservedValue[2][3] = 4.0;
            var engine = new InferenceEngine();
            engine.Compiler.ShowWarnings = false;
            var mPost = engine.Infer<SparseGaussianList>(m);

            Console.WriteLine(mPost);
            Assert.Equal(2, mPost.SparseValues.Count);
        }

        [Fact]
        public void GaussianOfSparseGaussianList2()
        {
            int n = 5;
            var mmean = SparseVector.Constant(n, 0.0);
            var mprec = SparseVector.Constant(n, 0.1);
            var sgl = Variable.New<SparseGaussianList>().Named("sgl");
            sgl.ObservedValue = SparseGaussianList.FromMeanAndPrecision(mmean, mprec, 0.0001);
            var m = Variable<ISparseList<double>>.Random(sgl).Named("mean");
            var p = SparseVector.Constant(n, 1.0) as ISparseList<double>;

            Range d = new Range(3).Named("d");
            var y = Variable.Array<ISparseList<double>>(d).Named("y");
            //y.AddAttribute(new MarginalPrototype(SparseGaussianList.FromSize(n, 0.01)));
            //y.SetSparsity(Sparsity.ApproximateWithTolerance(0.01));

            y[d] = Variable.GaussianListFromMeanAndPrecision(m, p).ForEach(d);
            y.ObservedValue = new ISparseList<double>[]
                {
                    SparseVector.Constant(n, 1.0), SparseVector.Constant(n, 1.0), SparseVector.Constant(n, 1.0)
                };
            y.ObservedValue[0][2] = 2.0;
            y.ObservedValue[1][2] = 3.0;
            y.ObservedValue[2][3] = 4.0;
            var engine = new InferenceEngine();
            engine.Compiler.ShowWarnings = false;
            var mPost = engine.Infer<SparseGaussianList>(m);

            Console.WriteLine(mPost);
            Assert.Equal(2, mPost.SparseValues.Count);
        }


        [Fact]
        public void SparseZip()
        {
            var x = SparseList<int>.FromSparseValues(10, 1, new List<ValueAtIndex<int>>()
                {
                    new ValueAtIndex<int> {Index = 1, Value = 2},
                    new ValueAtIndex<int> {Index = 4, Value = 3},
                    new ValueAtIndex<int> {Index = 5, Value = 4}
                });
            var y = SparseList<int>.FromSparseValues(10, 0, new List<ValueAtIndex<int>>()
                {
                    new ValueAtIndex<int> {Index = 2, Value = 1},
                    new ValueAtIndex<int> {Index = 4, Value = 2}
                });
            var z = x.ListZip(y, (i, j) => i + j);
            var z2 = SparseList<int>.FromSize(x.Count);
            z2.SetToFunction(x, y, (i, j) => i + j);
            Assert.True(z.Count == x.Count);
            Assert.True(z2.Count == x.Count);
            for (int i = 0; i < 10; i++)
            {
                //Console.WriteLine(x[i] + " + " + y[i] + " = " + z[i]); 
                Assert.True(x[i] + y[i] == z[i]);
                Assert.True(x[i] + y[i] == z2[i]);
            }
        }


        [Fact]
        public void SparseZipThree()
        {
            var x = SparseList<int>.FromSparseValues(10, 1, new List<ValueAtIndex<int>>()
                {
                    new ValueAtIndex<int> {Index = 1, Value = 2},
                    new ValueAtIndex<int> {Index = 4, Value = 3},
                    new ValueAtIndex<int> {Index = 5, Value = 4}
                });
            var y = SparseList<int>.FromSparseValues(10, 0, new List<ValueAtIndex<int>>()
                {
                    new ValueAtIndex<int> {Index = 2, Value = 1},
                    new ValueAtIndex<int> {Index = 4, Value = 2}
                });
            var z = SparseList<int>.FromSparseValues(10, 0, new List<ValueAtIndex<int>>()
                {
                    new ValueAtIndex<int> {Index = 3, Value = 2},
                    new ValueAtIndex<int> {Index = 4, Value = 3}
                });
            var a = x.ListZip(y, z, (i, j, k) => i + j + k);
            var a2 = SparseList<int>.FromSize(x.Count);
            a2.SetToFunction(x, y, z, (i, j, k) => i + j + k);
            Assert.True(a.Count == x.Count);
            Assert.True(a2.Count == x.Count);
            for (int i = 0; i < 10; i++)
            {
                //Console.WriteLine(x[i] + " + " + y[i] + " = " + z[i]); 
                Assert.True(x[i] + y[i] + z[i] == a[i]);
                Assert.True(x[i] + y[i] + z[i] == a2[i]);
            }
        }

        [Fact]
        public void SparseZipFour()
        {
            var x = SparseList<int>.FromSparseValues(10, 1, new List<ValueAtIndex<int>>()
                {
                    new ValueAtIndex<int> {Index = 1, Value = 2},
                    new ValueAtIndex<int> {Index = 4, Value = 3},
                    new ValueAtIndex<int> {Index = 5, Value = 4}
                });
            var y = SparseList<int>.FromSparseValues(10, 0, new List<ValueAtIndex<int>>()
                {
                    new ValueAtIndex<int> {Index = 2, Value = 1},
                    new ValueAtIndex<int> {Index = 4, Value = 2}
                });
            var z = SparseList<int>.FromSparseValues(10, 0, new List<ValueAtIndex<int>>()
                {
                    new ValueAtIndex<int> {Index = 3, Value = 2},
                    new ValueAtIndex<int> {Index = 4, Value = 3}
                });
            var w = SparseList<int>.FromSparseValues(10, 0, new List<ValueAtIndex<int>>()
                {
                    new ValueAtIndex<int> {Index = 0, Value = 2},
                    new ValueAtIndex<int> {Index = 9, Value = 3}
                });

            var a2 = SparseList<int>.FromSize(x.Count);
            a2.SetToFunction(x, y, z, w, (i, j, k, l) => i + j + k + l);
            //var a = x.ListZip(y, z, w, (i, j, k, l) => i + j + k + l);
            //Assert.True(a.Count == x.Count);
            //Assert.True(a2.Count == x.Count);
            //for (int i = 0; i < 10; i++)
            //{
            //    //Console.WriteLine(x[i] + " + " + y[i] + " = " + z[i]); 
            //    Assert.True(x[i] + y[i] + z[i] + w[i] == a[i]);
            //    Assert.True(x[i] + y[i] + z[i] + w[i] == a2[i]);
            //}
        }


        private SparseGaussianList SparseGaussianListWithSoftmax(Range n, Range d, int[] data)
        {
            var mmean = Variable.Constant(SparseVector.Constant(n.SizeAsInt, 0.0) as ISparseList<double>).Named("mmean");
            var mprec = Variable.Constant(SparseVector.Constant(n.SizeAsInt, 0.1) as ISparseList<double>).Named("mprec");
            var m = Variable.GaussianListFromMeanAndPrecision(mmean, mprec).Named("mean");
            m.Attrib(new MarginalPrototype(SparseGaussianList.FromSize(n.SizeAsInt)));

            var p = SparseVector.Constant(n.SizeAsInt, 1.0) as ISparseList<double>;

            var h = Variable.Array<ISparseList<double>>(d).Named("h");
            var s = Variable.Array<Vector>(d).Named("s");
            //s.AddAttribute(new MarginalPrototype(Dirichlet.Uniform(n.SizeAsInt, Sparsity.Sparse)));
            s.SetSparsity(Sparsity.Sparse);
            var y = Variable.Array<int>(d).Named("y");
            y.SetValueRange(n);
            using (Variable.ForEach(d))
            {
                h[d] = Variable.GaussianListFromMeanAndPrecision(m, p);
                s[d] = Variable.Softmax(h[d]);
                y[d] = Variable.Discrete(s[d]);
            }
            y.ObservedValue = data;
            var engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1000;
            engine.Compiler.GivePriorityTo(typeof (SoftmaxOp_KM11_Sparse));
            //engine.Compiler.GivePriorityTo(typeof(SaulJordanSoftmaxOp_NCVMP)); 
            return engine.Infer<SparseGaussianList>(m);
        }

        public DistributionArray<Gaussian> GaussianListWithSoftmax(Range n, Range d, int[] data)
        {
            var m = Variable.Array<double>(n).Named("mean");
            m[n] = Variable.GaussianFromMeanAndPrecision(0.0, 0.1).ForEach(n);
            var h = Variable.Array(Variable.Array<double>(n), d).Named("h");
            var s = Variable.Array<Vector>(d).Named("s");
            var y = Variable.Array<int>(d).Named("y");
            y.SetValueRange(n);
            h[d][n] = Variable.GaussianFromMeanAndPrecision(m[n], 1.0).ForEach(d);
            s[d] = Variable.Softmax(h[d]);
            y[d] = Variable.Discrete(s[d]);
            y.ObservedValue = data;
            var engine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 1000;
            //engine.Compiler.GivePriorityTo(typeof(SaulJordanSoftmaxOp_NCVMP_Sparse)); 
            engine.Compiler.GivePriorityTo(typeof (SoftmaxOp_KM11));
            return engine.Infer<DistributionArray<Gaussian>>(m);
        }

        [Fact]
        public void SparseGaussianListWithSoftmaxTest()
        {
            var data = new int[] {2, 3, 3};
            Range n = new Range(5);
            Range d = new Range(3).Named("d");

            Rand.Restart(12345);
            var mPostSparse = SparseGaussianListWithSoftmax(n, d, data);
            Rand.Restart(12345);
            var mPostDense = GaussianListWithSoftmax(n, d, data);

            Console.WriteLine("mPostSparse:");
            Console.WriteLine(mPostSparse);
            Assert.Equal(2, mPostSparse.SparseValues.Count);
            Console.WriteLine("mPostDense:");
            Console.WriteLine(mPostDense);

            for (int i = 0; i < n.SizeAsInt; i++)
            {
                Assert.True(mPostSparse[i].MaxDiff(mPostDense[i]) < 1e-4);
            }
        }
    }
}