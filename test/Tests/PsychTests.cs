// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using System.IO;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Xunit.Assert;
    using Range = Microsoft.ML.Probabilistic.Models.Range;


    public class PsychTests
    {
        internal void LogisticIrtTest()
        {
            Variable<int> numStudents = Variable.New<int>().Named("numStudents");
            Range student = new Range(numStudents);
            VariableArray<double> ability = Variable.Array<double>(student).Named("ability");
            ability[student] = Variable.GaussianFromMeanAndPrecision(0, 1e-6).ForEach(student);
            Variable<int> numQuestions = Variable.New<int>().Named("numQuestions");
            Range question = new Range(numQuestions);
            VariableArray<double> difficulty = Variable.Array<double>(question).Named("difficulty");
            difficulty[question] = Variable.GaussianFromMeanAndPrecision(0, 1e-6).ForEach(question);
            VariableArray<double> discrimination = Variable.Array<double>(question).Named("discrimination");
            discrimination[question] = Variable.Exp(Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(question));
            VariableArray2D<bool> response = Variable.Array<bool>(student, question).Named("response");
            response[student, question] = Variable.BernoulliFromLogOdds(((ability[student] - difficulty[question]).Named("minus")*discrimination[question]).Named("product"));
            bool[,] data;
            double[] discriminationTrue = new double[0];
            bool useDummyData = false;
            if (useDummyData)
            {
                data = new bool[4,2];
                for (int i = 0; i < data.GetLength(0); i++)
                {
                    for (int j = 0; j < data.GetLength(1); j++)
                    {
                        data[i, j] = (i > j);
                    }
                }
            }
            else
            {
                // simulated data
                // also try IRT2PL_10_250.mat
                //TODO: change path for cross platform using
                Dictionary<string, object> dict = MatlabReader.Read(@"..\..\..\Tests\Data\IRT2PL_10_1000.mat");
                Matrix m = (Matrix) dict["Y"];
                data = ConvertToBool(m.ToArray());
                m = (Matrix) dict["discrimination"];
                discriminationTrue = Util.ArrayInit(data.GetLength(1), i => m[i]);
            }
            numStudents.ObservedValue = data.GetLength(0);
            numQuestions.ObservedValue = data.GetLength(1);
            response.ObservedValue = data;
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            Console.WriteLine(StringUtil.JoinColumns(engine.Infer(discrimination), " should be ", StringUtil.ToString(discriminationTrue)));
        }

        public static bool[,] ConvertToBool(double[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            bool[,] result = new bool[rows,cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = (array[i, j] > 0);
                }
            }
            return result;
        }

        /// <summary>
        /// Nonconjugate VMP crashes with improper message on the first iteration.
        /// </summary>
        internal void LogisticIrtTestWithTruncatedGaussian()
        {
            Variable<int> numStudents = Variable.New<int>().Named("numStudents");
            Range student = new Range(numStudents);
            VariableArray<double> ability = Variable.Array<double>(student).Named("ability");
            ability[student] = Variable.GaussianFromMeanAndPrecision(0, 1e-2).ForEach(student);
            Variable<int> numQuestions = Variable.New<int>().Named("numQuestions");
            Range question = new Range(numQuestions);
            VariableArray<double> difficulty = Variable.Array<double>(question).Named("difficulty");
            difficulty[question] = Variable.GaussianFromMeanAndPrecision(0, 1e-2).ForEach(question);
            var response = Variable.Array<bool>(student, question).Named("response");
            var minus = Variable.Array<double>(student, question).Named("minus");
            //var discrimination = Variable.Array<double>(question).Named("discrimination");
            //discrimination[question] = Variable.TruncatedGaussian(1, 1/1.5e-3, 0, double.PositiveInfinity).ForEach(question);
            var disc2 = Variable.Array<double>(question).Named("disc2");
            //disc2[question] = Variable.Copy(discrimination[question]);
            //disc2.AddAttribute(new MarginalPrototype(new Gaussian())); 
            disc2[question] = Variable.GaussianFromMeanAndVariance(1, 1/1.5e-3).ForEach(question);
            Variable.ConstrainPositive(disc2[question]);
            minus[student, question] = (ability[student] - difficulty[question]);
            var product = Variable.Array<double>(student, question).Named("product");
            product[student, question] = minus[student, question]*disc2[question];
            response[student, question] = Variable.BernoulliFromLogOdds(product[student, question]);
            //response.AddAttribute(new MarginalPrototype(new Gaussian())); 
            bool[,] data;
            double[] discriminationTrue = new double[0];
            bool useDummyData = false;
            if (useDummyData)
            {
                data = new bool[4,2];
                for (int i = 0; i < data.GetLength(0); i++)
                {
                    for (int j = 0; j < data.GetLength(1); j++)
                    {
                        data[i, j] = (i > j);
                    }
                }
            }
            else
            {
                // simulated data
                // also try IRT2PL_10_250.mat
                //TODO: change path for cross platform using
                Dictionary<string, object> dict = MatlabReader.Read(@"..\..\..\Tests\Data\IRT2PL_10_1000.mat");
                Matrix m = (Matrix) dict["Y"];
                data = ConvertToBool(m.ToArray());
                m = (Matrix) dict["discrimination"];
                discriminationTrue = Util.ArrayInit(data.GetLength(1), i => m[i]);
            }
            numStudents.ObservedValue = data.GetLength(0);
            numQuestions.ObservedValue = data.GetLength(1);
            response.ObservedValue = data;
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            engine.ShowTimings = true;
            Console.WriteLine("Compare inferred logDiscrimination to ground truth:");
            //var marg = engine.Infer<DistributionArray<TruncatedGaussian>>(discrimination);
            var marg = engine.Infer<DistributionArray<Gaussian>>(disc2);
            for (int i = 0; i < data.GetLength(1); i++)
                Console.WriteLine(marg[i].GetMean() + " \t " + discriminationTrue[i]);
        }

        internal void LogisticIrtProductExpTest()
        {
            int numStudents = 20;
            Range student = new Range(numStudents).Named("students");
            var ability = Variable.Array<double>(student).Named("ability");
            ability[student] = Variable.GaussianFromMeanAndPrecision(0, 1e-6).ForEach(student);
            int numQuestions = 4;
            Range question = new Range(numQuestions).Named("questions");
            var difficulty = Variable.Array<double>(question).Named("difficulty");
            difficulty[question] = Variable.GaussianFromMeanAndPrecision(0, 1e-6).ForEach(question);
            var logDisc = Variable.Array<double>(question).Named("logDisc");
            logDisc[question] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(question);
            var response = Variable.Array<bool>(student, question).Named("response");
            var minus = Variable.Array<double>(student, question).Named("minus");
            minus[student, question] = (ability[student] - difficulty[question]);
            var product = Variable.Array<double>(student, question).Named("product");
            product[student, question] = Variable.ProductExp(minus[student, question], logDisc[question]);
            response[student, question] = Variable.BernoulliFromLogOdds(product[student, question]);
            bool[,] data = new bool[numStudents,numQuestions];
            for (int i = 0; i < numStudents; i++)
            {
                for (int j = 0; j < numQuestions; j++)
                {
                    data[i, j] = (i > j);
                }
            }
            response.ObservedValue = data;
            InferenceEngine engine = new InferenceEngine();
            engine.ShowFactorGraph = true;
            engine.Algorithm = new VariationalMessagePassing();
            Console.WriteLine(engine.Infer(logDisc));
        }
    }
}