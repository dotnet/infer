// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace ClickThroughModel
{
  public class DocumentStatistics
  {
    public Beta inferredRelevance;
    public Beta inferredAppeal;
  }

  class Inference
  {
    private InferenceEngine ie;
    private VariableArray<bool>[] examine;
    private Variable<double>[] appeal;
    private Variable<double>[] relevance;
    private Variable<double> probNextIfNotClick;
    private Variable<double> probNextIfClickNotRel;
    private Variable<double> probNextIfClickRel;
    private Variable<int> nUsers;
    private VariableArray<bool>[] click;
    private VariableArray<bool>[] isRel;
    private int nRanks;

    public Inference(int numRanks)
    {
      nRanks = numRanks;
      probNextIfNotClick = Variable.New<double>();
      probNextIfClickNotRel = Variable.New<double>();
      probNextIfClickRel = Variable.New<double>();
      nUsers = Variable.New<int>().Named("nUsers");
      Range u = new Range(nUsers);

      appeal = new Variable<double>[nRanks];
      relevance = new Variable<double>[nRanks];
      examine = new VariableArray<bool>[nRanks];
      click = new VariableArray<bool>[nRanks];
      isRel = new VariableArray<bool>[nRanks];

      // user independent variables
      for (int d = 0; d < nRanks; d++)
      {
        appeal[d] = Variable.Beta(1, 1);
        relevance[d] = Variable.Beta(1, 1);
      }

      // Main model code
      for (int d = 0; d < nRanks; d++)
      {
        examine[d] = Variable.Array<bool>(u);
        click[d] = Variable.Array<bool>(u);
        isRel[d] = Variable.Array<bool>(u);
        if (d == 0)
          examine[d][u] = Variable.Bernoulli(1).ForEach(u);
        else
          using (Variable.ForEach(u))
          {
            var nextIfClick = Variable.New<bool>();
            using (Variable.If(isRel[d - 1][u]))
              nextIfClick.SetTo(Variable.Bernoulli(probNextIfClickRel));
            using (Variable.IfNot(isRel[d - 1][u]))
              nextIfClick.SetTo(Variable.Bernoulli(probNextIfClickNotRel));
            var nextIfNotClick = Variable.Bernoulli(probNextIfNotClick);
            var next = 
              (((!click[d - 1][u]) & nextIfNotClick) | (click[d - 1][u] & nextIfClick));
            examine[d][u] = examine[d - 1][u] & next;
          }

        using (Variable.ForEach(u))
        {
          click[d][u] = examine[d][u] & Variable.Bernoulli(appeal[d]);
          isRel[d][u] = click[d][u] & Variable.Bernoulli(relevance[d]);
        }
      }

      ie = new InferenceEngine();
    }

    public DocumentStatistics[] performInference(UserData user)
    {
      ie.NumberOfIterations = user.nIters;
      ie.ShowProgress = false;
      DocumentStatistics[] docStats = new DocumentStatistics[nRanks];
      for (int i = 0; i < nRanks; i++)
      {
        docStats[i] = new DocumentStatistics();
      }

      nUsers.ObservedValue = user.nUsers;
      probNextIfNotClick.ObservedValue = user.probExamine[0];
      probNextIfClickNotRel.ObservedValue = user.probExamine[1];
      probNextIfClickRel.ObservedValue = user.probExamine[2];

      for (int d = 0; d < nRanks; d++)
        click[d].ObservedValue = user.clicks[d];

      try
      {
        for (int d = 0; d < nRanks; d++)
        {
          docStats[d].inferredRelevance = ie.Infer<Beta>(relevance[d]);
          docStats[d].inferredAppeal = ie.Infer<Beta>(appeal[d]);
        }

        return docStats;
      }
      catch (Exception e)
      {
        Console.WriteLine(e);
        return null;
      }
    }
  }
}
