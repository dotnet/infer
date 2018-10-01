---
layout: default
---

[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md) : [Conference reviewer model](Calibrating reviews of conference submissions.md)

## Reviewer model

```csharp
// (C) Copyright 2009-2010 Microsoft Research Cambridge
// Authors: John Guiver and Ralf Herbrich
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace MicrosoftResearch.Conference
{
  /// <summary>
  /// This class contains the model and inference code for reviewer calibration.
  /// </summary>
  public class ReviewerCalibration
  {
    /// <summary>
    /// Shape parameter for prior of expertise precisions.
    /// </summary>
    public double k_e { get; set; }
    /// <summary>
    /// Rate parameter for prior of expertise precisions
    /// </summary>
    public double beta_e { get; set; }
    /// <summary>
    /// Shape parameter for prior of accuracy.
    /// </summary>
    public double k_a { get; set; }
    /// <summary>
    /// Prior variance for reviewer-dependent precisions. Avoid setting this too low relative
    /// to mean - otherwise you may get improper message exceptions.
    /// </summary>
    public double beta_a { get; set; }
    /// <summary>
    /// Default constructor.
    /// </summary>
    public ReviewerCalibration()
    {
        k_e = 10.0;
        beta_e = 10.0;
        k_a = 10.0;
        beta_a = 10.0;
    }
    /// <summary>
    /// Builds and runs the reviewer calibration model.
    /// </summary>
    /// <param name="reviews">Stream of reviews.</param>
    /// <returns>Results of running Expectation Propagation on the reviewer/submission graph.</returns>
    public ReviewerCalibrationResults Run(IEnumerable<Review> reviews)
    {
      Recommendation[] recommendationLevels =
        (Recommendation[])Enum.GetValues(typeof(Recommendation));
      Expertise[] expertiseLevels = (Expertise[])Enum.GetValues(typeof(Expertise));
      // Number of thresholds is one less than the number of recommendation levels
      int T = recommendationLevels.Length - 1; // Number of thresholds
      // Nominal thresholds
      double[] th0 = new double[T];
      for (int i = 0; i < T; i++) th0[i] = 1.5 + (double)i;
      // Mean/precision for quality prior
      double m_q = (0.5 * T) + 1.0; double p_q = 1.0;
      // Get distinct submissions and reviewers
      Submission[] submissions = (from rev in reviews select rev.Submission).Distinct().ToArray();
      Reviewer[] reviewers = (from rev in reviews select rev.Reviewer).Distinct().ToArray();
      // Build a dictionary which maps submission to index for model
      var submissionToIndex = Utility.ArrayToDictionary(submissions);
      // Build a dictionary which maps reviewer to index for model
       var reviewerToIndex = Utility.ArrayToDictionary(reviewers);
      // Ranges for the model
      int R = reviews.Count();
      int S = submissions.Length;
      int E = expertiseLevels.Length;
      int J = reviewers.Length;
      Range s = new Range(S);
      Range j = new Range(J);
      Range r = new Range(R);
      Range e = new Range(E);
      Range t = new Range(T);
      // Observations - convert from recommendation to an array of bool
      bool[][] obs = new bool[T][];
      for (int i=0; i < T; i++)
          obs[i] = (from rev in reviews select (i < (int)rev.Recommendation - 1)).ToArray();
      // Constant variable arrays
      var sOf = Variable.Observed(
        (from rev in reviews select submissionToIndex[rev.Submission]).ToArray(), r);
      var jOf = Variable.Observed(
        (from rev in reviews select reviewerToIndex[rev.Reviewer]).ToArray(), r);
      var eOf = Variable.Observed(
        (from rev in reviews select ((int)rev.Expertise - 1)).ToArray(), r);
      var theta0 = Variable.Constant<double>(th0, t);
      var observation = Variable.Array(Variable.Array<bool>(r), t);
      observation.ObservedValue = obs;
      // Declarations
      var quality = Variable.Array<double>(s);
      var expertise = Variable.Array<double>(e);
      var score = Variable.Array<double>(r);
      var accuracy = Variable.Array<double>(j);
      var theta = Variable.Array(Variable.Array<double>(t), j);
      // The model
      quality[s] = Variable.GaussianFromMeanAndPrecision(m_q, p_q).ForEach(s);
      expertise[e] = Variable.GammaFromShapeAndRate(k_e, beta_e).ForEach(e);
      score[r] = Variable.GaussianFromMeanAndPrecision(quality[sOf[r]], expertise[eOf[r]]);
      accuracy[j] = Variable.GammaFromShapeAndRate(k_a, beta_a).ForEach(j);
      theta[j][t] = Variable.GaussianFromMeanAndPrecision(theta0[t], accuracy[j]);
      observation[t][r] = score[r] > theta[jOf[r]][t];
      // The inference
      var engine = new InferenceEngine();
      engine.NumberOfIterations = 20;
      var qualityPosterior = engine.Infer<Gaussian[]>(quality);
      var expertisePosterior = engine.Infer<Gamma[]>(expertise);
      var accuracyPosterior = engine.Infer<Gamma[]>(accuracy);
      var thetaPosterior = engine.Infer<Gaussian[][]>(theta);
      return (new ReviewerCalibrationResults(
        Utility.KeyValueArraysToDictionary(submissions, qualityPosterior),
        Utility.KeyValueArraysToDictionary(reviewers, thetaPosterior),
        Utility.KeyValueArraysToDictionary(expertiseLevels, expertisePosterior),
        Utility.KeyValueArraysToDictionary(reviewers, accuracyPosterior)));
    }
  }

  /// <summary>
  /// A class which holds the results of running the reviewer calibration model.
  /// </summary>
  public class ReviewerCalibrationResults
  {
    /// <summary>
    /// The quality score of each submission.
    /// </summary>
    public IDictionary<Submission, Gaussian> Quality { get; private set; }
    /// <summary>
    /// The thresholds of all reviewers.
    /// </summary>
    public IDictionary<Reviewer, Gaussian[]> Thresholds { get; private set; }
    /// <summary>
    /// The precisions of the expertise level.
    /// </summary>
    public IDictionary<Expertise, Gamma> ExpertPrecision { get; private set; }
    /// <summary>
    /// The accuracy of each reviewer.
    /// </summary>
    public IDictionary<Reviewer, Gamma> Accuracy { get; private set; }
    /// <summary>
    /// Non-default constructor.
    /// </summary>
    /// <param name="scores">The quality of each submission.</param>
    /// <param name="thresholds">The thresholds for each reviewers.</param>
    /// <param name="expertPrec">The precisions of the expertise levels.</param>
    /// <param name="reviewerPrec">The accuracy of each reviewer.</param>
    public ReviewerCalibrationResults(
      IDictionary<Submission, Gaussian> quality,
      IDictionary<Reviewer, Gaussian[]> thresholds,
      IDictionary<Expertise, Gamma> expertPrec,
      IDictionary<Reviewer, Gamma> accuracy)
    {
      Quality = quality;
      Thresholds = thresholds;
      ExpertPrecision = expertPrec;
      Accuracy = accuracy;
    }
 }

/// <summary>
/// This class stores all the information about a reviewer.
/// </summary>
public class Reviewer
{
  /// <summary>
  /// The name of the reviewer.
  /// </summary>
  public string Name { get; private set; }
  /// <summary>
  /// Non-default constructor based on a unique reviewer index.
  /// </summary>
  /// <param name="index">Internal index of the reviewer.</param>
  public Reviewer(int index)
  {
    Name = "Reviewer " + index;
  }
}
/// <summary>
/// This class stores all the information about a submission.
/// </summary>
public class Submission
{
  /// <summary>
  /// Title of the submission.
  /// </summary>
  public string Title { get; private set; }
  /// <summary>
  /// Non-default constructor based on a unique submission index.
  /// </summary>
  /// <param name="index">Internal index of the submission.</param>
  public Submission(int index)
  {
    Title = "Submission " + index;
  }
}
/// <summary>
/// The type of recommendations that a reviewer can give for a submission.
/// </summary>
public enum Recommendation
{
  StrongReject=1,
  Reject=2,
  WeakReject=3,
  WeakAccept=4,
  Accept=5,
  StrongAccept=6
}
/// <summary>
/// The type of expertise levels of a reviewer in their review.
/// </summary>
public enum Expertise
{
  InformedOutsider=1,
  Knowledgeable=2,
  Expert=3
}
/// <summary>
/// This class stores all the information about a review.
/// </summary>
public class Review
{
  /// <summary>
  /// Reviewer of a submission.
  /// </summary>
  public Reviewer Reviewer { get; private set; }
  /// <summary>
  /// Submission that is reviewed.
  /// </summary>
  public Submission Submission { get; private set; }
  /// <summary>
  /// Recommendation of the submission by the reviewer - larger means better.
  /// </summary>
  public Recommendation Recommendation { get; private set; }
  /// <summary>
  /// Recommendation of the submission by the reviewer - larger means better.
  /// </summary>
  public Expertise Expertise { get; private set; }
  /// <summary>
  /// Non-default constructor for a review.
  /// </summary>
  /// <param name="reviewer">Reviewer of the submission.</param>
  /// <param name="submission">Submission that is reviewed.</param>
  /// <param name="recommendation">Recommendation of the reviewer for the submission.</param>
  /// <param name="expertise">Expertise of the reviewer for this recommendation.</param>
  public Review(Reviewer reviewer, Submission submission, int recommendation, int expertise)
  {
    Reviewer = reviewer;
    Submission = submission;
    switch (recommendation)
    {
      case 1: Recommendation = Recommendation.StrongReject; break;
      case 2: Recommendation = Recommendation.Reject; break;
      case 3: Recommendation = Recommendation.WeakReject; break;
      case 4: Recommendation = Recommendation.WeakAccept; break;
      case 5: Recommendation = Recommendation.Accept; break;
      case 6: Recommendation = Recommendation.StrongAccept; break;
      default: throw new ArgumentException(
        "recommendation",
        "Recommendations have to be in the range {1,2,3,4,5,6}.");
    }
    switch (expertise)
    {
      case 1: Expertise = Expertise.InformedOutsider; break;
      case 2: Expertise = Expertise.Knowledgeable; break;
      case 3: Expertise = Expertise.Expert; break;
      default: throw new ArgumentException(
        "expertise", 
        "Expertise have to be in the range {1,2,3}.");
    }
  }
  /// <summary>
  /// A class that holds some useful utility functions.
  /// </summary>
  internal class Utility
  {
    /// <summary>
    /// Transforms an array of items into a dictionary that maps an entry to the integer index.
    /// </summary>
    /// <param name="values">Array of values</param>
    /// <returns>A dictionary of lookups from values to integer indices.</returns>
    internal static Dictionary<T, int> ArrayToDictionary<T>(T[] values)
    {
      var dict = new Dictionary<T, int>();
      for (int i = 0; i < values.Length; i++) dict.Add(values[i],i);
      return dict;
    }
    /// <summary>
    /// Transforms an array into a dictionary provided the lookup is known
    /// </summary>
    /// <param name="keys">Array of keys.</param>
    /// <param name="values">Array of values</param>
    /// <returns>A dictionary of lookups from keys to values.</returns>
    internal static Dictionary<K, V> KeyValueArraysToDictionary<K, V>(K[] keys, V[] values)
    {
      if (keys.Length != values.Length) 
        throw new ArgumentException("keys", "Keys and Values should be of same length");
      Dictionary<K, V> dict = new Dictionary<K, V>();
      for (int i = 0; i < keys.Length; i++) dict.Add(keys[i], values[i]);
      return dict;
    }
  }
}
```