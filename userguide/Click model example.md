---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

Page 1 \| [Page 2](Click Model 1.md) \| [Page 3](Click model 2.md) \| [Page 4](Click Model Prediction.md)

## Click model example

This example is from the field of Information Retrieval and builds a model which reconciles document click counts and human relevance judgements of documents. You can run the code in this tutorial either using the [**Examples Browser**]() or by opening the Tutorials solution in Visual Studio and executing **ClickModel.cs**.

#### The problem

When a user submits a query to an information retrieval system, the search engine returns a list of document hyperlinks to the user, along with a title and query-related snippet extracted from the document. The user looks at the list, and based on title and snippet, decides whether to click on a document in the list or whether to pass over it. These decisions are recorded in click logs, and the decision of a user to click or not click on a document in the list gives a small indication - an individual datum - as to whether the document is relevant or not.

The relevance of a document to a given query can also be determined by human judgements. Typically these judgements are in the form of a set of labels with associated numeric values; for example 'Not Relevant' (0), 'Possibly Relevant' (1), and 'Relevant' (2). Building a successful search engine typically requires the collection of many human relevance judgements to train and validate a document ranking system. These human judgements are much more expensive to collect than click log data.

The problem that is addressed in this example is to reconcile the two different types of relevance data via a relevance 'score' variable which captures the correlations between the two. Sometimes the human judgement and the click data will disagree - for example, a document judged to be highly relevant is never clicked on. The model naturally learns the uncertainties in relevance based on these discrepancies. When deployed, and in the absence of human judgements, the model can predict a relevance score along with its associated uncertainty. The uses of this model are (a) to identify possible bad human judgements which can then be discarded or re-examined, and/or (b) to provide cleaned and more extensive data from which to train a ranking model by using the relevance scores as target data.

This example problem illustrates the following aspects of Infer.NET:

*   Gaussian, Gamma, and Beta Distributions
*   Ranges
*   Constrain.Between and Constrain.EqualRandom constraints
*   Sharing variables between models
*   Controlling convergence
*   Deploying a model

#### The data

The data for this example can be found in the [src\\Tutorials\\TutorialData folder](https://github.com/dotnet/infer/tree/master/src/Tutorials/TutorialData). The data file ("ClickData.txt") is a tab-delimited text file with three columns. Each row after the header row represent three data fields for a particular query/document pair. The first field is the human relevance judgement (always having a value of 0, 1, or 2), the second field is the number of times the document was clicked for the query, and the third field is the number of times the document was examined (i.e. it or something later in the list was clicked). The file contains data for 4604 query/document pairs. The first few lines are shown here:

 ```
Judgement #Clicks   #Exams
1         0         0
1         0         0
2         0         0
2         24        34
1         0         0
0         0         0
2         0         0
0         0         0
2         0         0
0         0         0
2         0         2
1         39        75
1         39        75
0         0         0
2         0         0
0         2         4
0         0         0
1         0         0
1         8         17
```

The human judgement in each row always provides a bit of evidence. However, note that a large majority of the rows have no examinations and therefore provide no supporting evidence from clicks. We might want to just include data records which include examinations because we are primarily interested in the correlation between the two sources of relevance information (there are 522 such rows). On the other hand, the human relevance judgements on their own provide information about the distribution of human judgements. When designing the model, we can defer a decision on which data we show to the model as long as we make the data an observed variable rather than a constant. The C# code below provides a function **LoadData** which supports both options. We will choose to skip over data which has non-zero examinations.

```csharp
// Method to read click data. This assumes a header row  
// followed by data rows with tab or comma separated text  
static private void LoadData( string ifn, // The file name  
    bool allowNoExams, // Allow records with no examinations  
    out int[] labels, // Labels  
    out int[] clicks, // Clicks  
    out int[] exams) // Examinations
```

#### **The model**

We will build two models to solve this problem. These models are the same except that the second one uses shared variables. The models are described in [Model 1](Click Model 1.md) and [Model 2](Click model 2.md) (which assumes that you have read the documentation for model 1). The two models should give identical results provided the inference converges.

<br/>
Page 1 \| [Page 2](Click Model 1.md) \| [Page 3](Click model 2.md) \| [Page 4](Click Model Prediction.md)
