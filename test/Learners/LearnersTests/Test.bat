:: Licensed to the .NET Foundation under one or more agreements.
:: The .NET Foundation licenses this file to you under the MIT license.
:: See the LICENSE file in the project root for more information.
@echo on

set CONFIGURATION=Debug
set TARGET_PLATFORM=net472
set RUN_DIR=bin\%CONFIGURATION%\%TARGET_PLATFORM%
set RUNNER=%RUN_DIR%\Learner.exe Recommender
set DATA=..\Runners\Evaluator\Data\Recommendation\MovieLens.dat
set DATA_TRAIN=%RUN_DIR%\Train.dat
set DATA_TEST=%RUN_DIR%\Test.dat
set MODEL=%RUN_DIR%\Model.dat
set PREDICTIONS=%RUN_DIR%\Predictions.dat
set REPORT=%RUN_DIR%\Predictions.dat

%RUNNER% SplitData --input-data %DATA% --output-data-train %DATA_TRAIN% --output-data-test %DATA_TEST% --ignored-users 0.8 --training-users 0.1
%RUNNER% Train --training-data %DATA_TRAIN% --trained-model %MODEL% --traits 4 --iterations 20 --batches 1 --use-user-features --use-item-features

%RUNNER% PredictRatings --data %DATA_TEST% --model %MODEL% --predictions %PREDICTIONS%
%RUNNER% EvaluateRatingPrediction --test-data %DATA_TEST% --predictions %PREDICTIONS% --report %REPORT%

%RUNNER% RecommendItems --data %DATA_TEST% --model %MODEL% --predictions %PREDICTIONS%
%RUNNER% EvaluateItemRecommendation --test-data %DATA_TEST% --predictions %PREDICTIONS% --report %REPORT%

%RUNNER% FindRelatedUsers --data %DATA_TEST% --model %MODEL% --predictions %PREDICTIONS%
%RUNNER% EvaluateFindRelatedUsers --test-data %DATA_TEST% --predictions %PREDICTIONS% --report %REPORT%

%RUNNER% FindRelatedItems --data %DATA_TEST% --model %MODEL% --predictions %PREDICTIONS%
%RUNNER% EvaluateFindRelatedItems --test-data %DATA_TEST% --predictions %PREDICTIONS% --report %REPORT%