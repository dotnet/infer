// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

import java.io.*;
import java.util.*;

import org.apache.mahout.common.*;
import org.apache.mahout.cf.taste.common.*;
import org.apache.mahout.cf.taste.recommender.*;
import org.apache.mahout.cf.taste.model.*;
import org.apache.mahout.cf.taste.neighborhood.*;
import org.apache.mahout.cf.taste.similarity.*;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.*;
import org.apache.mahout.cf.taste.impl.recommender.*;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.*;
import org.apache.mahout.cf.taste.impl.recommender.svd.*;
import org.apache.mahout.cf.taste.impl.similarity.*;

public class MahoutRunner {
    private static Random random = new Random(239017);
    
    public static void main(String args[]) {
        try {
            if (args.length == 0) {
                reportUsageError();
            }
            
            String mode = args[0];
			if (mode.compareToIgnoreCase("PredictRatings_UserBased") == 0) {
				predictRatingsUserBasedMode(args);
			} else if (mode.compareToIgnoreCase("PredictRatings_ItemBased") == 0) {
				predictRatingsItemBasedMode(args);
			} else if (mode.compareToIgnoreCase("PredictRatings_SlopeOne") == 0) {
				predictRatingsSlopeOneMode(args);
			} else if (mode.compareToIgnoreCase("PredictRatings_Svd") == 0) {
				predictRatingsSvd(args);
			} else if (mode.compareToIgnoreCase("FindRelatedUsers") == 0) {
                findRelatedUsersMode(args);
            } else if (mode.compareToIgnoreCase("FindRelatedItems") == 0) {
                findRelatedItemsMode(args);
            } else {
                reportUsageError();
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
    
	private static void predictRatingsUserBasedMode(String args[]) throws Exception {
        if (args.length != 6) {
            reportUsageError();
        }
		
		String trainingDatasetFile = args[1];
        String testDatasetFile = args[2];
        String predictionsFile = args[3];
        String similarityFunc = args[4];
		int userNeighborhoodSize = Integer.parseInt(args[5]);
		
		// Load the training data
        FileDataModel model =  new FileDataModel(new File(trainingDatasetFile));
        
        // Train the recommender
        UserSimilarity similarity = (UserSimilarity) similarityFromString(similarityFunc, model);
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(userNeighborhoodSize, similarity, model);
        GenericUserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
		
		// Generate the predictions
        predictRatingsModeImpl(testDatasetFile, predictionsFile, recommender);
	}
	
	private static void predictRatingsItemBasedMode(String args[]) throws Exception {
        if (args.length != 5) {
            reportUsageError();
        }
		
		String trainingDatasetFile = args[1];
        String testDatasetFile = args[2];
        String predictionsFile = args[3];
        String similarityFunc = args[4];
				
		// Load the training data
        FileDataModel model =  new FileDataModel(new File(trainingDatasetFile));
        
        // Train the recommender
        ItemSimilarity similarity = (ItemSimilarity) similarityFromString(similarityFunc, model);
        GenericItemBasedRecommender recommender = new GenericItemBasedRecommender(model, similarity);
		
		// Generate the predictions
        predictRatingsModeImpl(testDatasetFile, predictionsFile, recommender);
	}
	
	private static void predictRatingsSlopeOneMode(String args[]) throws Exception {
        if (args.length != 4) {
            reportUsageError();
        }
		
		String trainingDatasetFile = args[1];
        String testDatasetFile = args[2];
        String predictionsFile = args[3];
        				
		// Load the training data
        FileDataModel model =  new FileDataModel(new File(trainingDatasetFile));
        
        // Train the recommender
        SlopeOneRecommender recommender = new SlopeOneRecommender(model);
		
		// Generate the predictions
        predictRatingsModeImpl(testDatasetFile, predictionsFile, recommender);
	}
	
	private static void predictRatingsSvd(String args[]) throws Exception {
        if (args.length != 6) {
            reportUsageError();
        }
		
		String trainingDatasetFile = args[1];
        String testDatasetFile = args[2];
        String predictionsFile = args[3];
		int traitCount = Integer.parseInt(args[4]);
		int iterationCount = Integer.parseInt(args[5]);
        				
		// Load the training data
        FileDataModel model =  new FileDataModel(new File(trainingDatasetFile));

		// Create a factorizer
		Factorizer factorizer = new RatingSGDFactorizer(model, traitCount, iterationCount); // SVDPlusPlusFactorizer
        
        // Train the recommender
        SVDRecommender recommender = new SVDRecommender(model, factorizer);
		
		// Generate the predictions
        predictRatingsModeImpl(testDatasetFile, predictionsFile, recommender);
	}

    private static void findRelatedUsersMode(String args[]) throws Exception {
        if (args.length != 6) {
            reportUsageError();
        }
        
        String trainingDatasetFile = args[1];
        String testDatasetFile = args[2];
        String predictionsFile = args[3];
        int maxRelatedUserCount = Integer.parseInt(args[4]);
        String similarityFunc = args[5];
                        
        // Load the training data
        FileDataModel model =  new FileDataModel(new File(trainingDatasetFile));
        
        // Train the recommender
        UserSimilarity similarity = (UserSimilarity) similarityFromString(similarityFunc, model);
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(1, similarity, model);  // Neighborhood size is irrelevant for related user search
        GenericUserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
        
        // Generate the predictions
        findRelatedEntitiesModeImpl(testDatasetFile, predictionsFile, new RelatedUserFinder(recommender), maxRelatedUserCount);
    }
    
    private static void findRelatedItemsMode(String args[]) throws Exception {
        if (args.length != 6) {
            reportUsageError();
        }
        
        String trainingDatasetFile = args[1];
        String testDatasetFile = args[2];
        String predictionsFile = args[3];
        int maxRelatedItemCount = Integer.parseInt(args[4]);
        String similarityFunc = args[5];
                                
        // Load the training data
        FileDataModel model =  new FileDataModel(new File(trainingDatasetFile));
        
        // Train the recommender
        ItemSimilarity similarity = (ItemSimilarity) similarityFromString(similarityFunc, model);
        GenericItemBasedRecommender recommender = new GenericItemBasedRecommender(model, similarity);
                
        // Generate the predictions
        findRelatedEntitiesModeImpl(testDatasetFile, predictionsFile, new RelatedItemFinder(recommender), maxRelatedItemCount);
    }
    
	private static void predictRatingsModeImpl(String testDatasetFile, String predictionsFile, Recommender recommender) throws Exception {
		Scanner lineScanner = new Scanner(new File(testDatasetFile));
        PrintStream predictionsPrintStream = new PrintStream(new FileOutputStream(predictionsFile)); 
        while (lineScanner.hasNextLine()) {
            // User id is followed by the item id
            String line = lineScanner.nextLine();
            String[] lineParts = line.split(",");
            long userId = Long.parseLong(lineParts[0]);
			long itemId = Long.parseLong(lineParts[1]);
                        
            // Make the prediction
            float preference = recommender.estimatePreference(userId, itemId);
                        
            // Write the prediction to the output file
            predictionsPrintStream.format("%d,%d,%f", userId, itemId, preference);
            predictionsPrintStream.println();
        }
        
        predictionsPrintStream.close();
	}
	
    private static void findRelatedEntitiesModeImpl(String testDatasetFile, String predictionsFile, RelatedEntityFinder relatedEntityFinder, int maxRelatedEntityCount) throws Exception {
        Scanner lineScanner = new Scanner(new File(testDatasetFile));
        PrintStream predictionsPrintStream = new PrintStream(new FileOutputStream(predictionsFile)); 
        while (lineScanner.hasNextLine()) {
            // Query entity id is followed by the ids of the entities that can be predicted as related
            String line = lineScanner.nextLine();
            String[] lineParts = line.split(",");
            long queryEntityId = Long.parseLong(lineParts[0]);
            HashSet<Long> allowedRelatedEntityIds = new HashSet<Long>();
            for (int i = 1; i < lineParts.length; ++i) {
                allowedRelatedEntityIds.add(Long.parseLong(lineParts[i]));
            }
            
            // Make the predictions
            long[] relatedEntityIds = relatedEntityFinder.findRelatedEntities(queryEntityId, allowedRelatedEntityIds, maxRelatedEntityCount);
            
            // If the prediction list is incomplete, we will add some random predictions from the allowed id list.
            // Without such additional random predictions other recommenders will have an NDCG advantage.
            int requiredRelatedEntityListSize = Math.min(maxRelatedEntityCount, allowedRelatedEntityIds.size());
            if (relatedEntityIds.length < requiredRelatedEntityListSize) {
                long[] allowedRelatedEntityIdsArray = setToArray(allowedRelatedEntityIds);
                permute(allowedRelatedEntityIdsArray);
                HashSet<Long> relatedEntityIdSet = arrayToSet(relatedEntityIds);
                int i = 0;
                while (relatedEntityIdSet.size() < requiredRelatedEntityListSize) {
                    if (!relatedEntityIdSet.contains(allowedRelatedEntityIdsArray[i])) {
                        relatedEntityIdSet.add(allowedRelatedEntityIdsArray[i]);
                    }
                    ++i;
                }
                relatedEntityIds = setToArray(relatedEntityIdSet);
            }
            assert relatedEntityIds.length == requiredRelatedEntityListSize;
            
            // Write the predictions to the output file
            predictionsPrintStream.format("%d", queryEntityId);
            for (int i = 0; i < relatedEntityIds.length; ++i) {
                predictionsPrintStream.format(",%d", relatedEntityIds[i]);
            }
            predictionsPrintStream.println();
        }
        
        predictionsPrintStream.close();
    }
    
    private static Object similarityFromString(String str, DataModel model) throws Exception {
        if (str.compareToIgnoreCase("Euclidean") == 0) {
            return new EuclideanDistanceSimilarity(model);
        } else if (str.compareToIgnoreCase("Manhattan") == 0) {
            return new CityBlockSimilarity(model);
        } else if (str.compareToIgnoreCase("PearsonCorrelation") == 0) {
            return new PearsonCorrelationSimilarity(model);
        } else {
            throw new Exception(String.format("Unknown similarity function: %s", str));
        }
    }
    
    private static void permute(long[] array) {
        for (int i = 0; i < array.length; ++i) {
            int swapIndex = i + random.nextInt(array.length - i);
            long temp = array[swapIndex];
            array[swapIndex] = array[i];
            array[i] = temp;
        }
    }
    
    private static long[] setToArray(HashSet<Long> set) {
        long[] result = new long[set.size()];
        int index = 0;
        for (Long l : set) {
            result[index++] = l;
        }
        return result;
    }
    
    private static HashSet<Long> arrayToSet(long[] array) {
        HashSet<Long> result = new HashSet<Long>();
        for (int i = 0; i < array.length; ++i) {
            result.add(array[i]);
        }
        return result;
    }
    
    private static void reportUsageError() {
        System.err.println("Usage error. Valid usage examples:");
		System.err.println("  runMahout PredictRatings_UserBased <train_dataset> <test_dataset> <predictions> <similarity> <user_neighborhood_size>");
		System.err.println("  runMahout PredictRatings_ItemBased <train_dataset> <test_dataset> <predictions> <similarity>");
		System.err.println("  runMahout PredictRatings_SlopeOne <train_dataset> <test_dataset> <predictions> <similarity>");
        System.err.println("  runMahout FindRelatedUsers <train_dataset> <test_dataset> <predictions> <max_related_users> <similarity>");
        System.err.println("  runMahout FindRelatedItems <train_dataset> <test_dataset> <predictions> <max_related_items> <similarity>");
        System.err.println("  Supported similarity functions: Euclidean, Manhattan, PearsonCorrelation.");
        System.exit(1);
    }
    
    private static class EntitySkipper implements Rescorer<LongPair> {
        private HashSet<Long> allowedIds;
        
        public EntitySkipper(HashSet<Long> allowedIds) {
            this.allowedIds = allowedIds;
        }
        
        public boolean isFiltered(LongPair idPair) {
            return !this.allowedIds.contains(idPair.getSecond());
        }
        
        public double rescore(LongPair idPair, double originalScore) {
            return originalScore;
        }
    }
    
    private interface RelatedEntityFinder {
        public long[] findRelatedEntities(long queryEntity, HashSet<Long> allowedResultEntities, int maxRelatedEntityCount) throws Exception;
    }
    
    private static class RelatedUserFinder implements RelatedEntityFinder {
        private GenericUserBasedRecommender recommender;
                
        public RelatedUserFinder(GenericUserBasedRecommender recommender) {
            this.recommender = recommender;
        }
        
        public long[] findRelatedEntities(long queryEntity, HashSet<Long> allowedResultEntities, int maxRelatedEntityCount) throws Exception {
            return this.recommender.mostSimilarUserIDs(queryEntity, maxRelatedEntityCount, new EntitySkipper(allowedResultEntities));
        }
    }
    
    private static class RelatedItemFinder implements RelatedEntityFinder {
        private GenericItemBasedRecommender recommender;
                
        public RelatedItemFinder(GenericItemBasedRecommender recommender) {
            this.recommender = recommender;
        }
        
        public long[] findRelatedEntities(long queryEntity, HashSet<Long> allowedResultEntities, int maxRelatedEntityCount) throws Exception {
            List<RecommendedItem> similarItems = this.recommender.mostSimilarItems(queryEntity, maxRelatedEntityCount, new EntitySkipper(allowedResultEntities));
            long result[] = new long[similarItems.size()];
            for (int i = 0; i < similarItems.size(); ++i) {
                result[i] = similarItems.get(i).getItemID();
            }
            
            return result;
        }
    }
}
