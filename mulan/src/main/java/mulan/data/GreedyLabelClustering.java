/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    GreedyLabelClustering.java
 */
package mulan.data;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.meta.SubsetLearner;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.SubsetAccuracy;
import weka.classifiers.Classifier;

/**
 * A class for clustering dependent label pairs into disjoint subsets. <br>
 * <br>
 * The type of the learned dependencies is determined by the
 * {@link mulan.data.LabelPairsDependenceIdentifier} supplied to the class constructor. The
 * clustering process is straightforward: initially all labels are assumed to be independent. Then
 * we start group the label pairs according to their dependence score from most to least dependent.
 * An SubsetLearner is build for each new partition and its accuracy is evaluated in terms of the
 * {@link #measure}. The process of grouping labels continues as long as the accuracy improves (or
 * at least is not reduced). A number of steps specified by {@link #allowedNonImprovementSteps}
 * without seeking any concomitant improvement in the accuracy is allowed. Such a �non-useful�
 * partitions are filtered out and the algorithm continues to evaluate subsequent pairs of dependent
 * labels until one of the stop conditions is reached. The possible stop conditions are: <br>
 * - no more label pairs to consider; <br>
 * - all labels are clustered into one single group; <br>
 * - pair dependence score is below the specified {@link #criticalValue}; <br>
 * - the number of {@link #allowedNonImprovementSteps} is exceeded.
 * 
 * @author Lena Chekina (lenat@bgu.ac.il)
 * @version 05.05.2011
 */
public class GreedyLabelClustering implements LabelClustering, Serializable {
	/** Classifier that will be used for single label training and predictions */
	private Classifier singleLabelLearner;
	/** Classifier that will be used for multi-label training and predictions */
	private MultiLabelLearner multiLabelLearner;
	/** Defines the type of dependence identification process. */
	private LabelPairsDependenceIdentifier depLabelsIdentifier;
	/**
	 * Critical value below which the label pairs are considered independent. If set to 0 - the
	 * critical value returned by
	 * {@link mulan.data.LabelPairsDependenceIdentifier#getCriticalValue()} is used.
	 */
	private double criticalValue = 0;
	/** Number of folds for evaluation of SubsetLearner models */
	private int numFolds = 10;
	/** Number of allowed concurrent steps with reduced accuracy */
	private int allowedNonImprovementSteps = 10;
	/** Measure by which models are compared */
	private Measure measure = new SubsetAccuracy();
	/** Enable SubsetLearner caching mechanism */
	private boolean useSubsetLearnerCache = true;
	/** Enable debug output of the internal SubsetLearner */
	private boolean internalSubsetLearnerDebug = true;

	/**
	 * Initialize the GreedyLabelClustering with multilabel and single label learners and a method
	 * for labels dependence identification.
	 * 
	 * @param aMultiLabelLearner - a learner for multilabel classification
	 * @param aSingleLabelLearner - a learner for single label classification
	 * @param dependenceIdentifier - a method for label pairs dependence identification
	 */
	public GreedyLabelClustering(MultiLabelLearner aMultiLabelLearner,
			Classifier aSingleLabelLearner, LabelPairsDependenceIdentifier dependenceIdentifier) {
		multiLabelLearner = aMultiLabelLearner;
		depLabelsIdentifier = dependenceIdentifier;
		singleLabelLearner = aSingleLabelLearner;
	}

	/**
	 * Determines labels partitioning into dependent sets. It clusters label pairs according to
	 * their dependence score and evaluates the related models. The clustering process continues as
	 * long as the accuracy improves. The finally selected labels partition is returned.
	 * 
	 * @param trainingSet the training data set
	 */
	@Override
	public int[][] determineClusters(MultiLabelInstances trainingSet) {
		HashMap<String, int[][]> evaluatedSubsets = new HashMap<String, int[][]>();
		LabelsPair[] labelPairs;
		int[][] currClusters;
		int[][] newClusters;
		SubsetLearner currClassif;
		Evaluator eval = new Evaluator();
		MultipleEvaluation results;
		Double newAcc;
		Double currAcc;
		if (criticalValue == 0) {
			criticalValue = depLabelsIdentifier.getCriticalValue();
		}
		// compute dependency level between all label pairs
		labelPairs = depLabelsIdentifier.calculateDependence(trainingSet);
		int numLabels = trainingSet.getNumLabels();
		// build initial combination set (each label in a separate group)
		currClusters = buildInitialSet(numLabels);
		String currSubsetsStr = partitionToString(currClusters);
		System.out.println("Evaluating initial model: " + currSubsetsStr);
		currClassif = new SubsetLearner(currClusters, multiLabelLearner, singleLabelLearner);
		currClassif.setDebug(internalSubsetLearnerDebug);
		currClassif.setUseCache(useSubsetLearnerCache);
		// cross-validate initial combination
		results = eval.crossValidate(currClassif, trainingSet, numFolds);
		results.calculateStatistics();
		currAcc = results.getMean(measure.getName());
		System.out.println("Model's " + measure.getName() + " = " + currAcc);
		evaluatedSubsets.put(currSubsetsStr, currClusters);
		int noImprovementCounter = 0;

		// take next labels pair, create new combination and build a model
		for (LabelsPair pair : labelPairs) {
			Double score = pair.getScore();
			if (score < criticalValue) {
				System.out.println("Pairs dependence score: " + score
						+ " is below the criticalValue: " + criticalValue
						+ ". Stop the clustering process!");
				break; // stop the process
			}
			if (noImprovementCounter > allowedNonImprovementSteps) {
				System.out.println("noImprovementCounter: " + noImprovementCounter
						+ " is above the allowed: " + allowedNonImprovementSteps
						+ ". Stop the clustering process!");
				break; // stop the process
			}
			int[] comb = pair.getPair();
			int length = currClusters.length;
			if (length == 1) {
				System.out
						.println("All labels are in the same group. Stop the clustering process!");
				break; // no more combinations possible - stop the process
			}
			// construct new label set partition
			newClusters = buildCombinationSet(currClusters, comb);
			for (int[] newCluster : newClusters) { // sort the labels within each group
				Arrays.sort(newCluster);
			}
			String newSubsetsStr = partitionToString(newClusters);
			if (!evaluatedSubsets.containsKey(newSubsetsStr)) {
				// if was not evaluated already -> build new model and evaluate it
				System.out.println("Evaluating model:" + newSubsetsStr);
				currClassif.resetSubsets(newClusters);
				// cross-validate new model
				results = eval.crossValidate(currClassif, trainingSet, numFolds);
				evaluatedSubsets.put(newSubsetsStr, newClusters);
				results.calculateStatistics();
				newAcc = results.getMean(measure.getName());
				System.out.println("Model's " + measure.getName() + " = " + newAcc);
				if (newAcc >= currAcc) { // make a decision
					currClusters = newClusters; // accept the new partition
					currAcc = newAcc;
					noImprovementCounter = 0; // and reset the counter
				} else {
					noImprovementCounter++;
				}
			}
		}
		System.out.println("Returning  the final labels partition: "
				+ partitionToString(currClusters) + '\n');
		return currClusters;
	}

	/**
	 * Returns a string representation of the labels partition.
	 * 
	 * @param partition - a label set partition
	 * @return a string representation of the labels partition
	 */
	public static String partitionToString(int[][] partition) {
		StringBuilder result = new StringBuilder();
		for (int[] aGroup : partition) {
			result.append(Arrays.toString(aGroup));
			result.append(", ");
		}
		return result.toString();
	}

	/**
	 * Build initial label set partition - each label in a separate group. For example for
	 * numLabels=4 , it returns an array {{0},{1},{2},{3}}
	 * 
	 * @param numLabels number of labels in the trainingSet
	 * @return two dimensional array of size numLabels, when each inner array is of size 1
	 */
	private static int[][] buildInitialSet(int numLabels) {
		int[][] res = new int[numLabels][1];
		for (int i = 0; i < numLabels; i++) {
			res[i][0] = i;
		}
		return res;
	}

	/**
	 * Clusters a new pair of labels and integrates the new group into the given labels partition.
	 * 
	 * @param partition - label set partition
	 * @param pair - labels pair
	 * @return a new partition with clustered labels of the pair
	 */
	private static int[][] buildCombinationSet(int[][] partition, int[] pair) {
		int[][] newClusters = new int[partition.length - 1][];
		int[][] tmpClusters = new int[partition.length][];
		int i1 = -1;
		int i2 = -1;
		for (int i = 0; i < partition.length; i++) { // identify indexes of pair values in the
			// partition
			for (int j = 0; j < partition[i].length; j++) {
				if (partition[i][j] == pair[0]) {
					i1 = i;
				}
				if (partition[i][j] == pair[1]) {
					i2 = i;
				}
			}
		}
		if (i1 == i2) // if both labels already in the same set -> there is no change
			return partition;
		for (int k = 0; k < partition.length; k++) { // copy unchanged sets and unify sets with
			// values from pair
			if (i1 > i2) { // ensure that i1 is index of first occurrence of one of the values from
				// pair
				int temp = i1;
				i1 = i2;
				i2 = temp;
			}
			if (k != i1) { // if set's values not in pair -> copy as is
				tmpClusters[k] = partition[k];
			} else { // set new set to be a union of two previous sets
				tmpClusters[k] = new int[partition[i1].length + partition[i2].length];
				int m;
				for (m = 0; m < partition[i1].length; m++) {
					tmpClusters[k][m] = partition[i1][m];
				}
				int n;
				for (n = 0; n < partition[i2].length; n++) {
					tmpClusters[k][m + n] = partition[i2][n];
				}
			}
		}
		// delete the set which labels were added to another set:
		System.arraycopy(tmpClusters, 0, newClusters, 0, i2);
		// move all sets appearing after eliminated set into one index smaller
		System.arraycopy(tmpClusters, i2 + 1, newClusters, i2, newClusters.length - i2);
		return newClusters;
	}

        /**
         * 
         * @return Number of folds
         */
        public int getNumFolds() {
		return numFolds;
	}

        /**
         * 
         * @param numFolds Number of folds
         */
        public void setNumFolds(int numFolds) {
		this.numFolds = numFolds;
	}

        /**
         * 
         * @return Measure by which models are compared
         */
        public Measure getMeasure() {
		return measure;
	}

        /**
         * 
         * @param measure the measure by which models are compared
         */
        public void setMeasure(Measure measure) {
		this.measure = measure;
	}

        /**
         * 
         * @return Number of allowed concurrent steps with reduced accuracy
         */
        public int getAllowedNonImprovementSteps() {
		return allowedNonImprovementSteps;
	}

        /**
         * 
         * @param allowedNonImprovementSteps the number of allowed concurrent steps with reduced accuracy
         */
        public void setAllowedNonImprovementSteps(int allowedNonImprovementSteps) {
		this.allowedNonImprovementSteps = allowedNonImprovementSteps;
	}

        /**
         * 
         * @return Critical value below which the label pairs are considered independent
         */
        public double getCriticalValue() {
		return criticalValue;
	}

        /**
         * 
         * @param criticalValue Critical value below which the label pairs are considered independent
         */
        public void setCriticalValue(double criticalValue) {
		this.criticalValue = criticalValue;
	}

        /**
         * 
         * @return Classifier for single label training and predictions
         */
        public Classifier getSingleLabelLearner() {
		return singleLabelLearner;
	}

        /**
         * 
         * @return Classifier for multi label training and predictions
         */
        public MultiLabelLearner getMultiLabelLearner() {
		return multiLabelLearner;
	}

        /**
         * 
         * @return If SubsetLearner caching mechanism is enabled
         */
        public boolean isUseSubsetLearnerCache() {
		return useSubsetLearnerCache;
	}

        /**
         * 
         * @param useSubsetLearnerCache Whether SubsetLearner caching mechanism is enabled or not
         */
        public void setUseSubsetLearnerCache(boolean useSubsetLearnerCache) {
		this.useSubsetLearnerCache = useSubsetLearnerCache;
	}

        /**
         * 
         * @return If debug output of the internal SubsetLearner is enabled
         */
        public boolean isInternalSubsetLearnerDebug() {
		return internalSubsetLearnerDebug;
	}

        /**
         * 
         * @param internalSubsetLearnerDebug whether to enable debug output of the internal SubsetLearner or not
         */
        public void setInternalSubsetLearnerDebug(boolean internalSubsetLearnerDebug) {
		this.internalSubsetLearnerDebug = internalSubsetLearnerDebug;
	}
}