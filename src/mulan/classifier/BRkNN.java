package mulan.classifier;

import java.util.ArrayList;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Simple BR implementation of the KNN algorithm
 * <!-- globalinfo-start -->
 * 
 * <pre>
 * Class implementing the base BRkNN algorithm and its 2 extensions BRkNN-a and BRkNN-b.
 * </pre>
 * 
 * For more information:
 * 
 * <pre>
 * E. Spyromitros, G. Tsoumakas, I. Vlahavas, “An Empirical Study of Lazy Multilabel Classification Algorithms”,
 * Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008), Springer, Syros, Greece, 2008.
 * http://mlkd.csd.auth.gr/multilabel.html
 * </pre>
 * 
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <p/> <!-- technical-bibtex-end -->
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * 
 */
@SuppressWarnings("serial")
public class BRkNN extends MultiLabelKNN {

	int avgPredictedLabels;
	
	protected Random Rand;

	protected int selectedMethod;

	public static final int BR = 0;

	public static final int BRexta = 2;

	public static final int BRextb = 3;

	
	/**
	 * The default constructor. (The base algorithm)
	 * @param numLabels
	 * @param numOfNeighbors
	 */
	public BRkNN(int numLabels, int numOfNeighbors) {
		super(numLabels, numOfNeighbors);
		distanceWeighting = WEIGHT_NONE; //weight none
		selectedMethod = BR; //the default method 
		Rand = new Random(1);
	}

	/**
	 * Constructor giving the option to select an extension of the base version
	 * 
	 * @param numLabels
	 * @param numOfNeighbors
	 * @param method			2 for BRkNN-a 3 for BRkNN-b		
	 */
	public BRkNN(int numLabels, int numOfNeighbors, int method) {
		super(numLabels, numOfNeighbors);
		distanceWeighting = WEIGHT_NONE; //weight none
		selectedMethod = method;
		Rand = new Random(1);
	}

	public void buildClassifier(Instances train) throws Exception {
		super.buildClassifier(train);
	}

	/**
	 * weka Ibk style prediction
	 */
	public Prediction makePrediction(Instance instance) throws Exception {

		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train);
		lnn.setMeasurePerformance(false);

		//in cross-validation test-train instances does not belong to the same data set
		//Instance instance2 = new Instance(instance);

		Instances knn = lnn.kNearestNeighbours(instance, numOfNeighbors);

		double[] distances = lnn.getDistances();
		double[] confidences = getConfidences(knn, distances);
		double[] predictions = null;

		if (selectedMethod == 0) {//BRknn
			predictions = labelsFromConfidences(confidences);
		} else if (selectedMethod == 2) {//BRknn-a 
			predictions = labelsFromConfidences2(confidences);
		} else if (selectedMethod == 3) {//BRknn-b
			predictions = labelsFromConfidences3(confidences);
		}
		Prediction results = new Prediction(predictions, confidences);
		return results;

	}

	private double[] getConfidences(Instances neighbours, double[] distances) {
		double total = 0, weight;
		double neighborLabels = 0;
		double[] confidences = new double[numLabels];

		// Set up a correction to the estimator
		for (int i = 0; i < numLabels; i++) {
			confidences[i] = 1.0 / Math.max(1, train.numInstances());
		}
		total = (double) numLabels / Math.max(1, train.numInstances());

		for (int i = 0; i < neighbours.numInstances(); i++) {
			// Collect class counts
			Instance current = neighbours.instance(i);
			distances[i] = distances[i] * distances[i];
			distances[i] = Math.sqrt(distances[i] / this.predictors);
			switch (distanceWeighting) {
			case WEIGHT_INVERSE:
				weight = 1.0 / (distances[i] + 0.001); // to avoid division by zero
				break;
			case WEIGHT_SIMILARITY:
				weight = 1.0 - distances[i];
				break;
			default: // WEIGHT_NONE:
				weight = 1.0;
				break;
			}
			weight *= current.weight();

			for (int j = 0; j < numLabels; j++) {
				double value = Double.parseDouble(current.attribute(predictors + j).value(
						(int) current.value(predictors + j)));
				if (Utils.eq(value, 1.0)) {
					confidences[j] += weight;
					neighborLabels += weight;
				}
			}
			total += weight;
		}

		avgPredictedLabels = (int) Math.round(neighborLabels / total);
		// Normalise distribution
		if (total > 0) {
			Utils.normalize(confidences, total);
		}
		return confidences;
	}

	/**
	 * old style prediction  (not in use)
	 * 
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public Prediction makePrediction2(Instance instance) throws Exception {
		double[] confidences = new double[numLabels];
		double[] predictions = new double[numLabels];

		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train);
		lnn.setMeasurePerformance(false);

		double[] votes = new double[numLabels];

		//in cross-validation test-train instances does not belong to the same data set
		//Instance instance2 = new Instance(instance);

		Instances knn = new Instances(lnn.kNearestNeighbours(instance, numOfNeighbors));

		for (int i = 0; i < numLabels; i++) {
			int aces = 0; // num of aces in Knn for i
			for (int k = 0; k < numOfNeighbors; k++) {
				double value = Double.parseDouble(train.attribute(predictors + i).value(
						(int) knn.instance(k).value(predictors + i)));
				if (Utils.eq(value, 1.0)) {
					aces++;
				}
			}
			votes[i] = aces;
		}

		for (int i = 0; i < numLabels; i++) {
			confidences[i] = (double) votes[i] / numOfNeighbors;
		}

		predictions = labelsFromConfidences(confidences);

		Prediction results = new Prediction(predictions, confidences);
		return results;
	}

	/**
	 * used for BRknn-a
	 */
	protected double[] labelsFromConfidences2(double[] confidences) {
		double[] result = new double[confidences.length];
		boolean flag = false; //check the case that no label is true

		for (int i = 0; i < result.length; i++) {
			if (confidences[i] >= threshold) {
				result[i] = 1.0;
				flag = true;
			}
		}
		//assign the class with the greater confidence
		if (flag == false) {
			int index = RandomIndexOfMax(confidences,Rand);
			result[index] = 1.0;
		}
		return result;
	}

	/**
	 * used for BRkNN-b (break ties arbitrarily)
	 */
	protected double[] labelsFromConfidences3(double[] confidences) {
		double[] result = new double[numLabels];

		int[] indices = Utils.stableSort(confidences);

		ArrayList<Integer> lastindices = new ArrayList<Integer>();

		int counter = 0;
		int i = numLabels - 1;

		while (i > 0) {
			if (confidences[indices[i]] > confidences[indices[numLabels - avgPredictedLabels]]) {
				result[indices[i]] = 1.0;
				counter++;
			} else if (confidences[indices[i]] == confidences[indices[numLabels
					- avgPredictedLabels]]) {
				lastindices.add(indices[i]);
			} else {
				break;
			}
			i--;
		}

		int size = lastindices.size();
	
		int j = avgPredictedLabels - counter;
		while (j > 0) {
			int next = Rand.nextInt(size);
			if (result[lastindices.get(next)] != 1.0) {
				result[lastindices.get(next)] = 1.0;
				j--;
			}
		}

		return result;
	}

	/**
	 * old style used for BRkNN-b (not in use)
	 */
	protected double[] labelsFromConfidences3old(double[] confidences) {
		double[] result = new double[numLabels];

		double[] conf2 = new double[numLabels];
		for (int i = 0; i < numLabels; i++) {
			conf2[i] = confidences[i];
		}

		for (int i = 0; i < avgPredictedLabels; i++) {
			int maxindex = Utils.maxIndex(conf2);
			result[maxindex] = 1.0;
			conf2[maxindex] = -1.0;
		}
		return result;
	}

}
