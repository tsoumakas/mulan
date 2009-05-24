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
 *    BRkNN.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.classifier.lazy;

import java.util.ArrayList;
import java.util.Arrays;

import mulan.classifier.MultiLabelOutput;
import mulan.core.Util;
import mulan.core.data.MultiLabelInstances;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Simple BR implementation of the KNN algorithm <!-- globalinfo-start -->
 * 
 * <pre>
 * Class implementing the base BRkNN algorithm and its 2 extensions BRkNN-a and BRkNN-b.
 * </pre>
 * 
 * For more information:
 * 
 * <pre>
 * E. Spyromitros, G. Tsoumakas, I. Vlahavas, An Empirical Study of Lazy Multilabel Classification Algorithms,
 * Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008), Springer, Syros, Greece, 2008.
 * http://mlkd.csd.auth.gr/multilabel.html
 * </pre>
 * 
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#064;inproceedings{1428385,
 *    author = {Spyromitros, Eleftherios and Tsoumakas, Grigorios and Vlahavas, Ioannis},
 *    title = {An Empirical Study of Lazy Multilabel Classification Algorithms},
 *    booktitle = {SETN '08: Proceedings of the 5th Hellenic conference on Artificial Intelligence},
 *    year = {2008},
 *    isbn = {978-3-540-87880-3},
 *    pages = {401--406},
 *    doi = {http://dx.doi.org/10.1007/978-3-540-87881-0_40},
 *    publisher = {Springer-Verlag},
 *    address = {Berlin, Heidelberg},
 * }
 * 
 * </pre>
 * 
 * <!-- technical-bibtex-end -->
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * 
 */
@SuppressWarnings("serial")
public class BRkNN extends MultiLabelKNN {

	/**
	 * Stores the average number of labels among the knn for each instance Used
	 * in BRkNN-b extension
	 */
	int avgPredictedLabels;

	/**
	 * The value of kNN provided by the user. This may differ from
	 * numOfNeighbors if cross-validation is being used.
	 */
	private int cvMaxK;

	/**
	 * Whether to select k by cross validation.
	 */
	private boolean cvkSelection = false;

	/**
	 * Meaningful values are 0,2 and 3
	 */
	protected int selectedMethod;

	public static final int BR = 0;

	public static final int BRexta = 2;

	public static final int BRextb = 3;

	/**
	 * The default constructor. (The base algorithm)
	 * 
	 * @param numOfNeighbors
	 */
	public BRkNN(int numOfNeighbors) {
		super(numOfNeighbors);
		distanceWeighting = WEIGHT_NONE; // weight none
		selectedMethod = BR; // the default method
	}

	/**
	 * Constructor giving the option to select an extension of the base version
	 * 
	 * @param numOfNeighbors
	 * @param method
	 *            (2 for BRkNN-a 3 for BRkNN-b)
	 * 
	 */
	public BRkNN(int numOfNeighbors, int method) {
		super(numOfNeighbors);
		distanceWeighting = WEIGHT_NONE; // weight none
		selectedMethod = method;
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(
				Type.INPROCEEDINGS);
		result
				.setValue(Field.AUTHOR,
						"Eleftherios Spyromitros, Grigorios Tsoumakas, Ioannis Vlahavas");
		result
				.setValue(Field.TITLE,
						"An Empirical Study of Lazy Multilabel Classification Algorithms");
		result
				.setValue(Field.BOOKTITLE,
						"Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008)");
		result.setValue(Field.LOCATION, "Syros, Greece");
		result.setValue(Field.YEAR, "2008");

		return result;
	}

	@Override
	protected void buildInternal(MultiLabelInstances aTrain) throws Exception {
		super.buildInternal(aTrain);

		lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train.getDataSet());
		lnn.setMeasurePerformance(false);

		if (cvkSelection == true) {
			crossValidate();
		}
	}

	/**
	 * 
	 * @param flag
	 *            if true the k is selected via cross-validation
	 */
	public void setkSelectionViaCV(boolean flag) {
		cvkSelection = flag;
	}

	/**
	 * Select the best value for k by hold-one-out cross-validation. Hamming
	 * Loss is minimized
	 * 
	 * @throws Exception
	 */
	protected void crossValidate() throws Exception {
		try {
			// the performance for each different k
			double[] hammingLoss = new double[cvMaxK];

			for (int i = 0; i < cvMaxK; i++) {
				hammingLoss[i] = 0;
			}

			Instances dataSet = train.getDataSet();
			Instance instance; // the hold out instance
			Instances neighbours; // the neighboring instances
			double[] origDistances, convertedDistances;
			for (int i = 0; i < dataSet.numInstances(); i++) {
				if (getDebug() && (i % 50 == 0)) {
					debug("Cross validating " + i + "/"
							+ dataSet.numInstances() + "\r");
				}
				instance = dataSet.instance(i);
				neighbours = lnn.kNearestNeighbours(instance, cvMaxK);
				origDistances = lnn.getDistances();

				// gathering the true labels for the instance
				boolean[] trueLabels = new boolean[numLabels];
				for (int counter = 0; counter < numLabels; counter++) {
					int classIdx = labelIndices[counter];
					String classValue = instance.attribute(classIdx).value(
							(int) instance.value(classIdx));
					trueLabels[counter] = classValue.equals("1");
				}
				// calculate the performance metric for each different k
				for (int j = cvMaxK; j > 0; j--) {
					convertedDistances = new double[origDistances.length];
					System.arraycopy(origDistances, 0, convertedDistances, 0,
							origDistances.length);
					double[] confidences = this.getConfidences(neighbours,
							convertedDistances);
					boolean[] bipartition = null;

					if (selectedMethod == BR) {// BRknn
						bipartition = labelsFromConfidences(confidences);
					} else if (selectedMethod == BRexta) {// BRknn-a
						bipartition = labelsFromConfidences2(confidences);
					} else if (selectedMethod == BRextb) {// BRknn-b
						bipartition = labelsFromConfidences3(confidences);
					}

					double symmetricDifference = 0; // |Y xor Z|
					for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
						boolean actual = trueLabels[labelIndex];
						boolean predicted = bipartition[labelIndex];

						if (predicted != actual) {
							symmetricDifference++;
						}
					}
					hammingLoss[j - 1] += (symmetricDifference / numLabels);

					neighbours = new IBk().pruneToK(neighbours,
							convertedDistances, j - 1);
				}
			}

			// Display the results of the cross-validation
			if (getDebug()) {
				for (int i = cvMaxK; i > 0; i--) {
					debug("Hold-one-out performance of " + (i) + " neighbors ");
					debug("(Hamming Loss) = " + hammingLoss[i - 1]
							/ dataSet.numInstances());
				}
			}

			// Check through the performance stats and select the best
			// k value (or the lowest k if more than one best)
			double[] searchStats = hammingLoss;

			double bestPerformance = Double.NaN;
			int bestK = 1;
			for (int i = 0; i < cvMaxK; i++) {
				if (Double.isNaN(bestPerformance)
						|| (bestPerformance > searchStats[i])) {
					bestPerformance = searchStats[i];
					bestK = i + 1;
				}
			}
			numOfNeighbors = bestK;
			if (getDebug()) {
				System.err.println("Selected k = " + bestK);
			}

		} catch (Exception ex) {
			throw new Error("Couldn't optimize by cross-validation: "
					+ ex.getMessage());
		}
	}

	/**
	 * weka Ibk style prediction
	 */

	public MultiLabelOutput makePrediction(Instance instance) throws Exception {

		Instances knn = lnn.kNearestNeighbours(instance, numOfNeighbors);

		double[] distances = lnn.getDistances();
		double[] confidences = getConfidences(knn, distances);
		boolean[] bipartition = null;

		if (selectedMethod == BR) {// BRknn
			bipartition = labelsFromConfidences(confidences);
		} else if (selectedMethod == BRexta) {// BRknn-a
			bipartition = labelsFromConfidences2(confidences);
		} else if (selectedMethod == BRextb) {// BRknn-b
			bipartition = labelsFromConfidences3(confidences);
		}
		MultiLabelOutput results = new MultiLabelOutput(bipartition,
				confidences);
		return results;

	}

	/**
	 * Calculates the confidences of the labels, based on the neighboring
	 * instances
	 * 
	 * @param neighbours
	 *            the list of nearest neighboring instances
	 * @param distances
	 *            the distances of the neighbors
	 * @return the confidences of the labels
	 */
	private double[] getConfidences(Instances neighbours, double[] distances) {
		double total = 0, weight;
		double neighborLabels = 0;
		double[] confidences = new double[numLabels];

		// Set up a correction to the estimator
		for (int i = 0; i < numLabels; i++) {
			confidences[i] = 1.0 / Math.max(1, train.getDataSet()
					.numInstances());
		}
		total = (double) numLabels
				/ Math.max(1, train.getDataSet().numInstances());

		for (int i = 0; i < neighbours.numInstances(); i++) {
			// Collect class counts
			Instance current = neighbours.instance(i);
			distances[i] = distances[i] * distances[i];
			distances[i] = Math.sqrt(distances[i]
					/ (train.getDataSet().numAttributes() - numLabels));
			switch (distanceWeighting) {
			case WEIGHT_INVERSE:
				weight = 1.0 / (distances[i] + 0.001); // to avoid division by
				// zero
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
				double value = Double.parseDouble(current.attribute(
						labelIndices[j]).value(
						(int) current.value(labelIndices[j])));
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
	 * old style prediction (not in use)
	 * 
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public MultiLabelOutput makePredictionOld(Instance instance)
			throws Exception {
		double[] confidences = new double[numLabels];
		boolean[] bipartition = new boolean[numLabels];

		double[] votes = new double[numLabels];

		Instances knn = new Instances(lnn.kNearestNeighbours(instance,
				numOfNeighbors));

		for (int i = 0; i < numLabels; i++) {
			int aces = 0; // num of aces in Knn for i
			for (int k = 0; k < numOfNeighbors; k++) {
				double value = Double.parseDouble(train.getDataSet().attribute(
						labelIndices[i]).value(
						(int) knn.instance(k).value(labelIndices[i])));
				if (Utils.eq(value, 1.0)) {
					aces++;
				}
			}
			votes[i] = aces;
		}

		for (int i = 0; i < numLabels; i++) {
			confidences[i] = (double) votes[i] / numOfNeighbors;
		}

		bipartition = labelsFromConfidences(confidences);

		MultiLabelOutput results = new MultiLabelOutput(bipartition,
				confidences);
		return results;
	}

	/**
	 * Derive output labels from distributions.
	 * 
	 * @param confidences
	 * @return
	 */
	protected boolean[] labelsFromConfidences(double[] confidences) {
		if (thresholds == null) {
			thresholds = new double[numLabels];
			Arrays.fill(thresholds, threshold);
		}

		boolean[] bipartition = new boolean[numLabels];
		for (int i = 0; i < numLabels; i++) {
			if (confidences[i] >= thresholds[i]) {
				bipartition[i] = true;
			}
		}
		return bipartition;
	}

	/**
	 * used for BRknn-a
	 */
	protected boolean[] labelsFromConfidences2(double[] confidences) {
		boolean[] bipartition = new boolean[numLabels];
		boolean flag = false; // check the case that no label is true

		for (int i = 0; i < numLabels; i++) {
			if (confidences[i] >= threshold) {
				bipartition[i] = true;
				flag = true;
			}
		}
		// assign the class with the greater confidence
		if (flag == false) {
			int index = Util.RandomIndexOfMax(confidences, random);
			bipartition[index] = true;
		}
		return bipartition;
	}

	/**
	 * used for BRkNN-b (break ties arbitrarily)
	 */
	protected boolean[] labelsFromConfidences3(double[] confidences) {
		boolean[] bipartition = new boolean[numLabels];

		int[] indices = Utils.stableSort(confidences);

		ArrayList<Integer> lastindices = new ArrayList<Integer>();

		int counter = 0;
		int i = numLabels - 1;

		while (i > 0) {
			if (confidences[indices[i]] > confidences[indices[numLabels
					- avgPredictedLabels]]) {
				bipartition[indices[i]] = true;
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
			int next = random.nextInt(size);
			if (bipartition[lastindices.get(next)] != true) {
				bipartition[lastindices.get(next)] = true;
				j--;
			}
		}

		return bipartition;
	}

	/**
	 * set the maximum number of neighbors to be evaluated via cross-validation
	 * 
	 * @param cvMaxK
	 */
	public void setCvMaxK(int cvMaxK) {
		this.cvMaxK = cvMaxK;
	}

}