package mulan.classifier;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Simple BR implementation of the KNN algorithm
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * 
 */
@SuppressWarnings("serial")
public class BRknn extends MultiLabelKNN {

	public BRknn(int numLabels, int numOfNeighbors) {
		super(numLabels, numOfNeighbors);
		m_DistanceWeighting = WEIGHT_NONE; //weight none
	}

	public void buildClassifier(Instances train) throws Exception {
		super.buildClassifier(train);
	}

	//IBk style
	public Prediction makePrediction(Instance instance) throws Exception {

		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train);
		lnn.setMeasurePerformance(false);

		// for cross-validation where test-train instances belong to the same data set
		Instance instance2 = new Instance(instance);

		Instances knn = lnn.kNearestNeighbours(instance2, numOfNeighbors);

		double[] distances = lnn.getDistances();
		double[] confidences = getConfidences(knn, distances);

		double[] predictions = labelsFromConfidences(confidences);

		Prediction results = new Prediction(predictions, confidences);
		return results;

	}

	private double[] getConfidences(Instances neighbours, double[] distances) {
		double total = 0, weight;
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
			switch (m_DistanceWeighting) {
			case WEIGHT_INVERSE:
				weight = 1.0 / (distances[i] + 0.001); // to avoid div by zero
				break;
			case WEIGHT_SIMILARITY:
				weight = 1.0 - distances[i];
				break;
			default: // WEIGHT_NONE:
				weight = 1.0;
				break;
			}
			weight *= current.weight();
			
			for(int j=0;j<numLabels;j++){
				double value = Double.parseDouble(current.attribute(predictors + j).value(
						(int) current.value(predictors + j)));
				if (Utils.eq(value, 1.0)) {
					confidences[j] += weight;
				}
			}

			total += weight;
		}
		// Normalise distribution
		if (total > 0) {
			Utils.normalize(confidences, total);
		}
		return confidences;
	}
	
	//old style
	public Prediction makePrediction2(Instance instance) throws Exception {
		double[] confidences = new double[numLabels];
		double[] predictions = new double[numLabels];

		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train);
		lnn.setMeasurePerformance(false);

		double[] votes = new double[numLabels];

		// for cross-validation where test-train instances belong to the same data set
		Instance instance2 = new Instance(instance);

		Instances knn = new Instances(lnn.kNearestNeighbours(instance2, numOfNeighbors));

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
}
