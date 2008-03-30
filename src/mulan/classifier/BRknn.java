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
	}

	public void buildClassifier(Instances train) throws Exception {
		super.buildClassifier(train);
	}

	public Prediction makePrediction(Instance instance) throws Exception {
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
