package mulan.classifier;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;

@SuppressWarnings("serial")
public class Experimentalknn extends MultiLabelKNN {

	public Experimentalknn() {
		// TODO Auto-generated constructor stub
	}

	public Experimentalknn(int numLabels, int numOfNeighbors) {
		super(numLabels, numOfNeighbors);
		// TODO Auto-generated constructor stub
	}
	
	public void buildClassifier(Instances train) throws Exception {
		super.buildClassifier(train);
	}

	public Prediction makePrediction(Instance instance) throws Exception {
		double[] confidences = new double[numLabels];
		double[] predictions = new double[numLabels];

		Instances newtrain = new Instances(this.train);
		//System.out.println(newtrain.numInstances());

		int result;
		do {
			result = toplabel(instance, newtrain, predictions);
			if (result != -1) {
				predictions[result] = 1;
				newtrain = new Instances(filterwithlabel(result, newtrain));
				sumedLabels++;
			}
			//System.out.println(newtrain.numInstances());
		} while (result != -1 && newtrain.numInstances() >= numOfNeighbors);

		Prediction results = new Prediction(predictions, confidences);
		return results;
	}
	
	public int toplabel(Instance instance, Instances train2, double[] predictions) throws Exception {

		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train2);
		lnn.setMeasurePerformance(false);

		double[] votes = new double[numLabels];
		int noclass = 0;

		// for cross-validation where test-train instances belong to the same data set
		Instance instance2 = new Instance(instance);

		Instances knn = new Instances(lnn.kNearestNeighbours(instance2, numOfNeighbors));

		for (int i = 0; i < numLabels; i++) {
			if (Utils.eq(predictions[i], 0)) {
				// compute sum of aces in KNN
				int aces = 0; // num of aces in Knn for i
				for (int k = 0; k < numOfNeighbors; k++) {
					double value = Double.parseDouble(train2.attribute(predictors + i).value(
							(int) knn.instance(k).value(predictors + i)));
					if (Utils.eq(value, 1.0)) {
						aces++;
					}
				}
				votes[i] = aces; // ranking function
			}
		}
		for (int k = 0; k < numOfNeighbors; k++) {
			boolean ace = false;
			for (int i = 0; i < numLabels; i++) {
				if (Utils.eq(predictions[i], 0)) {
					double value = Double.parseDouble(train2.attribute(predictors + i).value(
							(int) knn.instance(k).value(predictors + i)));
					if (Utils.eq(value, 1.0)) {
						ace = true;
						break;
					}
				}
			}
			if (ace == false) {
				noclass++;
			}
		}
		int result = Utils.maxIndex(votes);

		if (votes[result] >= noclass ) // && votes[result] >= 5
			return result;
		else
			return -1;
	}
	
	public Instances filterwithlabel(int j, Instances init) {
		//make a copy of the supplied dataset
		Instances transformed = new Instances(init);

		//delete instances without label j
		for (int i = 0; i < init.numInstances(); i++) {
			double value = Double.parseDouble(init.attribute(predictors + j).value(
					(int) init.instance(i).value(predictors + j)));
			if (!Utils.eq(value, 1.0)) {
				init.delete(i);
			}
		}

		//delete label j
		//transformed.deleteAttributeAt(predictors+j);

		//System.out.println(transformed);
		return init;

	}

}
