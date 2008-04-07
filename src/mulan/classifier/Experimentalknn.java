package mulan.classifier;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;

@SuppressWarnings("serial")
public class Experimentalknn extends MultiLabelKNN {

	public Experimentalknn() {
	}

	public Experimentalknn(int numLabels, int numOfNeighbors) {
		super(numLabels, numOfNeighbors);
	}

	public void buildClassifier(Instances train) throws Exception {
		super.buildClassifier(train);
	}

	public Prediction makePrediction(Instance instance) throws Exception {
		double[] confidences = new double[numLabels];
		double[] predictions = new double[numLabels];

		Instances newtrain = new Instances(this.train);
		//System.out.println(newtrain.numInstances());

		int[] votes = new int[numLabels + 1]; //for the null class 

		int result;
		int counter=0;
		
		while (newtrain.numInstances() >= numOfNeighbors){
			counter++;
			votes = votessofar(instance, newtrain, predictions, votes);
			
			result = Utils.maxIndex(votes);//most voted

			if (result == numLabels || votes[result] < numOfNeighbors/2 ) // && votes[result] >= 5
				break;
			else {
				confidences[result] = (double)votes[result]/ numOfNeighbors * counter;
				predictions[result] = 1;
				newtrain = new Instances(filterwithlabel(result, newtrain));
				sumedLabels++;
			}
			//System.out.println(newtrain.numInstances());
			votes[result] = 0;
		} 
		
		//calculate confidences
		for(int i=0 ;i < numLabels ; i++){
			if(confidences[i]== 0.0){
			confidences[i] = (double)votes[i]/ numOfNeighbors * counter;
			//System.out.println("1");
			}
		}

		Prediction results = new Prediction(predictions, confidences);
		//System.out.println("ONE prediction completed");
		return results;
	}

	public int[] votessofar(Instance instance, Instances train2, double[] predictions, int[] votes)
			throws Exception {

		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train2);
		lnn.setMeasurePerformance(false);

		//double[] votes = new double[numLabels+1]; //+1 for no class
		//int noclass = 0;

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
				votes[numLabels]++;
			}
		}

		return votes;
	}

	public Instances filterwithlabel(int j, Instances init) {
		//delete instances without label j
		for (int i = 0; i < init.numInstances(); i++) {
			double value = Double.parseDouble(init.attribute(predictors + j).value(
					(int) init.instance(i).value(predictors + j)));
			if (!Utils.eq(value, 1.0)) {
				init.delete(i);
			}
		}
		//System.out.println(transformed);
		return init;

	}

}
