package mulan.classifier;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 *
 */
public class MultiKnn extends AbstractMultiLabelClassifier {

	public long sumedlabels;

	private int predictors;

	protected LinearNNSearch lnn;

	private EuclideanDistance dfunc = null;

	private int numofNeighbours;

	private Instances train;

	public MultiKnn(int labels, Instances train, int k) {
		numLabels = labels;
		numofNeighbours = k;
	}

	public void buildClassifier(Instances train) {
		this.train = train;
		predictors = train.numAttributes() - numLabels;

		dfunc = new EuclideanDistance();
		dfunc.setDontNormalize(false);
		dfunc.setAttributeIndices("first-" + predictors);
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

		Instances knn = new Instances(lnn.kNearestNeighbours(instance2, numofNeighbours));

		for (int i = 0; i < numLabels; i++) {
			if (Utils.eq(predictions[i], 0)) {
				// compute sum of aces in KNN
				int aces = 0; // num of aces in Knn for i
				for (int k = 0; k < numofNeighbours; k++) {
					double value = Double.parseDouble(train2.attribute(predictors + i).value(
							(int) knn.instance(k).value(predictors + i)));
					if (Utils.eq(value, 1.0)) {
						aces++;
					}
				}
				votes[i] = aces; // ranking function
			}
		}
		for (int k = 0; k < numofNeighbours; k++) {
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

		if (votes[result] > noclass ) // && votes[result] >= 5
			return result;
		else
			return -1;
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
				sumedlabels++;
			}
			//System.out.println(newtrain.numInstances());
		} while (result != -1 && newtrain.numInstances() >= numofNeighbours);

		Prediction results = new Prediction(predictions, confidences);
		return results;
	}
	
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

		Instances knn = new Instances(lnn.kNearestNeighbours(instance2, numofNeighbours));

		for (int i = 0; i < numLabels; i++) {
				int aces = 0; // num of aces in Knn for i
				for (int k = 0; k < numofNeighbours; k++) {
					double value = Double.parseDouble(train.attribute(predictors + i).value(
							(int) knn.instance(k).value(predictors + i)));
					if (Utils.eq(value, 1.0)) {
						aces++;
					}
				}
				votes[i] = aces; 
		}
		
		for (int i = 0; i < numLabels; i++){
			if (votes[i]>numofNeighbours/2){
				predictions[i]=1.0;
				sumedlabels++;
			}
		}

		Prediction results = new Prediction(predictions, confidences);
		return results;
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
