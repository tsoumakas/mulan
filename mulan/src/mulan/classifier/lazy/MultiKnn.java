package mulan.classifier.lazy;


import mulan.classifier.MultiLabelOutput;
import mulan.core.data.MultiLabelInstances;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Class implementing an experimental knn-based multi-label classifier
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 *
 */
@SuppressWarnings("serial")
public class MultiKnn extends MultiLabelKNN {

	public long sumedlabels;

	private int predictors;

	protected LinearNNSearch lnn;

	private EuclideanDistance dfunc = null;

	private int numofNeighbours;

	//private Instances train;

	public MultiKnn(int k) {
		super(k);
	}

    @Override
	protected void buildInternal(MultiLabelInstances train) {
		//this.train = train;
		predictors = train.getDataSet().numAttributes() - numLabels;

		dfunc = new EuclideanDistance();
		dfunc.setDontNormalize(false);
		dfunc.setAttributeIndices("first-" + predictors);
	}

	public int toplabel(Instance instance, Instances train, boolean[] bipartition) throws Exception {

		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train);
		lnn.setMeasurePerformance(false);

		double[] votes = new double[numLabels];
		int noclass = 0;

		//in cross-validation test-train instances does not belong to the same data set
		//Instance instance2 = new Instance(instance);

		Instances knn = new Instances(lnn.kNearestNeighbours(instance, numofNeighbours));

		//calculation of the votes of each label
		for (int i = 0; i < numLabels; i++) {
			if (!bipartition[i]) { //calculate votes for the rest of the labels only
				// compute sum of aces in KNN
				int aces = 0; // num of aces in Knn for i
				for (int k = 0; k < numofNeighbours; k++) {
					double value = Double.parseDouble(train.attribute(predictors + i).value(
							(int) knn.instance(k).value(predictors + i)));
					if (Utils.eq(value, 1.0)) {
						aces++;
					}
				}
				votes[i] = aces; // ranking function
			}
		}
		//calculation of the votes for no label  
		for (int k = 0; k < numofNeighbours; k++) {
			boolean ace = false;
			for (int i = 0; i < numLabels; i++) {
				if (!bipartition[i]) {//calculate votes for the rest of the labels only
					double value = Double.parseDouble(train.attribute(predictors + i).value(
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

		//if the class with the most votes has less votes than the no class...
		if (votes[result] > noclass ) // && votes[result] >= 5
			return result;
		else
			return -1;
	}

//	public MultiLabelOutput makePrediction2(Instance instance) throws Exception {
//		double[] confidences = new double[numLabels];
//		boolean[] bipartition = new boolean[numLabels];
//
//		Instances newtrain = new Instances(this.train);
//		//System.out.println(newtrain.numInstances());
//
//		int result;
//		do {
//			result = toplabel(instance, newtrain, bipartition);
//			if (result != -1) {
//				bipartition[result] = true;
//		//		newtrain = new Instances(filterwithlabel(result, newtrain));
//				sumedlabels++;
//			}
//			//System.out.println(newtrain.numInstances());
//		} while (result != -1 && newtrain.numInstances() >= numofNeighbours);
//
//        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
//		return mlo;
//	}
	/*
	public Bipartition makePrediction(Instance instance) throws Exception {
		double[] confidences = new double[numLabels];
		double[] predictions = new double[numLabels];

		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train);
		lnn.setMeasurePerformance(false);

		double[] votes = new double[numLabels];
		int noclass = 0;
		Instances allnn = new Instances(lnn.kNearestNeighbours(instance, train.numInstances()));
		int result;
		
		do {
			//calculation of the votes of each label
			for (int i = 0; i < numLabels; i++) {
				if (Utils.eq(predictions[i], 0)) { //calculate votes for the rest of the labels only
					// compute sum of aces in KNN
					int aces = 0; // num of aces in Knn for i
					for (int k = 0; k < numofNeighbours; k++) {
						double value = Double.parseDouble(train.attribute(predictors + i).value(
								(int) allnn.instance(k).value(predictors + i)));
						if (Utils.eq(value, 1.0)) {
							aces++;
						}
					}
					votes[i] = aces; // ranking function
				}
			}
			//calculation of the votes for no label  
			for (int k = 0; k < numofNeighbours; k++) {
				boolean ace = false;
				for (int i = 0; i < numLabels; i++) {
					if (Utils.eq(predictions[i], 0)) {//calculate votes for the rest of the labels only
						double value = Double.parseDouble(train.attribute(predictors + i).value(
								(int) allnn.instance(k).value(predictors + i)));
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
			result = Utils.maxIndex(votes);//TODO: Random index of max 

			//if the class with the most votes has less votes than the no class...
			if (votes[result] <= noclass) { 
				break;
			}
			
			predictions[result] = 1.0;
			allnn = new Instances(filterwithlabel(result, allnn));
			sumedlabels++;
			
			votes = new double[numLabels];
			noclass = 0;

		} while (allnn.numInstances() >= numofNeighbours);
		
		Prediction results = new Prediction(predictions, confidences);
		return results;
	}

	//only keep the instances that have label j = 1.0
	public Instances filterwithlabel(int j, Instances init) {
		//make a copy of the supplied dataset
		//Instances transformed = new Instances(init);

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

	}*/

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public MultiLabelOutput makePrediction(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
