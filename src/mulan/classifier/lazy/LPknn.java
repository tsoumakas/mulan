package mulan.classifier.lazy;

import java.util.HashSet;

import mulan.classifier.Prediction;
import mulan.core.LabelSet;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * 
 * Label Powerset Classification method (knn style)
 *
 */
@SuppressWarnings("serial")
public class LPknn extends MultiLabelKNN {

	public LPknn(int numLabels, int numOfNeighbors) {
		super(numLabels, numOfNeighbors);
	}

	public Prediction makePrediction(Instance instance) throws Exception {

		double[] predictions = new double[numLabels];

		double[][] dblLabels = new double[numOfNeighbors][numLabels];

		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train);
		lnn.setMeasurePerformance(false);

		// for cross-validation where test-train instances belong to the same data set
		Instance instance2 = new Instance(instance);

		Instances knn = lnn.kNearestNeighbours(instance2, numOfNeighbors);

		//double[] distances = lnn.getDistances();

		// gather distinct label combinations
		HashSet<LabelSet> labelSets = new HashSet<LabelSet>();
		for (int i = 0; i < numOfNeighbors; i++) {
			// construct label set
			for (int j = 0; j < numLabels; j++)
				dblLabels[i][j] = Double.parseDouble(knn.attribute(predictors + j).value(
						(int) knn.instance(i).value(predictors + j)));
			LabelSet labelSet = new LabelSet(dblLabels[i]);

			// add label set if not already present
			labelSets.add(labelSet);
		}

		// gather knn votes for each distinct label combination
		int[][] votes = new int[labelSets.size()][2];

		//get all distinct label sets in an array
		LabelSet[] distinctLabelSets = new LabelSet[labelSets.size()];

		Object[] odistincLabelSets = labelSets.toArray();

		for (int i = 0; i < labelSets.size(); i++) {
			distinctLabelSets[i] = (LabelSet) odistincLabelSets[i];
		}

		// count the votes of knn for each distinct labelset
		for (int i = 0; i < numOfNeighbors; i++) {
			LabelSet labelSet = new LabelSet(dblLabels[i]);
			for (int j = 0; j < labelSets.size(); j++) {
				if (labelSet.equals(distinctLabelSets[j])) {
					votes[j][0]++;
					votes[j][1]+= (i+1);
				}
			}
		}

		//the latest subsets are better because they are the subsets of the 
		//nearest neighbors
		int max = 0;
		for (int i = 1; i < labelSets.size(); i++) {
			if (votes[i][0] > votes[max][0]) {
				max = i;
			}
			else if(votes[i][0] == votes[max][0] && votes[i][1]<= votes[max][1]){
				max = i;
			}
		}

		predictions = distinctLabelSets[max].toDoubleArray();
		
		// the confidences for the true labels are 1. The rest are 0.
		Prediction results = new Prediction(predictions, predictions);
		return results;
	}
}
