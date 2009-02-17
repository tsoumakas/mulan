package mulan.classifier.lazy;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import mulan.classifier.MultiLabelOutput;
import mulan.core.LabelSet;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Rakel algorithm implementation (knn style)
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * 
 */
@SuppressWarnings("serial")
public class RAKELknn extends MultiLabelKNN {
	
	double[] sumVotes;
	double[] lengthVotes;
	int numOfModels;
	int sizeOfSubset;
	int[][] classIndicesPerSubset;

	public RAKELknn(int labels, int neighbors, int models, int subset) {
		super(labels, neighbors);
		numOfModels = models;
		sizeOfSubset = subset;
		classIndicesPerSubset = new int[numOfModels][sizeOfSubset];
		sumVotes = new double[numLabels];
		lengthVotes = new double[numLabels];
	}
/*
	public Bipartition makePrediction(Instance instance) throws Exception {
		double[][] predictions = new double[numOfModels][numLabels];

		double[][][] dblLabels = new double[numOfModels][numOfNeighbors][numLabels];

		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train);
		lnn.setMeasurePerformance(false);

		// for cross-validation where test-train instances belong to the same data set
		Instance instance2 = new Instance(instance);

		Instances knn = lnn.kNearestNeighbours(instance2, numOfNeighbors);

		//double[] distances = lnn.getDistances();

		//build the models of k-label sets
		for(int i=0;i<numOfModels;i++){
		HashSet<String>	combinations = new HashSet<String>();
		
		Random rnd = new Random();	

		// --select a random subset of classes not seen before
		boolean[] selected;
		do {
			selected = new boolean[numLabels];
			for (int j=0; j<sizeOfSubset; j++) {
				int randomLabel;
	           	randomLabel = Math.abs(rnd.nextInt() % numLabels);
	            while (selected[randomLabel] != false) {
	            	randomLabel = Math.abs(rnd.nextInt() % numLabels);
	            }
				selected[randomLabel] = true;
				//System.out.println("label: " + randomLabel);
				classIndicesPerSubset[i][j] = randomLabel;
			}
			Arrays.sort(classIndicesPerSubset[i]);
		} while (combinations.add(Arrays.toString(classIndicesPerSubset[i])) == false);
		System.out.println("Building model " + i + ", subset: " + Arrays.toString(classIndicesPerSubset[i]));	

		}
		
		for (int k = 0; k < numOfModels; k++) {
			// gather distinct label combinations
			HashSet<LabelSet> labelSets = new HashSet<LabelSet>();
			for (int i = 0; i < numOfNeighbors; i++) {
				// construct label set
				for (int j = 0; j < numLabels; j++) {
					for (int l = 0; l < sizeOfSubset; l++) {
						if (classIndicesPerSubset[k][l] == j) {
							dblLabels[k][i][j] = Double.parseDouble(knn.attribute(predictors + j)
									.value((int) knn.instance(i).value(predictors + j)));
							break;
						}
					}
				}
				LabelSet labelSet = new LabelSet(dblLabels[k][i]);

				// add label set if not already present
				labelSets.add(labelSet);
			}
			
			// gather knn votes for each distinct label combination
			int[] votes = new int[labelSets.size()];

			//get all distinct label sets in an array
			LabelSet[] distinctLabelSets = new LabelSet[labelSets.size()];

			Object[] odistincLabelSets = labelSets.toArray();

			for (int i = 0; i < labelSets.size(); i++) {
				distinctLabelSets[i] = (LabelSet) odistincLabelSets[i];
			}

			// count the votes of knn for each distinct labelset
			for (int i = 0; i < numOfNeighbors; i++) {
				LabelSet labelSet = new LabelSet(dblLabels[k][i]);
				for (int j = 0; j < labelSets.size(); j++) {
					if (labelSet.equals(distinctLabelSets[j])) {
						votes[j]++;
					}
				}
			}

			//the latest subsets are better because they are the subsets of the 
			//nearest neighbors
			int max = 0;
			for (int i = 1; i < labelSets.size(); i++) {
				if (votes[i] >= votes[max]) {
					max = i;
				}
			}

			predictions[k] = distinctLabelSets[max].toDoubleArray();
		}
		
		for (int i=0; i<numOfModels; i++) {
			for (int j=0; j<sizeOfSubset; j++) {
				sumVotes[classIndicesPerSubset[i][j]] += predictions[i][j];
				lengthVotes[classIndicesPerSubset[i][j]]++;
			}
		}
		
		double[] confidences = new double[numLabels];
		double[] labels = new double[numLabels];
		for (int i=0; i<numLabels; i++) {
			confidences[i] = sumVotes[i]/lengthVotes[i];
			if (confidences[i] >= 0.5)
				labels[i] = 1;
			else
				labels[i] = 0;
		}
		
		Prediction results = new Prediction(labels, confidences);
		return results;
		
	}
*/
    public MultiLabelOutput makePrediction(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
