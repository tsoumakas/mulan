package mulan.classifier;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * class that implements the ML-KNN (Multi-Label K Nearest Neighbours )
 * algorithm
 * <p>
 * @author Eleftherios Spyromitros Xioufis 
 * @version $Revision: 1.1 $
 */
@SuppressWarnings("serial")//testing svn notify No2
public class MultiLabelKNN extends AbstractMultiLabelClassifier {

	private double[] PriorProbabilities;
	private double[] PriorNProbabilities;
	private double[][] CondProbabilities;
	private double[][] CondNProbabilities;

	// private int numLabels;
	private int numofNeighbours;
	private int predictors;
	private double smooth;
	private boolean dontnormalize;
	private LinearNNSearch lnn = null;
	private EuclideanDistance dfunc = null;
	private Instances train = null;

	public MultiLabelKNN() {
	}

	public MultiLabelKNN(int labels, int k, double s) {
		numLabels = labels;
		numofNeighbours = k;
		smooth = s;
		dontnormalize = true;
		PriorProbabilities = new double[numLabels];
		PriorNProbabilities = new double[numLabels];
		CondProbabilities = new double[numLabels][numofNeighbours + 1];
		CondNProbabilities = new double[numLabels][numofNeighbours + 1];
	}

	public void buildClassifier(Instances train) throws Exception {
		this.train = train;
		predictors = train.numAttributes() - numLabels;

		dfunc = new EuclideanDistance();
		dfunc.setDontNormalize(dontnormalize);
		dfunc.setAttributeIndices("first-" + predictors);

		ComputePrior(train);
		ComputeCond(train);

	}

	public void setdontnormalize(boolean norm) {
		dontnormalize = norm;
	}

	/**
	 * Computing Prior and PriorN Probabilities for each class of the training
	 * set
	 * 
	 * @param train :
	 *            the training dataset
	 */
	private void ComputePrior(Instances train) {
		for (int i = 0; i < numLabels; i++) {
			int temp_Ci = 0;
			for (int j = 0; j < train.numInstances(); j++) {
				double value = Double.parseDouble(train.attribute(predictors + i).value(
						(int) train.instance(j).value(predictors + i)));
				if (Utils.eq(value, 1.0)) {
					temp_Ci++;
				}
			}
			PriorProbabilities[i] = (smooth + temp_Ci) / (smooth * 2 + train.numInstances());
			PriorNProbabilities[i] = 1 - PriorProbabilities[i];
		}
	}

	/**
	 * Computing Cond and CondN Probabilities for each class of the training set
	 * 
	 * @param train :
	 *            the training dataset
	 */
	private void ComputeCond(Instances train) throws Exception {

		lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(train);
		// lnn.setMeasurePerformance(false);
		// lnn.setSkipIdentical(true); this implementation doesn't need it

		// c[k] counts the number of training instances with label i whose k
		// nearest neighbours contain exactly k instances with label i
		int[][] temp_Ci = new int[numLabels][numofNeighbours + 1];
		int[][] temp_NCi = new int[numLabels][numofNeighbours + 1];

		for (int i = 0; i < train.numInstances(); i++) {

			Instances knn = new Instances(lnn
					.kNearestNeighbours(train.instance(i), numofNeighbours));

			// now compute values of temp_Ci and temp_NCi for every class label
			for (int j = 0; j < numLabels; j++) {

				int aces = 0; // num of aces in Knn for j
				for (int k = 0; k < numofNeighbours; k++) {
					double value = Double.parseDouble(train.attribute(predictors + j).value(
							(int) knn.instance(k).value(predictors + j)));
					if (Utils.eq(value, 1.0)) {
						aces++;
					}
				}
				// raise the counter of temp_Ci[j][aces] and temp_NCi[j][aces]
				// by 1
				if (Utils.eq(Double.parseDouble(train.attribute(predictors + j).value(
						(int) train.instance(i).value(predictors + j))), 1.0)) {
					temp_Ci[j][aces]++;
				} else {
					temp_NCi[j][aces]++;
				}
			}
		}

		// compute CondProbabilities[i][..] for labels based on temp_Ci[]
		for (int i = 0; i < numLabels; i++) {
			int temp1 = 0;
			int temp2 = 0;
			for (int j = 0; j < numofNeighbours + 1; j++) {
				temp1 += temp_Ci[i][j];
				temp2 += temp_NCi[i][j];
			}
			for (int j = 0; j < numofNeighbours + 1; j++) {
				CondProbabilities[i][j] = (smooth + temp_Ci[i][j])
						/ (smooth * (numofNeighbours + 1) + temp1);
				CondNProbabilities[i][j] = (smooth + temp_NCi[i][j])
						/ (smooth * (numofNeighbours + 1) + temp2);
			}
		}
	}

	public Prediction makePrediction(Instance instance) throws Exception {

		double[] confidences = new double[numLabels];
		double[] predictions = new double[numLabels];

		// for cross-validation where test-train instances belong to the same data set
		Instance instance2 = new Instance(instance);

		Instances knn = new Instances(lnn.kNearestNeighbours(instance2, numofNeighbours));

		for (int i = 0; i < numLabels; i++) {
			// compute sum of aces in KNN
			int aces = 0; // num of aces in Knn for i
			for (int k = 0; k < numofNeighbours; k++) {
				double value = Double.parseDouble(train.attribute(predictors + i).value(
						(int) knn.instance(k).value(predictors + i)));
				if (Utils.eq(value, 1.0)) {
					aces++;
				}
			}
			double Prob_in = PriorProbabilities[i] * CondProbabilities[i][aces];
			double Prob_out = PriorNProbabilities[i] * CondNProbabilities[i][aces];
			confidences[i] = Prob_in / (Prob_in + Prob_out); // ranking function

			if (confidences[i] >= 0.5) {
				predictions[i] = 1;
			} else {
				predictions[i] = 0;
			}
		}
		Prediction result = new Prediction(predictions, confidences);
		return result;
	}

	public void output() {
		System.out.println("Computed Prior Probabilities");
		for (int i = 0; i < numLabels; i++) {
			System.out.println("Label " + (i + 1) + ": " + PriorProbabilities[i]);
		}
		System.out.println("Computed Posterior Probabilities");
		for (int i = 0; i < numLabels; i++) {
			System.out.println("Label " + (i + 1));
			for (int j = 0; j < numofNeighbours + 1; j++) {
				System.out.println(j + " neighbours: " + CondProbabilities[i][j]);
				System.out.println(j + " neighbours: " + CondNProbabilities[i][j]);
			}
		}
	}
}
