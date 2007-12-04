package mulan.classifier;

import weka.filters.*;
import weka.filters.unsupervised.attribute.Remove;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;
import weka.core.neighboursearch.CoverTree;
import weka.core.Attribute;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;

/**
 * Class that implements the ML-KNN (Multi-Label K Nearest Neighbours )
 * algorithm
 * <p>
 * @author Eleftherios Spyromitros Xioufis
 */

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
	private CoverTree mlCoverTree = null;
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
		this.train=train;
		predictors = train.numAttributes() - numLabels;

		ComputePrior(train);
		ComputeCond(train);

	}
	
	public void setdontnormalize (boolean norm) {
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
				double value = train.instance(j).value(predictors + i);
				if (Utils.eq(value, 1.0)) {
					temp_Ci++;
				}
			}
			PriorProbabilities[i] = (smooth + temp_Ci)
					/ (smooth * 2 + train.numInstances());
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
		// -1 einai o class index
		// diladi den exei oristei clasi
		// kai to covertree douleui giati vriskei tinapostasi me vasi ta
		// attributes apo 0 mexrin numattributes kai aplos prosperna tin
		// klasi. alla afou einai -1 den ti vriskei pote
		// System.out.println(datalabels.classIndex());

		Instances attributes = transform(train, true);// crop class labels

		Attribute fakeclass = new Attribute("position"); // create new attr
		attributes.insertAttributeAt(fakeclass, 0);

		for (int i = 0; i < train.numInstances(); i++)
			// give values
			attributes.instance(i).setValue(0, i);

		attributes.setClassIndex(0);

		//distance function intitiallization
		EuclideanDistance dfunc = new EuclideanDistance();
		dfunc.setDontNormalize(dontnormalize);
		
		mlCoverTree = new CoverTree();
		mlCoverTree.setDistanceFunction(dfunc);
		
		mlCoverTree.setInstances(attributes);

		// c[k] counts the number of training instances with label i whose k
		// nearest neighbours contain exactly k instances with label i
		int[][] temp_Ci = new int[numLabels][numofNeighbours + 1];
		int[][] temp_NCi = new int[numLabels][numofNeighbours + 1];

		

		for (int i = 0; i < train.numInstances(); i++) {
			// it also counts the instance itself, so we compute one n more and
			// then crop it
			Instances tempknn = new Instances(mlCoverTree.kNearestNeighbours(
					attributes.instance(i), numofNeighbours + 1 ));

			// now compute values of temp_Ci and temp_NCi for every class label
			for (int j = 0; j < numLabels; j++) {
				// compute sum of aces in KNN (starts from 1 to bypass the extra
				// neighbour)
				int tempaces = 0; // num of aces in Knn for j
				// tempknn.numInstances()= numofNeighbours+1
				for (int k = 1; k < numofNeighbours + 1; k++) {
					int index = (int) tempknn.instance(k).value(0);
					double value = train.instance(index).value(predictors + j);
					if (Utils.eq(value, 1.0)) {
						tempaces++;
					}
				}
				// raise the counter of temp_Ci[j][tempaces] //
				// temp_NCi[j][tempaces] by 1
				if ((train.instance(i).value(predictors + j)) == 1) {
					temp_Ci[j][tempaces]++;
				} else {
					temp_NCi[j][tempaces]++;
				}
			}
		}

		// finally compute CondProbabilities[i][..] for labels based on temp_Ci
		// array
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

		Instance instance2 = new Instance(instance);
		// transform instance (delete class attributes)
		for (int i = 0; i < numLabels; i++)
			instance2.deleteAttributeAt(predictors + numLabels - i -1);
		 // create new attr
		instance2.insertAttributeAt(0);

		// identify knn
		Instances knn = new Instances(mlCoverTree.kNearestNeighbours(instance2,
				numofNeighbours));
		for (int i = 0; i < numLabels; i++) {
			// compute sum of aces in KNN
			int tempaces = 0; // num of aces in Knn for i
			for (int k = 0; k < numofNeighbours; k++) {
				int index = (int) knn.instance(k).value(0);
				double value = train.instance(index).value(predictors + i);
				if (Utils.eq(value, 1.0)) {
					tempaces++;
				}
			}
			double value1 = PriorProbabilities[i]
					* CondProbabilities[i][tempaces];
			double value2 = PriorNProbabilities[i]
					* CondNProbabilities[i][tempaces];
			if (value1/(value1+value2) >= 0.5){
				predictions[i] = 1;
				//confidences[i] = value1 / ( value1 + value2 );
			}
			else{
				predictions[i] = 0;
				//confidences[i] =1-( value1 / ( value1 + value2));
			}
		}

		Prediction result = new Prediction(predictions, confidences);
		return result;

	}

	/**
	 * Remove all non - label attributes or all label attributes depending on
	 * parameter
	 * 
	 * @param option
	 *            if True select attributes if False select labels
	 * @param train
	 */
	private Instances transform(Instances train, boolean option)
			throws Exception {
		// Indices of attributes to keep or to remove
		int predictors = train.numAttributes() - numLabels;
		int indices[] = new int[predictors];

		for (int j = 0; j < predictors; j++) {
			indices[j] = j;
		}

		Remove remove = new Remove();
		remove.setInvertSelection(option);
		remove.setAttributeIndicesArray(indices);
		remove.setInputFormat(train);
		Instances result = Filter.useFilter(train, remove);
		return result;
	}

	public void output() {
		System.out.println("Computed Prior Probabilities");
		for (int i = 0; i < numLabels; i++) {
			System.out.println("Label " + (i + 1) + ": "
					+ PriorProbabilities[i]);
		}
		System.out.println("Computed Posterior Probabilities");
		for (int i = 0; i < numLabels; i++) {
			System.out.println("Label " + (i + 1));
			for (int j = 0; j < numofNeighbours + 1; j++) {
				System.out.println(j + " neighbours: "
						+ CondProbabilities[i][j]);
				System.out.println(j + " neighbours: "
						+ CondNProbabilities[i][j]);
			}
		}
	}

}
