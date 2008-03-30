package mulan.classifier;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Superclass of all KNN based multi-label algorithms
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 *
 */
@SuppressWarnings("serial")
public class MultiLabelKNN extends AbstractMultiLabelClassifier {
	
	/**
	 * Sum of predicted labels for all instances
	 */
	protected long sumedLabels;
	/**
	 * Number of predictor attributes
	 */
	protected int predictors;
	/**
	 * Number of neighbors used in the k-nearest neighbor algorithm
	 */
	protected int numOfNeighbors;
	/**
	 * Whether to use normalized Euclidean distance
	 */
	protected boolean dontNormalize;
	/**
	 * Class implementing the brute force search algorithm for nearest neighbour
	 * search. Default value is true.
	 */
	protected LinearNNSearch lnn = null;
	/**
	 * Implementing Euclidean distance (or similarity) function.
	 */
	protected EuclideanDistance dfunc = null;
	/**
	 * The training instances
	 */
	protected Instances train = null;
	
	public MultiLabelKNN(){
	}
	
	public MultiLabelKNN(int numLabels, int numOfNeighbors) {
		super(numLabels);
		this.numOfNeighbors = numOfNeighbors;
	}

	public void buildClassifier(Instances train) throws Exception {
		this.train = train;
		predictors = train.numAttributes() - numLabels;

		dfunc = new EuclideanDistance();
		dfunc.setDontNormalize(dontNormalize);
		dfunc.setAttributeIndices("first-" + predictors);
	}

	/**
	 * @return the dontNormalize
	 */
	public boolean isDontNormalize() {
		return dontNormalize;
	}


	/**
	 * @param dontNormalize the dontNormalize to set
	 */
	public void setDontNormalize(boolean dontNormalize) {
		this.dontNormalize = dontNormalize;
	}


	/**
	 * @return the sumedLabels
	 */
	public long getSumedLabels() {
		return sumedLabels;
	}


	/**
	 * @return the predictors
	 */
	public int getPredictors() {
		return predictors;
	}


	/**
	 * @return the numOfNeighbors
	 */
	public int getNumOfNeighbors() {
		return numOfNeighbors;
	}


	@Override
	protected Prediction makePrediction(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

}
