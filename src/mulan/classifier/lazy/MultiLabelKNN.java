package mulan.classifier.lazy;

import java.util.Random;

import mulan.classifier.AbstractMultiLabelClassifier;
import mulan.classifier.Prediction;

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

	/** Whether the neighbors should be distance-weighted. */
	protected int distanceWeighting;
	/** no weighting. */
	public static final int WEIGHT_NONE = 1;
	/** weight by 1/distance. */
	public static final int WEIGHT_INVERSE = 2;
	/** weight by 1-distance. */
	public static final int WEIGHT_SIMILARITY = 4;
	/**
	 * Random number generator.
	 */
	Random random = null;
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

	public MultiLabelKNN() {
	}

	public MultiLabelKNN(int numLabels, int numOfNeighbors) {
		super(numLabels);
		this.numOfNeighbors = numOfNeighbors;
		random = new Random(1); // seed is always 1 to reproduce results
	}

    @Override
    public void buildClassifier(Instances train) throws Exception {
        this.train = train;
        predictors = train.numAttributes() - numLabels;

        dfunc = new EuclideanDistance();
        dfunc.setDontNormalize(dontNormalize);
        dfunc.setAttributeIndices("first-" + predictors);
    }

	/**
	 * Derive output labels from distribution
	 * currently not in use
	 */
	protected double[] labelsFromConfidencesRandom(double[] confidences) {
		double[] result = new double[confidences.length];
		for (int i = 0; i < result.length; i++) {
			if (confidences[i] > threshold) {
				result[i] = 1.0;
			} else if (confidences[i] == threshold) {
				result[i] = random.nextInt(2);
			}
		}
		return result;
	}

	/**
	 * @return the dontNormalize
	 */
	public boolean isDontNormalize() {
		return dontNormalize;
	}

	/**
	 * @param dontNormalize
	 *            the dontNormalize to set
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
		return null;
	}

	/**
	 * @return the distanceWeighting
	 */
	public int getDistanceWeighting() {
		return distanceWeighting;
	}

    /**
     * @param distanceWeighting the distanceWeighting to set
     */
    public void setDistanceWeighting(int distanceWeighting) {
        this.distanceWeighting = distanceWeighting;
    }

	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}


}
