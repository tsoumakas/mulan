/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    MultiLabelKNN.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.classifier.lazy;

import java.util.Random;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.core.data.MultiLabelInstances;
import weka.core.EuclideanDistance;
import weka.core.TechnicalInformation;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Superclass of all KNN based multi-label algorithms
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * 
 */
@SuppressWarnings("serial")
public abstract class MultiLabelKNN extends MultiLabelLearnerBase  {
    double threshold = 0.5;
    double[] thresholds;

	/** Whether the neighbors should be distance-weighted. */
	protected int distanceWeighting;
	/** no weighting. */
	public static final int WEIGHT_NONE = 1;
	/** weight by 1/distance. */
	public static final int WEIGHT_INVERSE = 2;
	/** weight by 1-distance. */
	public static final int WEIGHT_SIMILARITY = 4;
	//TODO weight each neighbor's vote according to the inverse square of its distance
	/**
	 * Random number generator.
	 */
	Random random = null;
	/**
	 * Sum of predicted labels for all instances
	 */
	protected long sumedLabels;
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
	protected MultiLabelInstances train = null;


	public MultiLabelKNN(int numOfNeighbors) {
		this.numOfNeighbors = numOfNeighbors;
		random = new Random(1); // seed is always 1 to reproduce results
	}

    protected void buildInternal(MultiLabelInstances trainSet) throws Exception {
    	train = trainSet;

        dfunc = new EuclideanDistance();
        dfunc.setDontNormalize(dontNormalize);
        //dfunc.setAttributeIndices("first-" + predictors);
        // TODO: use the meta-data to set the predictor attributes (as they may not appear first)
        int [] labelIndices = train.getLabelIndices();
        String labelIndicesString = "";
        for (int i =0;i<numLabels-1;i++){
        	labelIndicesString += (labelIndices[i]+1) + ",";
        }
        labelIndicesString += (labelIndices[numLabels-1]+1);
        dfunc.setAttributeIndices(labelIndicesString);
        dfunc.setInvertSelection(true);
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
	 * @return the numOfNeighbors
	 */
	public int getNumOfNeighbors() {
		return numOfNeighbors;
	}
/*
	protected MultiLabelOutput makePrediction(Instance instance) throws Exception {
		return null;
	}
*/
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
    
    public TechnicalInformation getTechnicalInformation(){
		return null;//TODO: implement in subclasses
    	
    }
}
