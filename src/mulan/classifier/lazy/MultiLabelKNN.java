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
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.lazy;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.data.MultiLabelInstances;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Superclass of all KNN based multi-label algorithms
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * 
 */
@SuppressWarnings("serial")
public abstract class MultiLabelKNN extends MultiLabelLearnerBase {

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
    protected Instances train;
    
    /**
     * The default constructor
     */
    public MultiLabelKNN() {
        this.numOfNeighbors = 10;
    }

    /**
     * Initializes the number of neighbors
     *
     * @param numOfNeighbors the number of neighbors
     */
    public MultiLabelKNN(int numOfNeighbors) {
        this.numOfNeighbors = numOfNeighbors;
    }

    protected void buildInternal(MultiLabelInstances trainSet) throws Exception {
        train = new Instances(trainSet.getDataSet());

        dfunc = new EuclideanDistance();
        dfunc.setDontNormalize(dontNormalize);

        // label attributes don't influence distance estimation
        String labelIndicesString = "";
        for (int i = 0; i < numLabels - 1; i++) {
            labelIndicesString += (labelIndices[i] + 1) + ",";
        }
        labelIndicesString += (labelIndices[numLabels - 1] + 1);
        dfunc.setAttributeIndices(labelIndicesString);
        dfunc.setInvertSelection(true);

        lnn = new LinearNNSearch();
        lnn.setDistanceFunction(dfunc);
        lnn.setInstances(train);
        lnn.setMeasurePerformance(false);
    }
    
    @Override
    public boolean isUpdatable() {
        return true;
    }

    /**
     * Sets normalization off or on
     *
     * @param dontNormalize the value of dontNormalize
     */
    public void setDontNormalize(boolean dontNormalize) {
        this.dontNormalize = dontNormalize;
    }

    /**
     * @param distanceWeighting the distanceWeighting to set
     */
    public void setDistanceWeighting(int distanceWeighting) {
        this.distanceWeighting = distanceWeighting;
    }
}
