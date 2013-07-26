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
package mulan.classifier.lazy;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.data.MultiLabelInstances;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Superclass of all KNN based multi-label algorithms
 *
 * @author Eleftherios Spyromitros-Xioufis
 * @author Grigorios Tsoumakas
 */
@SuppressWarnings("serial")
public abstract class MultiLabelKNN extends MultiLabelLearnerBase {

    /**
     * Whether the neighbors should be distance-weighted.
     */
    protected int distanceWeighting;
    /**
     * no weighting.
     */
    public static final int WEIGHT_NONE = 1;
    /**
     * weight by 1/distance.
     */
    public static final int WEIGHT_INVERSE = 2;
    /**
     * weight by 1-distance.
     */
    public static final int WEIGHT_SIMILARITY = 4;
    // TODO weight each neighbor's vote according to the inverse square of its distance
    /**
     * Number of neighbors used in the k-nearest neighbor algorithm
     */
    protected int numOfNeighbors;
    /**
     * Class implementing the brute force search algorithm for nearest neighbor
     * search. Default value is true.
     */
    protected LinearNNSearch lnn = null;
    /**
     * Implementing Euclidean distance (or similarity) function.
     */
    protected DistanceFunction dfunc = null;

    /**
     * Sets a distance function
     * @param dfunc the distance function
     */
    public void setDfunc(DistanceFunction dfunc) {
        this.dfunc = dfunc;
    }
    /**
     * The training instances
     */
    protected Instances train;

    /**
     * The default constructor
     */
    public MultiLabelKNN() {
        this.numOfNeighbors = 10;
        dfunc = new EuclideanDistance();
    }

    /**
     * Initializes the number of neighbors
     *
     * @param numOfNeighbors the number of neighbors
     */
    public MultiLabelKNN(int numOfNeighbors) {
        this.numOfNeighbors = numOfNeighbors;
        dfunc = new EuclideanDistance();
    }

    protected void buildInternal(MultiLabelInstances trainSet) throws Exception {
        if (trainSet.getNumInstances() < numOfNeighbors) {
            throw new IllegalArgumentException("The number of training instances is less than the number of requested nearest neighbours");
        }
        train = new Instances(trainSet.getDataSet());

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
     * @param distanceWeighting the distanceWeighting to set
     */
    public void setDistanceWeighting(int distanceWeighting) {
        this.distanceWeighting = distanceWeighting;
    }
}