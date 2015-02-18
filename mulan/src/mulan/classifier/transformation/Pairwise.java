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
package mulan.classifier.transformation;

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.transformations.PairwiseTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.*;

/**
 * <p>Class implementing the Ranking by Pairwise Comparisons (RPC) algorithm.
 * For more information, see <em>H&uuml;llermeier, E.; F&uuml;rnkranz, J.;
 * Cheng, W.; Brinker, K. (2008) Label ranking by learning pairwise preferences,
 * Artificial Intelligence 172(16-17):1897-1916</em></p>
 *
 * @author Grigorios Tsoumakas
 * @version 2012.11.1
 */
public class Pairwise extends TransformationBasedMultiLabelLearner {

    /**
     * whether to consider soft or binary outputs
     */
    protected boolean soft;
    /**
     * array holding the one vs one models
     */
    protected Classifier[] oneVsOneModels;
    /**
     * number of one vs one models
     */
    private int numModels;
    /**
     * whether no data exist for one-vs-one learning
     */
    private boolean[] nodata;
    /**
     * transformation
     */
    private PairwiseTransformation pt;

    /**
     * Constructor that initializes the learner with a base algorithm
     *
     * @param classifier the binary classification algorithm to use
     */
    public Pairwise(Classifier classifier) {
        super(classifier);
        soft = false;
    }

    /**
     * Default constructor using J48 as underlying classifier
     */
    public Pairwise() {
        this(new J48());
    }

    public Classifier getModel(int modelIndex) {
        return oneVsOneModels[modelIndex];
    }
    
    public boolean noData(int index) {        
        return nodata[index];
    }
    
    public PairwiseTransformation getTransformation() {
        return pt;
    }

    /**
     * Sets whether to consider the outputs as soft [0..1] or hard {0,1}
     *
     * @param value true for setting soft outputs and false for hard outputs
     */
    public void setSoft(boolean value) {
        soft = value;
    }

    @Override
    protected void buildInternal(MultiLabelInstances train) throws Exception {
        numModels = ((numLabels) * (numLabels - 1)) / 2;
        oneVsOneModels = AbstractClassifier.makeCopies(getBaseClassifier(), numModels);
        nodata = new boolean[numModels];

        debug("preparing shell");
        pt = new PairwiseTransformation(train);

        int counter = 0;
        // Creation of one-vs-one models
        for (int label1 = 0; label1 < numLabels - 1; label1++) {
            for (int label2 = label1 + 1; label2 < numLabels; label2++) {
                debug("Building one-vs-one model " + (counter + 1) + "/" + numModels);
                // initialize training set
                Instances dataOneVsOne = pt.transformInstances(label1, label2);

                // build model label1 vs label2
                if (dataOneVsOne.size() > 0) {
                    oneVsOneModels[counter].buildClassifier(dataOneVsOne);
                } else {
                    nodata[counter] = true;
                }
                counter++;
            }
        }
    }

    /**
     * Calculates the sum of votes/scores for each label by querying all
     * pairwise models
     *
     * @param instance an instance who has passed through the pairwise
     * transformation filter
     * @return an array with the sum of scores/votes for the labels
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public double[] calculateScores(Instance instance) throws Exception {
        Instance transformed = pt.transformInstance(instance);
        double[] scores = new double[numLabels];

        int counter = 0;
        for (int label1 = 0; label1 < numLabels - 1; label1++) {
            for (int label2 = label1 + 1; label2 < numLabels; label2++) {
                if (!nodata[counter]) {
                    double distribution[];
                    distribution = oneVsOneModels[counter].distributionForInstance(transformed);
                    if (!soft) {
                        int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;
                        if (maxIndex == 1) {
                            scores[label1]++;
                        } else {
                            scores[label2]++;
                        }
                    } else {
                        scores[label1] += distribution[1];
                        scores[label2] += distribution[0];
                    }
                    counter++;
                }
            }
        }
        return scores;
    }

    /**
     * This method does a prediction for an unlabeled instance
     *
     * @param instance unlabeled instance
     * @return prediction
     */
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] scores = calculateScores(instance);
        for (int i = 0; i < scores.length; i++) {
            scores[i] /= (numLabels - 1);
        }
        MultiLabelOutput mlo = new MultiLabelOutput(scores);
        return mlo;
    }
}