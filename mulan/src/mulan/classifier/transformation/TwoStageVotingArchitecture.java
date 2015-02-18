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
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.*;

/**
 * <p>Implementation of the Two Stage Voting Architecture (TSVA) algorithm.</p>
 * <p>For more information see <em>Madjarov, Gj; Gjorgjevikj, D.; Dzeroski, S.
 * (2012) Two stage architecture for multi-label learning. Pattern Recognition.
 * 45(3):1019-1034.</em></p>
 *
 * @author Gjorgji Madjarov
 * @version 2012.11.25
 */
public class TwoStageVotingArchitecture extends BinaryAndPairwise {

    /**
     * number of one vs one models
     */
    protected int numModels;
    /**
     * temporary training data for each one vs one model
     */
    protected Instances trainingdata;
    /**
     * headers of the training sets of the one vs one models
     */
    protected Instances[] metaDataTest;
    /**
     * In two stage architecture how many models from the first stage forwards a
     * test example to the second stage
     */
    protected int avgForwards = 0;
    /**
     * threshold for efficient two stage strategy
     */
    private double threshold = 0.2;

    /**
     * Default constructor using J48 as underlying classifier
     */
    public TwoStageVotingArchitecture() {
        super(new J48());
    }

    /**
     * Constructor that initializes the learner with a base algorithm
     *
     * @param classifier the binary classification algorithm to use
     */
    public TwoStageVotingArchitecture(Classifier classifier) {
        super(classifier);
    }

    /**
     * Get threshold of Two Stage Voting Architecture.
     *
     * @return the actual value of the threshold
     */
    public double getTreshold() {
        return threshold;
    }

    /**
     * Set threshold to concrete value.
     *
     * @param threshold the threshold
     */
    public void setTreshold(double threshold) {
        this.threshold = threshold;
    }

    /**
     * This method does a prediction for an instance with the values of label
     * missing Temporary included to switch between standard voting and
     * qweighted multilabel voting
     *
     * @param instance the instance used
     * @return prediction the prediction made
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];


        int[] noVoteLabel = new int[numLabels + 1];

        MultiLabelOutput virtualMLO = getOneVsRestModels().makePrediction(instance);
        boolean[] virtualBipartition = virtualMLO.getBipartition();

        //number of classifiers of the first layer that forward the instance to the second layer
        int forwards = 0;

        int voteVirtual = 0;
        int[] voteLabel = new int[numLabels];
        double[] confidenceFromVirtualModels = virtualMLO.getConfidences();
        for (int i = 0; i < numLabels; i++) {
            if (virtualBipartition[i]) {
                voteLabel[i]++;
            } else {
                voteVirtual++;
            }

            if (confidenceFromVirtualModels[i] > threshold) {
                forwards++;
            }
        }

        Instance transformed = getOneVsOneModels().getTransformation().transformInstance(instance);

        int counter = 0;
        for (int label1 = 0; label1 < numLabels - 1; label1++) {
            for (int label2 = label1 + 1; label2 < numLabels; label2++) {
                if (!getOneVsOneModels().noData(counter)) {
                    if (confidenceFromVirtualModels[label1] > threshold && confidenceFromVirtualModels[label2] > threshold) {
                        double distribution[];
                        distribution = getOneVsOneModels().getModel(counter).distributionForInstance(transformed);
                        int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;
                        if (maxIndex == 1) {
                            voteLabel[label1]++;
                        } else {
                            voteLabel[label2]++;
                        }
                    } else if (confidenceFromVirtualModels[label1] > threshold) {
                        voteLabel[label1]++;
                    } else if (confidenceFromVirtualModels[label2] > threshold) {
                        voteLabel[label2]++;
                    } else {
                        noVoteLabel[label1]++;
                        noVoteLabel[label2]++;
                    }
                }

                counter++;
            }

        }

        avgForwards += forwards;

        for (int i = 0; i < numLabels; i++) {
            if (voteLabel[i] >= voteVirtual) {
                bipartition[i] = true;
                confidences[i] = (1.0 * voteLabel[i]) / (numLabels - noVoteLabel[i]);
            } else {
                bipartition[i] = false;
                confidences[i] = 1.0 * confidenceFromVirtualModels[i] / numLabels;
            }
        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }
}