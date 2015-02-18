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
import weka.core.*;

/**
 * <p>Implementation of the Calibrated Label Ranking (CLR) algorithm.</p> <p>For
 * more information, see <em> F&uuml;rnkranz, J.; H&uuml;llermeier, E.; Loza
 * Menc&iacute;a, E.; Brinker, K. (2008) Multilabel classification via
 * calibrated label ranking. Machine Learning. 73(2):133-153.</em></p>
 *
 * @author Elise Rairat
 * @author Sang-Hyeun Park
 * @author Grigorios Tsoumakas
 * @version 2012.10.31
 */
public class CalibratedLabelRanking extends BinaryAndPairwise {

    /**
     * whether to consider soft or binary outputs
     */
    private boolean soft;
    /**
     * whether to use standard voting or the fast qweighted algorithm
     */
    private boolean useStandardVoting;

    /**
     * Constructor that initializes the learner with a base algorithm
     *
     * @param classifier the binary classification algorithm to use
     */
    public CalibratedLabelRanking(Classifier classifier) {
        super(classifier);
        useStandardVoting = true;
        soft = false;
    }

    /**
     * Sets whether to consider the outputs as soft [0..1] or hard {0,1}
     *
     * @param value <code>true</code> for setting soft outputs and
     * <code>false</code> for hard outputs
     */
    public void setSoft(boolean value) {
        soft = value;
    }

    /**
     * Set Prediction to standard voting mode.
     *
     * @param standardVoting <code>true</code> if standard voting should be used
     */
    public void setStandardVoting(boolean standardVoting) {
        useStandardVoting = standardVoting;
    }

    /**
     * Get whether standard voting is turned on.
     *
     * @return <code>true</code> if standard voting is on
     */
    public boolean getStandardVoting() {
        return useStandardVoting;
    }

    /**
     * This method does a prediction for an instance with the values of label
     * missing Temporary included to switch between standard voting and
     * qweighted multilabel voting
     *
     * @param instance the instance for which the prediction is made
     * @return prediction the prediction made
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        if (useStandardVoting) {
            return makePredictionStandard(instance);
        } else {
            if (!soft) {
                return makePredictionQW(instance);
            } else {
                return makePredictionQWSoft(instance);
            }
        }
    }

    /**
     * This method does a prediction for an instance with the values of label
     * missing
     *
     * @param instance the instance for which the prediction is made
     * @return prediction the prediction made
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    public MultiLabelOutput makePredictionStandard(Instance instance) throws Exception {
        double[] scores = getOneVsOneModels().calculateScores(instance);

        double scoreVirtual = 0;
        MultiLabelOutput virtualMLO = getOneVsRestModels().makePrediction(instance);
        if (!soft) {
            boolean[] virtualBipartition = virtualMLO.getBipartition();
            for (int i = 0; i < numLabels; i++) {
                if (virtualBipartition[i]) {
                    scores[i]++;
                } else {
                    scoreVirtual++;
                }
            }
        } else {
            double[] virtualScores = virtualMLO.getConfidences();
            for (int i = 0; i < numLabels; i++) {
                scores[i] += virtualScores[i];
                scoreVirtual += (1 - virtualScores[i]);
            }
        }

        boolean[] bipartition = new boolean[numLabels];
        for (int i = 0; i < numLabels; i++) {
            if (scores[i] >= scoreVirtual) {
                bipartition[i] = true;
            } else {
                bipartition[i] = false;
            }
        }

        for (int i = 0; i < scores.length; i++) {
            scores[i] /= numLabels;
        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, scores);
        return mlo;
    }

    /**
     * This method does a prediction for an instance with the values of label
     * missing according to QWeighted algorithm for Multilabel Classification
     * (QCMLPP2), which is described in : Loza Mencia, E., Park, S.-H., and
     * Fuernkranz, J. (2009) Efficient voting prediction for pairwise multilabel
     * classification. In Proceedings of 17th European Symposium on Artificial
     * Neural Networks (ESANN 2009), Bruges (Belgium), April 2009
     *
     * This method reduces the number of classifier evaluations and guarantees
     * the same Multilabel Output as ordinary Voting. But: the estimated
     * confidences are only approximated. Therefore, ranking-based performances
     * are worse than ordinary voting.
     *
     * @param instance the instance for which the prediction is made
     * @return prediction the prediction made
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    public MultiLabelOutput makePredictionQW(Instance instance) throws Exception {
        int[] voteLabel = new int[numLabels];
        int[] played = new int[numLabels + 1];
        int[][] playedMatrix = new int[numLabels + 1][numLabels + 1];
        int[] sortarr;
        double[] limits = new double[numLabels];
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];
        int voteVirtual = 0;
        double limitVirtual;
        boolean allEqualClassesFound = false;

        // evaluate all classifiers of the calibrated label beforehand, #numLabels 1 vs. A evaluations
        MultiLabelOutput virtualMLO = getOneVsRestModels().makePrediction(instance);
        boolean[] virtualBipartition = virtualMLO.getBipartition();
        for (int i = 0; i < numLabels; i++) {
            if (virtualBipartition[i]) {
                voteLabel[i]++;
            } else {
                voteVirtual++;
            }

            played[i]++;
            playedMatrix[i][numLabels] = 1;
            playedMatrix[numLabels][i] = 1;
            limits[i] = played[i] - voteLabel[i];
        }
        limitVirtual = numLabels - voteVirtual;
        played[numLabels] = numLabels;

        Instance transformed = getOneVsOneModels().getTransformation().transformInstance(instance);
        // apply QWeighted iteratively to estimate all relevant labels until the
        // calibrated label is found
        boolean found = false;
        int pos = 0;
        int player1 = -1;
        int player2;
        while (!allEqualClassesFound && pos < numLabels) {
            while (!found) {

                // opponent selection process: pair best against second best w.r.t. to number of "lost games"
                // player1 = pick player with min(limits[player]) && player isn't ranked
                sortarr = Utils.sort(limits);
                player1 = sortarr[0];
                player2 = -1;
                int i = 1;

                // can we found unplayed matches of player1 ?
                if (played[player1] < numLabels) {
                    // search for best opponent
                    while (player2 == -1 && i < sortarr.length) {
                        // already played ??
                        if (playedMatrix[player1][sortarr[i]] == 0) {
                            player2 = sortarr[i];
                        }
                        i++;
                    }

                    // play found Pairing and update stats
                    int modelIndex = getRRClassifierIndex(player1, player2);
                    double[] distribution = getOneVsOneModels().getModel(modelIndex).distributionForInstance(transformed);
                    int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;
                    if (maxIndex == 1) {
                        voteLabel[player1 > player2 ? player2 : player1]++;
                    } else {
                        voteLabel[player1 > player2 ? player1 : player2]++;
                    }

                    // update stats
                    played[player1]++;
                    played[player2]++;
                    playedMatrix[player1][player2] = 1;
                    playedMatrix[player2][player1] = 1;
                    limits[player1] = played[player1] - voteLabel[player1];
                    limits[player2] = played[player2] - voteLabel[player2];
                } // full played, there are no opponents left
                else {
                    found = true;
                }
            }

            //arrange already as relevant validated labels at the end of possible opponents
            limits[player1] = Double.MAX_VALUE;

            //check for possible labels, which can still gain greater or equal votes as the calibrated label
            allEqualClassesFound = true;
            for (int i = 0; i < numLabels; i++) {
                if (limits[i] <= limitVirtual) {
                    allEqualClassesFound = false;
                }
            }

            // search for next relevant label
            found = false;
            pos++;
        }

        //Generate Multilabel Output
        for (int i = 0; i < numLabels; i++) {
            if (voteLabel[i] >= voteVirtual) {
                bipartition[i] = true;
            } else {
                bipartition[i] = false;
            }
            confidences[i] = 1.0 * voteLabel[i] / numLabels;
        }
        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }

    /**
     * This method does a prediction for an instance with the values of label
     * missing according to QWeighted algorithm for Multilabel Classification
     * (QCMLPP2), which is described in : Loza Mencia, E., Park, S.-H., and
     * Fuernkranz, J. (2009) Efficient voting prediction for pairwise multilabel
     * classification. In Proceedings of 17th European Symposium on Artificial
     * Neural Networks (ESANN 2009), Bruges (Belgium), April 2009
     *
     * This method reduces the number of classifier evaluations and guarantees
     * the same Multilabel Output as ordinary Voting. But: the estimated
     * confidences are only approximated. Therefore, ranking-based performances
     * are worse than ordinary voting.
     *
     * @param instance the instance for which the prediction is made
     * @return prediction the prediction made
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    public MultiLabelOutput makePredictionQWSoft(Instance instance) throws Exception {
        double[] voteLabel = new double[numLabels];
        int[] played = new int[numLabels + 1];
        int[][] playedMatrix = new int[numLabels + 1][numLabels + 1];
        int[] sortarr;
        double[] limits = new double[numLabels];
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];
        double voteVirtual = 0;
        double limitVirtual;
        boolean allEqualClassesFound = false;

        // evaluate all classifiers of the calibrated label beforehand, #numLabels 1 vs. A evaluations
        MultiLabelOutput virtualMLO = getOneVsRestModels().makePrediction(instance);
        double[] virtualScores = virtualMLO.getConfidences();
        //boolean[] virtualBipartition = virtualMLO.getBipartition();
        for (int i = 0; i < numLabels; i++) {
            voteLabel[i] += virtualScores[i];
            voteVirtual += (1 - virtualScores[i]);

            played[i]++;
            playedMatrix[i][numLabels] = 1;
            playedMatrix[numLabels][i] = 1;
            limits[i] = played[i] - voteLabel[i];
        }
        limitVirtual = numLabels - voteVirtual;
        played[numLabels] = numLabels;

        Instance transformed = getOneVsOneModels().getTransformation().transformInstance(instance);
        // apply QWeighted iteratively to estimate all relevant labels until the
        // calibrated label is found
        boolean found = false;
        int pos = 0;
        int player1 = -1;
        int player2;
        while (!allEqualClassesFound && pos < numLabels) {
            while (!found) {

                // opponent selection process: pair best against second best w.r.t. to number of "lost games"
                // player1 = pick player with min(limits[player]) && player isn't ranked
                sortarr = Utils.sort(limits);
                player1 = sortarr[0];
                player2 = -1;
                int i = 1;

                // can we found unplayed matches of player1 ?
                if (played[player1] < numLabels) {
                    // search for best opponent
                    while (player2 == -1 && i < sortarr.length) {
                        // already played ??
                        if (playedMatrix[player1][sortarr[i]] == 0) {
                            player2 = sortarr[i];
                        }
                        i++;
                    }

                    // play found Pairing and update stats
                    int modelIndex = getRRClassifierIndex(player1, player2);
                    double[] distribution = getOneVsOneModels().getModel(modelIndex).distributionForInstance(transformed);

                    if (player1 > player2) {
                        voteLabel[player1] += distribution[0];
                        voteLabel[player2] += distribution[1];
                    } else {
                        voteLabel[player1] += distribution[1];
                        voteLabel[player2] += distribution[0];
                    }

                    // update stats
                    played[player1]++;
                    played[player2]++;
                    playedMatrix[player1][player2] = 1;
                    playedMatrix[player2][player1] = 1;
                    limits[player1] = played[player1] - voteLabel[player1];
                    limits[player2] = played[player2] - voteLabel[player2];
                } // full played, there are no opponents left
                else {
                    found = true;
                }
            }

            //arrange already as relevant validated labels at the end of possible opponents
            limits[player1] = Double.MAX_VALUE;

            //check for possible labels, which can still gain greater or equal votes as the calibrated label
            allEqualClassesFound = true;
            for (int i = 0; i < numLabels; i++) {
                if (limits[i] <= limitVirtual) {
                    allEqualClassesFound = false;
                }
            }

            // search for next relevant label
            found = false;
            pos++;
        }

        //Generate Multilabel Output
        for (int i = 0; i < numLabels; i++) {
            if (voteLabel[i] >= voteVirtual) {
                bipartition[i] = true;
            } else {
                bipartition[i] = false;
            }
            confidences[i] = 1.0 * voteLabel[i] / numLabels;
        }
        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }

    /**
     * a function to get the classifier index for label1 vs label2 (single
     * Round-Robin) in the array of classifiers, oneVsOneModels
     *
     * @param label1 the first label
     * @param label2 the second label
     * @return index of classifier (label1 vs label2)
     */
    private int getRRClassifierIndex(int label1, int label2) {
        int l1 = label1 > label2 ? label2 : label1;
        int l2 = label1 > label2 ? label1 : label2;

        if (l1 == 0) {
            return (l2 - 1);
        } else {
            int temp = 0;
            for (int i = l1; i > 0; i--) {
                temp += (numLabels - i);
            }
            temp += l2 - (l1 + 1);
            return temp;
        }
    }
}