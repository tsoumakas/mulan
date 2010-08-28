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
 *    CalibratedLabelRanking.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.transformation;

import java.util.Arrays;

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;

/**
 *
 * <!-- globalinfo-start -->
 *
 * <pre>
 * Class implementing the Calibrated Label Ranking algorithm.
 * </pre>
 *
 * For more information:
 *
 * <pre>
 * Fuernkranz, J., Huellermeier, E., Loza Mencia, E., and Brinker, K. (2008)
 * Multilabel classification via calibrated label ranking.
 * Machine Learning 73(2), 133-153
 * </pre>
 *
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#064;article{furnkranze+etal:2008,
 *    author = {Fuernkranz, J. and Huellermeier, E. and Loza Mencia, E. and Brinker, K.},
 *    title = {Multilabel classification via calibrated label ranking},
 *    journal = {Machine Learning},
 *    volume = {73},
 *    number = {2},
 *    year = {2008},
 *    pages = {133--153},
 * }
 * </pre>
 *
 * <p/> <!-- technical-bibtex-end -->
 *
 * @author Elise Rairat
 * @author Grigorios Tsoumakas
 * @author Sang-Hyeun Park
 * @version $Revision: 1.0 $
 */
public class CalibratedLabelRanking extends TransformationBasedMultiLabelLearner {

    /** array holding the one vs one models */
    protected Classifier[] oneVsOneModels;
    /** number of one vs one models */
    protected int numModels;
    /** temporary training data for each one vs one model */
    protected Instances trainingdata;
    /** headers of the training sets of the one vs one models */
    protected Instances[] metaDataTest;
    /** binary relevance models for the virtual label */
    protected BinaryRelevance virtualLabelModels;
    /** whether to use standard voting or the fast qweighted algorithm */
    private boolean useStandardVoting = true;
    /** whether no data exist for one-vs-one learning */
    protected boolean[] nodata;

    /**
     * Constructor that initializes the learner with a base algorithm
     *
     * @param classifier the binary classification algorithm to use
     */
    public CalibratedLabelRanking(Classifier classifier) {
        super(classifier);
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

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        // Virtual label models
        debug("Building calibration label models");
        virtualLabelModels = new BinaryRelevance(getBaseClassifier());
        virtualLabelModels.setDebug(getDebug());
        virtualLabelModels.build(trainingSet);

        // One-vs-one models
        numModels = ((numLabels) * (numLabels - 1)) / 2;
        oneVsOneModels = AbstractClassifier.makeCopies(getBaseClassifier(), numModels);
        nodata = new boolean[numModels];
        metaDataTest = new Instances[numModels];

        Instances trainingData = trainingSet.getDataSet();

        int counter = 0;
        // Creation of one-vs-one models
        for (int label1 = 0; label1 < numLabels - 1; label1++) {
            // Attribute of label 1
            Attribute attrLabel1 = trainingData.attribute(labelIndices[label1]);
            for (int label2 = label1 + 1; label2 < numLabels; label2++) {
                debug("Building one-vs-one model " + (counter + 1) + "/" + numModels);
                // Attribute of label 2
                Attribute attrLabel2 = trainingData.attribute(labelIndices[label2]);

                // initialize training set
                Instances dataOneVsOne = new Instances(trainingData, 0);
                // filter out examples with no preference
                for (int i = 0; i < trainingData.numInstances(); i++) {
                    Instance tempInstance;
                    if (trainingData.instance(i) instanceof SparseInstance) {
                        tempInstance = new SparseInstance(trainingData.instance(i));
                    } else {
                        tempInstance = new DenseInstance(trainingData.instance(i));
                    }

                    int nominalValueIndex;
                    nominalValueIndex = (int) tempInstance.value(labelIndices[label1]);
                    String value1 = attrLabel1.value(nominalValueIndex);
                    nominalValueIndex = (int) tempInstance.value(labelIndices[label2]);
                    String value2 = attrLabel2.value(nominalValueIndex);

                    if (!value1.equals(value2)) {
                        tempInstance.setValue(attrLabel1, value1);
                        dataOneVsOne.add(tempInstance);
                    }
                }

                // remove all labels apart from label1 and place it at the end
                Reorder filter = new Reorder();
                int numPredictors = trainingData.numAttributes() - numLabels;
                int[] reorderedIndices = new int[numPredictors + 1];
                for (int i = 0; i < numPredictors; i++) {
                    reorderedIndices[i] = featureIndices[i];
                }
                reorderedIndices[numPredictors] = labelIndices[label1];
                filter.setAttributeIndicesArray(reorderedIndices);
                filter.setInputFormat(dataOneVsOne);
                dataOneVsOne = Filter.useFilter(dataOneVsOne, filter);
                //System.out.println(dataOneVsOne.toString());
                dataOneVsOne.setClassIndex(numPredictors);

                // build model label1 vs label2
                if (dataOneVsOne.size() > 0) {
                    oneVsOneModels[counter].buildClassifier(dataOneVsOne);
                } else {
                    nodata[counter] = true;
                }
                dataOneVsOne.delete();
                metaDataTest[counter] = dataOneVsOne;
                counter++;
            }
        }
    }

    /**
     * This method does a prediction for an instance with the values of label missing
     * Temporary included to switch between standard voting and qweighted multilabel voting
     * @param instance
     * @return prediction
     * @throws java.lang.Exception
     */
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        if (useStandardVoting) {
            return makePredictionStandard(instance);
        } else {
            return makePredictionQW(instance);
        }
    }

    /**
     * This method does a prediction for an instance with the values of label missing
     * @param instance
     * @return prediction
     * @throws java.lang.Exception
     */
    public MultiLabelOutput makePredictionStandard(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];
        int[] voteLabel = new int[numLabels + 1];

        //System.out.println("Instance:" + instance.toString());

        // delete all labels and add a new atribute at the end
        Instance newInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
        newInstance.insertAttributeAt(newInstance.numAttributes());

        //initialize the array voteLabel
        Arrays.fill(voteLabel, 0);

        int counter = 0;
        for (int label1 = 0; label1 < numLabels - 1; label1++) {
            for (int label2 = label1 + 1; label2 < numLabels; label2++) {
                if (!nodata[counter]) {
                    double distribution[] = new double[2];
                    try {
                        newInstance.setDataset(metaDataTest[counter]);
                        distribution = oneVsOneModels[counter].distributionForInstance(newInstance);
                    } catch (Exception e) {
                        System.out.println(e);
                        return null;
                    }
                    int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;
                    // Ensure correct predictions both for class values {0,1} and {1,0}
                    Attribute classAttribute = metaDataTest[counter].classAttribute();

                    if (classAttribute.value(maxIndex).equals("1")) {
                        voteLabel[label1]++;
                    } else {
                        voteLabel[label2]++;
                    }
                }

                counter++;
            }

        }

        int voteVirtual = 0;
        MultiLabelOutput virtualMLO = virtualLabelModels.makePrediction(instance);
        boolean[] virtualBipartition = virtualMLO.getBipartition();
        for (int i = 0; i < numLabels; i++) {
            if (virtualBipartition[i]) {
                voteLabel[i]++;
            } else {
                voteVirtual++;
            }
        }

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
     * This method does a prediction for an instance with the values of label missing
     * according to QWeighted algorithm for Multilabel Classification (QCMLPP2), which is
     * described in :
     * Loza Mencia, E., Park, S.-H., and Fuernkranz, J. (2009)
     * Efficient voting prediction for pairwise multilabel classification.
     * In Proceedings of 17th European Symposium on Artificial
     * Neural Networks (ESANN 2009), Bruges (Belgium), April 2009
     *
     * This method reduces the number of classifier evaluations and guarantees the same
     * Multilabel Output as ordinary Voting. But: the estimated confidences are only
     * approximated. Therefore, ranking-based performances are worse than ordinary voting.
     * @param instance
     * @return prediction
     * @throws java.lang.Exception
     */
    public MultiLabelOutput makePredictionQW(Instance instance) throws Exception {

        int[] voteLabel = new int[numLabels];
        int[] played = new int[numLabels + 1];
        int[][] playedMatrix = new int[numLabels + 1][numLabels + 1];
        int[] sortarr = new int[numLabels + 1];
        double[] limits = new double[numLabels];
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];
        int voteVirtual = 0;
        double limitVirtual = 0.0;
        boolean allEqualClassesFound = false;

        // delete all labels and add a new atribute at the end
        Instance newInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
        newInstance.insertAttributeAt(newInstance.numAttributes());

        //initialize the array voteLabel
        Arrays.fill(voteLabel, 0);

        // evaluate all classifiers of the calibrated label beforehand, #numLabels 1 vs. A evaluations
        MultiLabelOutput virtualMLO = virtualLabelModels.makePrediction(instance);
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

        // apply QWeighted iteratively to estimate all relevant labels until the
        // calibrated label is found
        boolean found = false;
        int pos = 0;
        int player1 = -1;
        int player2 = -1;
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
                    newInstance.setDataset(metaDataTest[modelIndex]);
                    double[] distribution = oneVsOneModels[modelIndex].distributionForInstance(newInstance);
                    int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

                    // Ensure correct predictions both for class values {0,1} and {1,0}
                    Attribute classAttribute = metaDataTest[modelIndex].classAttribute();

                    if (classAttribute.value(maxIndex).equals("1")) {
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
     * a function to get the classifier index for label1 vs label2 (single Round-Robin)
     * in the array of classifiers, oneVsOneModels
     * @param label1
     * @param label2
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
