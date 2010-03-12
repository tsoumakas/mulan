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
 *    SubsetLearner.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.meta;

import java.util.ArrayList;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;
import mulan.core.MulanRuntimeException;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * <!-- globalinfo-start -->
 *
 * <pre>
 * Class that splits the set of labels into disjoint subsets according to user
 * specifications and trains a different multi-label learner for each subset
 * </pre>
 *
 *<!-- globalinfo-end -->
 *
 * @author Vasiloudis Theodoros
 * @version $Revision: 0.01 $
 */
public class SubsetLearner extends MultiLabelLearnerBase {

    /**Arraylist containing the MultiLabelLearners that we will train and use to make the predictions*/
    private ArrayList<MultiLabelLearner> multiLabelLearners;
    /**Arraylist containing the FilteredClassifiers that we will train and use to make the predictions*/
    private ArrayList<FilteredClassifier> singleLabelLearners;

    /** Array containing the way the labels will be split */
    private int[][] splitOrder;
    /**Array containing the indices of the labels we are going to remove*/
    private int[][] absoluteIndicesToRemove;
    /**Array containing the Remove objects used to remove the labels for each split*/
    private Remove[] remove;

    /**Base MultiLabelLearner that will be used for training and predictions*/
    protected final MultiLabelLearner baseMultiLabelLearner;
    /**Base Classifier that will be used for training and predictions*/
    protected final Classifier baseClassifier;

    public SubsetLearner(MultiLabelLearner aBaseMultiLabelLearner, Classifier aBaseClassifier, int[][] aSplitOrder) {

        if (aBaseMultiLabelLearner == null) {
            throw new ArgumentNullException("baseMultiLabelLearner");
        }
        if (aBaseClassifier == null) {
            throw new ArgumentNullException("baseClassifier");
        }
        if (aSplitOrder == null) {
            throw new ArgumentNullException("splitOrder");
        }

        baseClassifier = aBaseClassifier;
        baseMultiLabelLearner = aBaseMultiLabelLearner;
        splitOrder = aSplitOrder;
        absoluteIndicesToRemove = new int[splitOrder.length][];
    }

    public SubsetLearner(MultiLabelLearner getbaseMultiLabelLearner, int[][] aSplitOrder) {

        if (getbaseMultiLabelLearner == null) {
            throw new ArgumentNullException("baseMultiLabelLearner");
        }

        if (aSplitOrder == null) {
            throw new ArgumentNullException("splitOrder");
        }

        for (int i = 0; i < aSplitOrder.length; i++) {
            if (aSplitOrder[i].length == 1) {
                throw new MulanRuntimeException("Single label split detected");
            }
        }

        baseClassifier = null;
        baseMultiLabelLearner = getbaseMultiLabelLearner;
        splitOrder = aSplitOrder;
        absoluteIndicesToRemove = new int[splitOrder.length][];
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     * We get the initial dataset through trainingSet. Then for each split as specified by splitOrder
     * we remove the unneeded labels and train the classifiers using a different method for multi-label splits
     * and single label splits.
     * @param trainingSet The initial {@link MultiLabelInstances} dataset
     * @throws Exception
     */
    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {

        int countSingle = 0, countMulti = 0;
        remove = new Remove[splitOrder.length];

        //Get values into absoluteIndicesToRemove
        int numofSplits = splitOrder.length;//Number of sets the main is going to be split into

        for (int r = 0; r < splitOrder.length; r++) {//Initialization required to avoid NullPointer exception
            absoluteIndicesToRemove[r] = new int[numLabels - splitOrder[r].length];
        }

        //Initialize an array containing which labels we want
        boolean[][] Selected = new boolean[splitOrder.length][numLabels];
        for (int i = 0; i < numofSplits; i++) {//Set true for the labels we need to keep
            for (int j = 0; j < splitOrder[i].length; j++) {
                Selected[i][splitOrder[i][j]] = true;
            }
        }

        for (int i = 0; i < numofSplits; i++) {//Get the labels you need to KEEP
            int k = 0;
            for (int j = 0; j < numLabels; j++) {
                if (Selected[i][j] != true) {
                    absoluteIndicesToRemove[i][k] = labelIndices[j];
                    k++;
                }
            }
        }

        //Create the lists which will contain the learners
        multiLabelLearners = new ArrayList<MultiLabelLearner>();
        singleLabelLearners = new ArrayList<FilteredClassifier>();
        countSingle = 0;//Reset the values to zero and reuse the variables
        countMulti = 0;
        //TODO: Add more comments for the procedure
        for (int totalSplitNo = 0; totalSplitNo < splitOrder.length; totalSplitNo++) {
            debug("Building set " + (totalSplitNo + 1) + "/" + splitOrder.length);
            if (splitOrder[totalSplitNo].length > 1) {
                //Remove the unneeded labels
                Instances trainSubset = trainingSet.getDataSet();
                remove[totalSplitNo] = new Remove();
                remove[totalSplitNo].setAttributeIndicesArray(absoluteIndicesToRemove[totalSplitNo]);
                remove[totalSplitNo].setInputFormat(trainSubset);
                remove[totalSplitNo].setInvertSelection(false);
                trainSubset = Filter.useFilter(trainSubset, remove[totalSplitNo]);

                //Reintegrate dataset and train learner
                multiLabelLearners.add(baseMultiLabelLearner.makeCopy());
                multiLabelLearners.get(countMulti).build(trainingSet.reintegrateModifiedDataSet(trainSubset));
                countMulti++;
            } else {
                debug("Single Label model.");
                //Initialize the FilteredClassifiers
                singleLabelLearners.add(new FilteredClassifier());
                singleLabelLearners.get(countSingle).setClassifier(AbstractClassifier.makeCopy(baseClassifier));

                Instances trainSubset = trainingSet.getDataSet();
                //Set the remove filter for the FilteredClassifiers
                remove[totalSplitNo] = new Remove();
                remove[totalSplitNo].setAttributeIndicesArray(absoluteIndicesToRemove[totalSplitNo]);
                remove[totalSplitNo].setInputFormat(trainSubset);
                remove[totalSplitNo].setInvertSelection(false);
                singleLabelLearners.get(countSingle).setFilter(remove[totalSplitNo]);
                //Set the remaining label as the class index
                trainSubset.setClassIndex(labelIndices[splitOrder[totalSplitNo][0]]);

                //Train
                singleLabelLearners.get(countSingle).buildClassifier(trainSubset);
                countSingle++;
            }
        }
    }

 
    /**
     * We make a prediction using a different method depending on whether the split has one or more labels
     * @param instance
     * @return the MultiLabelOutput
     * @throws Exception
     */
    public MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        MultiLabelOutput[] MLO = new MultiLabelOutput[splitOrder.length];
        int singleSplitNo = 0, multiSplitNo = 0;
        boolean[][] BooleanSubsets = new boolean[splitOrder.length][];
        double[][] ConfidenceSubsets = new double[splitOrder.length][];
        for (int r = 0; r < splitOrder.length; r++) {//Initilization required to avoid NullPointer exception
            BooleanSubsets[r] = new boolean[splitOrder[r].length];
            ConfidenceSubsets[r] = new double[splitOrder[r].length];
        }
        boolean[] BipartitionOut = new boolean[numLabels];
        double[] ConfidenceOut = new double[numLabels];


        //We make a prediction for the instance in each seperate dataset
        //The learners have been trained for each seperate dataset in buildInternal
        for (int i = 0; i < splitOrder.length; i++) {
            if (splitOrder[i].length == 1) {//Prediction for single label splits
                double distribution[] = new double[2];
                try {
                    distribution = singleLabelLearners.get(singleSplitNo).distributionForInstance(instance);
                } catch (Exception e) {
                    System.out.println(e);
                    return null;
                }
                int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

                // Ensure correct predictions both for class values {0,1} and {1,0}
                Attribute classAttribute = singleLabelLearners.get(singleSplitNo).getFilter().getOutputFormat().classAttribute();
                BooleanSubsets[i][0] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

                // The confidence of the label being equal to 1
                ConfidenceSubsets[i][0] = distribution[classAttribute.indexOfValue("1")];
                singleSplitNo++;
            } else {//Prediction for multi label splits
                remove[i].input(instance);
                remove[i].batchFinished();
                Instance newInstance = remove[i].output();
                MLO[multiSplitNo] = multiLabelLearners.get(multiSplitNo).makePrediction(newInstance);
                BooleanSubsets[i] = MLO[multiSplitNo].getBipartition();//Get each array of Bipartitions, confidences  from each learner
                ConfidenceSubsets[i] = MLO[multiSplitNo].getConfidences();
                multiSplitNo++;
            }
        }
        //Concatenate the outputs while putting everything in its right place
        for (int i = 0; i < splitOrder.length; i++) {
            for (int j = 0; j < splitOrder[i].length; j++) {
                BipartitionOut[splitOrder[i][j]] = BooleanSubsets[i][j];
                ConfidenceOut[splitOrder[i][j]] = ConfidenceSubsets[i][j];
            }
        }

        MultiLabelOutput mlo = new MultiLabelOutput(BipartitionOut, ConfidenceOut);
        //MultiLabelOutput mlo =  new MultiLabelOutput(BipartitionOut);
        return mlo;

    }

}
