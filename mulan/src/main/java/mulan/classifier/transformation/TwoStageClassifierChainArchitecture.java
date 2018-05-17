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

import java.util.Arrays;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;

/**
 * <p>Implementation of the Two Stage Classifier Chain Architecture (TSCCA)
 * algorithm.</p> <p>For more information see <em>Madjarov, Gj; Gjorgjevikj, D.;
 * Dzeroski, S. (2012) Two stage architecture for multi-label learning. Pattern
 * Recognition. 45(3):1019-1034.</em></p>
 *
 * @author Gjorgji Madjarov
 * @version 2013.01.22
 */
public class TwoStageClassifierChainArchitecture extends TransformationBasedMultiLabelLearner {

    /**
     * array holding the one vs one models
     */
    protected Classifier[] oneVsOneModels;
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
     * binary relevance models for the virtual label
     */
    protected BinaryRelevance virtualLabelModels;
    /**
     * whether no data exist for one-vs-one learning
     */
    protected boolean[] nodata;
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
    public TwoStageClassifierChainArchitecture() {
        super(new J48());
    }

    /**
     * Constructor that initializes the learner with a base algorithm
     *
     * @param classifier the binary classification algorithm to use
     */
    public TwoStageClassifierChainArchitecture(Classifier classifier) {
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

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        // Virtual label models
        debug("Building calibration label models");
        virtualLabelModels = new BinaryRelevance(getBaseClassifier());
        virtualLabelModels.setDebug(getDebug());
        virtualLabelModels.build(trainingSet);

        //Generate the chain: Test the same dataset
        MultiLabelInstances tempTrainingSet = GenerateChain(trainingSet);

        labelIndices = tempTrainingSet.getLabelIndices();
        featureIndices = tempTrainingSet.getFeatureIndices();

        // One-vs-one models
        numModels = ((numLabels) * (numLabels - 1)) / 2;
        oneVsOneModels = AbstractClassifier.makeCopies(getBaseClassifier(), numModels);
        nodata = new boolean[numModels];
        metaDataTest = new Instances[numModels];

        Instances trainingData = tempTrainingSet.getDataSet();

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

                System.arraycopy(featureIndices, 0, reorderedIndices, 0, numPredictors);
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
     * This method does a prediction for an instance with the values of label
     * missing Temporary included to switch between standard voting and
     * qweighted multilabel voting
     *
     * @param instance the instance used 
     * @return prediction the prediction made
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        return makePredictionTSCCV(instance);
    }

    /**
     * This method does a prediction for an instance with the values of label
     * missing according to Two Stage Pruned Classifier Chain (TSPCCA), which is
     * described in : Madjarov, Gj., Gjorgjevikj, D. and Dzeroski, S. Two stage
     * architecture for multi-label learning Pattern Recognition, vol. 45, pp.
     * 1019–1034, 2012
     *
     * @param instance the instance used
     * @return prediction the prediction made
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    private MultiLabelOutput makePredictionTSCCV(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];
        int[] voteLabel = new int[numLabels + 1];
        int[] noVoteLabel = new int[numLabels + 1];
        int[] voteFromVirtualModels = new int[numLabels];
        double[] confidenceFromVirtualModels = new double[numLabels];

        //initialize the array voteLabel
        Arrays.fill(voteLabel, 0);
        Arrays.fill(noVoteLabel, 0);
        Arrays.fill(voteFromVirtualModels, 0);
        Arrays.fill(confidenceFromVirtualModels, 0.0);


        int voteVirtual = 0;
        MultiLabelOutput virtualMLO = virtualLabelModels.makePrediction(instance);
        boolean[] virtualBipartition = virtualMLO.getBipartition();

        //number of classifiers of the first layer that forward the instance to the second layer
        int forwards = 0;

        for (int i = 0; i < numLabels; i++) {
            if (virtualMLO.hasConfidences()) {
                confidenceFromVirtualModels[i] = virtualMLO.getConfidences()[i];
                //System.out.print(confidenceFromVirtualModels[i]);
                //System.out.print("\t");
            }
            if (virtualBipartition[i]) {
                voteLabel[i]++;
                voteFromVirtualModels[i]++;
            } else {
                voteVirtual++;
            }

            if (confidenceFromVirtualModels[i] > threshold) {
                forwards++;
            }
        }


        Instance newInstanceFirstStage;
        //add predictions from the vurtual models
        if (instance instanceof SparseInstance) {
            newInstanceFirstStage = modifySparseInstance(instance, virtualMLO.getConfidences());
        } else {
            newInstanceFirstStage = modifyDenseInstance(instance, virtualMLO.getConfidences());
        }

        // delete all labels and add a new atribute at the end
        Instance newInstance = RemoveAllLabels.transformInstance(newInstanceFirstStage, labelIndices);
        newInstance.insertAttributeAt(newInstance.numAttributes());

        int counter = 0;
        for (int label1 = 0; label1 < numLabels - 1; label1++) {
            for (int label2 = label1 + 1; label2 < numLabels; label2++) {
                if (!nodata[counter]) {
                    if (confidenceFromVirtualModels[label1] > threshold && confidenceFromVirtualModels[label2] > threshold) {
                        double distribution[];
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
                //confidences[i]=confidenceFromVirtualModels[i];
            }
            //System.out.println(bipartition[i]);
            //System.out.println(confidences[i]);
            //confidences[i]*=confidenceFromVirtualModels[i];
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

    private MultiLabelInstances GenerateChain(MultiLabelInstances trainingSet) throws Exception {
        MultiLabelInstances tempTrainingSet = new MultiLabelInstances(new Instances(trainingSet.getDataSet(), trainingSet.getDataSet().numInstances()), trainingSet.getLabelsMetaData());


        for (int i = trainingSet.getNumLabels() - 1; i >= 0; i--) {
            tempTrainingSet.getDataSet().insertAttributeAt(new Attribute("0vs" + i + 1), 0);
        }

        for (int i = 0; i < trainingSet.getDataSet().numInstances(); i++) {

            MultiLabelOutput output = virtualLabelModels.makePrediction(trainingSet.getDataSet().instance(i));

            Instance transformed;

            if (trainingSet.getDataSet().instance(i) instanceof SparseInstance) {
                transformed = modifySparseInstance(trainingSet.getDataSet().instance(i), output.getConfidences());
            } else {
                transformed = modifyDenseInstance(trainingSet.getDataSet().instance(i), output.getConfidences());
            }
            tempTrainingSet.getDataSet().add(transformed);
            transformed.setDataset(tempTrainingSet.getDataSet());
        }

        return tempTrainingSet;
    }

    private Instance modifySparseInstance(Instance instance, double[] confidences) {
        SparseInstance modifiedIns = new SparseInstance(instance);
        for (int i = confidences.length - 1; i >= 0; i--) {
            modifiedIns.insertAttributeAt(0);
            modifiedIns.setValue(0, confidences[i]);
        }
        return modifiedIns;
    }

    private Instance modifyDenseInstance(Instance instance, double[] confidences) {
        Instance modifiedIns = new DenseInstance(instance);
        for (int i = confidences.length - 1; i >= 0; i--) {
            modifiedIns.insertAttributeAt(0);
            modifiedIns.setValue(0, confidences[i]);
        }
        return modifiedIns;
    }
}