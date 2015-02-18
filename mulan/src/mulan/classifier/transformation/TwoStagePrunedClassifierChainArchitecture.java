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

import java.util.ArrayList;
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
 * <p>Implementation of the Two Stage Pruned Classifier Chain Architecture
 * (TSPCCA) algorithm.</p> <p>For more information see <em>Madjarov, Gj;
 * Gjorgjevikj, D.; Dzeroski, S. (2012) Two stage architecture for multi-label
 * learning. Pattern Recognition. 45(3):1019-1034.</em></p>
 *
 * @author Gjorgji Madjarov
 * @version 2013.01.22
 */
public class TwoStagePrunedClassifierChainArchitecture extends TransformationBasedMultiLabelLearner {

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
    public TwoStagePrunedClassifierChainArchitecture() {
        super(new J48());
    }

    /**
     * Constructor that initializes the learner with a base algorithm
     *
     * @param classifier the binary classification algorithm to use
     */
    public TwoStagePrunedClassifierChainArchitecture(Classifier classifier) {
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
     * @param threshold threshold value to set
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

        // One-vs-one models
        numModels = ((numLabels) * (numLabels - 1)) / 2;
        oneVsOneModels = AbstractClassifier.makeCopies(getBaseClassifier(), numModels);
        nodata = new boolean[numModels];
        metaDataTest = new Instances[numModels];

        ArrayList<MultiLabelOutput> predictions;
        predictions = predictLabels(trainingSet);

        int counter = 0;
        // Creation of one-vs-one models
        for (int label1 = 0; label1 < numLabels - 1; label1++) {
            for (int label2 = label1 + 1; label2 < numLabels; label2++) {
                //Generate the chain: Test the same dataset
                MultiLabelInstances tempTrainingSet = GenerateChain(trainingSet, label1, label2, predictions);

                Instances trainingData = tempTrainingSet.getDataSet();

                labelIndices = tempTrainingSet.getLabelIndices();
                featureIndices = tempTrainingSet.getFeatureIndices();

                // Attribute of label 1
                Attribute attrLabel1 = trainingData.attribute(labelIndices[label1]);

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
     * @param instance the instance for which the prediction is to be made
     * @return prediction the prediction made
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        return makePredictionTSCCA(instance);
    }

    /**
     * This method does a prediction for an instance with the values of label
     * missing according to Two Stage Voting Method (TSVM), which is described
     * in : Madjarov, Gj., Gjorgjevikj, D. and Dzeroski, S. Efficient two stage
     * voting architecture for pairwise multi-label classification. In AI 2010:
     * Advances in Artificial Intelligence (J. Li, ed.), vol. 6464 of Lecture
     * Notes in Computer Science, pp. 164–173, 2011
     *
     * @param instance
     * @return prediction
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    private MultiLabelOutput makePredictionTSCCA(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];
        int[] voteLabel = new int[numLabels + 1];
        int[] noVoteLabel = new int[numLabels + 1];
        int[] voteFromVirtualModels = new int[numLabels];
        double[] confidenceFromVirtualModels = new double[numLabels];

        //System.out.println("Instance:" + instance.toString());

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

        int counter = 0;
        for (int label1 = 0; label1 < numLabels - 1; label1++) {
            for (int label2 = label1 + 1; label2 < numLabels; label2++) {
                Instance newInstanceFirstStage;
                //add predictions from the vurtual models
                if (instance instanceof SparseInstance) {
                    newInstanceFirstStage = modifySparseInstance(instance, virtualMLO.getConfidences()[label1], virtualMLO.getConfidences()[label2]);
                } else {
                    newInstanceFirstStage = modifyDenseInstance(instance, virtualMLO.getConfidences()[label1], virtualMLO.getConfidences()[label2]);
                }

                // delete all labels and add a new atribute at the end
                Instance newInstance = RemoveAllLabels.transformInstance(newInstanceFirstStage, labelIndices);
                newInstance.insertAttributeAt(newInstance.numAttributes());

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

    private ArrayList<MultiLabelOutput> predictLabels(MultiLabelInstances trainingSet) throws Exception {
        ArrayList<MultiLabelOutput> preditions = new ArrayList<MultiLabelOutput>(trainingSet.getDataSet().numInstances());

        for (int i = 0; i < trainingSet.getDataSet().numInstances(); i++) {
            MultiLabelOutput output = virtualLabelModels.makePrediction(trainingSet.getDataSet().instance(i));
            preditions.add(output);
        }

        return preditions;
    }

    private MultiLabelInstances GenerateChain(MultiLabelInstances trainingSet, int label1, int label2, ArrayList<MultiLabelOutput> predictions) throws Exception {


        MultiLabelInstances tempTrainingSet = new MultiLabelInstances(new Instances(trainingSet.getDataSet(), trainingSet.getDataSet().numInstances()), trainingSet.getLabelsMetaData());

        tempTrainingSet.getDataSet().insertAttributeAt(new Attribute("0vs" + label1), 0);
        tempTrainingSet.getDataSet().insertAttributeAt(new Attribute("0vs" + label2), 0);

        for (int i = 0; i < trainingSet.getDataSet().numInstances(); i++) {

            Instance transformed;

            if (trainingSet.getDataSet().instance(i) instanceof SparseInstance) {
                transformed = modifySparseInstance(trainingSet.getDataSet().instance(i), predictions.get(i).getConfidences()[label1], predictions.get(i).getConfidences()[label2]);
            } else {
                transformed = modifyDenseInstance(trainingSet.getDataSet().instance(i), predictions.get(i).getConfidences()[label1], predictions.get(i).getConfidences()[label2]);
            }

            tempTrainingSet.getDataSet().add(transformed);
            transformed.setDataset(tempTrainingSet.getDataSet());
        }

        return tempTrainingSet;

//        Instances td = new Instances(trainingSet.getDataSet(), trainingSet.getDataSet().numInstances());

//        for(int i = trainingSet.getNumLabels()-1; i>=0; i--)
//        {
//            td.insertAttributeAt(new Attribute("0vs" + i+1), 0);
//        }
//
//        for (int i = 0; i < trainingSet.getDataSet().numInstances(); i++)
//        {
//
//            MultiLabelOutput output = virtualLabelModels.makePrediction(trainingSet.getDataSet().instance(i));
//
//            Instance transformed = modifyInstance(trainingSet.getDataSet().instance(i), output.getBipartition());
//            td.add(transformed);
//            transformed.setDataset(td);
//        }
//
//        return td;
    }

    private Instance modifySparseInstance(Instance ins, double value1, double value2) {
        SparseInstance modifiedIns = new SparseInstance(ins);
        modifiedIns.insertAttributeAt(0);
        modifiedIns.setValue(0, value1);
        modifiedIns.insertAttributeAt(0);
        modifiedIns.setValue(0, value2);
        return modifiedIns;
    }

    public Instance modifyDenseInstance(Instance ins, double value1, double value2) {
        Instance modifiedIns = new DenseInstance(ins);
        modifiedIns.insertAttributeAt(0);
        modifiedIns.setValue(0, value1);
        modifiedIns.insertAttributeAt(0);
        modifiedIns.setValue(0, value2);
        return modifiedIns;
    }
}