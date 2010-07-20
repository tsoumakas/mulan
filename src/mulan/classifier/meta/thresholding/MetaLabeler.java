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
 *    Meta.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.meta.thresholding;

import java.util.ArrayList;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 *
 * @author Marios Ioannou
 * @author George Sakkas
 * @author Grigorios Tsoumakas
 * @version 0.1
 */
public class MetaLabeler extends Meta {

    /** the type of the class*/
    private String classChoice;

    /**
     * Constructor that initializes the learner
     *
     * @param baseLearner the underlying multi-label learner
     * @param classifier the binary classification
     * @param kFolds the number of folds for cross validation
     */
    public MetaLabeler(MultiLabelLearner baseLearner, Classifier classifier, int kFolds) {
        super(baseLearner, classifier, kFolds);
        classChoice = "Nominal-Class";
    }

    public void chooseNumericClass() {
        classChoice = "Numeric-Class";
    }

    public void chooseNominalClass() {
        classChoice = "Nominal-Class";
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Lei Tang and Sugu Rajan and Yijay K. Narayanan");
        result.setValue(Field.TITLE, "Large scale multi-label classification via metalabeler");
        result.setValue(Field.BOOKTITLE, "Proceedings of the 18th international conference on World wide web ");
        result.setValue(Field.PAGES, "211-220");
        result.setValue(Field.LOCATION, "Madrid, Spain");
        result.setValue(Field.YEAR, "2009");
        return result;
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        MultiLabelOutput mlo = baseLearner.makePrediction(instance);
        int[] arrayOfRankink = new int[numLabels];
        boolean[] predictedLabels = new boolean[numLabels];
        Instance modifiedIns = modifiedInstanceX(instance, metaDatasetChoice);

        modifiedIns.insertAttributeAt(modifiedIns.numAttributes());
        // set dataset to instance
        modifiedIns.setDataset(header);
        //get the bipartition_key after classify the instance
        int bipartition_key;
        if (classChoice.compareTo("Nominal-Class") == 0) {
            double classify_key = classifier.classifyInstance(modifiedIns);
            String s = header.attribute(header.numAttributes() - 1).value((int) classify_key);
            bipartition_key = Integer.valueOf(s);
        } else { //Numeric-Class
            double classify_key = classifier.classifyInstance(modifiedIns);
            bipartition_key = (int) Math.round(classify_key);
        }
        if (mlo.hasRanking()) {
            arrayOfRankink = mlo.getRanking();
            for (int i = 0; i < numLabels; i++) {
                if (arrayOfRankink[i] <= bipartition_key) {
                    predictedLabels[i] = true;
                } else {
                    predictedLabels[i] = false;
                }
            }
        }
        MultiLabelOutput final_mlo = new MultiLabelOutput(predictedLabels, mlo.getConfidences());
        return final_mlo;
    }

    public Instances transformData(MultiLabelInstances trainingData) throws Exception {
        // copy existing attributes
        ArrayList<Attribute> atts = createHeader(trainingData, metaDatasetChoice, classChoice);

        // initialize  classifier instances
        Instances classifierInstances = new Instances(trainingData.getDataSet().relationName(), atts,
                trainingData.getDataSet().numInstances());
        classifierInstances.setClassIndex(classifierInstances.numAttributes() - 1);

        for (int k = 0; k < kFoldsCV; k++) {
            //Split data to train and test sets
            MultiLabelInstances mlTest;
            if (kFoldsCV == 1) {
                mlTest = trainingData.clone();
                baseLearner.build(mlTest);
            } else {
                Instances train = trainingData.getDataSet().trainCV(kFoldsCV, k);
                Instances test = trainingData.getDataSet().testCV(kFoldsCV, k);
                MultiLabelInstances mlTrain = new MultiLabelInstances(train, trainingData.getLabelsMetaData());
                mlTest = new MultiLabelInstances(test, trainingData.getLabelsMetaData());
                baseLearner.build(mlTrain);
            }

            // copy features and labels, set metalabels
            for (int instanceIndex = 0; instanceIndex < mlTest.getDataSet().numInstances(); instanceIndex++) {
                // initialize new class values
                double[] newValues = new double[classifierInstances.numAttributes()];
                // copy features
                valuesX(mlTest, newValues, metaDatasetChoice, instanceIndex);
                //set the number of true labels of an instance
                int countTrueLabels = 0;
                for (int i = 0; i < numLabels; i++) {
                    int labelIndice = labelIndices[i];
                    if (mlTest.getDataSet().attribute(labelIndice).value((int) mlTest.getDataSet().instance(instanceIndex).value(labelIndice)).equals("1")) {
                        countTrueLabels++;
                    }
                }

                if (classChoice.compareTo("Nominal-Class") == 0) {
                    newValues[newValues.length - 1] = classifierInstances.attribute(classifierInstances.numAttributes() - 1).indexOfValue("" + countTrueLabels);
                } else if (classChoice.compareTo("Numeric-Class") == 0) {
                    newValues[newValues.length - 1] = countTrueLabels;
                }
                // add the new insatnce to  classifierInstances
                Instance newInstance = DataUtils.createInstance(mlTest.getDataSet().instance(instanceIndex), mlTest.getDataSet().instance(instanceIndex).weight(), newValues);
                classifierInstances.add(newInstance);
            }
        }
        return classifierInstances;
    }
}
