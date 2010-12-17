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
 *    MetaLabeler.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.meta.thresholding;

import java.util.ArrayList;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;

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
 * @version 2010.12.14
 */
public class MetaLabeler extends Meta {

    /** the type of the class*/
    private String classChoice;

    /**
     * Constructor that initializes the learner
     *
     * @param baseLearner the underlying multi-label learner
     * @param classifier the binary classification
     * @param metaDataChoice the type of meta-data
     * @param aClassChoice the type of the class
     */
    public MetaLabeler(MultiLabelLearner baseLearner, Classifier classifier, String metaDataChoice, String aClassChoice) {
        super(baseLearner, classifier, metaDataChoice);
        if (!metaDataChoice.equals("Content-Based")) {
            try {
                foldLearner = baseLearner.makeCopy();
            } catch (Exception ex) {
                Logger.getLogger(MetaLabeler.class.getName()).log(Level.SEVERE, null, ex);
            }
            kFoldsCV = 3;
        }
        classChoice = aClassChoice;
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
        //System.out.println(instance);
        MultiLabelOutput mlo = baseLearner.makePrediction(instance);
        int[] arrayOfRankink = new int[numLabels];
        boolean[] predictedLabels = new boolean[numLabels];
        Instance modifiedIns = modifiedInstanceX(instance, metaDatasetChoice);
        //System.out.println(modifiedIns);
        modifiedIns.insertAttributeAt(modifiedIns.numAttributes());
        // set dataset to instance
        modifiedIns.setDataset(classifierInstances);
        //get the bipartition_key after classify the instance
        int bipartition_key;
        if (classChoice.compareTo("Nominal-Class") == 0) {
            double classify_key = classifier.classifyInstance(modifiedIns);
            String s = classifierInstances.attribute(classifierInstances.numAttributes() - 1).value((int) classify_key);
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

    private int countTrueLabels(Instance instance) {
        int numTrueLabels = 0;
        for (int i = 0; i < numLabels; i++) {
            int labelIndice = labelIndices[i];
            if (instance.dataset().attribute(labelIndice).value((int) instance.value(labelIndice)).equals("1")) {
                numTrueLabels++;
            }
        }
        return numTrueLabels;
    }

    protected Instances transformData(MultiLabelInstances trainingData) throws Exception {
        // initialize  classifier instances
        classifierInstances = RemoveAllLabels.transformInstances(trainingData);
        classifierInstances = new Instances(classifierInstances, 0);
        Attribute target = null;
        if (classChoice.equals("Nominal-Class")) {
            int countTrueLabels = 0;
            Set<Integer> treeSet = new TreeSet();
            for (int instanceIndex = 0; instanceIndex < trainingData.getDataSet().numInstances(); instanceIndex++) {
                countTrueLabels = 0;
                for (int i = 0; i < numLabels; i++) {
                    int labelIndice = labelIndices[i];
                    if (trainingData.getDataSet().attribute(labelIndice).value((int) trainingData.getDataSet().instance(instanceIndex).value(labelIndice)).equals("1")) {
                        countTrueLabels++;
                    }
                }
                treeSet.add(countTrueLabels);
            }
            ArrayList<String> classlabel = new ArrayList<String>();
            for (Integer x : treeSet) {
                classlabel.add(x.toString());
            }
            target = new Attribute("Class", classlabel);
        } else if (classChoice.equals("Numeric-Class")) {
            target = new Attribute("Class");
        }
        classifierInstances.insertAttributeAt(target, classifierInstances.numAttributes());
        classifierInstances.setClassIndex(classifierInstances.numAttributes() - 1);

        // create instances
        if (metaDatasetChoice.equals("Content-Based")) {
            for (int instanceIndex = 0; instanceIndex < trainingData.getNumInstances(); instanceIndex++) {
                Instance instance = trainingData.getDataSet().instance(instanceIndex);
                double[] values = instance.toDoubleArray();
                double[] newValues = new double[classifierInstances.numAttributes()];
                for (int i = 0; i < featureIndices.length; i++) {
                    newValues[i] = values[featureIndices[i]];
                }

                //set the number of true labels of an instance
                int numTrueLabels = countTrueLabels(instance);
                if (classChoice.compareTo("Nominal-Class") == 0) {
                    newValues[newValues.length - 1] = classifierInstances.attribute(classifierInstances.numAttributes() - 1).indexOfValue("" + numTrueLabels);
                } else if (classChoice.compareTo("Numeric-Class") == 0) {
                    newValues[newValues.length - 1] = numTrueLabels;
                }
                Instance newInstance = DataUtils.createInstance(instance, instance.weight(), newValues);
                classifierInstances.add(newInstance);
            }
        } else {
            for (int k = 0; k < kFoldsCV; k++) {
                //Split data to train and test sets
                MultiLabelLearner tempLearner;
                MultiLabelInstances mlTest;
                if (kFoldsCV == 1) {
                    tempLearner = baseLearner;
                    mlTest = trainingData;
                } else {
                    Instances train = trainingData.getDataSet().trainCV(kFoldsCV, k);
                    Instances test = trainingData.getDataSet().testCV(kFoldsCV, k);
                    MultiLabelInstances mlTrain = new MultiLabelInstances(train, trainingData.getLabelsMetaData());
                    mlTest = new MultiLabelInstances(test, trainingData.getLabelsMetaData());
                    tempLearner = foldLearner.makeCopy();
                    tempLearner.build(mlTrain);
                }

                // copy features and labels, set metalabels
                for (int instanceIndex = 0; instanceIndex < mlTest.getDataSet().numInstances(); instanceIndex++) {
                    Instance instance = mlTest.getDataSet().instance(instanceIndex);

                    // initialize new class values
                    double[] newValues = new double[classifierInstances.numAttributes()];

                    // create features
                    valuesX(tempLearner, instance, newValues, metaDatasetChoice);

                    //set the number of true labels of an instance   
                    int numTrueLabels = countTrueLabels(instance);
                    if (classChoice.compareTo("Nominal-Class") == 0) {
                        newValues[newValues.length - 1] = classifierInstances.attribute(classifierInstances.numAttributes() - 1).indexOfValue("" + numTrueLabels);
                    } else if (classChoice.compareTo("Numeric-Class") == 0) {
                        newValues[newValues.length - 1] = numTrueLabels;
                    }

                    // add the new instance to  classifierInstances
                    Instance newInstance = DataUtils.createInstance(mlTest.getDataSet().instance(instanceIndex), mlTest.getDataSet().instance(instanceIndex).weight(), newValues);
                    classifierInstances.add(newInstance);
                }
            }
        }

        return classifierInstances;
    }

    /**
     * Sets the number of folds for internal cv
     *
     * @param f the number of folds
     */
    public void setFolds(int f) {
        kFoldsCV = f;
    }
}
