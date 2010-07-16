/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */package mulan.classifier.meta.thresholding;

import mulan.classifier.meta.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;
import java.util.TreeSet;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Base class for instance-based prediction of a bipartition from
 * the labels' scores
 *
 * @author Marios Ioannou
 * @author George Sakkas
 * @author Grigorios Tsoumakas
 * @version 0.1
 */

public abstract class Meta extends MultiLabelMetaLearner {
    /** the classifier to learn the number of top labels or the threshold */
    protected Classifier classifier;

    /** the header of the instances */
    protected Instances header;

    /** the type for constructing the meta dataset*/
    protected String metaDatasetChoice;

    /**the number of folds for cross validation*/
    protected int kFoldsCV;

    /**
     * Constructor that initializes the learner 
     *
     * @param baseLearner the MultiLabelLearner
     * @param classifier the binary classification
     * @param kFolds the number of folds for cross validation
     */
    public Meta(MultiLabelLearner baseLearner, Classifier classifier, int kFolds) {
        super(baseLearner);
        metaDatasetChoice = "Content-Based";
        this.classifier = classifier;
        kFoldsCV = kFolds;
    }

    public void chooseContentBased() {
        metaDatasetChoice = "Content-Based";
    }

    public void chooseRankBased() {
        metaDatasetChoice = "Rank-Based";
    }

    public void chooseScoreBased() {
        metaDatasetChoice = "Score-Based";
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public Instances getHeader() {
        return header;
    }

    /**
     * abstract method that transform tha meta dataset
     *
     * @param trainingData the training data set
     * @throws Exception
     */
    protected abstract Instances transformData(MultiLabelInstances trainingData) throws Exception;

    /**
     * A method that modify an instance
     *
     * @param instance to modified
     * @param xBased the type for constructing the meta dataset
     */
    protected Instance modifiedInstanceX(Instance instance, String xBased) throws Exception {
        Instance modifiedIns;
        MultiLabelOutput mlo = null;
        if (xBased.compareTo("Content-Based") == 0) {
            Instance tempInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
            modifiedIns = DataUtils.createInstance(tempInstance, tempInstance.weight(), tempInstance.toDoubleArray());
        } else if (xBased.compareTo("Score-Based") == 0) {
            double[] arrayOfScores = new double[numLabels];
            mlo = baseLearner.makePrediction(instance);
            arrayOfScores = mlo.getConfidences();
            modifiedIns = DataUtils.createInstance(instance, numLabels);
            for (int i = 0; i < numLabels; i++) {
                modifiedIns.setValue(i, arrayOfScores[i]);
            }
        } else {       //Rank-Based
            double[] arrayOfScores = new double[numLabels];
            mlo = baseLearner.makePrediction(instance);
            arrayOfScores = mlo.getConfidences();
            ArrayList<Double> list = new ArrayList();
            for (int i = 0; i < numLabels; i++) {
                list.add(arrayOfScores[i]);
            }
            Collections.sort(list);
            modifiedIns = DataUtils.createInstance(instance, numLabels);
            int j = numLabels - 1;
            for (Double x : list) {
                modifiedIns.setValue(j, x);
                j--;
            }
        }
        return modifiedIns;
    }

    /**
     * A method that create a fast vector for the header of dataset
     *
     * @param @param trainingData The initial {@link MultiLabelInstances} dataset
     * @param xBased the type for constructing the meta dataset
     * @param xClass the type of the class
     * @return a list of attributes 
     */
    protected ArrayList<Attribute> createFastVector(MultiLabelInstances trainingData, String xBased, String xClass) {

        // copy existing attributes
        ArrayList<Attribute> atts;
        if (xBased.compareTo("Content-Based") == 0) {
            atts = new ArrayList<Attribute>();
            atts = new ArrayList<Attribute>(featureIndices.length+1);
            for (int i = 0; i < trainingData.getDataSet().numAttributes() - numLabels; i++) {
                atts.add(trainingData.getDataSet().attribute(i));   
            }
        } else {     //Score-Based or Rank-Based
            atts = new ArrayList<Attribute>(numLabels);
            for (int i = 0; i < numLabels; i++) {
                Attribute f = new Attribute("Label-" + i);   
                atts.add(f);
            }
        }
        // add metalabel attributes
        if (xClass.compareTo("Nominal-Class") == 0) {
            int countTrueLabels = 0;
            Set<Integer> treeSet = new TreeSet();
            for (int instanceIndex = 0; instanceIndex < trainingData.getDataSet().numInstances(); instanceIndex++) {
                countTrueLabels = 0;
                for (int i = 0; i < numLabels; i++) {
                    int labelIndice = labelIndices[i];
                    if (trainingData.getDataSet().attribute(labelIndice).value((int) trainingData.getDataSet().instance(instanceIndex).value(labelIndice)).equals("1")) {
                        countTrueLabels += 1;
                    }
                }
                treeSet.add(countTrueLabels);
            }
            ArrayList<String> classlabel = new ArrayList<String>();
            for (Integer x : treeSet) {
                classlabel.add(x.toString());
            }
            atts.add(new Attribute("Class", classlabel));
        } else if (xClass.compareTo("Numeric-Class") == 0)
            atts.add(new Attribute("Class"));
        
        return atts;
    }

    /**
     * A method that fill the array "newValues"
     *
     * @param mlTest the test dataset
     * @param newValues the array to fill
     * @param xBased the type for constructing the meta dataset
     * @param instanceIndex
     * @throws Exception
     */
    protected void valuesX(MultiLabelInstances mlTest, double[] newValues, String xBased, int instanceIndex) throws Exception {
        MultiLabelOutput mlo = null;
        if (metaDatasetChoice.compareTo("Content-Based") == 0) {
            double[] values = mlTest.getDataSet().instance(instanceIndex).toDoubleArray();
            System.arraycopy(values, 0, newValues, 0, values.length - numLabels);
        } else if (metaDatasetChoice.compareTo("Score-Based") == 0) {
            mlo = baseLearner.makePrediction(mlTest.getDataSet().instance(instanceIndex));
            double[] values = mlo.getConfidences();
            System.arraycopy(values, 0, newValues, 0, values.length);
        } else if (metaDatasetChoice.compareTo("Rank-Based") == 0) {
            mlo = baseLearner.makePrediction(mlTest.getDataSet().instance(instanceIndex));
            double[] values = mlo.getConfidences();
            ArrayList<Double> list = new ArrayList();
            for (int i = 0; i < numLabels; i++) {
                list.add(values[i]);
            }
            Collections.sort(list);
            int j = numLabels - 1;
            for (Double x : list) {
                newValues[j] = x;
                j--;
            }
        }
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingData) throws Exception {
        // build the base multilabel learner from the original training data
        baseLearner.build(trainingData);

        // transform MultiLabelInstances to classifierInstances
        Instances classifierInstances = transformData(trainingData);

        // build the prediction model
        classifier.buildClassifier(classifierInstances);

        // keep header information
        header = new Instances(classifierInstances, 0);
    }
}