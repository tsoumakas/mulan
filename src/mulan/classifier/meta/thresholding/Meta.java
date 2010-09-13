/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */package mulan.classifier.meta.thresholding;

import java.util.ArrayList;
import java.util.Collections;

import mulan.classifier.meta.*;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Base class for instance-based prediction of a bipartition from
 * the labels' scores
 *
 * @author Marios Ioannou
 * @author George Sakkas
 * @author Grigorios Tsoumakas
 * @version 12 September 2010
 */

public abstract class Meta extends MultiLabelMetaLearner {
    /** the classifier to learn the number of top labels or the threshold */
    protected Classifier classifier;

    /** the training instances for the single-label model */
    protected Instances classifierInstances;

    /** the type for constructing the meta dataset*/
    protected String metaDatasetChoice;

    /**the number of folds for cross validation*/
    protected int kFoldsCV;

    /** clean multi-label learner for cross-validation  */
    protected MultiLabelLearner foldLearner;

    /**
     * Constructor that initializes the learner 
     *
     * @param baseLearner the MultiLabelLearner
     * @param classifier the binary classification
     * @param kFolds the number of folds for cross validation
     */
    public Meta(MultiLabelLearner baseLearner, Classifier aClassifier, String aMetaDatasetChoice) {
        super(baseLearner);
        metaDatasetChoice = aMetaDatasetChoice;
        classifier = aClassifier;
    }

    public Classifier getClassifier() {
        return classifier;
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
     * A method that fill the array "newValues"
     *
     * @param mlTest the test dataset
     * @param newValues the array to fill
     * @param xBased the type for constructing the meta dataset
     * @param instanceIndex
     * @throws Exception
     */
    protected void valuesX(MultiLabelLearner learner, Instance instance, double[] newValues, String xBased) throws Exception {
        MultiLabelOutput mlo = null;
        if (metaDatasetChoice.compareTo("Content-Based") == 0) {
            double[] values = instance.toDoubleArray();
            for (int i=0; i<featureIndices.length; i++) 
                newValues[i] = values[featureIndices[i]];
        } else if (metaDatasetChoice.compareTo("Score-Based") == 0) {
            mlo = learner.makePrediction(instance);
            double[] values = mlo.getConfidences();
            System.arraycopy(values, 0, newValues, 0, values.length);
        } else if (metaDatasetChoice.compareTo("Rank-Based") == 0) {
            mlo = learner.makePrediction(instance);
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

        classifierInstances = transformData(trainingData);

        // build the prediction model
        classifier.buildClassifier(classifierInstances);

        // keep just the header information
        classifierInstances = new Instances(classifierInstances, 0);
    }

}