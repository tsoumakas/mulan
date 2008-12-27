package mulan.classifier;

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

import java.util.Arrays;
import java.util.Random;

import mulan.core.LabelSet;
import mulan.core.Transformations;
import mulan.core.Util;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

@SuppressWarnings("serial")
/**
 * Class that implements a label powerset classifier <p>
 *
 * @author Grigorios Tsoumakas 
 * @author Robert Friberg
 * @version $Revision: 0.05 $ 
 */
public class LabelPowerset extends TransformationBasedMultiLabelClassifier
{
    /**
     * The confidence values for each label are calculated in the following ways
     * 1: Confidence of 1/0 for all labels (1 if label true, 0 if label is false)
     * 2: Confidence calculated based on the distribution of probabilities 
     *    obtained from the base classifier, as introduced by the PPT algorithm
     */
    private int confidenceCalculationMethod = 2;

    /**
     * Whether the method introduced by the PPT algorithm will be used to
     * actually get the 1/0 output predictions based on the confidences
     * (requires a threshold)
     */
    protected boolean makePredictionsBasedOnConfidences=false;

    /**
     * Threshold used for deciding the 1/0 output value of each label based on
     * the corresponding confidences as calculated by the method introduced in
     * the PPT algorithm
     */
    protected double threshold=0.5;

    protected Instances metadataTrain;
    protected Instances metadataTest;
        
    protected Random Rand;

    public LabelPowerset(Classifier classifier, int numLabels) throws Exception
    {
        super(classifier, numLabels);
        Rand = new Random(1);
    }
   
    public void setMakePredictionsBasedOnConfidences(boolean value)
    {
        makePredictionsBasedOnConfidences = value;
    }

    /**
     * Setting a seed for random selection in case of ties during prediction
     */
    public void setSeed(int s) 
    {
        Rand = new Random(s);
    }
    

    public void setThreshold(double t)
    {
        threshold = t;
    }

    public void setConfidenceCalculationMethod(int method)
    {
        if (method == 1 || method == 2)
            confidenceCalculationMethod = method;
    }    
    
    public int indexOfClassValue(String value)
    {
        return metadataTest.attribute(metadataTest.numAttributes()-1).indexOfValue(value);
    }
    
    @Override
    public void buildClassifier(Instances train) throws Exception
    {
        metadataTrain = new Instances(train, 0);

        Transformations trans = new Transformations(numLabels);
        Instances newTrain = trans.LabelPowerset(train);               
        
        // keep the header of new dataset for classification
        metadataTest = new Instances(newTrain, 0);

        // check for unary class
        if (newTrain.attribute(newTrain.numAttributes()-1).numValues() > 1) {
            // build classifier on new dataset
            baseClassifier.buildClassifier(newTrain);
        }

    }

    /**
     * Extracted from makePrediction to support label subset mapping which 
     * needs access to this distribution. The distribution contains the prior
     * probabilities of all the label subsets when a probabilistic base
     * classifier is used.
     */
    protected double[] distributionFromBaseClassifier(Instance instance) throws Exception
    {
        //System.out.println("old instance:" + instance.toString());
        Instance newInstance;
        if (instance instanceof SparseInstance) 
            newInstance = (SparseInstance) instance.copy();            
        else 
            newInstance = (Instance) instance.copy();

        int numAttributes = instance.numAttributes();
        newInstance.setDataset(null);
        for (int i=0; i<numLabels-1; i++)
            newInstance.deleteAttributeAt(numAttributes-1-i);
        newInstance.setDataset(metadataTest);
        //System.out.println("new instance:" + newInstance.toString());
        
        return baseClassifier.distributionForInstance(newInstance); 		
    }

    public Prediction makePrediction(Instance instance) throws Exception {
        double predictions[] = null;
        double confidences[] = null;

        // check for unary class
        if (metadataTest.attribute(metadataTest.numAttributes()-1).numValues() == 1) {            
            String strClass = (metadataTest.classAttribute()).value(0);
            LabelSet labels = LabelSet.fromBitString(strClass);
            predictions = labels.toDoubleArray();
            confidences = Arrays.copyOf(predictions, predictions.length);
        } else {        
            double[] distribution = distributionFromBaseClassifier(instance);
            
            int classIndex = Util.RandomIndexOfMax(distribution,Rand);
            String strClass = (metadataTest.classAttribute()).value(classIndex);
            LabelSet labels = LabelSet.fromBitString(strClass);
            predictions = labels.toDoubleArray();
            
            switch (confidenceCalculationMethod) 
            {
                case 1: confidences = Arrays.copyOf(predictions, predictions.length);
                        break;
                case 2: confidences = new double[numLabels]; 
                        for (int i=0; i<distribution.length; i++)
                        {
                            strClass = (metadataTest.classAttribute()).value(i);
                            labels = LabelSet.fromBitString(strClass);
                            double[] predictionsTemp = labels.toDoubleArray();
                            double confidence = distribution[i];
                            for (int j=0; j<numLabels;j++)
                                if (predictionsTemp[j] == 1)
                                    confidences[j] += confidence;
                        }    
            }
        }

        if (makePredictionsBasedOnConfidences)
        {
            for (int i=0; i<confidences.length; i++)
                if (confidences[i] > threshold)
                    predictions[i] = 1;
                else
                    predictions[i] = 0;
        }

        Prediction result = new Prediction(predictions, confidences);
        
        return result;
    }

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

}

