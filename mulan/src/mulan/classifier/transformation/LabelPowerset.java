package mulan.classifier.transformation;

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
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.MultiLabelOutput;
import mulan.core.LabelSet;
import mulan.core.Util;
import mulan.core.data.MultiLabelInstances;
import mulan.transformations.LabelPowersetTransformation;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class that implements a label powerset classifier <p>
 *
 * @author Grigorios Tsoumakas 
 * @author Robert Friberg
 * @version $Revision: 0.05 $ 
 */
public class LabelPowerset extends TransformationBasedMultiLabelLearner 
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
     * actually get the 1/0 output bipartition based on the confidences
     * (requires a threshold)
     */
    protected boolean makePredictionsBasedOnConfidences=false;

    /**
     * Threshold used for deciding the 1/0 output value of each label based on
     * the corresponding confidences as calculated by the method introduced in
     * the PPT algorithm
     */
    protected double threshold=0.5;

    protected LabelPowersetTransformation transformation;
        
    protected Random Rand;

    public LabelPowerset(Classifier classifier) throws Exception
    {
        super(classifier);
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
    /*
    public int indexOfClassValue(String value)
    {
        return transformedFormat.attribute(transformedFormat.numAttributes()-1).indexOfValue(value);
    }*/
    
    protected void buildInternal(MultiLabelInstances mlData) throws Exception
    {
        Instances transformedData;
        transformation = new LabelPowersetTransformation();
        debug("Transforming the training set.");
        transformedData = transformation.transformInstances(mlData);

        //debug("Transformed training set: \n + transformedData.toString());

        // check for unary class
        debug("Building single-label classifier.");
        if (transformedData.attribute(transformedData.numAttributes()-1).numValues() > 1) {
            baseClassifier.buildClassifier(transformedData);
        }
    }

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public MultiLabelOutput makePrediction(Instance instance) throws Exception {
        boolean bipartition[] = null;
        double confidences[] = null;

        // check for unary class
        if (transformation.getTransformedFormat().classAttribute().numValues() == 1)
        {
            String strClass = transformation.getTransformedFormat().classAttribute().value(0);
            LabelSet labelSet = null;
            try {
                labelSet = LabelSet.fromBitString(strClass);
            } catch (Exception ex) {
                Logger.getLogger(LabelPowerset.class.getName()).log(Level.SEVERE, null, ex);
            }
            bipartition = labelSet.toBooleanArray();
            confidences = Arrays.copyOf(labelSet.toDoubleArray(), labelSet.size());
        } else {
            double[] distribution = null;
            try {
                //debug("old instance:" + instance.toString());
                Instance transformedInstance;
                transformedInstance = transformation.transformInstance(instance, labelIndices);
                distribution = baseClassifier.distributionForInstance(transformedInstance);
                //debug(Arrays.toString(distribution));
            } catch (Exception ex) {
                Logger.getLogger(LabelPowerset.class.getName()).log(Level.SEVERE, null, ex);
            }
            int classIndex = Util.RandomIndexOfMax(distribution,Rand);
            //debug("" + classIndex);
            String strClass = (transformation.getTransformedFormat().classAttribute()).value(classIndex);
            LabelSet labelSet = null;
            try {
                labelSet = LabelSet.fromBitString(strClass);
            } catch (Exception ex) {
                Logger.getLogger(LabelPowerset.class.getName()).log(Level.SEVERE, null, ex);
            }

            bipartition = labelSet.toBooleanArray();
            //debug(Arrays.toString(bipartition));

            switch (confidenceCalculationMethod)
            {
                case 1: confidences = Arrays.copyOf(labelSet.toDoubleArray(), labelSet.size());
                        break;
                case 2: confidences = new double[numLabels];
                        for (int i=0; i<distribution.length; i++)
                        {
                            strClass = (transformation.getTransformedFormat().classAttribute()).value(i);
                            try {
                                labelSet = LabelSet.fromBitString(strClass);
                            } catch (Exception ex) {
                                Logger.getLogger(LabelPowerset.class.getName()).log(Level.SEVERE, null, ex);
                            }
                            double[] predictionsTemp = labelSet.toDoubleArray();
                            double confidence = distribution[i];
                            for (int j=0; j<numLabels;j++)
                                if (predictionsTemp[j] == 1)
                                    confidences[j] += confidence;
                        }
            }

            if (makePredictionsBasedOnConfidences)
            {
                for (int i=0; i<confidences.length; i++)
                    if (confidences[i] > threshold)
                        bipartition[i] = true;
                    else
                        bipartition[i] = false;
            }
            
        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
 		return mlo;
    }


}

