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

import mulan.Transformations;
import mulan.LabelSet;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

@SuppressWarnings("serial")
/**
 * Class that implements a label powerset classifier <p>
 *
 * @author Grigorios Tsoumakas 
 * @author Robert Friberg
 * @version $Revision: 0.02 $ 
 */
public class LabelPowersetClassifier extends AbstractMultiLabelClassifier
{
    protected Instances metadataTrain;
    protected Instances metadataTest;

    public LabelPowersetClassifier(Classifier classifier, int numLabels) throws Exception
    {
        super(numLabels);
        this.baseClassifier = makeCopy(classifier);
    }

    public int indexOfClassValue(String value)
    {
        return metadataTest.attribute(metadataTest.numAttributes()-1).indexOfValue(value);
    }

    public void buildClassifier(Instances train) throws Exception
    {
        //super.buildClassifier(train);
        if (baseClassifier == null) baseClassifier = defaultClassifier(); 
        metadataTrain = new Instances(train, 0);

        Transformations trans = new Transformations(numLabels);
        Instances newTrain = trans.LabelPowerset(train);
        
        // build classifier on new dataset
        baseClassifier.buildClassifier(newTrain);

        // keep the header of new dataset for classification
        metadataTest = new Instances(newTrain, 0);
    }

    /**
     * Remove all label attributes 
     */
    private Instances removeAllLabels(Instances train) throws Exception
    {
        //Indices of attributes to remove
        int indices[] = new int[numLabels];
        int k = 0;
        for (int j = 0; j < numLabels; j++)
        {
            indices[k] = train.numAttributes() - numLabels + j;
            k++;
        }

        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indices);
        remove.setInputFormat(train);
        remove.setInvertSelection(true);
        Instances result = Filter.useFilter(train, remove);
        result.setClassIndex(result.numAttributes() - 1);
        return result;
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
        double predictions[];
        double confidences[] = new double[numLabels];

        double[] distribution = distributionFromBaseClassifier(instance);

        int classIndex = Utils.maxIndex(distribution);
        double confidence = distribution[classIndex];

        String strClass = (metadataTest.classAttribute()).value(classIndex);
        LabelSet labels = LabelSet.fromBitString(strClass);
        predictions = labels.toDoubleArray();
        
        for (int i = 0; i < numLabels; i++)
        {
                if (predictions[i] == 1) confidences[i] = confidence;
                else confidences[i] = 1-confidence;
        }
        
        Prediction result = new Prediction(predictions, confidences);
        
        return result;
    }

}

