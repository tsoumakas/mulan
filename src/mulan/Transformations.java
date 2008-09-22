/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.core;

import java.util.HashSet;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
/**
 *
 * @author greg
 */
public class Transformations {
    // Number of labels
    int numLabels;    
    
    public Transformations(int l) {
        numLabels = l;
    }
    
    public Instances LabelPowerset(Instances data) throws Exception
    {
        Instances newData = null;

        // gather distinct label combinations
        HashSet<LabelSet> labelSets = new HashSet();
        int numInstances = data.numInstances(); 
        int numPredictors = data.numAttributes() - numLabels;
        for (int i=0; i<numInstances; i++)
        {
            // construct labelset
            double[] dblLabels = new double[numLabels];
            for (int j=0; j<numLabels; j++)
                dblLabels[j] = Double.parseDouble(data.attribute(numPredictors+j).value((int) data.instance(i).value(numPredictors + j)));                         
            LabelSet labelSet = new LabelSet(dblLabels);                   
            
            // add labelset if not already present
            labelSets.add(labelSet);
        }

        // construct class attribute
        FastVector classValues = new FastVector(numLabels);
        for(LabelSet subset : labelSets)
            classValues.addElement(subset.toBitString());
        Attribute newClass = new Attribute("class", classValues);

        // create new instances
        newData = removeAllLabels(data); 
        newData.insertAttributeAt(newClass, newData.numAttributes());
        newData.setClassIndex(newData.numAttributes() - 1);

        // add class values
        for (int i = 0; i < newData.numInstances(); i++) {
            String strClass = "";
                for (int j = 0; j < numLabels; j++)
                    strClass = strClass + data.attribute(data.numAttributes()-numLabels+j).value((int) data.instance(i).
                                      value(data.numAttributes() - numLabels + j));
                    newData.instance(i).setClassValue(strClass);
        }

        return newData;
    }

    public Instances removeAllLabels(Instances data) throws Exception
    {
        //Indices of attributes to remove
        int indices[] = new int[numLabels];
        int k = 0;
        for (int j = 0; j < numLabels; j++)
        {
            indices[k] = data.numAttributes() - numLabels + j;
            k++;
        }

        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indices);
        remove.setInputFormat(data);
        remove.setInvertSelection(true);
        Instances result = Filter.useFilter(data, remove);
        result.setClassIndex(result.numAttributes() - 1);
        return result;
    }

}
