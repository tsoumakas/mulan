/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.transformations;
import java.util.HashSet;
import mulan.core.LabelSet;
import weka.core.Instances;
/**
 * Class that implement the Label powerset (LP) transformation method
 * @author Stavros
 */
public class LabelPowersetTransformation  {

    public Instances transformInstances(Instances data, int numLabels) throws Exception
    {
        Instances newData = null;

        // gather distinct label combinations
        HashSet<LabelSet> labelSets = new HashSet<LabelSet>();
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
        
        RemoveAllLabels rmLabels = new RemoveAllLabels();
        newData = new Instances(rmLabels.transformInstances(data, numLabels));

        // construct class attribute
        CreateNewLabel cnLabels = new CreateNewLabel();
        newData = cnLabels.executeLP(newData, labelSets, numLabels);

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
}
