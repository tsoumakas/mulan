
package mulan.transformations;
import java.util.HashSet;
import mulan.core.LabelSet;
import weka.core.Attribute;
import weka.core.FastVector;
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
        
        // create class attribute
        FastVector classValues = new FastVector(labelSets.size());
        for(LabelSet subset : labelSets)
            classValues.addElement(subset.toBitString());
        Attribute newClass = new Attribute("class", classValues);

        // remove all labels
        RemoveAllLabels rmLabels = new RemoveAllLabels();
        newData = new Instances(rmLabels.transformInstances(data, numLabels));

        // add new class attribute
        newData.insertAttributeAt(newClass, newData.numAttributes());
        newData.setClassIndex(newData.numAttributes()-1);

        // add class values
        for (int i = 0; i < newData.numInstances(); i++) {
            String strClass = "";
            for (int j = 0; j < numLabels; j++) {
                int index = data.numAttributes()-numLabels+j;
                strClass = strClass + data.attribute(index).value((int) data.instance(i).value(index));
            }
            newData.instance(i).setClassValue(strClass);
        }
        return newData;
    }
}
