
package mulan.transformations;
import java.util.HashSet;
import mulan.core.data.LabelSet;
import mulan.core.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
/**
 * Class that implement the Label powerset (LP) transformation method
 * @author Stavros
 */
public class LabelPowersetTransformation  {
    private Instances transformedFormat;

    public Instances getTransformedFormat()
    {
        return transformedFormat;
    }

    public Instances transformInstances(MultiLabelInstances mlData) throws Exception
    {
        Instances data = mlData.getDataSet();
        int numLabels = mlData.getNumLabels();
        int[] labelIndices = mlData.getLabelIndices();

        Instances newData = null;

        // gather distinct label combinations
        HashSet<LabelSet> labelSets = new HashSet<LabelSet>();
        int numInstances = data.numInstances();
        for (int i=0; i<numInstances; i++)
        {
            // construct labelset
            double[] dblLabels = new double[numLabels];
            for (int j=0; j<numLabels; j++)
            {
                int index = labelIndices[j];
                dblLabels[j] = Double.parseDouble(data.attribute(index).value((int) data.instance(i).value(index)));
            }
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
        newData = RemoveAllLabels.transformInstances(data, labelIndices);

        // add new class attribute
        newData.insertAttributeAt(newClass, newData.numAttributes());
        newData.setClassIndex(newData.numAttributes()-1);

        // add class values
        for (int i = 0; i < newData.numInstances(); i++) {
            //System.out.println(newData.instance(i).toString());
            String strClass = "";
            for (int j=0; j<numLabels; j++)
            {
                int index = labelIndices[j];
                strClass = strClass + data.attribute(index).value((int) data.instance(i).value(index));
            }
            //System.out.println(strClass);
            newData.instance(i).setClassValue(strClass);
        }
        transformedFormat = new Instances(newData, 0);
        return newData;
    }

    public Instance transformInstance(Instance instance, int[] labelIndices) throws Exception {
        Instance transformedInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
        transformedInstance.setDataset(null);
        transformedInstance.insertAttributeAt(transformedInstance.numAttributes());
        transformedInstance.setDataset(transformedFormat);
        return transformedInstance;
    }
}
