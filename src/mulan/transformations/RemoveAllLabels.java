/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.transformations;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
/**
 *
 * @author Stavros
 */
public class RemoveAllLabels {

    public static Instances transformInstances(MultiLabelInstances mlData) throws Exception
    {
        Instances result;
        result = transformInstances(mlData.getDataSet(), mlData.getLabelIndices());
        return result;
    }

    public static Instances transformInstances(Instances dataSet, int[] labelIndices) throws Exception
    {
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(labelIndices);
        remove.setInputFormat(dataSet);
        Instances result = Filter.useFilter(dataSet, remove);
        return result;
    }

    public static Instance transformInstance(Instance instance, int[] labelIndices)
    {
        double[] oldValues = instance.toDoubleArray();
        double[] newValues = new double[oldValues.length-labelIndices.length];
        int counter1 = 0;
        int counter2 = 0;
        for (int i=0; i<oldValues.length; i++)
        {
            if (i == labelIndices[counter1]) {
                counter1++;
                continue;
            }
            newValues[counter2] = oldValues[i];
            counter2++;
        }
        return new Instance(instance.weight(), newValues);        
    }
    /*
    public static Instance transformInstance(Instance instance, int[] labelIndices) throws Exception
    {
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(labelIndices);
        remove.setInputFormat(instance.dataset());
        remove.input(instance);
        remove.batchFinished();
        return remove.output();
    }*/

}
