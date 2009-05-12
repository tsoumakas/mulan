/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.transformations;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
/**
 *
 * @author Stavros
 */
public class RemoveAllLabels {

    public static Instances transformInstances(Instances dataSet, int[] labelIndices) throws Exception
    {
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(labelIndices);
        remove.setInputFormat(dataSet);
        Instances result = Filter.useFilter(dataSet, remove);
        return result;
    }


    public static Instance transformInstance(Instance instance, int[] labelIndices) throws Exception
    {
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(labelIndices);
        remove.setInputFormat(instance.dataset());
        remove.input(instance);
        remove.batchFinished();
        return remove.output();
    }

}
