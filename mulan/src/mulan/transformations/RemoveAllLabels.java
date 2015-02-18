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
package mulan.transformations;

import mulan.data.MultiLabelInstances;
import mulan.data.DataUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * This transformation removes all the label attributes from a 
 * multi-label dataset
 * 
 * @author Stavros Mpakirtzoglou
 * @author Grigorios Tsoumakas
 * @version 2012.02.02
 */
public class RemoveAllLabels {

    /**
     * 
     * @param mlData multi-label data
     * @return transformed instances
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public static Instances transformInstances(MultiLabelInstances mlData) throws Exception {
        Instances result;
        result = transformInstances(mlData.getDataSet(), mlData.getLabelIndices());
        return result;
    }

    /**
     * Removes the labels from a set of instances
     * 
     * @param dataSet a multi-label dataset
     * @param labelIndices the indices of the labels
     * @return the transformed dataset
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public static Instances transformInstances(Instances dataSet, int[] labelIndices) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(labelIndices);
        remove.setInputFormat(dataSet);
        Instances result = Filter.useFilter(dataSet, remove);
        return result;
    }

    /**
     * 
     * @param instance the instance to be transformed
     * @param labelIndices the indices of labels to transform 
     * @return tranformed instance
     */
    public static Instance transformInstance(Instance instance, int[] labelIndices) {
        double[] oldValues = instance.toDoubleArray();
        double[] newValues = new double[oldValues.length - labelIndices.length];
        int counter1 = 0;
        int counter2 = 0;
        for (int i = 0; i < oldValues.length; i++) {
            if (counter1 < labelIndices.length) {
                if (i == labelIndices[counter1]) {
                    counter1++;
                    continue;
                }
            }
            newValues[counter2] = oldValues[i];
            counter2++;
        }
        return DataUtils.createInstance(instance, instance.weight(), newValues);
    }
}