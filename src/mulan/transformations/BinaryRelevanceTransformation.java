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

/*
 *    BinaryRelevanceTransformation.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.transformations;

import java.io.Serializable;

import mulan.data.DataUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Grigorios Tsoumakas
 */
public class BinaryRelevanceTransformation implements Serializable {

    int numOfLabels;

    public BinaryRelevanceTransformation(int num) {
        numOfLabels = num;
    }

    /**
     * Remove all label attributes except labelToKeep
     * @param instance 
     * @param labelToKeep 
     * @return transformed Instance
     */
    public Instance transformInstance(Instance instance, int labelToKeep) {
        Instance newInstance = DataUtils.createInstance(instance, instance.numAttributes());
        newInstance.setDataset(null);
        int numPredictors = instance.numAttributes() - numOfLabels;
        int skipLabel = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            if (labelIndex == labelToKeep) {
                skipLabel++;
                continue;
            }
            newInstance.deleteAttributeAt(numPredictors + skipLabel);
        }
        return newInstance;
    }

    /**
     * Remove all label attributes except labelToKeep
     * @param train 
     * @param labelToKeep 
     * @return transformed Instances object
     * @throws Exception 
     */
    public Instances transformInstances(Instances train, int labelToKeep) throws Exception {
        // Indices of attributes to remove
        int indices[] = new int[numOfLabels - 1];

        int k = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            if (labelIndex != labelToKeep) {
                indices[k] = train.numAttributes() - numOfLabels + labelIndex;
                k++;
            }
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
     * Remove all label attributes except that at indexOfLabelToKeep
     * @param train 
     * @param labelIndices 
     * @param indexToKeep 
     * @return transformed Instances object
     * @throws Exception 
     */
    public static Instances transformInstances(Instances train, int[] labelIndices, int indexToKeep) throws Exception {
        int numLabels = labelIndices.length;

        train.setClassIndex(indexToKeep);


        // Indices of attributes to remove
        int[] indicesToRemove = new int[numLabels - 1];
        int counter2 = 0;
        for (int counter1 = 0; counter1 < numLabels; counter1++) {
            if (labelIndices[counter1] != indexToKeep) {
                indicesToRemove[counter2] = labelIndices[counter1];
                counter2++;
            }
        }

        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indicesToRemove);
        remove.setInputFormat(train);
        remove.setInvertSelection(true);
        Instances result = Filter.useFilter(train, remove);
        return result;
    }

    /**
     * Remove all label attributes except label at position indexToKeep
     * @param instance 
     * @param labelIndices 
     * @param indexToKeep 
     * @return transformed Instance
     */
    public static Instance transformInstance(Instance instance, int[] labelIndices, int indexToKeep) {
        double[] values = instance.toDoubleArray();
        double[] transformedValues = new double[values.length - labelIndices.length + 1];

        int counterTransformed = 0;
        boolean isLabel = false;

        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < labelIndices.length; j++) {
                if (i == labelIndices[j] && i != indexToKeep) {
                    isLabel = true;
                    break;
                }
            }

            if (!isLabel) {
                transformedValues[counterTransformed] = instance.value(i);
                counterTransformed++;
            }
            isLabel = false;
        }

        Instance transformedInstance = DataUtils.createInstance(instance, 1, transformedValues);
        return transformedInstance;
    }
}
