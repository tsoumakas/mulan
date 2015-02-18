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
package mulan.transformations.regression;

import java.io.Serializable;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class that implements the single target transformation.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
 */
public class SingleTargetTransformation implements Serializable {

    private static final long serialVersionUID = 1L;

    private MultiLabelInstances data;
    private Instances shell;
    private Remove remove;
    private Add add;

    /**
     * Constructor
     * 
     * @param mlData a multi-target regression dataset
     */
    public SingleTargetTransformation(MultiLabelInstances mlData) {
        try {
            this.data = mlData;
            // any changes are applied to a copy of the original dataset
            Instances data = new Instances(mlData.getDataSet());
            remove = new Remove();
            int[] labelIndices = mlData.getLabelIndices();
            int[] indices = new int[labelIndices.length];
            System.arraycopy(labelIndices, 0, indices, 0, labelIndices.length);
            remove.setAttributeIndicesArray(indices);
            remove.setInvertSelection(false);
            remove.setInputFormat(data);
            shell = Filter.useFilter(data, remove);
            add = new Add();
            add.setAttributeIndex("last");
            add.setAttributeName("SingleTarget");
            add.setInputFormat(shell);
            shell = Filter.useFilter(shell, add);
            shell.setClassIndex(shell.numAttributes() - 1);
        } catch (Exception ex) {
            Logger.getLogger(SingleTargetTransformation.class.getName())
                    .log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Remove all target attributes except labelToKeep
     * 
     * @param instance the instance to be transformed
     * @param targetToKeep the target to keep
     * @return transformed Instance
     */
    public Instance transformInstance(Instance instance, int targetToKeep) {
        Instance transformedInstance;
        remove.input(instance);
        transformedInstance = remove.output();
        add.input(transformedInstance);
        transformedInstance = add.output();
        transformedInstance.setDataset(shell);

        int[] targetIndices = data.getLabelIndices();
        transformedInstance.setValue(shell.numAttributes() - 1,
                instance.value(targetIndices[targetToKeep]));

        return transformedInstance;
    }

    /**
     * Remove all target attributes except targetToKeep
     * 
     * @param targetToKeep the target to keep
     * @return transformed Instances object
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public Instances transformInstances(int targetToKeep) throws Exception {
        Instances shellCopy = new Instances(this.shell);
        int[] labelIndices = data.getLabelIndices();

        for (int j = 0; j < shellCopy.numInstances(); j++) {
            shellCopy.instance(j).setValue(shellCopy.numAttributes() - 1,
                    data.getDataSet().instance(j).value(labelIndices[targetToKeep]));

        }
        return shellCopy;
    }

    /**
     * Remove all target attributes except that at indexToKeep.
     * 
     * @param train -
     * @param targetIndices the target indices to tranform
     * @param indexToKeep the target to keep
     * @return transformed Instances object
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public static Instances transformInstances(Instances train, int[] targetIndices, int indexToKeep)
            throws Exception {
        int numTargets = targetIndices.length;

        train.setClassIndex(indexToKeep);

        // Indices of attributes to remove
        int[] indicesToRemove = new int[numTargets - 1];
        int counter2 = 0;
        for (int counter1 = 0; counter1 < numTargets; counter1++) {
            if (targetIndices[counter1] != indexToKeep) {
                indicesToRemove[counter2] = targetIndices[counter1];
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
     * Remove all target attributes except target at position indexToKeep.
     * 
     * @param instance the instance to be transformed
     * @param targetIndices the target indices to transform
     * @param indexToKeep the target to keep
     * @return transformed Instance
     */
    public static Instance transformInstance(Instance instance, int[] targetIndices, int indexToKeep) {
        double[] values = instance.toDoubleArray();
        double[] transformedValues = new double[values.length - targetIndices.length + 1];

        int counterTransformed = 0;
        boolean isTarget = false;

        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < targetIndices.length; j++) {
                if (i == targetIndices[j] && i != indexToKeep) {
                    isTarget = true;
                    break;
                }
            }

            if (!isTarget) {
                transformedValues[counterTransformed] = instance.value(i);
                counterTransformed++;
            }
            isTarget = false;
        }

        Instance transformedInstance = DataUtils.createInstance(instance, 1, transformedValues);
        return transformedInstance;
    }
}