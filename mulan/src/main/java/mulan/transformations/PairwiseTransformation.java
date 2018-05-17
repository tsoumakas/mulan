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

import java.io.Serializable;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.data.MultiLabelInstances;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class that implements the pairwise transformation
 *
 * @author Grigorios Tsoumakas
 * @version 2012.10.30
 */
public class PairwiseTransformation implements Serializable {

    private MultiLabelInstances data;
    private Instances shell;
    private Remove remove;
    private Add add;

    /**
     * Initializes a pairwise transformation object
     *
     * @param data a multi-label dataset
     */
    public PairwiseTransformation(MultiLabelInstances data) {
        try {
            this.data = data;
            remove = new Remove();
            int[] labelIndices = data.getLabelIndices();
            int[] indices = new int[labelIndices.length];
            System.arraycopy(labelIndices, 0, indices, 0, labelIndices.length);
            remove.setAttributeIndicesArray(indices);
            remove.setInvertSelection(false);
            remove.setInputFormat(data.getDataSet());
            shell = Filter.useFilter(data.getDataSet(), remove);
            add = new Add();
            add.setAttributeIndex("last");
            add.setNominalLabels("0,1");
            add.setAttributeName("PairwiseLabel");
            add.setInputFormat(shell);
            shell = Filter.useFilter(shell, add);
            shell.setClassIndex(shell.numAttributes() - 1);
        } catch (Exception ex) {
            Logger.getLogger(PairwiseTransformation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Keeps the predictors and adds one binary attrinute
     *
     * @param instance the instance to be transformed
     * @return transformed Instance
     */
    public Instance transformInstance(Instance instance) {
        Instance transformedInstance;
        remove.input(instance);
        transformedInstance = remove.output();
        add.input(transformedInstance);
        transformedInstance = add.output();
        transformedInstance.setDataset(shell);
        return transformedInstance;
    }

    /**
     * Prepares the training data for two labels. 
     *
     * @param label1 first label
     * @param label2 second label
     * @return transformed Instances object
     */
    public Instances transformInstances(int label1, int label2) {
        Instances transformed = new Instances(shell, 0);
        int[] labelIndices = data.getLabelIndices();

        int indexOfTrueLabel1;
        if (data.getDataSet().attribute(labelIndices[label1]).value(0).equals("1")) {
            indexOfTrueLabel1 = 0;
        } else {
            indexOfTrueLabel1 = 1;
        }
        int indexOfTrueLabel2;
        if (data.getDataSet().attribute(labelIndices[label2]).value(0).equals("1")) {
            indexOfTrueLabel2 = 0;
        } else {
            indexOfTrueLabel2 = 1;
        }

        for (int j = 0; j < shell.numInstances(); j++) {
            boolean value1 = ((int) data.getDataSet().instance(j).value(labelIndices[label1]) == indexOfTrueLabel1);
            boolean value2 = ((int) data.getDataSet().instance(j).value(labelIndices[label2]) == indexOfTrueLabel2);
            if (value1 != value2) {
                Instance tempInstance;
                if (shell.instance(j) instanceof SparseInstance) {
                    tempInstance = new SparseInstance(shell.instance(j));
                } else {
                    tempInstance = new DenseInstance(shell.instance(j));
                }
                tempInstance.setDataset(transformed);
                if (value1 == true) {
                    tempInstance.setClassValue(1);
                } 
                else {
                    tempInstance.setClassValue(0);
                }
                transformed.add(tempInstance);
            }
        }
        return transformed;
    }

}