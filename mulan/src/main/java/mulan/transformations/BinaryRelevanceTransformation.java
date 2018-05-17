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

import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

import java.io.Serializable;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Class that implements the binary relevance transformation
 *
 * @author Grigorios Tsoumakas
 * @version 2012.05.30
 */
public class BinaryRelevanceTransformation implements Serializable {

    private MultiLabelInstances data;
    private Instances shellBinary;
    private Instances shellNumeric;
    private Remove remove;
    private Add addBinary;
    private Add addNumeric;
    private boolean isContainBianry=false;
    private boolean isContainNumeric=false;

    /**
     * Constructor
     *
     * @param data a multi-label dataset
     */
    public BinaryRelevanceTransformation(MultiLabelInstances data) {
        try {
            this.data = data;
            remove = new Remove();
            int[] labelIndices = data.getLabelIndices();
            int[] indices = new int[labelIndices.length];
            System.arraycopy(labelIndices, 0, indices, 0, labelIndices.length);
            
            for(int i:labelIndices){
            	if(data.getDataSet().get(0).attribute(i).isNominal()){
            		isContainBianry=true;
            	}
            	if(data.getDataSet().get(0).attribute(i).isNumeric()){
            		isContainNumeric=true;
            	}
            	if(isContainBianry&&isContainNumeric){
            		break;
            	}
            }
            if(isContainBianry){
                remove.setAttributeIndicesArray(indices);
                remove.setInvertSelection(false);
                remove.setInputFormat(data.getDataSet());
                shellBinary = Filter.useFilter(data.getDataSet(), remove);
                
                addBinary = new Add();
                addBinary.setAttributeIndex("last");
                addBinary.setNominalLabels("0,1");  
                addBinary.setAttributeName("BinaryRelevanceLabel");
                addBinary.setInputFormat(shellBinary);
                shellBinary = Filter.useFilter(shellBinary, addBinary);
                
                shellBinary.setClassIndex(shellBinary.numAttributes() - 1);
            }
            if(isContainNumeric){
                remove.setAttributeIndicesArray(indices);
                remove.setInvertSelection(false);
                remove.setInputFormat(data.getDataSet());
                shellNumeric = Filter.useFilter(data.getDataSet(), remove);
                
                addNumeric = new Add();
                addNumeric.setAttributeIndex("last");
                addNumeric.setAttributeName("SignleTargetRegressorLabel");
                addNumeric.setInputFormat(shellNumeric);
                shellNumeric = Filter.useFilter(shellNumeric, addNumeric);
                
                shellNumeric.setClassIndex(shellNumeric.numAttributes() - 1);
            }
        } catch (Exception ex) {
            Logger.getLogger(BinaryRelevanceTransformation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Remove all label attributes except labelToKeep
     *
     * @param instance the instance from which labels are to be removed
     * @param labelToKeep the label to keep
     * @return transformed Instance
     */
    public Instance transformInstance(Instance instance, int labelToKeep) {
    	int[] labelIndices = data.getLabelIndices();
    	
    	Instance transformedInstance;
        remove.input(instance);
        transformedInstance = remove.output();
        
        if(instance.attribute(labelIndices[labelToKeep]).isNominal()){
        	 addBinary.input(transformedInstance);
             transformedInstance = addBinary.output();
             transformedInstance.setDataset(shellBinary);
             if (data.getDataSet().attribute(labelIndices[labelToKeep]).value(0).equals("1")) {
                 transformedInstance.setValue(shellBinary.numAttributes() - 1, 1 - instance.value(labelIndices[labelToKeep]));
             } else {
                 transformedInstance.setValue(shellBinary.numAttributes() - 1, instance.value(labelIndices[labelToKeep]));
             }
        }
        else if(instance.attribute(labelIndices[labelToKeep]).isNumeric()){
       	 	addNumeric.input(transformedInstance);
            transformedInstance = addNumeric.output(); 
            transformedInstance.setDataset(shellNumeric);
            transformedInstance.setValue(shellNumeric.numAttributes() - 1, instance.value(labelIndices[labelToKeep]));
            
        }
        return transformedInstance;
    }

    /**
     * Remove all label attributes except labelToKeep
     *
     * @param labelToKeep the label to keep
     * @return transformed Instances object
     * @throws Exception when removal fails
     */
    public Instances transformInstances(int labelToKeep) throws Exception {
    	int[] labelIndices = data.getLabelIndices();
    	Instances shellCopy=null;
    	
    	if(data.getDataSet().get(0).attribute(labelIndices[labelToKeep]).isNominal()){
    		shellCopy= new Instances(this.shellBinary);
    		boolean order10 = false;
            if (data.getDataSet().attribute(labelIndices[labelToKeep]).value(0).equals("1")) {
                order10 = true;
            }
            for (int j = 0; j < shellCopy.numInstances(); j++) {
                if (order10) {
                    shellCopy.instance(j).setValue(shellCopy.numAttributes() - 1, 1 - data.getDataSet().instance(j).value(labelIndices[labelToKeep]));
                } else {
                    shellCopy.instance(j).setValue(shellCopy.numAttributes() - 1, data.getDataSet().instance(j).value(labelIndices[labelToKeep]));
                }
            }
    	}
    	else if(data.getDataSet().get(0).attribute(labelIndices[labelToKeep]).isNumeric()){
    		shellCopy= new Instances(this.shellNumeric);
    		for (int j = 0; j < shellCopy.numInstances(); j++) {
    			shellCopy.instance(j).setValue(shellCopy.numAttributes() - 1, data.getDataSet().instance(j).value(labelIndices[labelToKeep]));
    		}
    	}
        return shellCopy;
    }

    /**
     * Remove all label attributes except that at indexOfLabelToKeep
     *
     * @param train -
     * @param labelIndices - 
     * @param indexToKeep the label to keep
     * @return transformed Instances object
     * @throws Exception when removal fails
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
        return Filter.useFilter(train, remove);
    }

    /**
     * Remove all label attributes except label at position indexToKeep
     *
     * @param instance the instance from which labels are to be removed
     * @param labelIndices the label indices to remove
     * @param indexToKeep the label to keep
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

        return DataUtils.createInstance(instance, 1, transformedValues);
    }
}
