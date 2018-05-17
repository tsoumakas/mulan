package mulan.transformations;

import java.io.Serializable;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class that implements the triple classes transformation of COCOA method
 *
 * @author Bin Liu
 * @version 2012.05.30
 */

public class COCOATripleClassTransformation implements Serializable {

    private MultiLabelInstances data;
    private Instances shell;
    private Remove remove;
    private Add add;

    /**
     * Constructor
     *
     * @param data a multi-label dataset
     */
    public COCOATripleClassTransformation(MultiLabelInstances data) {
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
            add.setNominalLabels("0,1,2");
            add.setAttributeName("TripleClassLabel");
            add.setInputFormat(shell);
            shell = Filter.useFilter(shell, add);
            shell.setClassIndex(shell.numAttributes() - 1);
        } catch (Exception ex) {
            Logger.getLogger(COCOATripleClassTransformation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Remove all label attributes except labelToKeep
     *
     * @param instance the instance from which labels are to be removed
     * @param labelToKeep the label to keep
     * @return transformed Instance
     */
    public Instance transformInstance(Instance instance, int label1, int label2) {
        Instance transformedInstance;
        remove.input(instance);
        transformedInstance = remove.output();
        add.input(transformedInstance);
        transformedInstance = add.output();
        transformedInstance.setDataset(shell);
             	
        String s1=instance.stringValue(label1);
    	String s2=instance.stringValue(label2);
    	if(s1.equals("0")){
    		if(s2.equals("0")){
    			transformedInstance.setClassValue("0");
    		}
    		else if(s2.equals("1")){
    			transformedInstance.setClassValue("1");
    		}
    	}
    	else if(s1.equals("1")){
    		transformedInstance.setClassValue("2");
    	}
        return transformedInstance;
    }

    
    /**
     * Prepares the training data of triple class for two labels.
     *
     * @param label1 first label
     * @param label2 second label
     * @return transformed Instances object
     */
    public Instances transformInstances(int label1, int label2) {
        Instances transformed = new Instances(shell, 0);
        for (int j = 0; j < shell.numInstances(); j++) {
        	Instance tempInstance=transformInstance(data.getDataSet().instance(j), label1, label2);
        	transformed.add(tempInstance);
        }
        return transformed;
    }
    
}