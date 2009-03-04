/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.transformations.multiclass;
import java.util.List;
import mulan.transformations.*;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * The inrterface of all transformations methods
 * @author Stavros
 */
public abstract class MultiClassTransformation {

    protected int numOfLabels;
    protected int numPredictors;

    MultiClassTransformation(int numOfLabels) {
        this.numOfLabels = numOfLabels;
    }

    public Instances transformInstances(Instances data) throws Exception {
        numPredictors = data.numAttributes()-numOfLabels;
        
        Instances transformed = new Instances(data, 0);
        
        // delete all labels
        RemoveAllLabels ral = new RemoveAllLabels();
        transformed = ral.transformInstances(data, numOfLabels);

        // add single label attribute
        FastVector classValues = new FastVector(numOfLabels);
        for(int x=0; x<numOfLabels; x++)
            classValues.addElement("Class"+(x+1));
        Attribute newClass = new Attribute("Class", classValues);
        transformed.insertAttributeAt(newClass, transformed.numAttributes());
        transformed.setClassIndex(transformed.numAttributes()-1);

        for (int instanceIndex=0; instanceIndex<data.numInstances(); instanceIndex++) {
            List<Instance> result = transformInstance(data.instance(instanceIndex));
            for (Instance instance : result)
                transformed.add(instance);
        }
        return transformed;
    }

    abstract List<Instance> transformInstance(Instance instance);
}
