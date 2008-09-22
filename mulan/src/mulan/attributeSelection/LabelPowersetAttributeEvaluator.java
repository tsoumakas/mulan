/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.attributeSelection;

import mulan.core.Transformations;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;

/**
 *
 * @author greg
 */
public class LabelPowersetAttributeEvaluator extends AttributeEvaluator {
    // number of labels
    int numLabels;
    // The single-label attributed evaluator to use underneath
    AttributeEvaluator baseAttributeEvaluator;
    
    public LabelPowersetAttributeEvaluator(int l) {
        numLabels = l;
    } 
    
    @Override
    public double evaluateAttribute(int attribute) throws Exception {
        return baseAttributeEvaluator.evaluateAttribute(attribute);
    }

    @Override
    public void buildEvaluator(Instances data) throws Exception {
        Transformations trans = new Transformations(numLabels);
        Instances newData = trans.LabelPowerset(data);
        baseAttributeEvaluator.buildEvaluator(newData);
    }

    public void setAttributeEvaluator(AttributeEvaluator x) {
        baseAttributeEvaluator = x;
    }
}
