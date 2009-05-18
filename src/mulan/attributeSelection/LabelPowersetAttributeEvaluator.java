/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.attributeSelection;

import mulan.core.data.MultiLabelInstances;
import mulan.transformations.LabelPowersetTransformation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;

/**
 *
 * @author greg
 */
public class LabelPowersetAttributeEvaluator extends AttributeEvaluator
{
    // number of labels
    int numLabels;
    // The single-label attributed evaluator to use underneath
    AttributeEvaluator baseAttributeEvaluator;

    public LabelPowersetAttributeEvaluator() {
    } 
    
    @Override
    public double evaluateAttribute(int attribute) throws Exception {
        return baseAttributeEvaluator.evaluateAttribute(attribute);
    }

    public void buildEvaluator(MultiLabelInstances mlData) throws Exception
    {
        LabelPowersetTransformation lbTrans = new LabelPowersetTransformation();
        Instances newData = lbTrans.transformInstances(mlData);
        baseAttributeEvaluator.buildEvaluator(newData);
    }

    public void setAttributeEvaluator(AttributeEvaluator x) {
        baseAttributeEvaluator = x;
    }

    @Override
    public void buildEvaluator(Instances arg0) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
