/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.attributeSelection;

import mulan.core.data.MultiLabelInstances;
import mulan.transformations.LabelPowersetTransformation;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;

/**
 *
 * @author greg
 */
public class LabelPowersetAttributeEvaluator extends ASEvaluation implements AttributeEvaluator
{
	private static final long serialVersionUID = -6751310731928159207L;
	// The single-label attributed evaluator to use underneath
    private AttributeEvaluator baseAttributeEvaluator;

    @Override
    public double evaluateAttribute(int attribute) throws Exception {
        return baseAttributeEvaluator.evaluateAttribute(attribute);
    }

    public void buildEvaluator(MultiLabelInstances mlData) throws Exception
    {
        LabelPowersetTransformation lbTrans = new LabelPowersetTransformation();
        Instances newData = lbTrans.transformInstances(mlData);
        ((ASEvaluation)baseAttributeEvaluator).buildEvaluator(newData);
    }

    public void setAttributeEvaluator(AttributeEvaluator x) {
        baseAttributeEvaluator = x;
    }

	@Override
	public void buildEvaluator(Instances data) throws Exception {
		throw new UnsupportedOperationException("The operation is not supported. " +
				"Use 'buildEvaluator(MultiLabelInstances)' API nstead.");
	}
}
