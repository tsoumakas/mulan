package mulan.attributeSelection;

import mulan.core.data.MultiLabelInstances;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;
import weka.core.Range;

/**
 * 
 * @author greg
 */
public class Ranker
{
	
	/**
	 * Calls a specifier {@link AttributeEvaluator} to evaluate each feature attribute
	 * of specified {@link MultiLabelInstances} data set. 
	 * Internally it uses {@link weka.attributeSelection.Ranker}, where 
	 * {@link weka.attributeSelection.Ranker#setStartSet(String)} is preset with range of
	 * only feature attributes indices.
	 * 
	 * @param attributeEval the attribute evaluator to guide the search
	 * @param mlData the multi-label instances data set
	 * @return an array (not necessarily ordered) of selected attribute indexes
	 * @throws Exception if an error occur in search
	 */
    public int[] search(AttributeEvaluator attributeEval, MultiLabelInstances mlData) throws Exception
    {
        Instances data = mlData.getDataSet();
        String startSet = Range.indicesToRangeList(mlData.getLabelIndices());
	    weka.attributeSelection.Ranker wekaRanker = new weka.attributeSelection.Ranker();
	    wekaRanker.setStartSet(startSet);
	    
        return wekaRanker.search((ASEvaluation)attributeEval, data);
    }
}
