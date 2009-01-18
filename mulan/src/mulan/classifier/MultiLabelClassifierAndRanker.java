package mulan.classifier;

import weka.core.Instance;

/**
 * Common interface of multi-label learner which is able to compute binary predictions and ranking
 * 
 * @author Jozef Vilcek
 */
public interface MultiLabelClassifierAndRanker extends MultiLabelLearner {

	/**
	 * Computes binary prediction of labels and their ranking for a specified 
	 * input {@link Instance}.
	 * The learner model must be build before prediction as called.
	 * 
	 * @param instance the input instance for which binary prediction and 
	 * 				   rankings are computed
	 * @return labels bipartition and ranking
	 * @see MultiLabelLearner#build(weka.core.Instances)
	 */
	public BipartitionAndRanking predictAndRank(Instance instance);
	
}
