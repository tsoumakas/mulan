package mulan.classifier;

import weka.core.Instance;

/**
 * Common interface for multi-label ranker.
 * 
 * @author Jozef Vilcek
 */
public interface MultiLabelRanker extends MultiLabelLearner {
	
	/**
	 * Computes ranking of labels for a specified input {@link Instance}.
	 * The ranker model must be build before prediction as called.
	 * 
	 * @param instance instance the input instance for which rankings are computed
	 * @return labels ranking
	 * @see MultiLabelLearner#build(weka.core.Instances)
	 */
	public Ranking rank(Instance instance);

}
