package mulan.classifier;

import weka.core.Instances;

/**
 * Common interface for all multi-label learner types.
 *  
 * @author Jozef Vilcek
 */
public interface MultiLabelLearner {

	/**
	 * Returns a number of labels the learner is configured for.
	 * Label attributes are assumed to be the last ones in {@link Instances} training data.
	 * @return number of labels
	 */
	public int getNumLabels();
	
	/**
	 * Builds the learner model. 
	 *  
	 * @param instances set of training data, upon which the classifier should be build
	 * @throws Exception if classifier was not created successfully
	 */
	public void build(Instances instances) throws Exception;
}
