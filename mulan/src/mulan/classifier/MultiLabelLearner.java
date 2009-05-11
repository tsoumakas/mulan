package mulan.classifier;

import mulan.core.data.MultiLabelInstances;
import weka.core.Instance;

/**
 * Common interface for all multi-label learner types.
 *  
 * @author Jozef Vilcek
 */
public interface MultiLabelLearner {
	
    /**
	 * Builds the learner model from specified {@link MultiLabelInstances} data. 
	 *  
	 * @param instances set of training data, upon which the classifier should be build
	 * @throws Exception if classifier was not created successfully
	 */
	public void build(MultiLabelInstances instances) throws Exception;
	
	/**
	 * Creates a deep copy of the given learner using serialization.
	 *
	 * @return a deep copy of the learner
	 * @exception Exception if an error occurs
	 */
	public MultiLabelLearner makeCopy() throws Exception;


	/**
	 * Returns the output .
	 *
	 * @return a deep copy of the learner
	 * @exception Exception if an error occurs
	 */
    public MultiLabelOutput makePrediction(Instance instance) throws Exception;
}
