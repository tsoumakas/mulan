package mulan.classifier;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Common interface for multi-label classifiers.
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * @author Jozef Vilcek
 */
public interface MultiLabelClassifier
{

	/**
	 * Returns a number of labels the classifier is configured for.
	 * The label attributes are assumed to be the last ones in {@link Instances} training data.
	 * @return
	 */
	public int getNumLabels();
	
	/**
	 * Computes the prediction of labels for a specified input {@link Instance}. 
	 * 
	 * @param instance the input instance for which the prediction is made
	 * @return the prediction
	 * @throws Exception if prediction was not successful
	 */
	public Prediction predict(Instance instance) throws Exception;
	
	/**
	 * Builds the classifier. 
	 *  
	 * @param instances set of training data, upon which the classifier should be build
	 * @throws Exception if classifier was not created successfully
	 */
	public void buildClassifier(Instances instances) throws Exception;

}